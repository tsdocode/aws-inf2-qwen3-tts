"""Fused CP step kernel vs the CPU step math (proto/cp_step_graph.Step, itself checked against the repo).

Teacher-forced: both sides consume the reference token each step, so per-step error does not compound.
Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/kernels/test_cp_kernels.py --batch 32 --steps 6
"""
import argparse, copy, sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from cp_graph import MODEL, WholeFrameCodePredictor  # noqa: E402
from cp_kernels import SLOTS, cp_step, cp_step_params  # noqa: E402
from cp_step_graph import GROUPS, Step, rms  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402


def ref_layers(step, x, kv, s):
    """Step.forward up to logits (CPU, repo-equivalent math). Returns logits fp32, hidden, new kv."""
    h = x
    new_kv = []
    slot = (torch.arange(GROUPS) == s).view(1, GROUPS, 1, 1)
    visible = (torch.arange(GROUPS) <= s).view(1, 1, 1, GROUPS)
    B = x.shape[0]
    for li, layer in enumerate(step.layers):
        a = layer.self_attn
        res = h
        q, k, v = a.project_qkv(rms(h, layer.input_layernorm))
        st = torch.tensor([s])
        q, k = step.rope(a.q_norm(q), st), step.rope(a.k_norm(k), st)
        kl = torch.where(slot, k.unsqueeze(1), kv[li, :, 0])
        vl = torch.where(slot, v.unsqueeze(1), kv[li, :, 1])
        new_kv.append(torch.stack((kl, vl), 1))
        rep = a.num_heads // a.num_kv_heads
        keys = kl.transpose(1, 2).repeat_interleave(rep, 1).float()
        vals = vl.transpose(1, 2).repeat_interleave(rep, 1)
        sc = (q.float().unsqueeze(2) @ keys.transpose(-1, -2)) * a.head_dim ** -0.5
        probs = torch.softmax(sc.masked_fill(~visible, -torch.inf), -1).to(vals.dtype)
        h = res + a.o_proj((probs @ vals).reshape(B, -1))
        h = h + layer.mlp(rms(h, layer.post_attention_layernorm))
    head = max(s - 1, 0)
    logits = rms(h, step.final_norm) @ step.f.predictor.lm_head_weight[head].t()
    return logits.float(), h, torch.stack(new_kv)


def to_kernel_kv(kv):  # repo [L,B,2,16,KVH,D] -> kernel [L,2,D,16,KVH,B]
    return kv.permute(0, 2, 5, 3, 4, 1).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fp8", default="", help="comma list of FP8 weight groups: qkv,o,gate_up,down,lm_heads")
    ap.add_argument("--steps", type=int, default=6)
    a = ap.parse_args()
    torch.manual_seed(0)
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    predictor, layer0 = loader.load_code_predictor(device=torch.device("cpu"), dtype=torch.bfloat16)
    frame = WholeFrameCodePredictor(predictor, layer0, greedy=True).eval()
    step = Step(frame, greedy=True).eval()
    step32 = copy.deepcopy(step).float()  # fp32 oracle: separates bf16 rounding from kernel bugs
    fp8 = [g for g in a.fp8.split(",") if g]
    prm = cp_step_params(predictor, fp8=fp8)
    eps = prm.pop("eps")
    heads_s_dev = prm.pop("lm_heads_s").to("neuron") if "lm_heads_s" in prm else None
    dev = {k: v.to("neuron") for k, v in prm.items() if k not in ("rope", "lm_heads")}
    rope_dev = prm["rope"].to("neuron")
    heads_dev = prm["lm_heads"].to("neuron")

    B = a.batch
    L = len(predictor.model.layers)
    attn = predictor.model.layers[0].self_attn
    kv = torch.zeros(L, B, 2, SLOTS, attn.num_kv_heads, attn.head_dim, dtype=torch.bfloat16)
    kv_dev = to_kernel_kv(kv).to("neuron")
    with torch.no_grad():
        x = predictor.small_to_mtp_projection((torch.randn(B, 2048) * 0.5).to(torch.bfloat16))
        tokens = torch.randint(0, 2048, (B,))
        kv32 = None
        for s in range(a.steps):
            if s > 0:  # teacher forcing: next input = projected embedding of the reference token
                x = F.embedding(tokens, frame.proj_layer0 if s == 1 else frame.proj_residual[s - 2])
            logits32, _, kv32 = ref_layers(step32, x.float(), kv32 if s else kv.float(), s)
            logits_ref, hidden_ref, kv = ref_layers(step, x, kv, s)
            logitsT, hidden = cp_step(x.t().contiguous().to("neuron"), kv_dev, **dev, lm_head=heads_dev[max(s - 1, 0)],
                                      rope=rope_dev[s], s=s, eps=eps,
                                      **(dict(lm_head_s=heads_s_dev[max(s - 1, 0)]) if heads_s_dev is not None else {}))
            got = logitsT.cpu().float().t()
            rel = ((got - logits_ref).norm() / logits_ref.norm()).item()
            kv_err = ((kv_dev.cpu().float() - to_kernel_kv(kv).float()).abs().max()).item()
            r32_cpu = ((logits_ref - logits32).norm() / logits32.norm()).item()
            r32_k = ((got - logits32).norm() / logits32.norm()).item()
            a32_cpu = (logits_ref.argmax(-1) == logits32.argmax(-1)).float().mean().item() * 100
            a32_k = (got.argmax(-1) == logits32.argmax(-1)).float().mean().item() * 100
            print(f"step {s:2d}: vs fp32 oracle -> CPU bf16 {r32_cpu:.2e} ({a32_cpu:5.1f}% argmax) | kernel {r32_k:.2e} "
                  f"({a32_k:5.1f}% argmax) | kernel vs CPU bf16 {rel:.2e} | kv diff {kv_err:.3f}", flush=True)
            tokens = logits_ref.argmax(-1)


if __name__ == "__main__":
    main()
