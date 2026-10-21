"""Prototype: ONE reusable Code Predictor step graph (step index is a tensor input).

Compiles ~1/16 of the whole-frame graph; a frame = 1 prep call + 16 step calls.
Checks greedy tokens vs the repo's CPU eager whole-frame path and times a frame.
Usage: python neuron/proto/cp_step_graph.py --batch 32 [--opt 1]
"""
import argparse, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from cp_graph import MODEL, WholeFrameCodePredictor, sample_topk  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402

GROUPS = 16


def rms(x, norm):
    v = x.float()
    return (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight.float()).to(x.dtype)


class Prep(nn.Module):
    """past_hidden -> step-0 input; layer0 token -> step-1 input and codec_sum seed."""

    def __init__(self, frame: WholeFrameCodePredictor):
        super().__init__()
        self.f = frame

    def forward(self, past_hidden, layer0_token):
        x0 = self.f.predictor.small_to_mtp_projection(past_hidden)
        return x0, F.embedding(layer0_token, self.f.proj_layer0), F.embedding(layer0_token, self.f.layer0_embed)


class Step(nn.Module):
    def __init__(self, frame: WholeFrameCodePredictor, greedy: bool):
        super().__init__()
        self.f, self.greedy = frame, greedy
        p = frame.predictor
        self.layers = p.model.layers
        self.final_norm = p.model.norm
        pad = torch.zeros_like(frame.proj_residual[:1])
        self.register_buffer("proj_next", torch.cat([frame.proj_residual, pad]))  # [15, 2048, 1024]
        attn = self.layers[0].self_attn
        hd = attn.head_dim
        self.register_buffer("inv_freq", 1.0 / (attn.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)))

    def rope(self, x, step):  # x [B, H, D], scalar position = step
        ang = step.float() * self.inv_freq
        cos, sin = torch.cat((ang.cos(), ang.cos())), torch.cat((ang.sin(), ang.sin()))
        a, b = x.float().chunk(2, -1)
        return (x.float() * cos + torch.cat((-b, a), -1) * sin).to(x.dtype)

    def forward(self, x, kv, step, step1_embed, codec_sum, uniform):
        # x [B,1024]; kv [L,B,2,16,KVH,D]; step int64 [1]
        B = x.shape[0]
        slot = (torch.arange(GROUPS, device=x.device) == step).view(1, GROUPS, 1, 1)  # write mask over [B,16,KVH,D]
        visible = (torch.arange(GROUPS, device=x.device) <= step).view(1, 1, 1, GROUPS)
        h = x
        new_kv = []
        for li, layer in enumerate(self.layers):
            a = layer.self_attn
            res = h
            q, k, v = a.project_qkv(rms(h, layer.input_layernorm))
            q, k = self.rope(a.q_norm(q), step), self.rope(a.k_norm(k), step)
            kl = torch.where(slot, k.unsqueeze(1), kv[li, :, 0])  # [B,16,KVH,D]
            vl = torch.where(slot, v.unsqueeze(1), kv[li, :, 1])
            new_kv.append(torch.stack((kl, vl), 1))
            rep = a.num_heads // a.num_kv_heads
            keys = kl.transpose(1, 2).repeat_interleave(rep, 1).float()      # [B,H,16,D]
            vals = vl.transpose(1, 2).repeat_interleave(rep, 1)
            scores = (q.float().unsqueeze(2) @ keys.transpose(-1, -2)) * a.head_dim ** -0.5  # [B,H,1,16]
            probs = torch.softmax(scores.masked_fill(~visible, -torch.inf), -1).to(vals.dtype)
            att = (probs @ vals).reshape(B, -1)
            h = res + a.o_proj(att)
            h = h + layer.mlp(rms(h, layer.post_attention_layernorm))
        head = (step - 1).clamp(min=0)
        hidden = rms(h, self.final_norm)
        logits = hidden @ self.f.predictor.lm_head_weight.index_select(0, head).squeeze(0).t()
        token = sample_topk(logits, uniform, self.f.temperature, self.f.top_k, self.greedy)
        is_res = step > 0
        add = F.embedding(token, self.f.residual_embed.index_select(0, head).squeeze(0))
        codec_sum = torch.where(is_res, codec_sum + add, codec_sum)
        nxt = F.embedding(token, self.proj_next.index_select(0, head).squeeze(0))
        nxt = torch.where(is_res, nxt, step1_embed)
        return torch.stack(new_kv), token, codec_sum, nxt


def run_frame(prep, step, past_hidden, layer0_token, uniforms, kv0, steps):
    x, step1_embed, codec_sum = prep(past_hidden, layer0_token)
    kv, tokens = kv0, [layer0_token]
    for s in range(GROUPS):
        kv, tok, codec_sum, x = step(x, kv, steps[s], step1_embed, codec_sum, uniforms[:, max(s - 1, 0)])
        if s > 0:
            tokens.append(tok)
    return torch.stack(tokens, 1), codec_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--opt", default=None, help="neuronx-cc -O level override (1/2/3)")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    if args.opt:
        os.environ["NEURON_CC_FLAGS"] = f"{os.environ.get('NEURON_CC_FLAGS', '')} -O{args.opt}".strip()
    torch.manual_seed(0)
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    predictor, layer0 = loader.load_code_predictor(device=torch.device("cpu"), dtype=torch.bfloat16)
    frame = WholeFrameCodePredictor(predictor, layer0, greedy=True).eval()
    B = args.batch
    past_hidden = torch.randn(B, 2048, dtype=torch.bfloat16)
    layer0_token = torch.randint(0, 2048, (B,))
    uniforms = torch.rand(B, 15)
    cp = predictor.model.layers[0].self_attn
    kv0 = torch.zeros(len(predictor.model.layers), B, 2, GROUPS, cp.num_kv_heads, cp.head_dim, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_tokens, ref_sum = frame(past_hidden, layer0_token, uniforms)  # repo math, CPU eager
        prep_m, step_m = Prep(frame), Step(frame, greedy=True)
        cpu_tokens, _ = run_frame(prep_m, step_m, past_hidden, layer0_token, uniforms, kv0,
                                  [torch.tensor([s]) for s in range(GROUPS)])
        print(f"CPU step-graph math vs repo eager: {(cpu_tokens == ref_tokens).float().mean().item() * 100:.2f}% tokens equal")

        dev = "neuron"
        frame.to(dev); step_m.to(dev)
        prep_c = torch.compile(prep_m, backend="neuron", fullgraph=True)
        step_c = torch.compile(step_m, backend="neuron", fullgraph=True)
        ins = (past_hidden.to(dev), layer0_token.to(dev), uniforms.to(dev), kv0.to(dev),
               [torch.tensor([s]).to(dev) for s in range(GROUPS)])
        t0 = time.perf_counter()
        tokens, csum = run_frame(prep_c, step_c, *ins)
        tokens = tokens.cpu()
        print(f"compile (prep + step) + first frame: {time.perf_counter() - t0:.1f} s")
        print(f"Neuron greedy vs repo CPU eager: {(tokens == ref_tokens).float().mean().item() * 100:.2f}% tokens equal; "
              f"per group {[round((tokens[:, g] == ref_tokens[:, g]).float().mean().item(), 2) for g in range(GROUPS)]}")
        print(f"codec_sum max abs diff: {(csum.cpu().float() - ref_sum.float()).abs().max().item():.4f}")
        for _ in range(3):
            run_frame(prep_c, step_c, *ins)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = run_frame(prep_c, step_c, *ins)
        out[0].cpu()
        ms = (time.perf_counter() - t0) / args.iters * 1e3
        print(f"B={B}: {ms:.2f} ms per frame-batch (17 launches; real-time budget 80 ms)")


if __name__ == "__main__":
    main()
