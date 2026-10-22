"""(1) per-token QK-norm+RoPE + dense path correctness vs torch for 1 layer (attention stub mirrored on CPU),
(2) 28-layer weight-stream timing at T tokens per step.
Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/bench/bench_talker_dense.py --T 32 64 128
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "kernels"))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from cp_graph import MODEL  # noqa: E402
from cp_step_graph import rms  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402
from talker_decode import rope  # noqa: E402
from talker_kernels import rope_tables, talker_dense_step, talker_params  # noqa: E402


def ref_one_layer_stub(talker, x, pos):
    layer = talker.model.layers[0]
    a = layer.self_attn
    inv_freq = 1.0 / (a.rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    res = x
    q, k, v = a.project_qkv(rms(x, layer.input_layernorm))
    q, k = rope(a.q_norm(q), pos, inv_freq), rope(a.k_norm(k), pos, inv_freq)
    att = torch.maximum(v.repeat_interleave(2, 1), q)
    att[:, :8] = torch.maximum(att[:, :8], k)
    h = res + a.o_proj(att.reshape(x.shape[0], -1))
    h = h + layer.mlp(rms(h, layer.post_attention_layernorm))
    return rms(h, talker.model.norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    talker = loader.load_talker(device=torch.device("cpu"), dtype=torch.bfloat16)
    del talker.model.text_embedding, talker.text_projection
    theta = talker.model.layers[0].self_attn.rope_theta

    # (1) correctness, 1 layer
    p1 = talker_params(talker, layers=1)
    eps = p1.pop("eps")
    d1 = {k: v.to("neuron") for k, v in p1.items()}
    T = 64
    x = (torch.randn(T, 2048) * 0.3).to(torch.bfloat16)
    pos = torch.randint(0, 900, (T,))
    cos, sin = rope_tables(pos, theta)
    with torch.no_grad():
        expect = ref_one_layer_stub(talker, x, pos).float()
    got = talker_dense_step(x.t().contiguous().to("neuron"), cos.to("neuron"), sin.to("neuron"), **d1, eps=eps).cpu().float().t()
    print(f"1-layer dense path (per-token RoPE, stub attention) vs torch: rel L2 {((got - expect).norm() / expect.norm()).item():.2e}", flush=True)

    # (2) 28-layer timing
    p = talker_params(talker)
    p.pop("eps")
    d = {k: v.to("neuron") for k, v in p.items()}
    wbytes = sum(v.numel() * v.element_size() for k, v in p.items() if k in ("qkv", "o", "gate_up", "down"))
    for T in a.T:
        x = (torch.randn(2048, T) * 0.3).to(torch.bfloat16).to("neuron")
        cos, sin = (t.to("neuron") for t in rope_tables(torch.randint(0, 900, (T,)), theta))
        t0 = time.perf_counter()
        talker_dense_step(x, cos, sin, **d, eps=eps)
        torch.neuron.synchronize()
        c = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(a.iters):
            talker_dense_step(x, cos, sin, **d, eps=eps)
        torch.neuron.synchronize()
        ms = (time.perf_counter() - t0) / a.iters * 1e3
        print(f"T={T:4d}: {ms:6.2f} ms per 28-layer step (no attention), weights {wbytes / 1e9:.2f} GB -> "
              f"{wbytes / ms / 1e6:.0f} GB/s  (first call {c:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
