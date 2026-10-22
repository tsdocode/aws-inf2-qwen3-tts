"""cp_frame (one launch per frame) vs the per-step path (16 cp_step launches + compiled _DeviceSampler): same uniforms,
same inputs -> tokens should match (greedy and sampled); also times both.

Usage: NEURON_RT_VISIBLE_CORES=0 python neuron/tests/test_cp_frame.py --batch 32
"""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "kernels"))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
sys.path.insert(0, str(Path(__file__).parents[1] / "engine"))
from cp_graph import MODEL  # noqa: E402
from cp_kernels import cp_frame, cp_step, cp_step_params  # noqa: E402
from neuron_engine import _DeviceSampler  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    torch._dynamo.config.automatic_dynamic_shapes = False
    B, d = a.batch, "neuron"
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    predictor, layer0 = loader.load_code_predictor(device=torch.device("cpu"), dtype=torch.bfloat16)
    with torch.no_grad():
        proj = predictor.small_to_mtp_projection
        proj_l0 = proj(layer0.weight).detach()
        proj_res = torch.stack([proj(e.weight) for e in predictor.model.codec_embedding[:-1]]).detach()
    prm = cp_step_params(predictor)
    eps = prm.pop("eps")
    rope, heads = prm.pop("rope").to(d), prm.pop("lm_heads").to(d)
    w = {k: v.to(d) for k, v in prm.items()}
    proj_l0_d = proj_l0.to(d)
    proj_res_pad = torch.cat([proj_res, torch.zeros_like(proj_res[:1])]).contiguous().to(d)
    proj_res_rows = proj_res.reshape(-1, proj_res.shape[-1]).contiguous().to(d)
    ident = torch.eye(128, dtype=torch.bfloat16).to(d)
    sampler = torch.compile(_DeviceSampler(50), backend="neuron", fullgraph=True)
    L = len(predictor.model.layers)
    g = torch.Generator().manual_seed(0)
    x0 = (torch.randn(B, 1024, generator=g) * 2).to(torch.bfloat16)
    tok0 = torch.randint(0, 2048, (B,), generator=g)
    u = torch.rand(15, B, generator=g)
    inv_t = torch.full((B, 1), 1.0 / 0.9)

    def per_step(greedy):
        kv = torch.zeros(L, 2, 128, 16, 8, B, dtype=torch.bfloat16).to(d)
        x, toks = x0.to(d), []
        gr, it, ud, t0 = torch.full((B,), greedy).to(d), inv_t.to(d), u.to(d), tok0.to(d)
        for st in range(16):
            lt, _ = cp_step(x.t().contiguous(), kv, **w, lm_head=heads[max(st - 1, 0)], rope=rope[st], s=st, eps=eps,
                            with_head=True)
            if st == 0:
                x = F.embedding(t0, proj_l0_d)
                continue
            tok, x = sampler(lt, ud[st - 1], it, gr, proj_res_pad[st - 1])
            toks.append(tok)
        return torch.stack(toks, 1).cpu()

    def frame(greedy):
        kv = torch.zeros(L, 2, 128, 16, 8, B, dtype=torch.bfloat16).to(d)
        out = cp_frame(x0.t().contiguous().to(d), tok0.float().view(B, 1).to(d), kv, **w, lm_heads=heads, rope=rope,
                       proj_l0=proj_l0_d, proj_res=proj_res_rows, u=u.t().contiguous().to(d), inv_t=inv_t.to(d),
                       greedy=torch.full((B, 1), float(greedy)).to(d), ident=ident, eps=eps)
        return out.cpu().round().long()

    for greedy in (True, False):
        ref, got = per_step(greedy), frame(greedy)
        agree = (ref == got).float().mean().item()
        first_bad = [(r, c) for r in range(B) for c in range(15) if ref[r, c] != got[r, c]][:3]
        print(f"greedy={greedy}: token agreement {agree * 100:.1f}% (first mismatches {first_bad})", flush=True)
    for name, fn in (("per-step (16 cp_step + 15 sampler)", per_step), ("cp_frame (1 launch)", frame)):
        fn(False)
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            fn(False)
        torch.neuron.synchronize()
        print(f"{name}: {(time.perf_counter() - t0) / a.iters * 1e3:.2f} ms per frame (B={B}, incl. D2H)")


if __name__ == "__main__":
    main()
