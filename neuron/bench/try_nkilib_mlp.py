"""Does NKI-Lib's TKG MLP (fused add + RMSNorm + SwiGLU) run correctly and fast on inf2 (gen2)?

Compares against torch math and the torch.compile(backend="neuron") baseline at decode shapes.
Usage: python neuron/bench/try_nkilib_mlp.py --tokens 32 --hidden 1024 --inter 3072
"""
import argparse, time

import torch
import torch._dynamo
import torch.nn.functional as F
from nkilib.core.mlp.mlp import mlp as nkl_mlp
from nkilib.core.utils.common_types import ActFnType, NormType

torch._dynamo.config.automatic_dynamic_shapes = False


def ref_mlp(x, residual, norm_w, wg, wu, wd, eps):
    h = (x.float() + residual.float())
    n = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w.float()).to(x.dtype)
    return (F.silu(n @ wg) * (n @ wu)) @ wd


def bench(fn, iters=50):
    for _ in range(5):
        out = fn()
    (out[0] if isinstance(out, (list, tuple)) else out).cpu()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    (out[0] if isinstance(out, (list, tuple)) else out).cpu()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--inter", type=int, default=3072)
    a = ap.parse_args()
    torch.manual_seed(0)
    T, H, I, eps = a.tokens, a.hidden, a.inter, 1e-6
    bf = torch.bfloat16
    x = torch.randn(1, T, H, dtype=bf) * 0.5
    res = torch.randn(1, T, H, dtype=bf) * 0.5
    norm_w = (1 + 0.1 * torch.randn(1, H)).to(bf)
    wg, wu = (torch.randn(H, I) * H ** -0.5).to(bf), (torch.randn(H, I) * H ** -0.5).to(bf)
    wd = (torch.randn(I, H) * I ** -0.5).to(bf)
    ref = ref_mlp(x, res, norm_w, wg, wu, wd, eps).float()

    d = [t.to("neuron") for t in (x, res, norm_w, wg, wu, wd)]
    call = lambda: nkl_mlp(d[0], d[3], d[4], d[5], normalization_weights_tensor=d[2], fused_add_tensor=d[1],
                           activation_fn=ActFnType.SiLU, normalization_type=NormType.RMS_NORM, eps=eps)
    try:
        t0 = time.perf_counter()
        out = call()
        y = (out[0] if isinstance(out, (list, tuple)) else out).float().cpu()
        print(f"NKI-Lib mlp compiled+ran on inf2 in {time.perf_counter() - t0:.1f} s; outputs: {len(out) if isinstance(out, (list, tuple)) else 1}")
        print(f"  rel L2 err vs torch ref: {((y.reshape(ref.shape) - ref).norm() / ref.norm()).item():.2e}")
        print(f"  NKI-Lib mlp: {bench(call):8.1f} us/call")
    except Exception as e:  # report, don't hide: gen2 paths are untested upstream
        print(f"NKI-Lib mlp FAILED on inf2: {type(e).__name__}: {str(e)[:600]}")

    tc = torch.compile(ref_mlp, backend="neuron", fullgraph=True)
    y2 = tc(*d, eps).float().cpu()
    print(f"torch.compile baseline: rel L2 err {((y2 - ref).norm() / ref.norm()).item():.2e}, {bench(lambda: tc(*d, eps)):8.1f} us/call")
    wbytes = 3 * H * I * 2
    print(f"bandwidth floor ({wbytes / 1e6:.1f} MB weights @ ~410 GB/s): {wbytes / 410e9 * 1e6:.1f} us; launch overhead ~140 us/call")


if __name__ == "__main__":
    main()
