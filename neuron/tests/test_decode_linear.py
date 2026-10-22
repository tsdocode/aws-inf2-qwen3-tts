"""Correctness + timing for decode_linear.rmsnorm_linear_t vs torch, on inf2.

Timing: eager NKI calls each pay a NEFF launch; we time a loop of calls and report both wall time
per call and (with --profile) device time from neuron-explorer.
Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/tests/test_decode_linear.py --K 1024 --N 4096 --B 32
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "kernels"))
from decode_linear import rmsnorm_linear_t, tile_weight  # noqa: E402


def ref(x, norm_w, w, eps):  # x [B,K], w [N,K] torch Linear layout
    v = x.float()
    n = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * norm_w.float()).to(x.dtype)
    return n @ w.t()


def wall_us(fn, iters=50):
    for _ in range(5):
        y = fn()
    y.cpu()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn()
    y.cpu()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--B", type=int, default=32)
    a = ap.parse_args()
    torch.manual_seed(0)
    bf = torch.bfloat16
    x = torch.randn(a.B, a.K, dtype=bf)
    norm_w = (1 + 0.1 * torch.randn(a.K)).to(bf)
    w = (torch.randn(a.N, a.K) * a.K ** -0.5).to(bf)
    expect = ref(x, norm_w, w, 1e-6).float()

    xT = x.t().contiguous().to("neuron")
    nw = norm_w.float().reshape(a.K, 1).to("neuron")
    wT = tile_weight(w).to("neuron")
    t0 = time.perf_counter()
    yT = rmsnorm_linear_t(xT, nw, wT)
    got = yT.cpu().float().t()
    print(f"compile+run {time.perf_counter() - t0:.1f}s; shape {tuple(got.shape)}")
    err = ((got - expect).norm() / expect.norm()).item()
    print(f"rel L2 err vs torch: {err:.2e}  max abs: {(got - expect).abs().max().item():.3e}")

    us = wall_us(lambda: rmsnorm_linear_t(xT, nw, wT))
    floor = a.K * a.N * 2 / 272e9 * 1e6
    print(f"K={a.K} N={a.N} B={a.B}: {us:.1f} us/call wall (incl. ~140 us launch); weight-bandwidth floor {floor:.1f} us")


if __name__ == "__main__":
    main()
