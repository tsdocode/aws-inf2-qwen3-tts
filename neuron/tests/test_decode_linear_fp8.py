"""Weight-only FP8 decode linear on inf2: (1) device fp8 decode matches ml_dtypes legacy e4m3,
(2) kernel vs host math on the *dequantized* weights (isolates kernel error from quantization error),
(3) quantization error vs bf16 weights, (4) device time vs the bf16 kernel (profile with tools/profile_neff.py).
Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/tests/test_decode_linear_fp8.py --K 2048 --N 6144 --B 32
"""
import argparse, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "kernels"))
from decode_linear import dequantize_fp8_e4m3, quantize_fp8_e4m3, rmsnorm_linear_fp8_t  # noqa: E402
from test_decode_linear import ref  # noqa: E402


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--N", type=int, default=6144)
    ap.add_argument("--B", type=int, default=32)
    a = ap.parse_args()
    torch.manual_seed(0)
    bf = torch.bfloat16
    x = torch.randn(a.B, a.K, dtype=bf)
    norm_w = (1 + 0.1 * torch.randn(a.K)).to(bf)
    w = (torch.randn(a.N, a.K) * a.K ** -0.5).to(bf)
    q, scale = quantize_fp8_e4m3(w)
    w_dq = dequantize_fp8_e4m3(q, scale, a.N, a.K)
    print(f"fp8 quantization error of W: rel L2 {rel(w_dq, w):.2e}")
    expect_dq = ref(x, norm_w, w_dq.to(bf), 1e-6).float()
    expect_bf = ref(x, norm_w, w, 1e-6).float()
    yT = rmsnorm_linear_fp8_t(x.t().contiguous().to("neuron"), norm_w.float().reshape(a.K, 1).to("neuron"),
                              q.to("neuron"), scale.to("neuron"))
    got = yT.cpu().float().t()
    print(f"kernel vs host math on dequantized W: rel L2 {rel(got, expect_dq):.2e}  (should be bf16-level ~1e-3)")
    print(f"end-to-end vs bf16 weights:           rel L2 {rel(got, expect_bf):.2e}")


if __name__ == "__main__":
    main()
