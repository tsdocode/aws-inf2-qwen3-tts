"""Measure achievable HBM->SBUF bandwidth on one inf2 NeuronCore (no compute).

Streams a [NB, 128, F] tensor block by block into SBUF; reduces one element per block so the loads
are not dead code. Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/bench/dma_ceiling.py --mb 8 --free 8192
"""
import argparse

import nki
import nki.isa as nisa
import nki.language as nl
import torch

P = 128


@nki.jit
def stream(src):
    NB, _, F = src.shape
    out = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.shared_hbm)
    acc = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
    for b in nl.affine_range(NB):
        t = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=src[b])
        nisa.tensor_copy(dst=acc[:, b:b + 1], src=t[:, 0:1])
    nisa.dma_copy(dst=out, src=acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=8)
    ap.add_argument("--free", type=int, default=8192, help="bf16 elements per partition per block")
    a = ap.parse_args()
    nb = max(1, int(a.mb * 1e6 / (P * a.free * 2)))
    x = torch.randn(nb, P, a.free, dtype=torch.bfloat16).to("neuron")
    stream(x).cpu()
    print(f"blocks={nb} x [128, {a.free}] bf16 = {nb * P * a.free * 2 / 1e6:.2f} MB "
          f"({a.free * 2 / 1024:.0f} KiB per partition per DMA)")


if __name__ == "__main__":
    main()
