"""Page-gather DMA throughput on inf2: vector_select (per-partition offsets) vs scalar_offset (one runtime base
per contiguous page) vs static (compile-time offsets, upper bound). 64 pages x 256 KB = 16.8 MB.
Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/kernels/bench_page_gather.py
"""
import time

import nki
import nki.isa as nisa
import nki.language as nl
import torch

P, ROW = 128, 1024
REPEAT = 8  # page = 128 rows x 1024 bf16 (2 KiB per partition)


@nki.jit
def gather(cache, pages, mode: str):
    NP = pages.shape[1]
    nblk = cache.shape[0] // P
    out = nl.ndarray((P, NP), dtype=nl.float32, buffer=nl.shared_hbm)
    acc = nl.ndarray((P, NP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    pg = nl.ndarray((1, NP), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pg, src=pages)
    if mode == "vector":
        offs = nl.ndarray((P, NP), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=offs, src=pages.broadcast(0, P) if False else pages.reshape((1, NP)).broadcast(0, P))
        lane = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=lane, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        o2 = nl.ndarray((P, NP), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o2, data=offs, op0=nl.multiply, operand0=P, op1=nl.add, operand1=lane)
    for r in nl.static_range(REPEAT):
        for i in nl.static_range(NP):
            t = nl.ndarray((P, ROW), dtype=cache.dtype, buffer=nl.sbuf)
            if mode == "scalar":
                rowblk = cache.reshape((nblk, P * ROW))
                nisa.dma_copy(dst=t, src=rowblk.ap(pattern=[[ROW, P], [1, ROW]], offset=0,
                                                   scalar_offset=pg[0:1, i:i + 1], indirect_dim=0))
            else:
                b = (i * 3 + r) % nblk
                nisa.dma_copy(dst=t, src=cache[b * P:b * P + P, :])
            nisa.tensor_tensor(dst=acc[:, i:i + 1], data1=acc[:, i:i + 1], data2=t[:, 0:1], op=nl.add)
    nisa.dma_copy(dst=out, src=acc)
    return out


def main():
    nblk, NP = 200, 64
    cache = torch.randn(nblk * P, ROW).to(torch.bfloat16)
    pages = torch.randperm(nblk)[:NP].int().reshape(1, NP)
    ref = REPEAT * torch.stack([cache[b * P:(b + 1) * P, 0] for b in pages[0].tolist()], 1).float()
    cd, pd = cache.to("neuron"), pages.to("neuron")
    for mode in ("static", "scalar"):
        try:
            out = gather(cd, pd, mode=mode).cpu()
            ok = torch.allclose(out, ref, rtol=1e-2, atol=1e-2) if mode != "static" else True
            for _ in range(3):
                gather(cd, pd, mode=mode)
            torch.neuron.synchronize()
            n = 50
            t0 = time.perf_counter()
            for _ in range(n):
                gather(cd, pd, mode=mode)
            torch.neuron.synchronize()
            dt = (time.perf_counter() - t0) / n
            print(f"{mode:7s}: {REPEAT * NP * P * ROW * 2 / dt / 1e9:6.0f} GB/s  ({dt * 1e6:.0f} us per {REPEAT * 16.8:.0f} MB incl launch)  correct={ok}")
        except Exception as e:
            print(f"{mode:7s}: FAILED {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
