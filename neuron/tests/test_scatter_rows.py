"""inf2: can ONE dma_copy scatter T token rows (tokens on partitions) to runtime rows via a per-partition
vector_offset on the destination? Out-of-range rows must be skipped (padding tokens).
Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/tests/test_scatter_rows.py
"""
import time

import nki
import nki.isa as nisa
import nki.language as nl
import torch

ROW = 1024  # KVH * D bf16 = 2 KiB per token row


@nki.jit
def scatter_rows(cache, rows_src, row_idx):
    """cache [NROWS, ROW] (in place); rows_src [T, ROW]; row_idx [T, 1] int32 (>= NROWS -> skipped)."""
    T = rows_src.shape[0]
    src = nl.ndarray((T, ROW), dtype=rows_src.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=src, src=rows_src)
    idx = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx, src=row_idx)
    nisa.dma_copy(dst=cache.ap(pattern=[[ROW, T], [1, ROW]], offset=0, vector_offset=idx, indirect_dim=0), src=src,
                  oob_mode=nisa.oob_mode.skip)
    done = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    z = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=z, value=1.0)
    nisa.dma_copy(dst=done, src=z)
    return done


def main():
    torch.manual_seed(0)
    nrows, T = 4096, 96
    cache = torch.randn(nrows, ROW).to(torch.bfloat16)
    rows = torch.randn(T, ROW).to(torch.bfloat16)
    idx = torch.randperm(nrows)[:T].int()
    idx[-3:] = nrows + 5                                   # padding tokens: out of range -> skipped
    expect = cache.clone()
    for t in range(T):
        if idx[t] < nrows:
            expect[idx[t]] = rows[t]
    cd = cache.to("neuron")
    try:
        scatter_rows(cd, rows.to("neuron"), idx.reshape(T, 1).to("neuron")).cpu()
        print("one-DMA scatter of 96 token rows (3 padded) exact:", torch.equal(cd.cpu(), expect))
        args = (cd, rows.to("neuron"), idx.reshape(T, 1).to("neuron"))
        for _ in range(3):
            scatter_rows(*args)
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            scatter_rows(*args)
        torch.neuron.synchronize()
        print(f"  {(time.perf_counter() - t0) / 50 * 1e6:.0f} us per call (incl ~140 us launch)")
    except Exception as e:
        print("scatter via dst vector_offset FAILED:", type(e).__name__, str(e)[:400])


if __name__ == "__main__":
    main()
