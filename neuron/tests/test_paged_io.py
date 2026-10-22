"""inf2 check of the two dynamic-DMA idioms the paged talker attention needs.

(1) per-token K/V writes at runtime (block, offset): K cache D-major [NBLK, D, BL, KVH] via scalar_offset,
    V cache token-major [NBLK, BL, KVH, D] via scalar_offset on a transposed (token-major) source;
(2) page gathers by a runtime block table with vector_select (partition p reads row blk*rows_per_blk + p).
Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/tests/test_paged_io.py
"""
import nki
import nki.isa as nisa
import nki.language as nl
import torch

P, D, BL, KVH = 128, 128, 128, 8


@nki.jit
def paged_io(k_cache, v_cache, k_newT, v_newT, k_widx, v_widx, pages):
    """k_newT/v_newT [D, KVH, T] bf16 (kernel layout); k_widx/v_widx [T, 1] int32 element-row indices;
    pages [NP] int32 block ids to gather. Writes the T tokens into the caches (in place), then returns the
    gathered K pages [D, NP, BL*KVH] and V pages [BL, NP, KVH*D] from HBM."""
    NBLK = k_cache.shape[0]
    T = k_newT.shape[2]
    NP = pages.shape[0]
    kn = nl.ndarray((D, KVH, T), dtype=k_newT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=kn, src=k_newT)
    vn = nl.ndarray((D, KVH, T), dtype=v_newT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=vn, src=v_newT)
    kw = nl.ndarray((1, T), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=kw, src=k_widx.reshape((1, T)))
    vw = nl.ndarray((1, T), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vw, src=v_widx.reshape((1, T)))

    # V to token-major on TensorE: [D, T] per kv head -> [T, D]
    ident = nl.shared_identity_matrix(n=P, dtype=nl.bfloat16)
    vtok = nl.ndarray((T, KVH, D), dtype=v_newT.dtype, buffer=nl.sbuf)
    for h in nl.static_range(KVH):
        tp = nl.ndarray((T, D), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=tp, stationary=vn[:, h, :], moving=ident, is_transpose=False)
        nisa.tensor_copy(dst=vtok[:, h, :], src=tp)

    k_rows = k_cache.reshape((NBLK * D * BL, KVH))          # row = (blk*D + d)*BL + off
    v_rows = v_cache.reshape((NBLK * BL, KVH * D))          # row = blk*BL + off
    for t in nl.static_range(T):
        nisa.dma_copy(dst=k_rows.ap(pattern=[[BL * KVH, D], [1, KVH]], offset=0, scalar_offset=kw[0:1, t:t + 1],
                                    indirect_dim=0), src=kn[:, :, t])
        nisa.dma_copy(dst=v_rows.ap(pattern=[[KVH * D, 1], [1, KVH * D]], offset=0, scalar_offset=vw[0:1, t:t + 1],
                                    indirect_dim=0), src=vtok[t:t + 1, :, :].reshape((1, KVH * D)))

    # gather pages by runtime block id: partition p reads row blk*D + p (K) / blk*BL + p (V)
    pg1 = nl.ndarray((1, NP), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pg1, src=pages.reshape((1, NP)))
    ones = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    pgf = nl.ndarray((1, NP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pgf, src=pg1)
    bc = nl.ndarray((P, NP), dtype=nl.float32, buffer=nl.psum)               # broadcast block ids to partitions
    nisa.nc_matmul(dst=bc, stationary=ones, moving=pgf)
    lane = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=lane, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    lanef = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lanef, src=lane)
    offs_f = nl.ndarray((P, NP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=offs_f, data=bc, op0=nl.multiply, operand0=float(D), op1=nl.add, operand1=lanef)
    offs = nl.ndarray((P, NP), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=offs, src=offs_f)

    k_pages = k_cache.reshape((NBLK * D, BL * KVH))
    v_pages = v_cache.reshape((NBLK * BL, KVH * D))
    kout = nl.ndarray((D, NP, BL * KVH), dtype=k_cache.dtype, buffer=nl.shared_hbm)
    vout = nl.ndarray((BL, NP, KVH * D), dtype=v_cache.dtype, buffer=nl.shared_hbm)
    for i in nl.static_range(NP):
        ks = nl.ndarray((P, BL * KVH), dtype=k_cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=ks, src=k_pages.vector_select(0, offs[:, i:i + 1]))   # D == BL == 128 rows per block
        nisa.dma_copy(dst=kout[:, i, :], src=ks)
        vs = nl.ndarray((P, KVH * D), dtype=v_cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=vs, src=v_pages.vector_select(0, offs[:, i:i + 1]))
        nisa.dma_copy(dst=vout[:, i, :], src=vs)
    return kout, vout


def main():
    torch.manual_seed(0)
    NBLK, T = 12, 5
    kc = torch.randn(NBLK, D, BL, KVH).to(torch.bfloat16)
    vc = torch.randn(NBLK, BL, KVH, D).to(torch.bfloat16)
    k_new = torch.randn(T, KVH, D).to(torch.bfloat16)
    v_new = torch.randn(T, KVH, D).to(torch.bfloat16)
    blk = torch.tensor([3, 3, 7, 0, 11]); off = torch.tensor([5, 6, 127, 0, 64])
    k_widx = ((blk * D) * BL + off).int().reshape(T, 1)            # row index into [NBLK*D*BL, KVH]; +d*BL via pattern
    v_widx = (blk * BL + off).int().reshape(T, 1)
    pages = torch.tensor([3, 7, 0, 11], dtype=torch.int32)
    # expected caches after writes
    ek, ev = kc.clone(), vc.clone()
    for t in range(T):
        ek[blk[t], :, off[t], :] = k_new[t].t()
        ev[blk[t], off[t]] = v_new[t]
    dev = lambda x: x.contiguous().to("neuron")  # noqa: E731
    kcd, vcd = dev(kc), dev(vc)
    ko, vo = paged_io(kcd, vcd, dev(k_new.permute(2, 1, 0)), dev(v_new.permute(2, 1, 0)), dev(k_widx), dev(v_widx), dev(pages))
    ko, vo = ko.cpu(), vo.cpu()
    print("K cache written in place:", torch.equal(kcd.cpu(), ek), "| V cache written in place:", torch.equal(vcd.cpu(), ev))
    exp_k = torch.stack([ek[p].reshape(D, BL * KVH) for p in pages.tolist()], 1)
    exp_v = torch.stack([ev[p].reshape(BL, KVH * D) for p in pages.tolist()], 1)
    print("K page gather exact:", torch.equal(ko, exp_k), "| V page gather exact:", torch.equal(vo, exp_v))


if __name__ == "__main__":
    main()
