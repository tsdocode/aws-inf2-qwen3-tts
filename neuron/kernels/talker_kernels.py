"""Fused Talker step (28 layers) as one NKI kernel for a MIXED token batch, inf2.

T tokens per step (static bucket): one decode token per active stream + prefill-chunk tokens, all in the
transposed layout ``[features on partitions, tokens on free]`` (see decode_linear / cp_kernels). Linears,
norms and the MLP are token-independent; only attention needs per-token metadata (sequence, position,
pages). One launch for all 28 layers keeps the weight stream continuous (per-launch fixed cost matters:
long streams sustain ~257 GB/s per core vs ~180 GB/s for short ones).

Per-token RoPE (tokens have different positions): rotate_half(x) is a fixed partition permutation, so
``rotate_half(w * x) = (Perm @ diag(w)) x`` is one matmul for all heads/tokens; cos/sin per token come from
the host as [128, T] tables. QK-RMSNorm's 1/rms is per (head, token) and applied first.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from cp_kernels import _add_residual, _linear, _rmsnorm
from decode_linear import P, tile_weight

HEAD_DIM = 128


# ----------------------------------------------------------------------------------------------- host side
def rotate_half_norm_matrix(norm_weight):
    """S fp32 [128,128] with nc_matmul(stationary=S, moving=x) == rotate_half(w * x)."""
    d, h = HEAD_DIM, HEAD_DIM // 2
    perm = torch.zeros(d, d, dtype=torch.float64)
    idx = torch.arange(h)
    perm[idx, idx + h] = -1.0       # out[i]   = -x[i+64]
    perm[idx + h, idx] = 1.0        # out[i+64] = x[i]
    M = perm @ torch.diag(norm_weight.detach().double())
    return M.t().float().contiguous()


def rope_tables(positions, theta):
    """positions [T] -> cos, sin [128, T] fp32 (non-interleaved Qwen convention, cat(f, f))."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    ang = positions.double().unsqueeze(0) * inv_freq.unsqueeze(1)          # [64, T]
    ang = torch.cat((ang, ang), 0)
    return ang.cos().float().contiguous(), ang.sin().float().contiguous()


def talker_params(talker, layers=None):
    """Stacked tiled bf16 weights + norm vectors for the Talker layers (host side)."""
    sel = talker.model.layers if layers is None else talker.model.layers[:layers]
    out = {k: [] for k in ("ln1", "qkv", "qn_w", "kn_w", "q_rot", "k_rot", "o", "ln2", "gate_up", "down")}
    for layer in sel:
        a = layer.self_attn
        out["ln1"].append(layer.input_layernorm.weight.detach().float().reshape(-1, 1))
        out["qkv"].append(tile_weight(a.qkv_proj.weight.detach().to(torch.bfloat16)))
        out["qn_w"].append(a.q_norm.weight.detach().float().reshape(-1, 1))
        out["kn_w"].append(a.k_norm.weight.detach().float().reshape(-1, 1))
        out["q_rot"].append(rotate_half_norm_matrix(a.q_norm.weight))
        out["k_rot"].append(rotate_half_norm_matrix(a.k_norm.weight))
        out["o"].append(tile_weight(a.o_proj.weight.detach().to(torch.bfloat16)))
        out["ln2"].append(layer.post_attention_layernorm.weight.detach().float().reshape(-1, 1))
        out["gate_up"].append(tile_weight(layer.mlp.gate_up_proj.weight.detach().to(torch.bfloat16)))
        out["down"].append(tile_weight(layer.mlp.down_proj.weight.detach().to(torch.bfloat16)))
    stacked = {k: torch.stack(v).contiguous() for k, v in out.items()}
    stacked["final_norm"] = talker.model.norm.weight.detach().float().reshape(-1, 1).contiguous()
    stacked["eps"] = float(talker.model.layers[0].input_layernorm.variance_epsilon)
    return stacked


# --------------------------------------------------------------------------------------------- kernel side
def _qk_norm_rope(qk, nheads, T, w_hbm, rot_hbm, cos, sin, eps, ones_f32):
    """qk [P, nheads, T] bf16 (raw projections) -> RoPE(RMSNorm_head(x) * w) bf16 [P, nheads, T]."""
    n = nheads * T
    hc = max(1, 512 // T)      # whole heads per <=512-wide matmul chunk (T need not divide 512)
    assert T <= 512, T
    sq = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=qk.reshape((P, n)))
    inv = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    for h0 in nl.static_range(0, nheads, hc):
        c, w_ = h0 * T, min(hc, nheads - h0) * T
        ss = nl.ndarray((P, w_), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ss, stationary=ones_f32, moving=sq[:, c:c + w_])
        nisa.activation(dst=inv[:, c:c + w_], op=nl.rsqrt, data=ss, scale=1.0 / HEAD_DIM, bias=eps)
    xn = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)                 # x / rms (no weight yet)
    nisa.tensor_tensor(dst=xn, data1=qk.reshape((P, n)), data2=inv, op=nl.multiply)
    w = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=w_hbm)
    rot_m = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_m, src=rot_hbm)
    xw = nl.ndarray((P, nheads, T), dtype=nl.float32, buffer=nl.sbuf)       # w * x / rms
    nisa.tensor_scalar(dst=xw.reshape((P, n)), data=xn, op0=nl.multiply, operand0=w)
    out = nl.ndarray((P, nheads, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    for h0 in nl.static_range(0, nheads, hc):
        hn = min(hc, nheads - h0)
        c, w_ = h0 * T, hn * T
        rot = nl.ndarray((P, w_), dtype=nl.float32, buffer=nl.psum)        # rotate_half(w * x / rms)
        nisa.nc_matmul(dst=rot, stationary=rot_m, moving=xn[:, c:c + w_])
        a = nl.ndarray((P, hn, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=a, data1=xw[:, h0:h0 + hn, :], data2=cos.expand_dim(1).broadcast(1, hn), op=nl.multiply)
        b = nl.ndarray((P, hn, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=b, data1=rot.reshape((P, hn, T)), data2=sin.expand_dim(1).broadcast(1, hn), op=nl.multiply)
        nisa.tensor_tensor(dst=out[:, h0:h0 + hn, :], data1=a, data2=b, op=nl.add)
    return out


@nki.jit
def talker_dense_step(xT, cos_hbm, sin_hbm, ln1, qkv, qn_w, kn_w, q_rot, k_rot, o, ln2, gate_up, down, final_norm,
                      eps: float, attention: str = "stub"):
    """Talker layers for T tokens with attention = ``stub`` (v of each token, GQA-repeated) — measures the
    weight stream and all token-local math; the paged mixed attention plugs in at ``att``.

    xT [2048, T] bf16; cos/sin [128, T] fp32 per-token RoPE tables. Returns hidden [2048, T] bf16 (final-normed).
    """
    H, T = xT.shape
    L = ln1.shape[0]
    KT = H // P
    NH, KVH = 16, 8
    nqk = NH + KVH

    ones_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f32, value=1.0)
    cos = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos, src=cos_hbm)
    sin = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sin, src=sin_hbm)
    x = nl.ndarray((P, KT, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=xT.reshape((KT, P, T)).permute((1, 0, 2)))

    for l in nl.static_range(L):
        hn = _rmsnorm(x, ln1[l], KT, T, eps, ones_f32)
        qkv_t = _linear(hn, KT, qkv[l], T, nl.bfloat16)                       # [P, 32, T]
        q = _qk_norm_rope(qkv_t[:, 0:NH, :], NH, T, qn_w[l], q_rot[l], cos, sin, eps, ones_f32)
        k = _qk_norm_rope(qkv_t[:, NH:nqk, :], KVH, T, kn_w[l], k_rot[l], cos, sin, eps, ones_f32)
        # ---- attention stub: att[h] = v[h // 2] * (1 + 0 * q + 0 * k) keeps q/k live without real attention
        att = nl.ndarray((P, NH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h in nl.static_range(NH):
            nisa.tensor_tensor(dst=att[:, h, :], data1=qkv_t[:, nqk + h // 2, :], data2=q[:, h, :], op=nl.maximum)
        nisa.tensor_tensor(dst=att[:, 0:KVH, :], data1=att[:, 0:KVH, :], data2=k, op=nl.maximum)
        attn_out = _linear(att, NH, o[l], T, nl.bfloat16)
        x = _add_residual(x, attn_out, KT, T)
        hn2 = _rmsnorm(x, ln2[l], KT, T, eps, ones_f32)
        gu = _linear(hn2, KT, gate_up[l], T, nl.bfloat16)                     # [P, 96, T]
        I_T = gu.shape[1] // 2
        act = nl.ndarray((P, I_T, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=act, op=nl.silu, data=gu[:, 0:I_T, :])
        hmid = nl.ndarray((P, I_T, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hmid, data1=act, data2=gu[:, I_T:2 * I_T, :], op=nl.multiply)
        mlp_out = _linear(hmid, I_T, down[l], T, nl.bfloat16)
        x = _add_residual(x, mlp_out, KT, T)

    xf = _rmsnorm(x, final_norm, KT, T, eps, ones_f32)
    hidden = nl.ndarray((H, T), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=hidden.reshape((KT, P, T)).permute((1, 0, 2)), src=xf)
    return hidden


# ================================================================================ mixed paged attention
PAGE = 128           # tokens per KV page (block)
MASK_NEG = -30000.0  # finite "minus infinity": gen2 PE transposes are not exact with inf (see NKI isa docs)


def build_group_metadata(seqs, qps, pps, num_blocks):
    """Host-side schedule metadata for one attention group.

    seqs: list of (block_ids list, context_len_before_step, n_query_tokens) per segment; padded to the group's
    static (len(seqs) segments, qps queries, pps pages). Query token j of a segment sits at position
    context_len_before_step + j, and its own K/V has already been written to the cache.
    Returns pages [1, nseg*pps] int32 block ids (pages are contiguous in HBM, gathered with one runtime
    ``scalar_offset`` each: 243 GB/s vs ~127 GB/s for per-partition ``vector_select`` on gen2) and the additive mask
    [128, nseg, pps, qps] fp32 (key k of page i at position i*128 + k is visible iff <= query position).
    Padding pages get id ``num_blocks`` (just past the cache): the DMA skips them (oob_mode.skip). Must be the
    *first* out-of-range id — a huge sentinel overflows the 32-bit element offset (id * 131072) and wraps
    back in bounds, silently reading real cache memory.
    """
    nseg = len(seqs)
    offs = torch.zeros(1, nseg * pps, dtype=torch.int32)
    mask = torch.full((PAGE, nseg, pps, qps), MASK_NEG)
    lane = torch.arange(PAGE)
    for s, (blocks, ctx, nq) in enumerate(seqs):
        for i in range(pps):
            offs[0, s * pps + i] = blocks[i] if i < len(blocks) else num_blocks
        for j in range(nq):
            qpos = ctx + j
            key_pos = torch.arange(pps).unsqueeze(0) * PAGE + lane.unsqueeze(1)          # [128, pps]
            mask[:, s, :, j] = torch.where(key_pos <= qpos, 0.0, MASK_NEG)
    return offs, mask


def _attend_group(q, att, tok0, nseg, qps, pps, offs_hbm, mask_hbm, k_pages, v_pages, ones_bf, ident_f32, seg_chunk,
                  dbg=None):
    """Paged attention for one static group, writes att[:, h, tok0 + s*qps + j] (bf16, [D, NH, T] layout)."""
    NH, KVH = 16, 8
    lanes = 2 * qps                                            # 2 query heads per KV head x query tokens
    scale = 1.0 / (HEAD_DIM ** 0.5)
    # Padding pages are skipped by the DMA (out-of-bounds id + oob_mode.skip), so their buffer slots keep old
    # contents: two persistent buffers zeroed once (finite stale data is harmless under the -30000 mask), used
    # ping-pong so chunk c+1's page DMAs can overlap chunk c's compute.
    max_npg = min(seg_chunk, nseg) * pps
    kbufs, vbufs = [], []
    for _ in nl.static_range(2):
        kb = nl.ndarray((P, max_npg, PAGE * KVH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=kb, value=0.0)
        vb = nl.ndarray((P, max_npg, KVH * HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=vb, value=0.0)
        kbufs.append(kb)
        vbufs.append(vb)
    for ci, c0 in enumerate(nl.static_range(0, nseg, seg_chunk)):
        sc = min(seg_chunk, nseg - c0)
        npg = sc * pps
        offs = nl.ndarray((1, npg), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=offs, src=offs_hbm[:, c0 * pps:c0 * pps + npg])
        mask = nl.ndarray((P, sc, pps, qps), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=mask, src=mask_hbm[:, c0:c0 + sc, :, :])
        kp = kbufs[ci % 2]                                     # [d, page, (key, g)]
        vp = vbufs[ci % 2]                                     # [key, page, (g, d)]
        for i in nl.static_range(npg):
            nisa.dma_copy(dst=kp[:, i, :], src=k_pages.ap(pattern=[[PAGE * 8, P], [1, PAGE * 8]], offset=0,
                                                          scalar_offset=offs[0:1, i:i + 1], indirect_dim=0),
                          oob_mode=nisa.oob_mode.skip)
            nisa.dma_copy(dst=vp[:, i, :], src=v_pages.ap(pattern=[[8 * HEAD_DIM, P], [1, 8 * HEAD_DIM]], offset=0,
                                                          scalar_offset=offs[0:1, i:i + 1], indirect_dim=0),
                          oob_mode=nisa.oob_mode.skip)
        kp4 = kp.reshape((P, max_npg, PAGE, KVH))
        vp4 = vp.reshape((P, max_npg, KVH, HEAD_DIM))
        for g in nl.static_range(KVH):
            # QK^T: keys on partitions, (seg, page, head-pair, token) on the free axis
            sps = nl.ndarray((P, sc, pps, 2, qps), dtype=nl.float32, buffer=nl.psum)
            for s in nl.static_range(sc):
                t0 = tok0 + (c0 + s) * qps
                qv = q[:, 2 * g:2 * g + 2, t0:t0 + qps]                                # [d, 2, qps]
                for i in nl.static_range(pps):
                    nisa.nc_matmul(dst=sps[:, s, i, :, :], stationary=kp4[:, s * pps + i, :, g], moving=qv)
            x = nl.ndarray((P, sc, pps, 2, qps), dtype=nl.float32, buffer=nl.sbuf)
            for hh in nl.static_range(2):
                nisa.tensor_tensor(dst=x[:, :, :, hh, :], data1=sps[:, :, :, hh, :], data2=mask, op=nl.add)
            # max over keys: free-axis reduce over pages, then one PE transpose for the partition axis
            m1 = nl.ndarray((P, sc, 2, qps), dtype=nl.float32, buffer=nl.sbuf)
            for hh in nl.static_range(2):  # vector ops take <= 3 free dims
                nisa.tensor_reduce(dst=m1[:, :, hh, :], op=nl.maximum, data=x[:, :, :, hh, :].permute((0, 1, 3, 2)), axis=3)
            L = sc * lanes
            mt = nl.ndarray((L, P), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=mt, stationary=m1.reshape((P, L)), moving=ident_f32)
            mrow = nl.ndarray((L, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mrow, op=nl.maximum, data=mt, axis=1)
            mbt = nl.ndarray((L, P), dtype=nl.float32, buffer=nl.sbuf)               # each lane's max, x128
            nisa.tensor_scalar(dst=mbt, data=ident_f32[0:L, :], op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=mrow)
            mb = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.psum)                # back: [keys, lanes]
            nisa.nc_matmul(dst=mb, stationary=mbt, moving=ident_f32[0:L, 0:L])
            e = nl.ndarray((P, sc, pps, 2, qps), dtype=nl.float32, buffer=nl.sbuf)
            mb4 = mb.reshape((P, sc, 2, qps))
            for hh in nl.static_range(2):
                nisa.tensor_tensor(dst=e[:, :, :, hh, :], data1=x[:, :, :, hh, :],
                                   data2=mb4[:, :, hh, :].expand_dim(2).broadcast(2, pps), op=nl.subtract)
            eb = nl.ndarray((P, sc, pps, 2, qps), dtype=nl.bfloat16, buffer=nl.sbuf)
            n_e = sc * pps * 2 * qps
            nisa.activation(dst=eb.reshape((P, n_e)), op=nl.exp, data=e.reshape((P, n_e)), scale=scale)
            # denominators: sum exp over pages on VectorE, then ONE ones-matmul per (chunk, head) sums over keys
            # (partitions) and broadcasts over d. PV accumulates over pages. Each accumulation gets its OWN PSUM
            # tile: accumulate=True into a slice of a larger PSUM tile overwrote instead of accumulating on gen2.
            esum = nl.ndarray((P, sc, 2, qps), dtype=nl.bfloat16, buffer=nl.sbuf)
            for hh in nl.static_range(2):
                nisa.tensor_reduce(dst=esum[:, :, hh, :], op=nl.add, data=eb[:, :, :, hh, :].permute((0, 1, 3, 2)), axis=3)
            dps = nl.ndarray((P, sc * lanes), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=dps, stationary=ones_bf, moving=esum.reshape((P, sc * lanes)))
            rden = nl.ndarray((P, sc, 2, qps), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rden.reshape((P, sc * lanes)), data=dps)    # VectorE: keeps ScalarE's table on exp
            num = nl.ndarray((P, sc, 2, qps), dtype=nl.float32, buffer=nl.sbuf)
            for s in nl.static_range(sc):
                nps = nl.ndarray((P, 2, qps), dtype=nl.float32, buffer=nl.psum)
                for i in nl.static_range(pps):
                    nisa.nc_matmul(dst=nps, stationary=vp4[:, s * pps + i, g, :], moving=eb[:, s, i, :, :], accumulate=(i > 0))
                nisa.tensor_copy(dst=num[:, s, :, :], src=nps)
            den = dps
            if dbg is not None and c0 == 0 and g == 0:   # test-only: dump intermediates of the first (chunk, head)
                for name, t in (("x", x), ("mb", mb), ("den", den.reshape((P, sc, 2, qps))), ("num", num), ("m1", m1), ("e", e), ("eb", eb)):
                    tmp = nl.ndarray(t.shape, dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=tmp, src=t)
                    nisa.dma_copy(dst=dbg[name], src=tmp)
            for s in nl.static_range(sc):
                t0 = tok0 + (c0 + s) * qps
                nisa.tensor_tensor(dst=att[:, 2 * g:2 * g + 2, t0:t0 + qps], data1=num[:, s, :, :], data2=rden[:, s, :, :],
                                   op=nl.multiply)


@nki.jit
def paged_attention(q_hbm, k_cache, v_cache, offs0, mask0, offs1=None, mask1=None, groups=(), seg_chunk: int = 4,
                    debug: bool = False):
    """One layer of mixed decode/prefill paged attention (standalone, for testing).

    q_hbm [D, NH, T] bf16 (RoPE'd, normed); caches: K [NBLK, D, PAGE, KVH], V [NBLK, PAGE, KVH, D] (new tokens
    already written). groups: static tuple of (nseg, qps, pps) (<= 2: decode, prefill) with tokens laid out group by
    group, segment by segment; offs<i>/mask<i> from ``build_group_metadata`` (tensors must be separate args: a list
    of tensors would become a static, value-keyed argument).
    Returns att [D, NH, T] bf16.
    """
    D, NH, T = q_hbm.shape
    NBLK = k_cache.shape[0]
    q = nl.ndarray((D, NH, T), dtype=q_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q, src=q_hbm)
    att = nl.ndarray((D, NH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    ones_bf = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_bf, value=1.0)
    ident_f32 = nl.shared_identity_matrix(n=P, dtype=nl.float32)
    k_pages = k_cache.reshape((NBLK, HEAD_DIM * PAGE * 8))   # one row per block: scalar_offset = block id
    v_pages = v_cache.reshape((NBLK, PAGE * 8 * HEAD_DIM))
    tok0 = 0
    meta = ((offs0, mask0), (offs1, mask1))
    dbg = None
    if debug:
        n0, q0, p0 = groups[0]
        sc0 = min(seg_chunk, n0)
        dbg = {"x": nl.ndarray((P, sc0, p0, 2, q0), dtype=nl.float32, buffer=nl.shared_hbm),
               "mb": nl.ndarray((P, sc0 * 2 * q0), dtype=nl.float32, buffer=nl.shared_hbm),
               "den": nl.ndarray((P, sc0, 2, q0), dtype=nl.float32, buffer=nl.shared_hbm),
               "num": nl.ndarray((P, sc0, 2, q0), dtype=nl.float32, buffer=nl.shared_hbm),
               "m1": nl.ndarray((P, sc0, 2, q0), dtype=nl.float32, buffer=nl.shared_hbm),
               "e": nl.ndarray((P, sc0, p0, 2, q0), dtype=nl.float32, buffer=nl.shared_hbm),
               "eb": nl.ndarray((P, sc0, p0, 2, q0), dtype=nl.float32, buffer=nl.shared_hbm)}
    for gi, (nseg, qps, pps) in enumerate(groups):
        _attend_group(q, att, tok0, nseg, qps, pps, meta[gi][0], meta[gi][1], k_pages, v_pages, ones_bf, ident_f32,
                      seg_chunk, dbg if gi == 0 else None)
        tok0 += nseg * qps
    out = nl.ndarray((D, NH, T), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=att)
    if debug:
        return out, dbg["x"], dbg["mb"], dbg["den"], dbg["num"], dbg["m1"], dbg["e"], dbg["eb"]
    return out


# ======================================================================================= full talker step
def kv_write_indices(blocks_per_token, offset_per_token, num_blocks):
    """Host: per-token write rows for the K view [NBLK*D*PAGE, KVH] (row = (blk*D)*PAGE + off; + d*PAGE via the
    access pattern) and the V view [NBLK*PAGE, KVH*D] (row = blk*PAGE + off). Padding tokens (block < 0) get
    out-of-range rows so their writes are skipped."""
    blk = torch.as_tensor(blocks_per_token).long()
    off = torch.as_tensor(offset_per_token).long()
    pad = blk < 0
    kw = torch.where(pad, num_blocks * HEAD_DIM * PAGE, blk * HEAD_DIM * PAGE + off)
    vw = torch.where(pad, num_blocks * PAGE, blk * PAGE + off)
    return kw.int().reshape(1, -1).contiguous(), vw.int().reshape(1, -1).contiguous()


def _write_kv(k, v, T, kw, vw, k_rows, v_rows, ident_bf):
    """Write this step's T tokens of K ([D, KVH, T]) and V into the paged caches at their runtime rows."""
    KVH = 8
    for t0 in nl.static_range(0, T, P):
        tn = min(P, T - t0)
        vtok = nl.ndarray((tn, KVH, HEAD_DIM), dtype=nl.bfloat16, buffer=nl.sbuf)     # token-major for V
        for h in nl.static_range(KVH):
            tp = nl.ndarray((tn, HEAD_DIM), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=tp, stationary=v[:, h, t0:t0 + tn], moving=ident_bf)
            nisa.tensor_copy(dst=vtok[:, h, :], src=tp)
        for j in nl.static_range(tn):
            t = t0 + j
            nisa.dma_copy(dst=k_rows.ap(pattern=[[PAGE * KVH, HEAD_DIM], [1, KVH]], offset=0, scalar_offset=kw[0:1, t:t + 1],
                                        indirect_dim=0), src=k[:, :, t], oob_mode=nisa.oob_mode.skip)
            nisa.dma_copy(dst=v_rows.ap(pattern=[[KVH * HEAD_DIM, 1], [1, KVH * HEAD_DIM]], offset=0,
                                        scalar_offset=vw[0:1, t:t + 1], indirect_dim=0),
                          src=vtok[j:j + 1, :, :].reshape((1, KVH * HEAD_DIM)), oob_mode=nisa.oob_mode.skip)


@nki.jit
def talker_step(xT, cos_hbm, sin_hbm, ln1, qkv, qn_w, kn_w, q_rot, k_rot, o, ln2, gate_up, down, final_norm,
                k_cache, v_cache, kw_hbm, vw_hbm, offs0, mask0, offs1=None, mask1=None, groups=(), eps: float = 1e-6,
                seg_chunk: int = 4):
    """Full Talker step (28 layers) for a MIXED token batch with paged KV.

    xT [2048, T] bf16 input embeddings (tokens ordered group by group, segment by segment); cos/sin [128, T];
    k_cache [L, NBLK, D, PAGE, KVH] / v_cache [L, NBLK, PAGE, KVH, D] bf16, updated in place with this step's K/V
    (rows from ``kv_write_indices``); groups + offs/mask from ``build_group_metadata``.
    Returns the final-normed hidden [2048, T] bf16.
    """
    H, T = xT.shape
    L = ln1.shape[0]
    NBLK = k_cache.shape[1]
    KT = H // P
    NH, KVH = 16, 8
    nqk = NH + KVH

    ones_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f32, value=1.0)
    ones_bf = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_bf, value=1.0)
    ident_f32 = nl.shared_identity_matrix(n=P, dtype=nl.float32)
    ident_bf = nl.shared_identity_matrix(n=P, dtype=nl.bfloat16)
    cos = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos, src=cos_hbm)
    sin = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sin, src=sin_hbm)
    kw = nl.ndarray((1, T), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=kw, src=kw_hbm)
    vw = nl.ndarray((1, T), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vw, src=vw_hbm)
    x = nl.ndarray((P, KT, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=xT.reshape((KT, P, T)).permute((1, 0, 2)))
    meta = ((offs0, mask0), (offs1, mask1))

    for l in nl.static_range(L):
        hn = _rmsnorm(x, ln1[l], KT, T, eps, ones_f32)
        qkv_t = _linear(hn, KT, qkv[l], T, nl.bfloat16)                       # [P, 32, T]
        q = _qk_norm_rope(qkv_t[:, 0:NH, :], NH, T, qn_w[l], q_rot[l], cos, sin, eps, ones_f32)
        k = _qk_norm_rope(qkv_t[:, NH:nqk, :], KVH, T, kn_w[l], k_rot[l], cos, sin, eps, ones_f32)
        v = qkv_t[:, nqk:nqk + KVH, :]
        k_rows = k_cache[l].reshape((NBLK * HEAD_DIM * PAGE, KVH))
        v_rows = v_cache[l].reshape((NBLK * PAGE, KVH * HEAD_DIM))
        _write_kv(k, v, T, kw, vw, k_rows, v_rows, ident_bf)
        att = nl.ndarray((P, NH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        k_pages = k_cache[l].reshape((NBLK, HEAD_DIM * PAGE * KVH))
        v_pages = v_cache[l].reshape((NBLK, PAGE * KVH * HEAD_DIM))
        tok0 = 0
        for gi, (nseg, qps, pps) in enumerate(groups):
            _attend_group(q, att, tok0, nseg, qps, pps, meta[gi][0], meta[gi][1], k_pages, v_pages, ones_bf, ident_f32,
                          seg_chunk)
            tok0 += nseg * qps
        attn_out = _linear(att, NH, o[l], T, nl.bfloat16)
        x = _add_residual(x, attn_out, KT, T)
        hn2 = _rmsnorm(x, ln2[l], KT, T, eps, ones_f32)
        gu = _linear(hn2, KT, gate_up[l], T, nl.bfloat16)                     # [P, 96, T]
        I_T = gu.shape[1] // 2
        act = nl.ndarray((P, I_T, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=act, op=nl.silu, data=gu[:, 0:I_T, :])
        hmid = nl.ndarray((P, I_T, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hmid, data1=act, data2=gu[:, I_T:2 * I_T, :], op=nl.multiply)
        mlp_out = _linear(hmid, I_T, down[l], T, nl.bfloat16)
        x = _add_residual(x, mlp_out, KT, T)

    xf = _rmsnorm(x, final_norm, KT, T, eps, ones_f32)
    hidden = nl.ndarray((H, T), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=hidden.reshape((KT, P, T)).permute((1, 0, 2)), src=xf)
    return hidden
