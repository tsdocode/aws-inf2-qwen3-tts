"""NKI kernel for the codec decoder's 8-layer sliding-window pre-transformer (streaming), inf2.

The compiled version of this transformer cost ~22 ms per call at B=32 (layout transposes / concat spills around
its KV state); everything else in the codec front stays compiled (~7 ms).

Layout: features on partitions, tokens (row-major: row b, frame f) on the free axis. Hidden 512 = 4 tiles; the
16 heads x 64 dims = 8 tiles, each tile holding a PAIR of heads (dims 0-63 head 2p, 64-127 head 2p+1).
Attention is matmul-free (like the Code Predictor kernel): q (.) K over each head's 64 partitions summed by a
block-diagonal ones matmul — which also broadcasts each head's score over its own 64 partitions, so both heads of a
pair are handled at once — softmax along keys on the free axis, then (p (.) V) reduced over keys lands directly in
the o_proj input layout (V's partitions are the same head dims). No transposes anywhere.

State per layer and head pair: K and V as [128 pair dims, B rows, 71 slots] (D-major, one DMA for all rows). Keys for a chunk of F frames
are [prior 71 | new F]; the new state is the last 71 columns (static DMA, no runtime offsets).
Residual stream fp32, matmul operands bf16 with fp32 accumulation.
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from cp_kernels import _linear
from decode_linear import P, tile_weight

HD = 64              # head dim
PAIRS = 8            # 16 heads / 2 per 128-partition tile
MASK_NEG = -30000.0


# ----------------------------------------------------------------------------------------------- host side
def _rot_half64_matrix(norm_weight):
    """S [128,128] fp32 with nc_matmul(stationary=S, moving=x) == rotate_half_64(w * x) on both 64-halves."""
    h = HD // 2
    perm = torch.zeros(HD, HD, dtype=torch.float64)
    idx = torch.arange(h)
    perm[idx, idx + h] = -1.0
    perm[idx + h, idx] = 1.0
    M = perm @ torch.diag(norm_weight.detach().double())
    full = torch.block_diag(M, M)
    return full.t().float().contiguous()


def front_transformer_params(transformer):
    """Stacked per-layer weights for ``front_transformer`` (host side)."""
    L = []
    a0 = transformer.layers[0].self_attn
    qk_norm = hasattr(a0.q_norm, "weight")              # Identity in the 12 Hz codec: RoPE only
    ones = torch.ones(HD)
    for layer in transformer.layers:
        a = layer.self_attn
        qw = a.q_norm.weight if qk_norm else ones
        kw = a.k_norm.weight if qk_norm else ones
        qkv = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).detach().to(torch.bfloat16)  # [3072, 512]
        gu = torch.cat([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight]).detach().to(torch.bfloat16)  # [2048, 512]
        L.append(dict(
            ln1=layer.input_layernorm.weight.detach().float().reshape(-1, 1),
            qkv=tile_weight(qkv),
            qn_w=qw.detach().float().repeat(2).reshape(-1, 1),
            kn_w=kw.detach().float().repeat(2).reshape(-1, 1),
            q_rot=_rot_half64_matrix(qw),
            k_rot=_rot_half64_matrix(kw),
            # out dim padded 512 -> 1024 (tile_weight blocks are 1024 columns); padded outputs are ignored
            o=tile_weight(torch.cat([a.o_proj.weight.detach(), torch.zeros(512, a.o_proj.weight.shape[1])]).to(torch.bfloat16)),
            attn_scale=layer.self_attn_layer_scale.scale.detach().float().reshape(-1, 1),
            ln2=layer.post_attention_layernorm.weight.detach().float().reshape(-1, 1),
            gate_up=tile_weight(gu),
            down=tile_weight(torch.cat([layer.mlp.down_proj.weight.detach(), torch.zeros(512, layer.mlp.down_proj.weight.shape[1])]).to(torch.bfloat16)),
            mlp_scale=layer.mlp_layer_scale.scale.detach().float().reshape(-1, 1),
        ))
    st = {k: torch.stack([l[k] for l in L]).contiguous() for k in L[0]}
    st["final_norm"] = transformer.norm.weight.detach().float().reshape(-1, 1).contiguous()
    st["eps"] = float(transformer.layers[0].input_layernorm.variance_epsilon)
    st["qk_norm"] = bool(qk_norm)
    return st


def front_rope_tables(pos, F, inv_freq):
    """pos [B] (frame position of the chunk's first frame) -> cos, sin [128, B*F] (64-dim cat(f,f), both heads)."""
    p = (pos.double().unsqueeze(1) + torch.arange(F, dtype=torch.float64)).reshape(-1)          # [B*F]
    ang = p.unsqueeze(0) * inv_freq.double().unsqueeze(1)                                         # [32, N]
    ang = torch.cat((ang, ang), 0)                                                                # [64, N]
    ang = torch.cat((ang, ang), 0)                                                                # [128, N]
    return ang.cos().float().contiguous(), ang.sin().float().contiguous()


def front_attention_mask(ctx, F, retained=71, window=72):
    """Additive mask [1, B, F, retained + F]: prior slot j valid iff j >= retained - ctx[b] and inside the window
    (j >= f); new frame j' visible iff j' <= f (causal)."""
    B = ctx.shape[0]
    j = torch.arange(retained)
    f = torch.arange(F)
    prior = (j.view(1, 1, -1) >= (retained - ctx).view(-1, 1, 1)) & (j.view(1, 1, -1) >= f.view(1, -1, 1) + retained + 1 - window)
    new = (torch.arange(F).view(1, 1, -1) <= f.view(1, -1, 1)).expand(B, F, F)
    ok = torch.cat([prior, new], -1)
    return torch.where(ok, 0.0, MASK_NEG).reshape(1, B, F, retained + F).contiguous()


# --------------------------------------------------------------------------------------------- kernel side
def _rmsnorm_f32(x, w_hbm, KT, N, eps, ones_f32):
    """x [P, KT, N] fp32 -> bf16 RMSNorm(x)*w (fp32 math)."""
    w = nl.ndarray((P, KT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=w_hbm.reshape((KT, P, 1)).permute((1, 0, 2)))
    ss = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.psum)
    for k in nl.static_range(KT):
        sq = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=x[:, k, :])
        nisa.nc_matmul(dst=ss, stationary=ones_f32, moving=sq, accumulate=(k > 0))
    inv = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=ss, scale=1.0 / (KT * P), bias=eps)
    out = nl.ndarray((P, KT, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    for k in nl.static_range(KT):
        t = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t, data1=x[:, k, :], data2=inv, op=nl.multiply)
        nisa.tensor_scalar(dst=out[:, k, :], data=t, op0=nl.multiply, operand0=w[:, k, :])
    return out


def _head_norm_rope(x, n_tiles, N, w_hbm, rot_hbm, cos, sin, eps, blk_ones, qk_norm=True):
    """x [P, n_tiles, N] bf16 (pairs of 64-dim heads) -> RoPE(RMSNorm_64(x) * w) bf16 (RoPE only if not qk_norm)."""
    w = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=w_hbm)
    rot_m = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rot_m, src=rot_hbm)
    out = nl.ndarray((P, n_tiles, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    for t in nl.static_range(n_tiles):
        xn = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        if qk_norm:
            sq = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sq, op=nl.square, data=x[:, t, :])
            ss = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ss, stationary=blk_ones, moving=sq)          # per-head (64-row) sums, broadcast
            inv = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=inv, op=nl.rsqrt, data=ss, scale=1.0 / HD, bias=eps)
            nisa.tensor_tensor(dst=xn, data1=x[:, t, :], data2=inv, op=nl.multiply)
        else:
            nisa.tensor_copy(dst=xn, src=x[:, t, :])
        xw = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=xw, data=xn, op0=nl.multiply, operand0=w)
        rot = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=rot, stationary=rot_m, moving=xn)
        a = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=a, data1=xw, data2=cos, op=nl.multiply)
        b = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=b, data1=rot, data2=sin, op=nl.multiply)
        nisa.tensor_tensor(dst=out[:, t, :], data1=a, data2=b, op=nl.add)
    return out


@nki.jit
def front_transformer(hT, cos_hbm, sin_hbm, mask_hbm, k_state, v_state, ln1, qkv, qn_w, kn_w, q_rot, k_rot, o,
                      attn_scale, ln2, gate_up, down, mlp_scale, final_norm, eps: float = 1e-5, row_chunk: int = 8,
                      qk_norm: bool = False):
    """hT [512, B*F] fp32 (after input_proj) -> (yT [512, B*F] fp32 after final norm, new k_state, new v_state).

    k_state / v_state [L, 8, 128, B, 71] bf16 (pair dims on partitions, all rows side by side); mask [1, B, F, 71+F] from ``front_attention_mask``;
    cos/sin [128, B*F] from ``front_rope_tables``.
    """
    D, N = hT.shape
    L, B = k_state.shape[0], k_state.shape[3]
    R = k_state.shape[4]
    F = N // B
    KEYS = R + F
    KT = D // P
    scale = 1.0 / math.sqrt(HD)

    ones_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f32, value=1.0)
    blk_ones = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)          # block-diag(ones64, ones64)
    nisa.memset(dst=blk_ones, value=0.0)
    nisa.memset(dst=blk_ones[0:HD, 0:HD], value=1.0)
    nisa.memset(dst=blk_ones[HD:P, HD:P], value=1.0)
    blk_bf = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=blk_bf, src=blk_ones)
    cos = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos, src=cos_hbm)
    sin = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sin, src=sin_hbm)
    mask = nl.ndarray((P, B, F, KEYS), dtype=nl.float32, buffer=nl.sbuf)       # same on every partition
    nisa.dma_copy(dst=mask, src=mask_hbm.broadcast(0, P))

    h = nl.ndarray((P, KT, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=h, src=hT.reshape((KT, P, N)).permute((1, 0, 2)))
    k_new_state = nl.ndarray(k_state.shape, dtype=k_state.dtype, buffer=nl.shared_hbm)
    v_new_state = nl.ndarray(v_state.shape, dtype=v_state.dtype, buffer=nl.shared_hbm)

    for l in nl.static_range(L):
        hn = _rmsnorm_f32(h, ln1[l], KT, N, eps, ones_f32)
        qkv_t = _linear(hn, KT, qkv[l], N, nl.bfloat16)                    # [P, 24, N]: q 0..7, k 8..15, v 16..23
        q = _head_norm_rope(qkv_t[:, 0:PAIRS, :], PAIRS, N, qn_w[l], q_rot[l], cos, sin, eps, blk_ones, qk_norm)
        k = _head_norm_rope(qkv_t[:, PAIRS:2 * PAIRS, :], PAIRS, N, kn_w[l], k_rot[l], cos, sin, eps, blk_ones, qk_norm)
        v = qkv_t[:, 2 * PAIRS:3 * PAIRS, :]
        att = nl.ndarray((P, PAIRS, N), dtype=nl.bfloat16, buffer=nl.sbuf)       # o_proj input (1024 features)
        for p in nl.static_range(PAIRS):
            # one DMA in / out per (layer, pair) for ALL rows: per-row state DMAs (8k per call) dominated the kernel
            # (staging copies on ScalarE: VectorE is the busy engine here)
            # staged through contiguous tiles: a DMA into the strided [:, :, 0:R] view splits into B*128 ~140-byte
            # descriptors (13 GB/s); contiguous [P, B*R] rows move as one 4.5 KB descriptor per partition
            kall = nl.ndarray((P, B, KEYS), dtype=nl.bfloat16, buffer=nl.sbuf)
            vall = nl.ndarray((P, B, KEYS), dtype=nl.bfloat16, buffer=nl.sbuf)
            kst = nl.ndarray((P, B, R), dtype=nl.bfloat16, buffer=nl.sbuf)
            vst = nl.ndarray((P, B, R), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=kst, src=k_state[l, p])
            nisa.dma_copy(dst=vst, src=v_state[l, p])
            nisa.activation(dst=kall[:, :, 0:R], op=nl.copy, data=kst)
            nisa.activation(dst=vall[:, :, 0:R], op=nl.copy, data=vst)
            nisa.tensor_copy(dst=kall[:, :, R:KEYS], src=k[:, p, :].reshape((P, B, F)))
            nisa.tensor_copy(dst=vall[:, :, R:KEYS], src=v[:, p, :].reshape((P, B, F)))
            kout = nl.ndarray((P, B, R), dtype=nl.bfloat16, buffer=nl.sbuf)
            vout = nl.ndarray((P, B, R), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=kout, op=nl.copy, data=kall[:, :, F:KEYS])
            nisa.activation(dst=vout, op=nl.copy, data=vall[:, :, F:KEYS])
            nisa.dma_copy(dst=k_new_state[l, p], src=kout)
            nisa.dma_copy(dst=v_new_state[l, p], src=vout)
            for r0 in nl.static_range(0, B, row_chunk):
                rc = min(row_chunk, B - r0)
                # scores for both heads of the pair: q (.) K, summed per 64-row half (block ones), on the free axis
                prod = nl.ndarray((P, rc, F, KEYS), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=prod, data1=kall[:, r0:r0 + rc, :].expand_dim(2).broadcast(2, F),
                                   data2=q[:, p, r0 * F:(r0 + rc) * F].reshape((P, rc, F)).expand_dim(3).broadcast(3, KEYS),
                                   op=nl.multiply)
                n = rc * F * KEYS
                sc = nl.ndarray((P, rc, F, KEYS), dtype=nl.float32, buffer=nl.sbuf)
                flat_p, flat_s = prod.reshape((P, n)), sc.reshape((P, n))
                for c in nl.static_range(0, n, 512):
                    w_ = min(512, n - c)
                    ps = nl.ndarray((P, w_), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=ps, stationary=blk_bf, moving=flat_p[:, c:c + w_])
                    nisa.tensor_tensor(dst=flat_s[:, c:c + w_], data1=ps, data2=mask[:, r0:r0 + rc, :, :].reshape((P, n))[:, c:c + w_],
                                       op=nl.add)
                mx = nl.ndarray((P, rc, F), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=mx, op=nl.maximum, data=sc, axis=3)
                e = nl.ndarray((P, rc, F, KEYS), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=e, data1=sc, data2=mx.expand_dim(3).broadcast(3, KEYS), op=nl.subtract)
                nisa.activation(dst=e.reshape((P, n)), op=nl.exp, data=e.reshape((P, n)), scale=scale)
                den = nl.ndarray((P, rc, F), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=den, op=nl.add, data=e, axis=3)
                rden = nl.ndarray((P, rc, F), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rden, data=den)
                pv = nl.ndarray((P, rc, F, KEYS), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=pv, data1=e, data2=vall[:, r0:r0 + rc, :].expand_dim(2).broadcast(2, F),
                                   op=nl.multiply)
                num = nl.ndarray((P, rc, F), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=num, op=nl.add, data=pv, axis=3)
                nisa.tensor_tensor(dst=att[:, p, r0 * F:(r0 + rc) * F], data1=num.reshape((P, rc * F)),
                                   data2=rden.reshape((P, rc * F)), op=nl.multiply)
        o_out = _linear(att, PAIRS, o[l], N, nl.float32)                     # [P, 8, N] (tiles 4..7 are zero pad)
        sa = nl.ndarray((P, KT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sa, src=attn_scale[l].reshape((KT, P, 1)).permute((1, 0, 2)))
        for kt in nl.static_range(KT):
            t = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=t, data=o_out[:, kt, :], op0=nl.multiply, operand0=sa[:, kt, :])
            nisa.tensor_tensor(dst=h[:, kt, :], data1=h[:, kt, :], data2=t, op=nl.add)
        hn2 = _rmsnorm_f32(h, ln2[l], KT, N, eps, ones_f32)
        gu = _linear(hn2, KT, gate_up[l], N, nl.float32)                      # [P, 16, N]: gate 0..7, up 8..15
        act = nl.ndarray((P, PAIRS, N), dtype=nl.bfloat16, buffer=nl.sbuf)     # 1024 intermediate features
        sg = nl.ndarray((P, PAIRS, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.silu, data=gu[:, 0:PAIRS, :])
        nisa.tensor_tensor(dst=act, data1=sg, data2=gu[:, PAIRS:2 * PAIRS, :], op=nl.multiply)
        d_out = _linear(act, PAIRS, down[l], N, nl.float32)                   # [P, 8, N] (tiles 4..7 zero pad)
        ms = nl.ndarray((P, KT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ms, src=mlp_scale[l].reshape((KT, P, 1)).permute((1, 0, 2)))
        for kt in nl.static_range(KT):
            t = nl.ndarray((P, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=t, data=d_out[:, kt, :], op0=nl.multiply, operand0=ms[:, kt, :])
            nisa.tensor_tensor(dst=h[:, kt, :], data1=h[:, kt, :], data2=t, op=nl.add)

    y = _rmsnorm_f32(h, final_norm, KT, N, eps, ones_f32)
    yf = nl.ndarray((P, KT, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=yf, src=y)
    out = nl.ndarray((D, N), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out.reshape((KT, P, N)).permute((1, 0, 2)), src=yf)
    return out, k_new_state, v_new_state
