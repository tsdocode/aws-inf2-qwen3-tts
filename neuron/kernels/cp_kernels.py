"""Fused Code Predictor depth step (5 layers + final norm + lm_head) as one NKI kernel, inf2.

Everything stays in the transposed layout ``[features on partitions, batch on free]`` used by
``decode_linear``: weights are the stationary operand, outputs land ready for the next matmul.

Per-layer tricks (vs the compiler's ~44k tiny instructions / 5.1k transposes per step):
- q/k RMSNorm weight + RoPE at the step's position fold into one 128x128 matrix ``M_s = R_s diag(w)``
  (every row of the batch shares position s), so q/k-norm+RoPE for all heads is 2 fp32 matmuls; the
  per-token 1/rms is applied after (it commutes with the linear map).
- Attention over the s+1 cached slots without per-row matmuls: elementwise ``q*K`` for all
  (head, slot, row) at once, then a ones-matrix matmul sums over the head dim *and* broadcasts the scores
  to every partition, softmax runs along the free axis, and ``p*V`` reduced over slots lands directly in
  the o_proj input layout.
- The step's new K/V slot is DMA'd straight into the cache (no full-buffer scatter).

Step index ``s`` is a compile-time int: one kernel specialization per depth step (16, each compiles in
seconds), so the attended slot count s+1 is static and needs no mask.
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from decode_linear import N_BLOCK, P, quantize_fp8_e4m3, tile_weight

HEAD_DIM = 128
SLOTS = 16
HEADS_PER_CHUNK = 2  # scores chunk = 2 heads x 16 slots x B fp32 in PSUM


# ----------------------------------------------------------------------------------------------- host side
def rope_norm_matrix(norm_weight, position, theta):
    """S = (R_pos @ diag(w)).T fp32 [128, 128]: nc_matmul(stationary=S, moving=x) == R_pos (w * x)."""
    d = HEAD_DIM
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = position * inv_freq
    cos, sin = torch.cos(ang), torch.sin(ang)
    R = torch.zeros(d, d, dtype=torch.float64)
    h = d // 2
    idx = torch.arange(h)
    R[idx, idx] = cos
    R[idx, idx + h] = -sin
    R[idx + h, idx + h] = cos
    R[idx + h, idx] = sin
    M = R @ torch.diag(norm_weight.detach().double())
    return M.t().float().contiguous()


def _prep_weight(weight, fp8):
    """-> (tiled weight, per-output-channel scale [N,1] fp32). bf16: scale of ones (unused by the kernel)."""
    if fp8:
        return quantize_fp8_e4m3(weight)
    return tile_weight(weight.detach().to(torch.bfloat16)), torch.ones(weight.shape[0], 1)


FP8_GROUPS = ("qkv", "o", "gate_up", "down", "lm_heads")


def cp_step_params(predictor, fp8=()):
    """Per-layer tiled weights + norm vectors for the Code Predictor, on the host.

    fp8: iterable of weight groups (subset of FP8_GROUPS) stored as weight-only legacy e4m3 (uint8) with
    per-output-channel scales ``<name>_s``; the rest stay bf16 and get no ``_s`` entry.
    """
    fp8 = set(fp8)
    layers = []
    for layer in predictor.model.layers:
        a = layer.self_attn
        entry = dict(ln1=layer.input_layernorm.weight.detach().float().reshape(-1, 1),
                     ln2=layer.post_attention_layernorm.weight.detach().float().reshape(-1, 1))
        for name, w in (("qkv", a.qkv_proj.weight), ("o", a.o_proj.weight),
                        ("gate_up", layer.mlp.gate_up_proj.weight), ("down", layer.mlp.down_proj.weight)):
            entry[name], scale = _prep_weight(w, name in fp8)
            if name in fp8:
                entry[name + "_s"] = scale
        layers.append(entry)
    stacked = {k: torch.stack([l[k] for l in layers]).contiguous() for k in layers[0]}
    stacked["final_norm"] = predictor.model.norm.weight.detach().float().reshape(-1, 1).contiguous()
    heads = [_prep_weight(w, "lm_heads" in fp8) for w in predictor.lm_head_weight]
    stacked["lm_heads"] = torch.stack([h[0] for h in heads]).contiguous()
    if "lm_heads" in fp8:
        stacked["lm_heads_s"] = torch.stack([h[1] for h in heads]).contiguous()
    a0 = predictor.model.layers[0].self_attn
    stacked["rope"] = torch.stack([
        torch.stack([torch.stack([rope_norm_matrix(l.self_attn.q_norm.weight, s, a0.rope_theta),
                                  rope_norm_matrix(l.self_attn.k_norm.weight, s, a0.rope_theta)])
                     for l in predictor.model.layers])
        for s in range(SLOTS)]).contiguous()                       # [16 steps, L, 2(q,k), 128, 128]
    stacked["eps"] = float(predictor.model.layers[0].input_layernorm.variance_epsilon)
    return stacked


# --------------------------------------------------------------------------------------------- kernel side
def _rmsnorm(x, w_hbm, KT, B, eps, ones_f32):
    """x [P, KT, B] (bf16) -> RMSNorm(x) bf16 [P, KT, B]; w_hbm [K, 1] fp32. fp32 statistics."""
    w = nl.ndarray((P, KT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=w_hbm.reshape((KT, P, 1)).permute((1, 0, 2)))
    sum_sq = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.psum)
    for k in nl.static_range(KT):
        sq = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=x[:, k, :])
        nisa.nc_matmul(dst=sum_sq, stationary=ones_f32, moving=sq, accumulate=(k > 0))
    inv = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=sum_sq, scale=1.0 / (KT * P), bias=eps)
    out = nl.ndarray((P, KT, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    for k in nl.static_range(KT):
        t = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t, data1=x[:, k, :], data2=inv, op=nl.multiply)
        nisa.tensor_scalar(dst=out[:, k, :], data=t, op0=nl.multiply, operand0=w[:, k, :])
    return out


def _linear(xn, KT, w_tiled, B, out_dtype, w_scale=None):
    """xn [P, KT, B] bf16 (SBUF) x tiled weight (HBM) -> [P, N/P, B] out_dtype (SBUF).

    w_scale [N, 1] given -> w_tiled is uint8 legacy-e4m3 FP8; the per-output-channel scale is applied on
    the PSUM copy-out (output channels are partitions in this layout).
    """
    NB = w_tiled.shape[0]
    NT = NB * (N_BLOCK // P)
    if w_scale is not None:
        w_tiled = w_tiled.view(nl.float8_e4m3)
        scale = nl.ndarray((P, NT, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=scale, src=w_scale.reshape((NT, P, 1)).permute((1, 0, 2)))
    out = nl.ndarray((P, NT, B), dtype=out_dtype, buffer=nl.sbuf)
    for nb in nl.static_range(NB):
        w_blk = nl.ndarray((P, KT, N_BLOCK), dtype=w_tiled.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_blk, src=w_tiled[nb])
        for j in nl.static_range(N_BLOCK // P):
            acc = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.psum)
            for k in nl.static_range(KT):
                nisa.nc_matmul(dst=acc, stationary=w_blk[:, k, j * P:(j + 1) * P], moving=xn[:, k, :],
                               accumulate=(k > 0))
            t = nb * (N_BLOCK // P) + j
            if w_scale is None:
                nisa.tensor_copy(dst=out[:, t, :], src=acc)
            else:
                nisa.tensor_scalar(dst=out[:, t, :], data=acc, op0=nl.multiply, operand0=scale[:, t, :])
    return out


def _add_residual(x, y, KT, B):
    """bf16 residual stream like the repo: x = bf16(x + y)."""
    out = nl.ndarray((P, KT, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=x, data2=y, op=nl.add)
    return out


def _cp_layers(x, kv, ln1, qkv, o, ln2, gate_up, down, rope, s, eps, ones_f32, ones_bf, qkv_s=None, o_s=None,
               gate_up_s=None, down_s=None):
    """All Code Predictor layers for depth step ``s`` (static): x [P, KT, B] bf16 -> x; K/V slot s written to kv."""
    sc_ = lambda t, l=None: None if t is None else (t if l is None else t[l])  # noqa: E731
    KT, B = x.shape[1], x.shape[2]
    L = kv.shape[0]
    KVH = kv.shape[4]
    NH = 16
    J = s + 1
    group = NH // KVH
    for l in nl.static_range(L):
        # ---- attention
        hn = _rmsnorm(x, ln1[l], KT, B, eps, ones_f32)
        qkv_t = _linear(hn, KT, qkv[l], B, nl.bfloat16, sc_(qkv_s, l))           # [P, 32, B]: q 0..15, k 16..23, v 24..31
        nqk = NH + KVH
        sq = nl.ndarray((P, nqk * B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=qkv_t[:, 0:nqk, :].reshape((P, nqk * B)))
        ss = nl.ndarray((P, nqk * B), dtype=nl.float32, buffer=nl.psum)
        for c in nl.static_range(0, nqk * B, 512):
            w_ = min(512, nqk * B - c)
            nisa.nc_matmul(dst=ss[:, c:c + w_], stationary=ones_f32, moving=sq[:, c:c + w_])
        inv = nl.ndarray((P, nqk, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=inv.reshape((P, nqk * B)), op=nl.rsqrt, data=ss, scale=1.0 / HEAD_DIM, bias=eps)

        rq = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rq, src=rope[l, 0])
        rk = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rk, src=rope[l, 1])
        qk32 = nl.ndarray((P, nqk, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qk32, src=qkv_t[:, 0:nqk, :])
        rot = nl.ndarray((P, nqk * B), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=rot[:, 0:NH * B], stationary=rq, moving=qk32[:, 0:NH, :].reshape((P, NH * B)))
        nisa.nc_matmul(dst=rot[:, NH * B:nqk * B], stationary=rk, moving=qk32[:, NH:nqk, :].reshape((P, KVH * B)))
        qk = nl.ndarray((P, nqk, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=qk.reshape((P, nqk * B)), data1=rot, data2=inv.reshape((P, nqk * B)), op=nl.multiply)

        # cache: this step's K/V into slot s (HBM), previous slots + this one gathered in SBUF
        nisa.dma_copy(dst=kv[l, 0, :, s, :, :], src=qk[:, NH:nqk, :])
        nisa.dma_copy(dst=kv[l, 1, :, s, :, :], src=qkv_t[:, nqk:nqk + KVH, :])
        k_all = nl.ndarray((P, J, KVH, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        v_all = nl.ndarray((P, J, KVH, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        if s > 0:
            nisa.dma_copy(dst=k_all[:, 0:s, :, :], src=kv[l, 0, :, 0:s, :, :])
            nisa.dma_copy(dst=v_all[:, 0:s, :, :], src=kv[l, 1, :, 0:s, :, :])
        nisa.tensor_copy(dst=k_all[:, s, :, :], src=qk[:, NH:nqk, :])
        nisa.tensor_copy(dst=v_all[:, s, :, :], src=qkv_t[:, nqk:nqk + KVH, :])

        att = nl.ndarray((P, NH, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        for h0 in nl.static_range(0, NH, HEADS_PER_CHUNK):
            hc = HEADS_PER_CHUNK
            prod = nl.ndarray((P, hc, J, B), dtype=nl.bfloat16, buffer=nl.sbuf)
            for hh in nl.static_range(hc):
                h = h0 + hh
                nisa.tensor_tensor(dst=prod[:, hh, :, :], data1=qk[:, h:h + 1, :].broadcast(1, J),
                                   data2=k_all[:, :, h // group, :], op=nl.multiply)
            n = hc * J * B
            sc = nl.ndarray((P, hc, J, B), dtype=nl.float32, buffer=nl.psum)
            flat_p, flat_s = prod.reshape((P, n)), sc.reshape((P, n))
            for c in nl.static_range(0, n, 512):
                w_ = min(512, n - c)
                nisa.nc_matmul(dst=flat_s[:, c:c + w_], stationary=ones_bf, moving=flat_p[:, c:c + w_])
            mx = nl.ndarray((P, hc, B), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mx, op=nl.maximum, data=sc.permute((0, 1, 3, 2)), axis=3)
            e = nl.ndarray((P, hc, J, B), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=e, data1=sc, data2=mx.expand_dim(2).broadcast(2, J), op=nl.subtract)
            nisa.activation(dst=e, op=nl.exp, data=e, scale=1.0 / math.sqrt(HEAD_DIM))
            den = nl.ndarray((P, hc, B), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=den, op=nl.add, data=e.permute((0, 1, 3, 2)), axis=3)
            rden = nl.ndarray((P, hc, B), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rden, op=nl.reciprocal, data=den)
            pv = nl.ndarray((P, hc, J, B), dtype=nl.float32, buffer=nl.sbuf)
            for hh in nl.static_range(hc):
                nisa.tensor_tensor(dst=pv[:, hh, :, :], data1=e[:, hh, :, :], data2=v_all[:, :, (h0 + hh) // group, :],
                                   op=nl.multiply)
            num = nl.ndarray((P, hc, B), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=num, op=nl.add, data=pv.permute((0, 1, 3, 2)), axis=3)
            nisa.tensor_tensor(dst=att[:, h0:h0 + hc, :], data1=num, data2=rden, op=nl.multiply)

        attn_out = _linear(att, NH, o[l], B, nl.bfloat16, sc_(o_s, l))          # [P, 8, B]
        x = _add_residual(x, attn_out, KT, B)

        # ---- MLP
        hn2 = _rmsnorm(x, ln2[l], KT, B, eps, ones_f32)
        gu = _linear(hn2, KT, gate_up[l], B, nl.float32, sc_(gate_up_s, l))           # [P, 48, B]: gate 0..23, up 24..47
        I_T = gu.shape[1] // 2
        act = nl.ndarray((P, I_T, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=act, op=nl.silu, data=gu[:, 0:I_T, :])
        hmid = nl.ndarray((P, I_T, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hmid, data1=act, data2=gu[:, I_T:2 * I_T, :], op=nl.multiply)
        mlp_out = _linear(hmid, I_T, down[l], B, nl.bfloat16, sc_(down_s, l))      # [P, 8, B]
        x = _add_residual(x, mlp_out, KT, B)

    return x


@nki.jit
def cp_step(xT, kv, ln1, qkv, o, ln2, gate_up, down, final_norm, lm_head, rope, s: int, eps: float,
            with_head: bool = True, qkv_s=None, o_s=None, gate_up_s=None, down_s=None, lm_head_s=None):
    """One Code Predictor depth step for B rows.

    xT [1024, B] bf16 (projected input embedding, transposed); kv [L, 2, 128, 16, 8, B] bf16 cache,
    slot ``s`` is written in place; per-layer weights stacked on dim 0 (see ``cp_step_params``);
    lm_head = tiled head for this step's group; rope [L, 2, 128, 128] = this step's matrices.
    A weight whose ``*_s`` scale is given is uint8 legacy-e4m3 FP8 (``cp_step_params(fp8=...)``).
    Returns logitsT [2048, B] fp32 (if with_head) and the final hidden [1024, B] bf16 (pre-norm).
    """
    H, B = xT.shape
    KT = H // P

    ones_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f32, value=1.0)
    ones_bf = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_bf, value=1.0)

    x = nl.ndarray((P, KT, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=xT.reshape((KT, P, B)).permute((1, 0, 2)))

    x = _cp_layers(x, kv, ln1, qkv, o, ln2, gate_up, down, rope, s, eps, ones_f32, ones_bf, qkv_s, o_s, gate_up_s,
                   down_s)

    hidden = nl.ndarray((H, B), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=hidden.reshape((KT, P, B)).permute((1, 0, 2)), src=x)
    if not with_head:
        return hidden
    xf = _rmsnorm(x, final_norm, KT, B, eps, ones_f32)
    logits = _linear(xf, KT, lm_head, B, nl.float32, lm_head_s)                    # [P, 16, B]
    NV = logits.shape[1]
    logitsT = nl.ndarray((NV * P, B), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=logitsT.reshape((NV, P, B)).permute((1, 0, 2)), src=logits)
    return logitsT, hidden


# ------------------------------------------------------------------------------------ whole frame (16 depth steps)
TOPK = 50
NEG_BIG = -1.0e30


def _head_rows(xf, lm_head, B):
    """xf [P, KT, B] bf16 x tiled lm_head -> logits [B, V] fp32 with ROWS on partitions (x is the stationary operand,
    weight blocks stream as moving: no transposes for the row-wise sampling that follows)."""
    KT = xf.shape[1]
    NB = lm_head.shape[0]
    lg = nl.ndarray((B, NB * N_BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    for nb in nl.static_range(NB):
        w_blk = nl.ndarray((P, KT, N_BLOCK), dtype=lm_head.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_blk, src=lm_head[nb])
        for j in nl.static_range(N_BLOCK // 512):
            acc = nl.ndarray((B, 512), dtype=nl.float32, buffer=nl.psum)
            for k in nl.static_range(KT):
                nisa.nc_matmul(dst=acc, stationary=xf[:, k, :], moving=w_blk[:, k, j * 512:(j + 1) * 512],
                               accumulate=(k > 0))
            c0 = nb * N_BLOCK + j * 512
            nisa.tensor_copy(dst=lg[:, c0:c0 + 512], src=acc)
    return lg


def _sample_rows(lg, u, inv_t, greedy, iota_k, ident_f, tri):
    """Row-wise top-k (k=TOPK) sampling, same algorithm as ``_DeviceSampler``: softmax(top-k * inv_t), inverse CDF at
    u * total, argmax for greedy rows. lg [B, V] fp32; u / inv_t / greedy [B, 1] fp32 -> token [B, 1] fp32.
    Top-k by 7 rounds of max8 + match_replace8 (56 sorted candidates, the first 50 used)."""
    B = lg.shape[0]
    R8 = (TOPK + 7) // 8
    vals = nl.ndarray((B, R8 * 8), dtype=nl.float32, buffer=nl.sbuf)
    idx = nl.ndarray((B, R8 * 8), dtype=nl.uint32, buffer=nl.sbuf)
    # (match_replace8 with dst_idx is unimplemented on gen2: indices come from find_index8 on an untouched copy)
    work = nl.ndarray(lg.shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=work, src=lg)
    for r in nl.static_range(R8):
        nisa.max8(dst=vals[:, r * 8:(r + 1) * 8], src=work)
        nisa.nc_find_index8(dst=idx[:, r * 8:(r + 1) * 8], data=lg, vals=vals[:, r * 8:(r + 1) * 8])
        if r < R8 - 1:
            nisa.nc_match_replace8(dst=work, data=work, vals=vals[:, r * 8:(r + 1) * 8], imm=NEG_BIG)
    e = nl.ndarray((B, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=e, data=vals[:, 0:TOPK], op0=nl.subtract, operand0=vals[:, 0:1], op1=nl.multiply,
                       operand1=inv_t)
    nisa.activation(dst=e, op=nl.exp, data=e)
    # cumsum along the free axis as two small PE matmuls (tensor_scalar_cumulative fails to lower on gen2 here):
    # eT = transpose(e), cdf = eT^T @ tri with tri[k, j] = (k <= j)
    eT_ps = nl.ndarray((TOPK, B), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=eT_ps, stationary=e, moving=ident_f[0:B, 0:B])
    eT = nl.ndarray((TOPK, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=eT, src=eT_ps)
    cdf_ps = nl.ndarray((B, TOPK), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=cdf_ps, stationary=eT, moving=tri)
    cdf = nl.ndarray((B, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cdf, src=cdf_ps)
    thr = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=thr, data1=u, data2=cdf[:, TOPK - 1:TOPK], op=nl.multiply)
    lt = nl.ndarray((B, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lt, data=cdf, op0=nl.less, operand0=thr)
    pick = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pick, op=nl.add, data=lt, axis=1)
    nisa.tensor_scalar(dst=pick, data=pick, op0=nl.minimum, operand0=float(TOPK - 1))
    oh = nl.ndarray((B, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=oh, data=iota_k, op0=nl.equal, operand0=pick)
    idx_f = nl.ndarray((B, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idx_f, src=idx[:, 0:TOPK])
    sel = nl.ndarray((B, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sel, data1=oh, data2=idx_f, op=nl.multiply)
    tok = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=tok, op=nl.add, data=sel, axis=1)
    # greedy rows: tok += greedy * (argmax - tok)
    dg = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dg, data1=idx_f[:, 0:1], data2=tok, op=nl.subtract)
    nisa.tensor_tensor(dst=dg, data1=dg, data2=greedy, op=nl.multiply)
    nisa.tensor_tensor(dst=tok, data1=tok, data2=dg, op=nl.add)
    return tok


def _sample_consts(B):
    """iota_k [B, TOPK] = 0..TOPK-1; ident_f [P, P] fp32 identity; tri [TOPK, TOPK] fp32 with tri[k, j] = (k <= j)."""
    ii = nl.ndarray((B, TOPK), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=ii, pattern=[[1, TOPK]], offset=0, channel_multiplier=0)
    iota_k = nl.ndarray((B, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=iota_k, src=ii)
    di = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)                     # j - k
    nisa.iota(dst=di, pattern=[[1, P]], offset=0, channel_multiplier=-1)
    df = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=df, src=di)
    ident_f = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ident_f, data=df, op0=nl.equal, operand0=0.0)
    tri = nl.ndarray((TOPK, TOPK), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tri, data=df[0:TOPK, 0:TOPK], op0=nl.greater_equal, operand0=0.0)
    return iota_k, ident_f, tri


def _embed_t(table_rows, tok_f, row_offset, ident_bf, KT, B):
    """Next depth input: rows ``table_rows[tok + row_offset]`` gathered by one indirect DMA (rows on partitions), then
    PE-transposed to the [P, KT, B] feature-major layout."""
    D = KT * P
    ti = nl.ndarray((B, 1), dtype=nl.int32, buffer=nl.sbuf)
    if row_offset:
        tf = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=tf, data=tok_f, op0=nl.add, operand0=float(row_offset))
        nisa.tensor_copy(dst=ti, src=tf)
    else:
        nisa.tensor_copy(dst=ti, src=tok_f)
    emb = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=emb, src=table_rows.ap(pattern=[[D, B], [1, D]], offset=0, vector_offset=ti, indirect_dim=0))
    x = nl.ndarray((P, KT, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    for k in nl.static_range(KT):
        tp = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=tp, stationary=emb[:, k * P:(k + 1) * P], moving=ident_bf[0:B, 0:B])
        nisa.tensor_copy(dst=x[:, k, :], src=tp)
    return x


@nki.jit
def cp_frame(xT, tok0, kv, ln1, qkv, o, ln2, gate_up, down, final_norm, lm_heads, rope, proj_l0, proj_res, u, inv_t,
             greedy, ident, eps: float):
    """One whole Code Predictor frame (16 depth steps + on-chip sampling + next-input lookups) in ONE launch.

    xT [1024, B] bf16 = small_proj(talker hidden); tok0 [B, 1] fp32 = talker token; kv [L, 2, 128, 16, 8, B] cache;
    lm_heads [15, ...] tiled; rope [16, L, 2, 128, 128]; proj_l0 [3072, 1024] and proj_res [14*2048, 1024] bf16 input
    tables; u [B, 15] fp32 host uniforms; inv_t / greedy [B, 1] fp32; ident [128, 128] bf16.
    Returns tokens [B, 15] fp32 for depth groups 1..15 (exact integers).
    """
    H, B = xT.shape
    KT = H // P
    V = lm_heads.shape[1] * N_BLOCK
    ones_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f32, value=1.0)
    ones_bf = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_bf, value=1.0)
    ident_bf = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=ident_bf, src=ident)
    iota_k, ident_f, tri = _sample_consts(B)
    u_sb = nl.ndarray((B, 15), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=u_sb, src=u)
    it_sb = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=it_sb, src=inv_t)
    gr_sb = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gr_sb, src=greedy)
    t0 = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t0, src=tok0)
    toks = nl.ndarray((B, 15), dtype=nl.float32, buffer=nl.sbuf)

    x = nl.ndarray((P, KT, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=xT.reshape((KT, P, B)).permute((1, 0, 2)))
    x = _cp_layers(x, kv, ln1, qkv, o, ln2, gate_up, down, rope[0], 0, eps, ones_f32, ones_bf)
    x = _embed_t(proj_l0, t0, 0, ident_bf, KT, B)                     # step-0 output is only its K/V
    for st in nl.static_range(1, 16):
        x = _cp_layers(x, kv, ln1, qkv, o, ln2, gate_up, down, rope[st], st, eps, ones_f32, ones_bf)
        xf = _rmsnorm(x, final_norm, KT, B, eps, ones_f32)
        lg = _head_rows(xf, lm_heads[st - 1], B)
        tok = _sample_rows(lg, u_sb[:, st - 1:st], it_sb, gr_sb, iota_k, ident_f, tri)
        nisa.tensor_copy(dst=toks[:, st - 1:st], src=tok)
        if st < 15:
            x = _embed_t(proj_res, tok, (st - 1) * (V), ident_bf, KT, B)
    out = nl.ndarray((B, 15), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=toks)
    return out
