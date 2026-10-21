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
    sq = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=qk.reshape((P, n)))
    inv = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    for c in nl.static_range(0, n, 512):
        w_ = min(512, n - c)
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
    for c in nl.static_range(0, n, 512):
        w_ = min(512, n - c)
        rot = nl.ndarray((P, w_), dtype=nl.float32, buffer=nl.psum)        # rotate_half(w * x / rms)
        nisa.nc_matmul(dst=rot, stationary=rot_m, moving=xn[:, c:c + w_])
        h0, hn = c // T, w_ // T
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
