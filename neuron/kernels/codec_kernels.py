"""NKI kernels for the Qwen3-TTS 12 Hz codec decoder on NeuronCore-v2 (inf2).

Layout: channels on SBUF partitions, time on the free axis. A dilated causal Conv1d is then a sum of
``kernel_size`` shifted matmuls accumulated in PSUM (shifts are free-axis offsets, no data movement),
and SnakeBeta's per-channel alpha/beta are per-partition scalars.

Precision: residual stream and SnakeBeta in fp32; matmul operands bf16 with fp32 PSUM accumulation.
The upstream decoder runs fully in fp32 (repo: checkpoint.py ``.float()``); plain bf16 lost ~87 dB SNR.

SnakeBeta(x) = x + sin(alpha*x)^2 / (beta + 1e-9), alpha = exp(log_alpha), beta = exp(log_beta).
The ScalarE ``sin`` is only valid on [-pi, pi] and |alpha*x| reaches ~142 on real audio; see
``_snake`` for the integer-cast range reduction (VectorE has no fmod).
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl
import torch

P = 128
T_TILE = 512  # output time steps per PSUM bank (512 fp32 per partition)


def channel_tiles(channels):
    return [(start, min(P, channels - start)) for start in range(0, channels, P)]


def snake_params(module):
    """SnakeBeta module -> (alpha/pi [C,1], sqrt(1/(beta+eps)) [C,1]) fp32, host-side once."""
    alpha = torch.exp(module.alpha.detach().float()) / math.pi
    inv_beta = 1.0 / (torch.exp(module.beta.detach().float()) + module.no_div_by_zero)
    return alpha.reshape(-1, 1).contiguous(), inv_beta.sqrt().reshape(-1, 1).contiguous()


def conv_weight_taps(conv):
    """Conv1d weight [Cout, Cin, K] -> [Cin, K, Cout] bf16 (rows = contraction channel)."""
    return conv.weight.detach().permute(1, 2, 0).contiguous().to(torch.bfloat16)


def residual_unit_params(unit):
    a1, ib1 = snake_params(unit.act1)
    a2, ib2 = snake_params(unit.act2)
    return dict(
        alpha1=a1, inv_beta_sqrt1=ib1,
        w1=conv_weight_taps(unit.conv1.conv), b1=unit.conv1.conv.bias.detach().float().reshape(-1, 1).contiguous(),
        alpha2=a2, inv_beta_sqrt2=ib2,
        w2=conv_weight_taps(unit.conv2.conv), b2=unit.conv2.conv.bias.detach().float().reshape(-1, 1).contiguous(),
        dilation=int(unit.conv1.dilation),
        split_snake=unit.conv1.conv.in_channels <= 192,
    )


CAST_ENGINE = nisa.engine.gpsimd  # fp32->bf16 casts off the busy VectorE
ROUND_MAGIC = 12582912.0  # 1.5 * 2**23: (q + M) - M rounds an fp32 |q| < 2**22 to the nearest integer


def _snake_fused(x, alpha_over_pi, inv_beta_sqrt, shape, cast_engine):
    """SnakeBeta as one bf16 operand (x + s): one matmul per tap. Better when TensorE-bound (C >= 384)."""
    xb, s2 = _snake_terms(x, alpha_over_pi, inv_beta_sqrt, shape, cast_engine, fp32_s2=True)
    a = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=a, data1=x, data2=s2, op=nl.add)
    return (a,)


def _snake_terms(x, alpha_over_pi, inv_beta_sqrt, shape, cast_engine, fp32_s2=False):
    """SnakeBeta split into its two additive terms, both bf16 for TensorE: (x, sin(alpha*x)^2 / beta).

    The next op is always a conv, and conv(x + s) == conv(x) + conv(s), so callers feed both terms as
    accumulating matmuls into one PSUM tile instead of spending a VectorE pass on the add (VectorE was
    the bottleneck at audio rate; TensorE had slack).

    Range reduction for the [-pi, pi] ScalarE sin: q = alpha*x/pi, r = q - round(q) in [-0.5, 0.5],
    sin^2(pi*r) == sin^2(alpha*x) because sin^2 has period pi (|alpha*x| reaches ~142 in practice).
    """
    q = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=q, op=nl.copy, data=x, scale=alpha_over_pi)
    rq = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rq, data=q, op0=nl.add, operand0=ROUND_MAGIC, op1=nl.subtract, operand1=ROUND_MAGIC)
    r = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=r, data1=q, data2=rq, op=nl.subtract)
    s = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=s, op=nl.sin, data=r, scale=math.pi)
    s2 = nl.ndarray(shape, dtype=nl.float32 if fp32_s2 else nl.bfloat16, buffer=nl.sbuf)
    nisa.activation(dst=s2, op=nl.square, data=s, scale=inv_beta_sqrt)
    if fp32_s2:
        return None, s2
    xb = nl.ndarray(shape, dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xb, src=x, engine=cast_engine)
    return xb, s2


@nki.jit
def residual_unit(x, alpha1, inv_beta_sqrt1, w1, b1, alpha2, inv_beta_sqrt2, w2, b2, dilation: int,
                  split_snake: bool = True):
    """Fused DecoderResidualUnit, whole-sequence causal (zero left history).

    x [B, C, T] fp32 -> y [B, C, T] fp32 with
    y = conv2(snake2(conv1(snake1(x)))) + x, conv1: K=7 dilated causal, conv2: K=1.
    Weights from ``residual_unit_params``. T % T_TILE == 0 or T < T_TILE. ``split_snake``: feed SnakeBeta's two terms
    as separate matmuls (VectorE-bound units, C <= 192) vs one fused add (TensorE-bound, C >= 384).
    """
    B, C, T = x.shape
    K = w1.shape[1]
    tt = min(T_TILE, T)
    halo = (K - 1) * dilation
    tiles = channel_tiles(C)
    out = nl.ndarray((B, C, T), dtype=x.dtype, buffer=nl.shared_hbm)

    # Weights and per-channel params stay resident in SBUF for the whole call.
    w1_sb, w2_sb, prm = [], [], []
    for ci, (c0, cs) in enumerate(tiles):
        t1 = nl.ndarray((cs, K, C), dtype=w1.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t1, src=w1[c0:c0 + cs])
        t2 = nl.ndarray((cs, C), dtype=w2.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t2, src=w2[c0:c0 + cs, 0, :])
        w1_sb.append(t1)
        w2_sb.append(t2)
        p = nl.ndarray((cs, 6), dtype=nl.float32, buffer=nl.sbuf)
        for j, src in enumerate((alpha1, inv_beta_sqrt1, b1, alpha2, inv_beta_sqrt2, b2)):
            nisa.dma_copy(dst=p[:, j:j + 1], src=src[c0:c0 + cs])
        prm.append(p)

    snake = _snake_terms if split_snake else _snake_fused

    def tile(b, t0, first: bool):
        # snake1(x) over [t0 - halo, t0 + tt); the first tile's left context is causal zero padding.
        act1, xres = [], []
        for ci, (c0, cs) in enumerate(tiles):
            xw = nl.ndarray((cs, halo + tt), dtype=nl.float32, buffer=nl.sbuf)
            if first:
                if halo:
                    nisa.memset(dst=xw[:, 0:halo], value=0.0)
                nisa.dma_copy(dst=xw[:, halo:halo + tt], src=x[b, c0:c0 + cs, 0:tt])
            else:
                nisa.dma_copy(dst=xw, src=x[b, c0:c0 + cs, t0 - halo:t0 + tt])
            xres.append(xw)
            act1.append(snake(xw, prm[ci][:, 0:1], prm[ci][:, 1:2], (cs, halo + tt), CAST_ENGINE))
        # conv1 (K taps x Cin tiles) + bias -> snake2 -> bf16
        act2 = []
        for co, (o0, os_) in enumerate(tiles):
            acc = nl.ndarray((os_, tt), dtype=nl.float32, buffer=nl.psum)
            first_mm = True
            for ci, (c0, cs) in enumerate(tiles):
                for k in nl.static_range(K):
                    for term in act1[ci]:
                        nisa.nc_matmul(dst=acc, stationary=w1_sb[ci][:, k, o0:o0 + os_],
                                       moving=term[:, k * dilation:k * dilation + tt], accumulate=not first_mm)
                        first_mm = False
            h = nl.ndarray((os_, tt), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=h, data=acc, op0=nl.add, operand0=prm[co][:, 2:3])  # Copy activation: immediate bias only
            act2.append(snake(h, prm[co][:, 3:4], prm[co][:, 4:5], (os_, tt), CAST_ENGINE))
        # conv2 (1x1) + bias + residual
        for co, (o0, os_) in enumerate(tiles):
            acc2 = nl.ndarray((os_, tt), dtype=nl.float32, buffer=nl.psum)
            first_mm = True
            for ci, (c0, cs) in enumerate(tiles):
                for term in act2[ci]:
                    nisa.nc_matmul(dst=acc2, stationary=w2_sb[ci][:, o0:o0 + os_], moving=term, accumulate=not first_mm)
                    first_mm = False
            y = nl.ndarray((os_, tt), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=y, data=acc2, op0=nl.add, operand0=prm[co][:, 5:6])
            y2 = nl.ndarray((os_, tt), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y2, data1=y, data2=xres[co][:, halo:halo + tt], op=nl.add)
            nisa.dma_copy(dst=out[b, o0:o0 + os_, t0:t0 + tt], src=y2)

    for b in nl.affine_range(B):
        tile(b, 0, True)
        for t in nl.affine_range(1, T // tt):
            tile(b, t * tt, False)
    return out


def transconv_weight_taps(conv):
    """ConvTranspose1d weight [Cin, Cout, K] -> [Cin, K, Cout] bf16."""
    return conv.weight.detach().permute(0, 2, 1).contiguous().to(torch.bfloat16)


def snake_transconv_params(snake, transconv_net):
    alpha, ibs = snake_params(snake)
    conv = transconv_net.conv
    return dict(alpha=alpha, inv_beta_sqrt=ibs, w=transconv_weight_taps(conv),
                b=conv.bias.detach().float().reshape(-1, 1).contiguous(), rate=int(conv.stride[0]),
                split_snake=conv.in_channels <= 192)


SBUF_WEIGHT_BUDGET = 96 * 1024  # bytes per partition for resident weights (SBUF: 192 KiB/partition)


@nki.jit
def snake_transconv(x, alpha, inv_beta_sqrt, w, b, rate: int, split_snake: bool = True):
    """Fused SnakeBeta + causal ConvTranspose1d(kernel=2*rate, stride=rate), whole-sequence.

    x [B, Cin, Tin] fp32 -> y [B, Cout, Tin*rate] fp32. Output phase m of input step n:
    y[:, n*rate + m] = W[:, :, m].T x[n] + W[:, :, m + rate].T x[n - 1] + b   (x[-1] = 0, causal)
    which equals upstream ConvTranspose1d followed by trimming ``rate`` samples on the right.
    Tin % T_TILE == 0 or Tin < T_TILE.
    """
    B, Cin, Tin = x.shape
    K, Cout = w.shape[1], w.shape[2]
    tt = min(T_TILE, Tin)
    ci_tiles, co_tiles = channel_tiles(Cin), channel_tiles(Cout)
    out = nl.ndarray((B, Cout, Tin * rate), dtype=x.dtype, buffer=nl.shared_hbm)
    snake = _snake_terms if split_snake else _snake_fused

    prm = []
    for c0, cs in ci_tiles:
        p = nl.ndarray((cs, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=p[:, 0:1], src=alpha[c0:c0 + cs])
        nisa.dma_copy(dst=p[:, 1:2], src=inv_beta_sqrt[c0:c0 + cs])
        prm.append(p)

    # Output-channel groups whose weights fit in SBUF together.
    per_tile_bytes = len(ci_tiles) * K * P * 2
    group = max(1, min(len(co_tiles), SBUF_WEIGHT_BUDGET // per_tile_bytes))
    for g0 in nl.static_range(0, len(co_tiles), group):
        grp = co_tiles[g0:g0 + group]
        lo, hi = grp[0][0], grp[-1][0] + grp[-1][1]
        w_sb, b_sb = [], []
        for c0, cs in ci_tiles:
            t = nl.ndarray((cs, K, hi - lo), dtype=w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=w[c0:c0 + cs, :, lo:hi])
            w_sb.append(t)
        for o0, os_ in grp:
            bt = nl.ndarray((os_, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=bt, src=b[o0:o0 + os_])
            b_sb.append(bt)

        def tile(bi, n0, first: bool):
            # snake(x) over input steps [n0 - 1, n0 + tt): one step of left context for the k >= rate taps.
            terms = []
            for ci, (c0, cs) in enumerate(ci_tiles):
                xw = nl.ndarray((cs, 1 + tt), dtype=nl.float32, buffer=nl.sbuf)
                if first:
                    nisa.memset(dst=xw[:, 0:1], value=0.0)
                    nisa.dma_copy(dst=xw[:, 1:1 + tt], src=x[bi, c0:c0 + cs, 0:tt])
                else:
                    nisa.dma_copy(dst=xw, src=x[bi, c0:c0 + cs, n0 - 1:n0 + tt])
                terms.append(snake(xw, prm[ci][:, 0:1], prm[ci][:, 1:2], (cs, 1 + tt), CAST_ENGINE))
            for gi, (o0, os_) in enumerate(grp):
                y = nl.ndarray((os_, tt, rate), dtype=nl.float32, buffer=nl.sbuf)
                for m in nl.static_range(rate):
                    acc = nl.ndarray((os_, tt), dtype=nl.float32, buffer=nl.psum)
                    first_mm = True
                    for ci, (c0, cs) in enumerate(ci_tiles):
                        for term in terms[ci]:
                            # current step n (tap m) and previous step n-1 (tap m + rate)
                            for k, off in ((m, 1), (m + rate, 0)):
                                nisa.nc_matmul(dst=acc, stationary=w_sb[ci][:, k, o0 - lo:o0 - lo + os_],
                                               moving=term[:, off:off + tt], accumulate=not first_mm)
                                first_mm = False
                    nisa.tensor_scalar(dst=y[:, :, m], data=acc, op0=nl.add, operand0=b_sb[gi])
                nisa.dma_copy(dst=out[bi, o0:o0 + os_, n0 * rate:(n0 + tt) * rate], src=y.reshape((os_, tt * rate)))

        for bi in nl.affine_range(B):
            tile(bi, 0, True)
            for t in nl.affine_range(1, Tin // tt):
                tile(bi, t * tt, False)
    return out


def snake_conv_out_params(snake, conv_net):
    alpha, ibs = snake_params(snake)
    conv = conv_net.conv
    return dict(alpha=alpha, inv_beta_sqrt=ibs, w=conv_weight_taps(conv),
                b=conv.bias.detach().float().reshape(1, 1).contiguous())


@nki.jit
def snake_conv_out(x, alpha, inv_beta_sqrt, w, b):
    """Decoder tail: clamp(conv(SnakeBeta(x)), -1, 1), conv Cin->1, K=7 causal; whole-sequence.

    x [B, Cin<=128, T] fp32 -> wav [B, 1, T] fp32. Snake terms fed as separate matmuls (VectorE-bound).
    """
    B, C, T = x.shape
    K = w.shape[1]
    halo = K - 1
    tt = min(T_TILE, T)
    out = nl.ndarray((B, 1, T), dtype=x.dtype, buffer=nl.shared_hbm)
    w_sb = nl.ndarray((C, K, 1), dtype=w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_sb, src=w)
    prm = nl.ndarray((C, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=prm[:, 0:1], src=alpha)
    nisa.dma_copy(dst=prm[:, 1:2], src=inv_beta_sqrt)
    b_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_sb, src=b)

    def tile(bi, t0, first: bool):
        xw = nl.ndarray((C, halo + tt), dtype=nl.float32, buffer=nl.sbuf)
        if first:
            nisa.memset(dst=xw[:, 0:halo], value=0.0)
            nisa.dma_copy(dst=xw[:, halo:halo + tt], src=x[bi, :, 0:tt])
        else:
            nisa.dma_copy(dst=xw, src=x[bi, :, t0 - halo:t0 + tt])
        terms = _snake_terms(xw, prm[:, 0:1], prm[:, 1:2], (C, halo + tt), CAST_ENGINE)
        acc = nl.ndarray((1, tt), dtype=nl.float32, buffer=nl.psum)
        first_mm = True
        for k in nl.static_range(K):
            for term in terms:
                nisa.nc_matmul(dst=acc, stationary=w_sb[:, k, :], moving=term[:, k:k + tt], accumulate=not first_mm)
                first_mm = False
        y = nl.ndarray((1, tt), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y, data=acc, op0=nl.add, operand0=b_sb, op1=nl.minimum, operand1=1.0)
        y2 = nl.ndarray((1, tt), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=y2, data=y, op0=nl.maximum, operand0=-1.0)
        nisa.dma_copy(dst=out[bi, :, t0:t0 + tt], src=y2)

    for bi in nl.affine_range(B):
        tile(bi, 0, True)
        for t in nl.affine_range(1, T // tt):
            tile(bi, t * tt, False)
    return out
