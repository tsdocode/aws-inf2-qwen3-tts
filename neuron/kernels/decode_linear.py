"""NKI decode kernels for NeuronCore-v2 (inf2), transposed-activation layout.

Activations live as ``xT [features, B]`` (features on SBUF partitions, batch on the free axis), so
weights are the TensorE *stationary* operand and every matmul output is already in the layout the
next matmul consumes: no transposes anywhere in a layer. With B <= 64 the MultiplyMoving cost is
the 64-cycle floor per 128x128 weight tile, so TensorE keeps up with HBM.

Weights are pre-tiled once at load time (see ``tile_weight``) so every weight DMA moves >= 2 KiB
contiguous per partition — below that, per-descriptor overhead caps trn1/inf2 DMA well under its
272 GB/s per-core peak (16 engines x 17 B/ns; nki-dma-bandwidth-guide).
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128          # partitions / TensorE rows
N_BLOCK = 1024   # weight columns per block; a block is [P, K/P, N_BLOCK] contiguous per partition


def tile_weight(weight):
    """torch Linear weight [N, K] -> [N/N_BLOCK, P, K/P, N_BLOCK] contiguous (host-side, once).

    Element [nb, p, kt, c] = W.T[kt*P + p, nb*N_BLOCK + c].
    """
    n, k = weight.shape
    assert k % P == 0 and n % N_BLOCK == 0, (n, k)
    wT = weight.t()
    return wT.reshape(k // P, P, n // N_BLOCK, N_BLOCK).permute(2, 1, 0, 3).contiguous()


@nki.jit
def rmsnorm_linear_t(xT, norm_w, w_tiled, eps=1e-6):
    """yT = (RMSNorm(x) @ W).T for decode.

    xT [K, B] bf16, norm_w [K, 1] fp32, w_tiled = tile_weight(W) bf16 -> yT [N, B] bf16.
    RMSNorm matches the repo: fp32 statistics, ``x.float() * rsqrt(mean(x^2) + eps) * w`` then one
    cast to bf16. K % 128 == 0, N % 512 == 0, B <= 128.
    """
    K, B = xT.shape
    NB = w_tiled.shape[0]
    N = NB * N_BLOCK
    KT = K // P

    out = nl.ndarray((N, B), dtype=xT.dtype, buffer=nl.shared_hbm)

    x = nl.ndarray((P, KT, B), dtype=xT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=xT.reshape((KT, P, B)).permute((1, 0, 2)))
    w_norm = nl.ndarray((P, KT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_norm, src=norm_w.reshape((KT, P, 1)).permute((1, 0, 2)))

    # sum_k x[k, b]^2, broadcast to every partition in one pass: ones[P, P] stationary.
    ones = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    sum_sq = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.psum)
    for k in nl.static_range(KT):
        sq = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=x[:, k, :])
        nisa.nc_matmul(dst=sum_sq, stationary=ones, moving=sq, accumulate=(k > 0))

    inv_rms = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_rms, op=nl.rsqrt, data=sum_sq, scale=1.0 / K, bias=eps)

    xn = nl.ndarray((P, KT, B), dtype=xT.dtype, buffer=nl.sbuf)
    for k in nl.static_range(KT):
        scaled = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=scaled, data1=x[:, k, :], data2=inv_rms, op=nl.multiply)
        nisa.tensor_scalar(dst=xn[:, k, :], data=scaled, op0=nl.multiply, operand0=w_norm[:, k, :])

    for nb in nl.affine_range(NB):
        w_blk = nl.ndarray((P, KT, N_BLOCK), dtype=w_tiled.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_blk, src=w_tiled[nb])
        for j in nl.static_range(N_BLOCK // P):
            acc = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.psum)
            for k in nl.static_range(KT):
                nisa.nc_matmul(dst=acc, stationary=w_blk[:, k, j * P:(j + 1) * P], moving=xn[:, k, :],
                               accumulate=(k > 0))
            y = nl.ndarray((P, B), dtype=xT.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=y, src=acc)
            col = nb * N_BLOCK + j * P
            nisa.dma_copy(dst=out[col:col + P, :], src=y)
    return out


def quantize_fp8_e4m3(weight):
    """Per-output-channel weight-only FP8 for NeuronCore-v2 (legacy IEEE-style e4m3, max 240).

    torch has no legacy-e4m3 dtype, so the encoding comes from ml_dtypes and travels as uint8; the
    kernel reinterprets it with ``.view(nl.float8_e4m3)``. Returns (tile_weight(q) as uint8, scale [N, 1] fp32).
    """
    import ml_dtypes
    import numpy as np
    import torch

    w = weight.detach().float()
    scale = (w.abs().amax(dim=1, keepdim=True) / 240.0).clamp_min(1e-12)
    q = (w / scale).numpy().astype(ml_dtypes.float8_e4m3).view(np.uint8)
    return tile_weight(torch.from_numpy(q.copy())), scale.contiguous()


def dequantize_fp8_e4m3(q_tiled, scale, n, k):
    """Inverse of quantize_fp8_e4m3 on the host (reference math for tests)."""
    import ml_dtypes
    import numpy as np
    import torch

    nb = q_tiled.shape[0]
    tiles = torch.from_numpy(q_tiled.numpy().view(ml_dtypes.float8_e4m3).astype(np.float32))
    wT = tiles.permute(2, 1, 0, 3).reshape(k, nb * N_BLOCK)
    return wT.t()[:n] * scale


@nki.jit
def rmsnorm_linear_fp8_t(xT, norm_w, w_tiled_u8, w_scale, eps=1e-6):
    """``rmsnorm_linear_t`` with weight-only FP8: stationary fp8 weights x moving bf16 activations
    (mixed dtypes are legal on v2; fp32 PSUM), per-output-channel scale applied on the PSUM copy-out
    (output channels are partitions here, so it is one per-partition multiply). Half the weight bytes.
    """
    w_tiled = w_tiled_u8.view(nl.float8_e4m3)
    K, B = xT.shape
    NB = w_tiled.shape[0]
    N = NB * N_BLOCK
    KT = K // P

    out = nl.ndarray((N, B), dtype=xT.dtype, buffer=nl.shared_hbm)
    x = nl.ndarray((P, KT, B), dtype=xT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=xT.reshape((KT, P, B)).permute((1, 0, 2)))
    w_norm = nl.ndarray((P, KT, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_norm, src=norm_w.reshape((KT, P, 1)).permute((1, 0, 2)))
    scale = nl.ndarray((P, N // P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale, src=w_scale.reshape((N // P, P, 1)).permute((1, 0, 2)))

    ones = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    sum_sq = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.psum)
    for k in nl.static_range(KT):
        sq = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq, op=nl.square, data=x[:, k, :])
        nisa.nc_matmul(dst=sum_sq, stationary=ones, moving=sq, accumulate=(k > 0))
    inv_rms = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_rms, op=nl.rsqrt, data=sum_sq, scale=1.0 / K, bias=eps)
    xn = nl.ndarray((P, KT, B), dtype=xT.dtype, buffer=nl.sbuf)
    for k in nl.static_range(KT):
        scaled = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=scaled, data1=x[:, k, :], data2=inv_rms, op=nl.multiply)
        nisa.tensor_scalar(dst=xn[:, k, :], data=scaled, op0=nl.multiply, operand0=w_norm[:, k, :])

    for nb in nl.affine_range(NB):
        w_blk = nl.ndarray((P, KT, N_BLOCK), dtype=w_tiled.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_blk, src=w_tiled[nb])
        for j in nl.static_range(N_BLOCK // P):
            acc = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.psum)
            for k in nl.static_range(KT):
                nisa.nc_matmul(dst=acc, stationary=w_blk[:, k, j * P:(j + 1) * P], moving=xn[:, k, :],
                               accumulate=(k > 0))
            y = nl.ndarray((P, B), dtype=xT.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=y, data=acc, op0=nl.multiply, operand0=scale[:, nb * (N_BLOCK // P) + j, :])
            col = nb * N_BLOCK + j * P
            nisa.dma_copy(dst=out[col:col + P, :], src=y)
    return out
