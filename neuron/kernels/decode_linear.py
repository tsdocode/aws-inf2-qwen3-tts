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
