"""Mixed decode + prefill paged attention (NKI) vs dense torch attention, one layer.

Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/tests/test_paged_attention.py --decode 32 --prefill 64 --pps 4
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "kernels"))
from talker_kernels import PAGE, build_group_metadata, paged_attention  # noqa: E402

D, NH, KVH = 128, 16, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--prefill", type=int, default=64, help="query tokens of the single prefill segment (0 = none)")
    ap.add_argument("--pps", type=int, default=4, help="pages per segment (static)")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    torch.manual_seed(0)
    max_ctx = a.pps * PAGE
    nblk = 1 + (a.decode + 1) * a.pps
    perm = (torch.randperm(nblk - 1) + 1).tolist()                       # block 0 = scratch
    k_cache = (torch.randn(nblk, PAGE, KVH, D) * 0.5).to(torch.bfloat16)
    v_cache = torch.randn(nblk, PAGE, KVH, D).to(torch.bfloat16)
    k_cache[0] = 0; v_cache[0] = 0

    def alloc(n_tokens):
        nb = (n_tokens + PAGE - 1) // PAGE
        return [perm.pop() for _ in range(nb)]

    dec = []
    for _ in range(a.decode):
        ctx = int(torch.randint(0, max_ctx - 1, (1,)))           # tokens before this step; new token at pos ctx
        dec.append((alloc(ctx + 1), ctx, 1))
    groups, offs, masks, segs = [(a.decode, 1, a.pps)], [], [], [dec]
    if a.prefill:
        ctx = max_ctx - a.prefill - 7
        pre = [(alloc(ctx + a.prefill), ctx, a.prefill)]
        groups.append((1, a.prefill, a.pps))
        segs.append(pre)
    for (nseg, qps, pps), seqs in zip(groups, segs):
        o, m = build_group_metadata(seqs, qps, pps, nblk)
        offs.append(o); masks.append(m)
    T = sum(n * q for n, q, _ in groups)
    q = torch.randn(D, NH, T).to(torch.bfloat16)

    # dense reference
    ref = torch.zeros(D, NH, T)
    tok = 0
    for (nseg, qps, pps), seqs in zip(groups, segs):
        for blocks, ctx, nq in seqs:
            K = torch.cat([k_cache[b] for b in blocks]).float()                    # [keys, KVH, D]
            V = torch.cat([v_cache[b] for b in blocks]).float()                    # [keys, KVH, D]
            for j in range(nq):
                n = ctx + j + 1
                for h in range(NH):
                    qv = q[:, h, tok + j].float()
                    p = torch.softmax((K[:n, h // 2] @ qv) / D ** 0.5, 0)
                    ref[:, h, tok + j] = p @ V[:n, h // 2]
            tok += qps
    dev = lambda x: x.contiguous().to("neuron")  # noqa: E731
    meta = [dev(t) for pair in zip(offs, masks) for t in pair]
    args = (dev(q), dev(k_cache), dev(v_cache), *meta)
    kw = dict(groups=tuple(groups))
    t0 = time.perf_counter()
    out = paged_attention(*args, **kw).cpu().float()
    print(f"groups {groups}, T={T}: compile+run {time.perf_counter() - t0:.1f}s")
    bad = ~torch.isfinite(out)
    if bad.any():
        toks = bad.any(0).any(0).nonzero().flatten().tolist()
        heads = bad.any(0).any(1).nonzero().flatten().tolist()
        print(f"NON-FINITE outputs: {bad.sum().item()} values, tokens {toks[:12]}{'...' if len(toks) > 12 else ''}, heads {heads}")
        fin = torch.isfinite(out).all(0).all(0)
        print(f"finite tokens rel L2: {((out[..., fin] - ref[..., fin]).norm() / ref[..., fin].norm()).item():.2e}")
    err = ((out - ref).norm() / ref.norm()).item()
    per_tok = ((out - ref).norm(dim=(0, 1)) / ref.norm(dim=(0, 1))).max().item()
    print(f"rel L2 vs dense torch: {err:.2e} (worst token {per_tok:.2e})")
    for _ in range(3):
        paged_attention(*args, **kw)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        paged_attention(*args, **kw)
    torch.neuron.synchronize()
    us = (time.perf_counter() - t0) / a.iters * 1e6
    pages = sum(n * p for n, _, p in groups)
    kv_bytes = pages * PAGE * KVH * D * 2 * 2
    real = sum(len(b) for seqs in segs for b, _, _ in seqs) * PAGE * KVH * D * 2 * 2
    print(f"{us:.0f} us per layer (wall, incl ~140 us launch); KV pages padded {kv_bytes / 1e6:.1f} MB, real {real / 1e6:.1f} MB -> "
          f"x28 layers = {us * 28 / 1e3:.1f} ms per step if standalone")


if __name__ == "__main__":
    main()
