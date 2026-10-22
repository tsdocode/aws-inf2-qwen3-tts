"""Full fused Talker step (28 layers, paged KV, mixed decode + prefill) vs the repo's Talker layers on CPU.

History: each sequence's prior context (real codec-embedding rows) runs through the CPU Talker to build real
per-layer K/V, which is scattered into the device paged caches. Then one mixed step (decode tokens + one
prefill chunk) runs on both sides. Reports kernel vs CPU bf16, and both vs a CPU fp32 oracle.
Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/kernels/test_talker_step.py --decode 4 --prefill 32
"""
import argparse, copy, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from cp_graph import MODEL  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402
from talker_decode import rope  # noqa: E402
from talker_kernels import PAGE, build_group_metadata, kv_write_indices, rope_tables, talker_params, talker_step  # noqa: E402


def cpu_talker(talker, x, pos, hist, fp32=False):
    """Causal Talker over new tokens x [n, 2048] at positions pos [n], given per-layer history
    hist[l] = (K [m, 8, 128], V [m, 8, 128]); returns final-normed hidden and the updated history.
    fp32: run as an fp32 oracle, converting ONE layer at a time (a full fp32 copy OOMs the 16 GB host)."""
    if fp32:
        x = x.float()
    a0 = talker.model.layers[0].self_attn
    dt = torch.float32 if fp32 else x.dtype
    inv_freq = 1.0 / (a0.rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    h, new_hist = x, []
    n = x.shape[0]
    for li, layer in enumerate(talker.model.layers):
        if fp32:
            layer = copy.deepcopy(layer).float()
        a = layer.self_attn
        res = h
        q, k, v = a.project_qkv(layer.input_layernorm(h))
        q, k = rope(a.q_norm(q), pos, inv_freq), rope(a.k_norm(k), pos, inv_freq)
        K = torch.cat([hist[li][0], k]) if hist else k
        V = torch.cat([hist[li][1], v]) if hist else v
        new_hist.append((K, V))
        m = K.shape[0]
        qpos = torch.arange(m - n, m)
        allowed = torch.arange(m).unsqueeze(0) <= qpos.unsqueeze(1)                 # [n, m]
        Kr, Vr = K.repeat_interleave(2, 1).float(), V.repeat_interleave(2, 1).float()
        sc = torch.einsum("nhd,mhd->hnm", q.float(), Kr) / 128 ** 0.5
        p = torch.softmax(sc.masked_fill(~allowed, -torch.inf), -1)
        att = torch.einsum("hnm,mhd->nhd", p, Vr).to(dt)
        h = res + a.o_proj(att.reshape(n, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        dt = h.dtype
    norm = copy.deepcopy(talker.model.norm).float() if fp32 else talker.model.norm
    return norm(h), new_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", type=int, default=4)
    ap.add_argument("--prefill", type=int, default=32)
    ap.add_argument("--max-ctx", type=int, default=200)
    ap.add_argument("--prefill-ctx", type=int, default=None, help="context before the prefill chunk (default max_ctx/2)")
    ap.add_argument("--prefill-real", type=int, default=None, help="real tokens in the prefill chunk; rest is padding")
    ap.add_argument("--prewrite", action="store_true", help="race test: put this step's correct K/V in the cache beforehand")
    ap.add_argument("--fp32", action="store_true", help="also run the layer-wise fp32 oracle (slow)")
    a = ap.parse_args()
    torch.manual_seed(0)
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    talker = loader.load_talker(device=torch.device("cpu"), dtype=torch.bfloat16).eval()
    del talker.model.text_embedding, talker.text_projection
    emb = talker.model.codec_embedding.weight.detach()
    L = len(talker.model.layers)
    theta = talker.model.layers[0].self_attn.rope_theta

    seqs = [int(c) for c in torch.randint(20, a.max_ctx, (a.decode,))]
    pre_ctx = a.max_ctx // 2 if a.prefill_ctx is None else a.prefill_ctx
    real = a.prefill if a.prefill_real is None else a.prefill_real
    all_ctx = seqs + ([pre_ctx] if a.prefill else [])
    pps = max((c + 1 + (a.prefill if i == len(seqs) else 0) + PAGE - 1) // PAGE for i, c in enumerate(all_ctx))
    nblk = 1 + len(all_ctx) * pps
    perm = (torch.randperm(nblk) ).tolist()

    t0 = time.perf_counter()
    hists, blocks, priors = [], [], []
    with torch.no_grad():
        for c in all_ctx:
            x_prior = emb[torch.randint(0, emb.shape[0], (c,))]
            priors.append(x_prior)
            _, hist = cpu_talker(talker, x_prior, torch.arange(c), []) if c > 0 else (None, [])
            hists.append(hist)
            blocks.append([perm.pop() for _ in range(pps)])
    print(f"built CPU KV history for contexts {all_ctx} in {time.perf_counter() - t0:.0f}s", flush=True)

    k_cache = torch.zeros(L, nblk, PAGE, 8, 128, dtype=torch.bfloat16)
    v_cache = torch.zeros(L, nblk, PAGE, 8, 128, dtype=torch.bfloat16)
    for hist, blk in zip(hists, blocks):
        for l, (K, V) in enumerate(hist or []):
            for t in range(K.shape[0]):
                b, o = blk[t // PAGE], t % PAGE
                k_cache[l, b, o] = K[t]
                v_cache[l, b, o] = V[t]

    # this step's tokens: one decode token per sequence, then the prefill chunk
    xs, pos, wblk, woff = [], [], [], []
    for i, c in enumerate(seqs):
        xs.append(emb[torch.randint(0, emb.shape[0], (1,))]); pos.append(torch.tensor([c]))
        wblk.append(blocks[i][c // PAGE]); woff.append(c % PAGE)
    if a.prefill:
        xp = torch.zeros(a.prefill, emb.shape[1], dtype=emb.dtype)
        xp[:real] = emb[torch.randint(0, emb.shape[0], (real,))]
        xs.append(xp); pos.append(torch.arange(pre_ctx, pre_ctx + a.prefill))
        for t in range(pre_ctx, pre_ctx + a.prefill):
            real_tok = t < pre_ctx + real
            wblk.append(blocks[-1][t // PAGE] if real_tok else -1); woff.append(t % PAGE)
    x = torch.cat(xs); positions = torch.cat(pos)
    with torch.no_grad():
        ref_parts, new_hists = [], []
        for i, c in enumerate(seqs):
            h, nh = cpu_talker(talker, xs[i], pos[i], hists[i]); ref_parts.append(h); new_hists.append((nh, c, 1))
        if a.prefill:
            h, nh = cpu_talker(talker, xs[-1][:real], pos[-1][:real], hists[-1])
            ref_parts.append(torch.cat([h, torch.zeros(a.prefill - real, h.shape[1], dtype=h.dtype)]))
            new_hists.append((nh, pre_ctx, real))
        if a.prewrite:
            for (nh, c0, n), blk in zip(new_hists, blocks):
                for l, (K, V) in enumerate(nh):
                    for t in range(c0, c0 + n):
                        b, o = blk[t // PAGE], t % PAGE
                        k_cache[l, b, o] = K[t]
                        v_cache[l, b, o] = V[t]
        ref = torch.cat(ref_parts).float()
        if a.fp32:
            parts = []
            for i, c in enumerate(all_ctx):
                _, h32 = cpu_talker(talker, priors[i], torch.arange(c), [], fp32=True) if c > 0 else (None, [])
                is_pre = a.prefill and i == len(all_ctx) - 1
                xi, pi = (xs[i][:real], pos[i][:real]) if is_pre else (xs[i], pos[i])
                hh, _ = cpu_talker(talker, xi, pi, h32, fp32=True)
                if is_pre:
                    hh = torch.cat([hh, torch.zeros(a.prefill - real, hh.shape[1])])
                parts.append(hh)
            ref32 = torch.cat(parts).float()

    groups = [(a.decode, 1, pps)] + ([(1, a.prefill, pps)] if a.prefill else [])
    metas = []
    gseqs = [[(blocks[i], c, 1) for i, c in enumerate(seqs)]] + ([[(blocks[-1], pre_ctx, real)]] if a.prefill else [])
    for (nseg, qps, p), sq in zip(groups, gseqs):
        metas += list(build_group_metadata(sq, qps, p, nblk * L))
    rows = kv_write_indices(wblk, woff, nblk * L)
    cos, sin = rope_tables(positions, theta)
    prm = talker_params(talker)
    eps = prm.pop("eps")
    dev = lambda t: t.contiguous().to("neuron")  # noqa: E731
    kcd, vcd = dev(k_cache), dev(v_cache)   # after the optional --prewrite
    t0 = time.perf_counter()
    out = talker_step(dev(x.t()), dev(cos), dev(sin), **{k: dev(v) for k, v in prm.items()}, k_cache=kcd, v_cache=vcd,
                      rows_hbm=dev(rows), offs0=dev(metas[0]), mask0=dev(metas[1]),
                      **({"offs1": dev(metas[2]), "mask1": dev(metas[3])} if a.prefill else {}),
                      groups=tuple(groups), eps=eps).cpu().float().t()
    print(f"talker_step groups {groups} T={x.shape[0]}: compile+run {time.perf_counter() - t0:.0f}s", flush=True)
    rel = lambda u, w: ((u - w).norm() / w.norm()).item()  # noqa: E731
    nd = len(seqs)
    keep = torch.cat([torch.arange(nd), nd + torch.arange(real)]) if a.prefill else torch.arange(nd)
    out, ref = out[keep], ref[keep]
    print(f"decode rows rel L2 {rel(out[:nd], ref[:nd]):.2e} | prefill rows rel L2 {rel(out[nd:], ref[nd:]) if a.prefill else 0:.2e}")
    if a.prefill:
        per = ((out[nd:] - ref[nd:]).norm(dim=1) / ref[nd:].norm(dim=1)).tolist()
        print("  prefill per-token rel L2:", " ".join(f"{v:.2f}" for v in per))
        if a.fp32:
            r32 = ref32[keep][nd:]
            per32 = ((ref[nd:] - r32).norm(dim=1) / r32.norm(dim=1)).tolist()
            perk = ((out[nd:] - r32).norm(dim=1) / r32.norm(dim=1)).tolist()
            print("  CPU bf16 vs fp32 per token:", " ".join(f"{v:.2f}" for v in per32))
            print("  kernel   vs fp32 per token:", " ".join(f"{v:.2f}" for v in perk))
    print(f"hidden rel L2: kernel vs CPU bf16 {rel(out, ref):.2e}" +
          ("" if not a.fp32 else f" | vs fp32 oracle: kernel {rel(out, ref32):.2e}, CPU bf16 {rel(ref, ref32):.2e}"))
    head = talker.codec_head.float()
    with torch.no_grad():
        lk, lr = head(out), head(ref)
        agree = (lk.argmax(-1) == lr.argmax(-1)).float().mean().item() * 100
        line = f"codec_head argmax agreement: kernel vs CPU bf16 {agree:.1f}%"
        if a.fp32:
            l32 = head(ref32)
            line += (f" | vs fp32: kernel {(lk.argmax(-1) == l32.argmax(-1)).float().mean().item() * 100:.1f}%, "
                     f"CPU bf16 {(lr.argmax(-1) == l32.argmax(-1)).float().mean().item() * 100:.1f}%")
    print(line)
    # did the step's own K/V land in the cache?
    kcpu = kcd.cpu()
    nz = sum(int(torch.allclose(kcpu[5, b, o].float(), kcpu[5, b, o].float()) and kcpu[5, b, o].abs().sum() > 0) for b, o in zip(wblk, woff))
    print(f"new-token K written into cache (layer 5): {nz}/{len(wblk)} slots non-zero")


if __name__ == "__main__":
    main()
