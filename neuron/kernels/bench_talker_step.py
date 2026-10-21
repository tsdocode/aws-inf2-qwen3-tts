"""Time the fused Talker step (28 layers, mixed decode + prefill, paged KV) on a serving-like batch.

Random caches/inputs (timing only; correctness is test_talker_step.py). Contexts uniform in [1, max_ctx).
Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/kernels/bench_talker_step.py --decode 32 --prefill 64 --max-ctx 512
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from cp_graph import MODEL  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402
from talker_kernels import PAGE, build_group_metadata, kv_write_indices, rope_tables, talker_params, talker_step  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--prefill", type=int, default=64)
    ap.add_argument("--max-ctx", type=int, default=512)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--fixed-ctx", action="store_true", help="every decode stream at max_ctx - 1 (like the compiled baseline)")
    ap.add_argument("--no-kv-write", action="store_true")
    ap.add_argument("--no-attention", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    talker = loader.load_talker(device=torch.device("cpu"), dtype=torch.bfloat16)
    del talker.model.text_embedding, talker.text_projection
    prm = talker_params(talker)
    eps = prm.pop("eps")
    theta = talker.model.layers[0].self_attn.rope_theta
    L = len(talker.model.layers)
    del talker

    pps = (a.max_ctx + PAGE - 1) // PAGE
    nblk = (a.decode + 1) * pps
    perm = torch.randperm(nblk).tolist()
    ctxs = [a.max_ctx - 1] * a.decode if a.fixed_ctx else [int(c) for c in torch.randint(1, a.max_ctx - 1, (a.decode,))]
    dec = []
    wblk, woff, pos = [], [], []
    for c in ctxs:
        blocks = [perm.pop() for _ in range((c + 1 + PAGE - 1) // PAGE)]
        dec.append((blocks, c, 1)); wblk.append(blocks[c // PAGE]); woff.append(c % PAGE); pos.append(c)
    groups, metas = [(a.decode, 1, pps)], list(build_group_metadata(dec, 1, pps, nblk))
    if a.prefill:
        pctx = max(0, a.max_ctx - a.prefill - 1) // 2
        blocks = [perm.pop() for _ in range((pctx + a.prefill + PAGE - 1) // PAGE)]
        groups.append((1, a.prefill, pps))
        metas += list(build_group_metadata([(blocks, pctx, a.prefill)], a.prefill, pps, nblk))
        for t in range(pctx, pctx + a.prefill):
            wblk.append(blocks[t // PAGE]); woff.append(t % PAGE); pos.append(t)
    T = len(pos)
    rows = kv_write_indices(wblk, woff, nblk)
    cos, sin = rope_tables(torch.tensor(pos), theta)
    dev = lambda t: t.contiguous().to("neuron")  # noqa: E731
    d = {k: dev(v) for k, v in prm.items()}
    kc = (torch.randn(L, nblk, PAGE, 8, 128) * 0.5).to(torch.bfloat16).to("neuron")
    vc = torch.randn(L, nblk, PAGE, 8, 128).to(torch.bfloat16).to("neuron")
    x = (torch.randn(2048, T) * 0.05).to(torch.bfloat16).to("neuron")
    kwargs = dict(k_cache=kc, v_cache=vc, rows_hbm=dev(rows), offs0=dev(metas[0]), mask0=dev(metas[1]),
                  groups=tuple(groups), eps=eps, do_kv_write=not a.no_kv_write, do_attention=not a.no_attention)
    if a.prefill:
        kwargs.update(offs1=dev(metas[2]), mask1=dev(metas[3]))
    t0 = time.perf_counter()
    talker_step(x, dev(cos), dev(sin), **d, **kwargs)
    torch.neuron.synchronize()
    print(f"compile + first step: {time.perf_counter() - t0:.0f}s", flush=True)
    c_, s_ = dev(cos), dev(sin)
    for _ in range(2):
        talker_step(x, c_, s_, **d, **kwargs)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        talker_step(x, c_, s_, **d, **kwargs)
    torch.neuron.synchronize()
    ms = (time.perf_counter() - t0) / a.iters * 1e3
    real_pages = sum(len(b) for b, _, _ in dec) + (len(blocks) if a.prefill else 0)
    kv = real_pages * PAGE * 8 * 128 * 2 * 2 * L
    w = 2.82e9
    tag = ("" if not a.no_kv_write else "[NO KV WRITE] ") + ("" if not a.no_attention else "[NO ATTENTION] ")
    print(f"{tag}decode {a.decode} + prefill {a.prefill} (T={T}), ctx < {a.max_ctx} (mean {sum(ctxs) / len(ctxs):.0f}): "
          f"{ms:.2f} ms per talker step | bytes: weights 2.82 GB + KV {kv / 1e9:.2f} GB -> {(w + kv) / ms / 1e6:.0f} GB/s "
          f"[compiled talker: 29.6 ms, decode-only B=32 ctx 512]")


if __name__ == "__main__":
    main()
