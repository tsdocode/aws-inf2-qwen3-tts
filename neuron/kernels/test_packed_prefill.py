"""Packed (varlen) fresh-prompt prefill vs the padded-segment path, same talker_step, real weights.

Decode streams + 3 prompts of different lengths: path A pads each prompt to a 64-token segment (groups
(S,1,pps) + (4,64,pps)); path B packs them back to back in one 128-token tile (groups (S,1,pps) + (1,128,0)).
Compares the final hidden at every real token and the K/V rows written to the caches.
Usage: NEURON_RT_VISIBLE_CORES=0 python neuron/kernels/test_packed_prefill.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from cp_graph import MODEL  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402
from test_talker_step import cpu_talker  # noqa: E402
from talker_kernels import (PAGE, build_group_metadata, kv_write_indices, packed_prefill_mask, rope_tables,  # noqa: E402
                            talker_params, talker_step)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    torch.manual_seed(0)
    talker = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True)) \
        .load_talker(device=torch.device("cpu"), dtype=torch.bfloat16)
    del talker.model.text_embedding, talker.text_projection
    prm = talker_params(talker)
    eps = prm.pop("eps")
    theta = talker.model.layers[0].self_attn.rope_theta
    L = len(talker.model.layers)
    d = {k: v.contiguous().to("neuron") for k, v in prm.items()}
    S, pps, lens = 4, 2, [37, 52, 20]
    nblk = 16
    oob = nblk * L
    kc0 = (torch.randn(L, nblk, PAGE, 8, 128) * 0.5).to(torch.bfloat16)
    vc0 = torch.randn(L, nblk, PAGE, 8, 128).to(torch.bfloat16)
    dec_ctx = [10, 100, 150, 200]
    dec_blocks = [[0, 1], [2, 3], [4, 5], [6, 7]]
    pre_blocks = [[8], [9], [10]]
    xd = torch.randn(S, 2048) * 0.05
    xp = [torch.randn(n, 2048) * 0.05 for n in lens]
    dec_meta = [(b, c, 1) for b, c in zip(dec_blocks, dec_ctx)]
    wd = [b[c // PAGE] for b, c in zip(dec_blocks, dec_ctx)]
    od = [c % PAGE for c in dec_ctx]

    def run(packed):
        xs, wblk, woff, pos, real = [xd], list(wd), list(od), list(dec_ctx), list(range(S))
        if packed:
            seg = []
            for i, n in enumerate(lens):
                real += list(range(S + len(seg), S + len(seg) + n))
                xs.append(xp[i]); wblk += [pre_blocks[i][0]] * n; woff += list(range(n)); pos += list(range(n))
                seg += [i] * n
            pad = PAGE - len(seg)
            xs.append(torch.zeros(pad, 2048)); wblk += [-1] * pad; woff += [0] * pad; pos += [0] * pad
            seg += [-1] * pad
            groups = ((S, 1, pps), (1, PAGE, 0))
            extra = dict(offs1=None, mask1=packed_prefill_mask(seg).to("neuron"))
        else:
            meta = []
            for i in range(4):
                n = lens[i] if i < len(lens) else 0
                if n:
                    real += list(range(S + 64 * i, S + 64 * i + n))
                    xs.append(xp[i])
                    meta.append((pre_blocks[i], 0, n))
                else:
                    meta.append(([], 0, 0))
                xs.append(torch.zeros(64 - n, 2048))
                wblk += ([pre_blocks[i][0]] * n if n else []) + [-1] * (64 - n)
                woff += list(range(n)) + [0] * (64 - n)
                pos += list(range(n)) + [0] * (64 - n)
            groups = ((S, 1, pps), (4, 64, pps))
            o1, m1 = build_group_metadata(meta, 64, pps, oob)
            extra = dict(offs1=o1.to("neuron"), mask1=m1.to("neuron"))
        o0, m0 = build_group_metadata(dec_meta, 1, pps, oob)
        x = torch.cat(xs).to(torch.bfloat16)
        cos, sin = rope_tables(torch.tensor(pos), theta)
        kc, vc = kc0.clone().to("neuron"), vc0.clone().to("neuron")
        h = talker_step(x.t().contiguous().to("neuron"), cos.to("neuron"), sin.to("neuron"), **d, k_cache=kc, v_cache=vc,
                        rows_hbm=kv_write_indices(wblk, woff, oob).to("neuron"), offs0=o0.to("neuron"),
                        mask0=m0.to("neuron"), **extra, groups=groups, eps=eps)
        return h.cpu().t()[real], kc.cpu(), vc.cpu()

    ha, ka, va = run(False)
    hb, kb, vb = run(True)
    print(f"hidden rel L2 (all {ha.shape[0]} real tokens): {rel(hb, ha):.2e}; decode rows {rel(hb[:S], ha[:S]):.2e}; "
          f"prompt rows {rel(hb[S:], ha[S:]):.2e}")
    with torch.no_grad():
        ref = torch.cat([cpu_talker(talker, x.to(torch.bfloat16), torch.arange(x.shape[0]), None, fp32=True)[0] for x in xp])
    print(f"vs fp32 oracle, prompt rows: padded path {rel(ha[S:], ref):.2e} | packed path {rel(hb[S:], ref):.2e}")
    print(f"K cache rel L2 {rel(kb, ka):.2e} | V cache rel L2 {rel(vb, va):.2e} | "
          f"cos(last prompt token) {torch.nn.functional.cosine_similarity(hb[-1].float(), ha[-1].float(), 0).item():.5f}")


if __name__ == "__main__":
    main()
