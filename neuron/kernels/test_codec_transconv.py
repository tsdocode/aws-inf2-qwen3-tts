"""Fused SnakeBeta + causal ConvTranspose1d (NKI) vs upstream fp32, every decoder block, real activations.

Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/kernels/test_codec_transconv.py --rows 4 --frames 16
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from codec_kernels import snake_transconv, snake_transconv_params  # noqa: E402
from cp_graph import MODEL  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402
from test_codec_kernels import GOLDEN, snr_db  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--frames", type=int, default=16)
    a = ap.parse_args()
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    decoder = loader.load_codec(device=torch.device("cpu"), dtype=torch.float32).float().decoder.eval()
    codes = torch.load(GOLDEN / "medium.codes.pt")
    chunk = torch.stack([codes[s:s + a.frames] for s in range(0, a.rows * 7, 7)]).transpose(1, 2).contiguous()
    blocks = {f"block{i}": decoder.decoder[1 + i] for i in range(4)}
    cap = {}
    hooks = []
    for n, blk in blocks.items():
        hooks.append(blk.block[0].register_forward_hook(lambda m, i, o, n=n: cap.__setitem__(n + ".in", i[0].detach().clone())))
        hooks.append(blk.block[1].register_forward_hook(lambda m, i, o, n=n: cap.__setitem__(n + ".out", o.detach().clone())))
    with torch.no_grad():
        decoder(chunk)
    for h in hooks:
        h.remove()
    for n, blk in blocks.items():
        x, ref = cap[n + ".in"], cap[n + ".out"]
        prm = snake_transconv_params(blk.block[0], blk.block[1])
        rate, split = prm.pop("rate"), prm.pop("split_snake")
        dev = {k: v.to("neuron") for k, v in prm.items()}
        t0 = time.perf_counter()
        y = snake_transconv(x.contiguous().to("neuron"), rate=rate, split_snake=split, **dev).cpu()
        print(f"{n}: x {tuple(x.shape)} -> {tuple(y.shape)} (ref {tuple(ref.shape)}), rate {rate}, "
              f"{time.perf_counter() - t0:.1f}s, SNR {snr_db(y, ref):.1f} dB", flush=True)


if __name__ == "__main__":
    main()
