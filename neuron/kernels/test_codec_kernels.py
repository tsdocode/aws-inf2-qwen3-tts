"""Fused codec residual unit (NKI) vs upstream fp32 module on real decoder activations.

Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/kernels/test_codec_kernels.py --rows 4 --frames 16
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from codec_kernels import residual_unit, residual_unit_params  # noqa: E402
from cp_graph import MODEL  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def snr_db(got, ref):
    return 10 * torch.log10(ref.pow(2).sum() / (got - ref).pow(2).sum().clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--only", default=None, help="substring of the unit name to test")
    a = ap.parse_args()
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    decoder = loader.load_codec(device=torch.device("cpu"), dtype=torch.float32).float().decoder.eval()
    codes = torch.load(GOLDEN / "medium.codes.pt")
    chunk = torch.stack([codes[s:s + a.frames] for s in range(0, a.rows * 7, 7)]).transpose(1, 2).contiguous()

    targets = {"block0.ru_d1 (C=768)": decoder.decoder[1].block[2], "block3.ru_d9 (C=96)": decoder.decoder[4].block[4]}
    captured = {}
    hooks = [m.register_forward_hook(lambda m, i, o, n=n: captured.__setitem__(n, (i[0].detach().clone(), o.detach().clone())))
             for n, m in targets.items()]
    with torch.no_grad():
        decoder(chunk)
    for h in hooks:
        h.remove()

    for name, unit in targets.items():
        if a.only and a.only not in name:
            continue
        x, ref = captured[name]
        prm = residual_unit_params(unit)
        dil = prm.pop("dilation")
        split = prm.pop("split_snake")
        dev = {k: v.to("neuron") for k, v in prm.items()}
        xd = x.contiguous().to("neuron")
        t0 = time.perf_counter()
        y = residual_unit(xd, dilation=dil, split_snake=split, **dev).cpu()
        print(f"{name}: x {tuple(x.shape)}, compile+run {time.perf_counter() - t0:.1f}s, "
              f"SNR vs upstream fp32 = {snr_db(y, ref):.1f} dB, max abs err {(y - ref).abs().max().item():.2e}", flush=True)
        for _ in range(3):
            residual_unit(xd, dilation=dil, split_snake=split, **dev)
        t0 = time.perf_counter()
        for _ in range(10):
            out = residual_unit(xd, dilation=dil, split_snake=split, **dev)
        out.cpu()
        B, C, T = x.shape
        flops = 2 * B * T * C * C * (7 + 1)
        ms = (time.perf_counter() - t0) / 10 * 1e3
        print(f"    {ms:.2f} ms/call wall; {flops / 1e9:.2f} GFLOP -> {flops / ms / 1e9:.1f} TFLOP/s effective", flush=True)


if __name__ == "__main__":
    main()
