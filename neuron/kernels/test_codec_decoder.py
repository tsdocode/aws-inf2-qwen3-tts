"""End-to-end codec decoder: NKI hybrid vs upstream fp32 (CPU) on golden codes; timing per chunk.

Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/kernels/test_codec_decoder.py --rows 32 --frames 16
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from codec_decoder import NeuronCodecDecoder, stage_breakdown  # noqa: E402
from cp_graph import MODEL  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402
from test_codec_kernels import GOLDEN, snr_db  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--front-bf16", action="store_true")
    a = ap.parse_args()
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    decoder = loader.load_codec(device=torch.device("cpu"), dtype=torch.float32).float().decoder.eval()
    codes = torch.load(GOLDEN / "medium.codes.pt")
    span = codes.shape[0] - a.frames
    chunk = torch.stack([codes[(r * 5) % span:(r * 5) % span + a.frames] for r in range(a.rows)]).transpose(1, 2).contiguous()
    with torch.no_grad():
        ref = decoder(chunk).float()
    neu = NeuronCodecDecoder(decoder, front_dtype=torch.bfloat16 if a.front_bf16 else torch.float32)
    x = chunk.to("neuron")
    t0 = time.perf_counter()
    wav = neu(x).float().cpu()
    print(f"compile + first run: {time.perf_counter() - t0:.1f}s; wav {tuple(wav.shape)} ref {tuple(ref.shape)}")
    per_row = [snr_db(wav[r], ref[r]) for r in range(a.rows)]
    print(f"audio SNR vs upstream fp32: overall {snr_db(wav, ref):.1f} dB, worst row {min(per_row):.1f} dB")
    for _ in range(2):
        neu(x)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        out = neu(x)
    torch.neuron.synchronize()
    ms = (time.perf_counter() - t0) / a.iters * 1e3
    print(f"B={a.rows} T={a.frames}: {ms:.1f} ms per chunk (wall, all launches) = {ms / a.frames:.2f} ms per frame-batch "
          f"[compiled baseline: 37.5 fp32 / 29.2 bf16 at B=32]")
    rows = stage_breakdown(neu, x)
    total = sum(ms for _, ms in rows)
    for name, ms in rows:
        print(f"    {name:28s} {ms:7.2f} ms  {ms / total * 100:5.1f}%")


if __name__ == "__main__":
    main()
