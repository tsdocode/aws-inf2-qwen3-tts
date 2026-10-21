"""Streaming codec (chunked, stateful) vs upstream whole-sequence fp32 decoder on golden codes.

Chunk schedule like nari's ttfa profile ([1, 2, 4, 8] then 12s). Usage:
NEURON_RT_VISIBLE_CORES=1 python neuron/kernels/test_codec_stream.py --rows 2 --schedule 1 2 4 8 12 12 12 12
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from codec_stream import StreamingCodecDecoder  # noqa: E402
from cp_graph import MODEL  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402
from test_codec_kernels import GOLDEN, snr_db  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--schedule", type=int, nargs="+", default=[1, 2, 4, 8, 12, 12, 12, 12])
    a = ap.parse_args()
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    decoder = loader.load_codec(device=torch.device("cpu"), dtype=torch.float32).float().decoder.eval()
    codes = torch.load(GOLDEN / "medium.codes.pt")                      # [74, 16]
    total = sum(a.schedule)
    assert total <= codes.shape[0], (total, codes.shape)
    rows = torch.stack([codes[r * 3:r * 3 + total] for r in range(a.rows)]).transpose(1, 2).contiguous()  # [B,16,F]
    with torch.no_grad():
        ref = decoder(rows).float()                                       # whole sequence, CPU fp32
    neu = StreamingCodecDecoder(decoder, batch=a.rows)
    chunks, f0 = [], 0
    t0 = time.perf_counter()
    for i, f in enumerate(a.schedule):
        ts = time.perf_counter()
        w = neu(rows[:, :, f0:f0 + f].contiguous().to("neuron")).cpu().float()
        chunks.append(w)
        print(f"  chunk {i}: {f:2d} frames -> {w.shape[-1]} samples ({(time.perf_counter() - ts) * 1e3:.0f} ms incl compile)", flush=True)
        f0 += f
    wav = torch.cat(chunks, -1)
    print(f"total {time.perf_counter() - t0:.0f}s; streamed {tuple(wav.shape)} vs whole {tuple(ref.shape)}")
    print(f"streaming vs whole-sequence upstream fp32: SNR {snr_db(wav, ref):.1f} dB, "
          f"worst row {min(snr_db(wav[r], ref[r]) for r in range(a.rows)):.1f} dB")
    # first-chunk boundary check: error concentrated at chunk seams would show here
    seam = sum(a.schedule[:1]) * 1920
    print(f"  SNR first chunk {snr_db(wav[..., :seam], ref[..., :seam]):.1f} dB, rest {snr_db(wav[..., seam:], ref[..., seam:]):.1f} dB")


if __name__ == "__main__":
    main()
