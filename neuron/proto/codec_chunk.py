"""Prototype: codec decoder (upstream whole-sequence forward) on a fixed-size chunk, Neuron vs CPU.

Cost baseline for the streaming codec stage: B rows x T frames -> T*1920 samples each.
Inputs are real golden codes. Usage: python neuron/proto/codec_chunk.py --batch 32 --frames 12 --dtype fp32
"""
import argparse, sys, time
from pathlib import Path

import torch
import torch._dynamo

sys.path.insert(0, str(Path(__file__).parent))
from cp_graph import MODEL  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402

torch._dynamo.config.automatic_dynamic_shapes = False
GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    codec = loader.load_codec(device=torch.device("cpu"), dtype=torch.float32).float()
    decoder = codec.decoder.eval()
    codes = torch.load(GOLDEN / "medium.codes.pt")                           # [frames, 16]
    T, B = args.frames, args.batch
    starts = torch.arange(B) % max(1, codes.shape[0] - T)
    chunk = torch.stack([codes[s:s + T] for s in starts.tolist()]).transpose(1, 2).contiguous()  # [B,16,T]
    with torch.no_grad():
        ref = decoder(chunk).float()                                            # CPU fp32
        decoder.to(dtype).to("neuron")
        c = torch.compile(decoder, backend="neuron", fullgraph=True)
        x = chunk.to("neuron")
        t0 = time.perf_counter()
        out = c(x).float().cpu()
        print(f"compile + first run: {time.perf_counter() - t0:.1f} s")
        err = ((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-9)).max().item()
        snr = 10 * torch.log10(ref.pow(2).sum() / (out - ref).pow(2).sum()).item()
        print(f"{args.dtype}: max per-row rel L2 err vs CPU fp32 = {err:.2e}, SNR = {snr:.1f} dB")
        for _ in range(2):
            c(x)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            o = c(x)
        o.cpu()
        ms = (time.perf_counter() - t0) / args.iters * 1e3
        audio_ms = T * 80
        print(f"B={B} T={T} {args.dtype}: {ms:.2f} ms per chunk = {ms / T:.2f} ms per frame-batch "
              f"({audio_ms} ms of audio per row; RTF {ms / audio_ms:.3f} for the batch)")


if __name__ == "__main__":
    main()
