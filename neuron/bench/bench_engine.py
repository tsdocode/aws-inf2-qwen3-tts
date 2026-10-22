"""Engine step-time breakdown and throughput with N concurrent synthetic requests.

Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/bench/bench_engine.py --slots 32 --requests 32 [--profile]
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "engine"))
from neuron_engine import NeuronTTSEngine, Sampling  # noqa: E402

MODEL = str(Path.home() / "models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
TEXTS = ["The quick brown fox jumps over the lazy dog, and then it runs back home before the rain starts.",
         "Hello from Nari Labs.", "Please confirm your appointment for tomorrow at three in the afternoon.",
         "Streaming speech synthesis needs low latency and steady real time playback."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec-every", type=int, default=2)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    a = ap.parse_args()
    torch._dynamo.config.automatic_dynamic_shapes = False
    eng = NeuronTTSEngine(MODEL, slots=a.slots, profile=a.profile, codec_every=a.codec_every)
    eng.warmup()
    streams = [eng.submit(TEXTS[i % len(TEXTS)], "ryan", "english", sampling=Sampling(max_new_tokens=a.max_new_tokens))
               for i in range(a.requests)]
    t0 = time.perf_counter()
    while eng.step():
        pass
    wall = time.perf_counter() - t0
    audio_s = sum(s.audio_frames for s in streams) * 0.08
    ttfa = sorted((s.t_first_audio - s.t_submit) * 1e3 for s in streams if s.t_first_audio)
    ms = np.array(eng.stats["step_ms"])
    print(f"{a.requests} requests, {a.slots} slots: {eng.stats['steps']} steps in {wall:.1f}s | audio {audio_s:.1f}s -> "
          f"{audio_s / wall:.1f}x real time aggregate | step median {np.median(ms):.1f} ms (80 ms = real time per stream)")
    print(f"TTFA ms: p50 {ttfa[len(ttfa) // 2]:.0f}  p95 {ttfa[int(len(ttfa) * 0.95) - 1]:.0f}  max {ttfa[-1]:.0f} "
          f"(all submitted at t=0; one prefill admitted per step)")
    if a.profile:
        print("step breakdown (median ms, device-synced):")
        for k, v in eng.phase_ms.items():
            print(f"    {k:48s} {np.median(v):7.2f}")


if __name__ == "__main__":
    main()
