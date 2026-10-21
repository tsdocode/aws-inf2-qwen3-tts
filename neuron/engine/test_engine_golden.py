"""End-to-end engine check on inf2: greedy generation of the golden prompts vs CPU fp32 upstream codes.

Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/engine/test_engine_golden.py --slots 4
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from neuron_engine import NeuronTTSEngine, Sampling  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
MODEL = str(Path.home() / "models/Qwen3-TTS-12Hz-1.7B-CustomVoice")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=4)
    a = ap.parse_args()
    torch._dynamo.config.automatic_dynamic_shapes = False
    eng = NeuronTTSEngine(MODEL, slots=a.slots)
    meta = json.loads((GOLDEN / "meta.json").read_text())
    streams, audio = {}, {}
    for name, m in meta.items():
        audio[name] = []
        sink = (lambda name: (lambda pcm: audio[name].append(pcm) if pcm is not None else None))(name)
        streams[name] = eng.submit(m["text"], m["speaker"], m["language"], sampling=Sampling(greedy=True), sink=sink)
    t0 = time.perf_counter()
    while eng.step():
        if eng.stats["steps"] % 20 == 0:
            print(f"  step {eng.stats['steps']}: last step {eng.stats['step_ms'][-1]:.0f} ms", flush=True)
    wall = time.perf_counter() - t0
    for name, s in streams.items():
        gold = torch.load(GOLDEN / f"{name}.codes.pt")
        got = torch.stack(s.frames) if s.frames else torch.zeros(0, 16, dtype=torch.long)
        n = min(len(gold), len(got))
        exact = [bool(torch.equal(got[i], gold[i])) for i in range(n)]
        first_bad = exact.index(False) if False in exact else n
        g0 = (got[:n, 0] == gold[:n, 0]).float().mean().item() * 100 if n else 0
        allg = (got[:n] == gold[:n]).float().mean().item() * 100 if n else 0
        wav = np.concatenate(audio[name]) if audio[name] else np.zeros(0)
        sf.write(GOLDEN.parent / "golden" / f"{name}.neuron.wav", wav, 24000)
        print(f"{name}: frames neuron {len(got)} vs golden {len(gold)} | first {first_bad} frames exact | "
              f"codebook-0 match {g0:.0f}% | all-code match {allg:.0f}% | TTFA {(s.t_first_audio - s.t_submit) * 1e3:.0f} ms "
              f"(incl. compile on first run) | audio {len(wav) / 24000:.2f}s")
    ms = eng.stats["step_ms"]
    print(f"{eng.stats['steps']} steps in {wall:.0f}s; steady step median {np.median(ms[len(ms) // 2:]):.1f} ms")


if __name__ == "__main__":
    main()
