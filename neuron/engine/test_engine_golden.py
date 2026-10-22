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


_REF = {}


def ref_codec():
    """Upstream codec decoder, CPU fp32 (loaded once)."""
    if "dec" not in _REF:
        from nari_qwen3_tts.config import ModelAssetConfig
        from nari_qwen3_tts.model.checkpoint import CheckpointLoader
        loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
        _REF["dec"] = loader.load_codec(device=torch.device("cpu"), dtype=torch.float32).float().decoder.eval()
    return _REF["dec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec-every", type=int, default=2)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--teacher", action="store_true", help="teacher-force golden frames; report per-step agreement")
    ap.add_argument("--no-mix", action="store_true")
    ap.add_argument("--audio-check", action="store_true",
                    help="SNR of the streamed audio vs the upstream whole-sequence fp32 decoder on the same frames")
    a = ap.parse_args()
    torch._dynamo.config.automatic_dynamic_shapes = False
    eng = NeuronTTSEngine(MODEL, slots=a.slots, mix_prefill=not a.no_mix, codec_every=a.codec_every)
    meta = json.loads((GOLDEN / "meta.json").read_text())
    streams, audio = {}, {}
    for name, m in meta.items():
        if a.only and name != a.only:
            continue
        audio[name] = []
        sink = (lambda name: (lambda pcm: audio[name].append(pcm) if pcm is not None else None))(name)
        forced = torch.load(GOLDEN / f"{name}.codes.pt") if a.teacher else None
        streams[name] = eng.submit(m["text"], m["speaker"], m["language"], sampling=Sampling(greedy=True), sink=sink,
                                   forced=forced)
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
        if a.audio_check:
            with torch.no_grad():
                ref = ref_codec()(got.t().unsqueeze(0)).float().reshape(-1).numpy()
            m_ = min(len(ref), len(wav))
            snr = 10 * np.log10((ref[:m_] ** 2).sum() / max(((wav[:m_] - ref[:m_]) ** 2).sum(), 1e-20))
            print(f"    streamed audio vs whole-sequence fp32 decode of the same frames: SNR {snr:.1f} dB "
                  f"({len(wav)} vs {len(ref)} samples)")
        print(f"{name}: frames neuron {len(got)} vs golden {len(gold)} | first {first_bad} frames exact | "
              f"codebook-0 match {g0:.0f}% | all-code match {allg:.0f}% | TTFA {(s.t_first_audio - s.t_submit) * 1e3:.0f} ms "
              f"(incl. compile on first run) | audio {len(wav) / 24000:.2f}s")
        if a.teacher:
            pred = torch.tensor([p for p in s.pred if len(p) == 16])
            n = min(len(pred), len(gold))
            agree = (pred[:n] == gold[:n]).float().mean(0) * 100
            print(f"    teacher-forced agreement: talker cb0 {agree[0]:.0f}% | CP groups 1-15 mean {agree[1:].mean():.0f}% "
                  f"(min {agree[1:].min():.0f}%) over {n} frames; EOS predicted at the golden end: "
                  f"{len(s.pred) > len(gold) and s.pred[len(gold)][0] == 2150}")
            per = (pred[:n] == gold[:n]).float().mean(1) * 100
            print("    per-frame agreement (all 16 codes):", " ".join(f"{v:.0f}" for v in per.tolist()))
    ms = eng.stats["step_ms"]
    print(f"{eng.stats['steps']} steps in {wall:.0f}s; steady step median {np.median(ms[len(ms) // 2:]):.1f} ms")


if __name__ == "__main__":
    main()
