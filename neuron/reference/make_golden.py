"""Golden reference for the Neuron port: upstream qwen_tts, CPU, fp32, greedy.

Saves per-prompt codec codes [frames, 16] and 24 kHz audio so Neuron stages
(talker, code predictor, codec) can be checked token-for-token / sample-close.
Usage: python neuron/reference/make_golden.py --model ~/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
"""
import argparse, json, time
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

PROMPTS = [
    ("short", "Hello from Nari Labs.", "ryan", "english"),
    ("medium", "The quick brown fox jumps over the lazy dog, and then it runs back home before the rain starts.", "aiden", "english"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "golden"))
    parser.add_argument("--max-new-tokens", type=int, default=400)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = Qwen3TTSModel.from_pretrained(args.model, device_map="cpu", dtype=torch.float32)

    captured = []
    decode = model.model.speech_tokenizer.decode
    def capture(items, *a, **k):
        captured.extend(item["audio_codes"].detach().cpu() for item in items)
        return decode(items, *a, **k)
    model.model.speech_tokenizer.decode = capture

    meta = {}
    for name, text, speaker, language in PROMPTS:
        captured.clear()
        start = time.perf_counter()
        wavs, sr = model.generate_custom_voice(
            text=text, speaker=speaker, language=language,
            do_sample=False, subtalker_dosample=False, max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.perf_counter() - start
        codes = captured[0]
        torch.save(codes, out / f"{name}.codes.pt")
        sf.write(out / f"{name}.wav", wavs[0], sr)
        meta[name] = dict(text=text, speaker=speaker, language=language, frames=int(codes.shape[0]),
                          code_shape=list(codes.shape), audio_s=len(wavs[0]) / sr, cpu_s=round(elapsed, 1))
        print(name, meta[name], flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
