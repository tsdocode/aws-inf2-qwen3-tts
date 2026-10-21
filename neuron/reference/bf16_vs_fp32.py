"""How early does upstream qwen_tts in bf16 (CPU, greedy) diverge from the fp32 golden codes?
Separates inherent bf16 drift from Neuron-port bugs. Usage: python neuron/reference/bf16_vs_fp32.py
"""
import json
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSModel

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
model = Qwen3TTSModel.from_pretrained(str(Path.home() / "models/Qwen3-TTS-12Hz-1.7B-CustomVoice"), device_map="cpu",
                                      dtype=torch.bfloat16)
cap = []
dec = model.model.speech_tokenizer.decode
model.model.speech_tokenizer.decode = lambda items, *a, **k: (cap.extend(i["audio_codes"].cpu() for i in items), dec(items, *a, **k))[1]
for name, m in json.loads((GOLDEN / "meta.json").read_text()).items():
    cap.clear()
    model.generate_custom_voice(text=m["text"], speaker=m["speaker"], language=m["language"], do_sample=False,
                                subtalker_dosample=False, max_new_tokens=400)
    got, gold = cap[0], torch.load(GOLDEN / f"{name}.codes.pt")
    n = min(len(got), len(gold))
    exact = [bool(torch.equal(got[i], gold[i])) for i in range(n)]
    first_bad = exact.index(False) if False in exact else n
    print(f"{name}: upstream bf16 CPU frames {len(got)} vs fp32 golden {len(gold)} | first {first_bad} frames exact | "
          f"codebook-0 match {(got[:n, 0] == gold[:n, 0]).float().mean().item() * 100:.0f}%", flush=True)
