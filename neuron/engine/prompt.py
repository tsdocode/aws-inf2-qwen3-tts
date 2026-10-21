"""Qwen3-TTS CustomVoice talker prompt layout (mirrors nari model/input_layout.py and upstream generate()).

Prefill slot i embeds  P(text_ids[i]) + (C(codec_ids[i]) if codec_mask[i] else 0), where
P = text_projection(text_embedding(.)) and C = talker codec_embedding. Decode step t (0-based) embeds
codec_sum(previous frame) + P(trailing[t] if t < len(trailing) else tts_pad).
"""

from dataclasses import dataclass

IM_START, ASSISTANT, IM_END = 151644, 77091, 151645
TTS_BOS, TTS_EOS, TTS_PAD = 151672, 151673, 151671
CODEC_PAD, CODEC_BOS, CODEC_EOS = 2148, 2149, 2150
CODEC_THINK, CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS = 2154, 2155, 2156, 2157


@dataclass
class Prompt:
    text_ids: list
    codec_ids: list
    codec_mask: list
    trailing: list


def codec_prefix(talker_cfg, language, speaker):
    spk = talker_cfg["spk_id"][speaker.lower()]
    lang = (language or "auto").lower()
    if lang == "auto":
        return [CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS], spk
    lang_id = talker_cfg["codec_language_id"][lang]
    dialect = talker_cfg.get("spk_is_dialect", {}).get(speaker.lower())
    if dialect and lang in ("chinese", "auto"):
        lang_id = talker_cfg["codec_language_id"][dialect]
    return [CODEC_THINK, CODEC_THINK_BOS, lang_id, CODEC_THINK_EOS], spk


def build_prompt(tokenizer, talker_cfg, text, speaker, language="auto", instruct="", non_streaming_mode=True):
    T = tokenizer(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    text_ids, codec_ids, mask = [], [], []
    if instruct:
        I = tokenizer(f"<|im_start|>user\n{instruct}<|im_end|>\n", add_special_tokens=False)["input_ids"]
        text_ids += I; codec_ids += [CODEC_PAD] * len(I); mask += [False] * len(I)
    text_ids += T[:3]; codec_ids += [CODEC_PAD] * 3; mask += [False] * 3
    prefix, spk = codec_prefix(talker_cfg, language, speaker)
    req = [*prefix, spk, CODEC_PAD]
    text_ids += [TTS_PAD] * (len(prefix) + 1) + [TTS_BOS]
    codec_ids += req; mask += [True] * len(req)
    target = T[3:-5]
    if non_streaming_mode:
        text_ids += target + [TTS_EOS, TTS_PAD]
        codec_ids += [CODEC_PAD] * (len(target) + 1) + [CODEC_BOS]
        mask += [True] * (len(target) + 2)
        trailing = [TTS_PAD]
    else:
        text_ids += target[:1]; codec_ids += [CODEC_BOS]; mask += [True]
        trailing = target[1:] + [TTS_EOS]
    return Prompt(text_ids, codec_ids, mask, trailing)
