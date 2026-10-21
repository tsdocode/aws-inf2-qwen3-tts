"""Lockstep continuous-batching Qwen3-TTS engine on one inf2 NeuronCore (v1).

Every ``step()``:
  1. ONE fused talker step (kernels/talker_kernels.talker_step): a decode token for each active slot + at most one
     waiting request's prefill (<= PREFILL_TOKENS tokens) in the same launch (mixed batch, paged KV).
  2. codec_head + talker sampling on host (suppress mask, repetition penalty, temperature, top-k, top-p).
  3. Code Predictor frame for slots that did not emit EOS: 16 fused step kernels (cp_kernels.cp_step), host
     top-k sampling between steps (v1; on-device sampling is a TODO).
  4. ONE streaming codec frame for every slot (codec_stream.StreamingCodecDecoder, batch = slots, F = 1),
     PCM for active slots is handed to their sinks.
Protocol mirrors nari (see engine/prompt.py): EOS blocked for the first 3 sampled tokens, decode input =
codec_sum(previous frame) + P(trailing text or tts_pad), positions = prefill 0..L-1 then L + t.
"""

import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kernels"))
sys.path.insert(0, str(HERE.parent / "proto"))

from codec_stream import StreamingCodecDecoder  # noqa: E402
from cp_kernels import SLOTS as CP_STEPS, cp_step, cp_step_params  # noqa: E402
from prompt import CODEC_EOS, TTS_PAD, build_prompt  # noqa: E402
from talker_kernels import PAGE, build_group_metadata, kv_write_indices, rope_tables, talker_params, talker_step  # noqa: E402

PREFILL_TOKENS = 64
PPS_BUCKETS = (2, 4, 8)            # pages per segment (256 / 512 / 1024 tokens of context)
MAX_CTX = PPS_BUCKETS[-1] * PAGE


@dataclass
class Sampling:
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    sub_temperature: float = 0.9
    sub_top_k: int = 50
    max_new_tokens: int = 2048
    greedy: bool = False


@dataclass
class Stream:
    rid: int
    prompt: object
    sampling: Sampling
    sink: object                                  # callable(pcm_float32_numpy) / None
    t_submit: float
    slot: int = -1
    blocks: list = field(default_factory=list)
    ctx: int = 0                                   # tokens in KV
    step: int = 0                                  # decode generation step t
    n_sampled: int = 0
    seen: torch.Tensor = None
    last_token: int = -1
    codec_sum: torch.Tensor = None                 # [2048] bf16, next decode input (codec part)
    frames: list = field(default_factory=list)
    done: bool = False
    finished_audio: bool = False
    t_first_audio: float = None
    t_done: float = None
    audio_frames: int = 0
    forced: torch.Tensor = None                    # teacher forcing (tests): [n, 16] frames to feed instead of samples
    pred: list = field(default_factory=list)       # teacher forcing: model's own pick per frame [16]


class _DeviceSampler(torch.nn.Module):
    """On-device Code Predictor sampling + next-input lookup (one compiled graph, reused for all 15 steps).

    top-k (static k) with per-row temperature, inverse-CDF draw with host-supplied uniforms (inf2 has no on-device
    RNG), argmax for greedy rows; then the projected embedding of the chosen token for the next depth step.
    """

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, logitsT, u, inv_temp, greedy, table):
        logits = logitsT.t().float()                                       # [S, V]
        vals, idx = torch.topk(logits, self.k, dim=-1)
        probs = torch.softmax(vals * inv_temp, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        pick = (cdf < (u * cdf[:, -1]).unsqueeze(-1)).sum(-1).clamp(max=self.k - 1)
        tok = torch.where(greedy, idx[:, 0], idx.gather(-1, pick.unsqueeze(-1)).squeeze(-1))
        return tok, F.embedding(tok, table)


def _sample(logits, temperature, top_k, top_p, greedy, gen):
    if greedy:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, -torch.inf)
    if top_p < 1.0:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sl, -1).cumsum(-1)
        drop = cum - torch.softmax(sl, -1) > top_p
        logits = logits.scatter(-1, si, sl.masked_fill(drop, -torch.inf))
    return torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(-1)


class NeuronTTSEngine:
    def __init__(self, model_dir, slots=32, device="neuron", seed=0, log=print, mix_prefill=True, profile=False):
        from nari_qwen3_tts.config import ModelAssetConfig
        from nari_qwen3_tts.model.checkpoint import CheckpointLoader
        from transformers import Qwen2TokenizerFast

        self.log, self.device, self.S = log, device, slots
        self.mix_prefill = mix_prefill
        self.profile = profile
        self.phase_ms = {}
        self.gen = torch.Generator().manual_seed(seed)
        t0 = time.perf_counter()
        self.cfg = json.load(open(Path(model_dir) / "config.json"))["talker_config"]
        self.tok = Qwen2TokenizerFast.from_pretrained(model_dir, fix_mistral_regex=True)
        loader = CheckpointLoader(ModelAssetConfig(model_id=str(model_dir), device="cpu", require_h100=False,
                                                   local_files_only=True))
        talker = loader.load_talker(device=torch.device("cpu"), dtype=torch.bfloat16).eval()
        predictor, layer0 = loader.load_code_predictor(device=torch.device("cpu"), dtype=torch.bfloat16)
        codec = loader.load_codec(device=torch.device("cpu"), dtype=torch.float32).float().decoder.eval()

        # ---- host tables
        self.text_emb = talker.model.text_embedding.weight.detach()
        self.text_proj = talker.text_projection.eval()
        self._p_cache = {}
        self.codec_emb = talker.model.codec_embedding.weight.detach()                  # talker codec emb (group 0)
        with torch.no_grad():
            proj = predictor.small_to_mtp_projection
            self.res_emb = torch.stack([e.weight.detach() for e in predictor.model.codec_embedding])  # [15, 2048, 2048]
            self.proj_l0 = proj(layer0.weight).detach()                                 # [3072, 1024]
            self.proj_res = torch.stack([proj(e.weight) for e in predictor.model.codec_embedding[:-1]]).detach()
        self.small_proj = predictor.small_to_mtp_projection.to(device)
        self.proj_l0_dev = self.proj_l0.to(device)
        self.proj_res_dev = torch.cat([self.proj_res, torch.zeros_like(self.proj_res[:1])]).contiguous().to(device)
        self.sampler = torch.compile(_DeviceSampler(50), backend="neuron", fullgraph=True)
        self.codec_head = talker.codec_head.to(device)
        suppress = torch.zeros(self.cfg["vocab_size"], dtype=torch.bool)
        suppress[2048:] = True
        suppress[CODEC_EOS] = False
        self.suppress = suppress

        # ---- device weights / kernels
        tp = talker_params(talker)
        self.eps = tp.pop("eps")
        self.theta = talker.model.layers[0].self_attn.rope_theta
        self.tw = {k: v.to(device) for k, v in tp.items()}
        cp = cp_step_params(predictor)
        self.cp_eps = cp.pop("eps")
        self.cp_rope = cp.pop("rope").to(device)
        self.cp_heads = cp.pop("lm_heads").to(device)
        self.cpw = {k: v.to(device) for k, v in cp.items()}
        self.L = len(talker.model.layers)
        self.nblk = slots * PPS_BUCKETS[-1] + 8
        self.k_cache = torch.zeros(self.L, self.nblk, PAGE, 8, 128, dtype=torch.bfloat16).to(device)
        self.v_cache = torch.zeros(self.L, self.nblk, PAGE, 8, 128, dtype=torch.bfloat16).to(device)
        self.free_blocks = deque(range(self.nblk))
        self.cp_kv = torch.zeros(len(predictor.model.layers), 2, 128, CP_STEPS, 8, slots, dtype=torch.bfloat16).to(device)
        self.codec = StreamingCodecDecoder(codec, batch=slots, device=device)
        del talker, predictor, codec

        self.waiting, self.slots = deque(), [None] * slots
        self.next_rid = 0
        self.stats = {"steps": 0, "step_ms": []}
        log(f"engine loaded in {time.perf_counter() - t0:.0f}s ({slots} slots, {self.nblk} KV pages)")

    # ------------------------------------------------------------------------------------------------- helpers
    def P(self, ids):
        """text_projection(text_embedding(ids)) [n, 2048] bf16, computed lazily per token id and cached."""
        miss = sorted({i for i in ids if i not in self._p_cache})
        if miss:
            with torch.no_grad():
                rows = self.text_proj(self.text_emb[torch.tensor(miss)])
            for i, r in zip(miss, rows):
                self._p_cache[i] = r
        return torch.stack([self._p_cache[i] for i in ids])

    def submit(self, text, speaker="ryan", language="auto", instruct="", non_streaming_mode=True, sampling=None,
               sink=None, forced=None):
        prompt = build_prompt(self.tok, self.cfg, text, speaker, language, instruct, non_streaming_mode)
        if len(prompt.text_ids) > PREFILL_TOKENS:
            raise ValueError(f"prompt of {len(prompt.text_ids)} tokens > {PREFILL_TOKENS} (chunked prefill TODO)")
        s = Stream(self.next_rid, prompt, sampling or Sampling(), sink, time.perf_counter())
        s.seen = torch.zeros(self.cfg["vocab_size"], dtype=torch.bool)
        s.forced = forced
        self.next_rid += 1
        self.waiting.append(s)
        return s

    def _alloc(self, s, upto):
        need = (upto + PAGE - 1) // PAGE - len(s.blocks)
        for _ in range(max(0, need)):
            s.blocks.append(self.free_blocks.popleft())

    def _release(self, s):
        self.free_blocks.extend(s.blocks)
        s.blocks = []
        self.slots[s.slot] = None

    @torch.no_grad()
    def warmup(self):
        """Compile every talker variant (pps bucket x with/without prefill) + CP/codec/sampler before serving."""
        t0 = time.perf_counter()
        S, d, oob = self.S, self.device, self.nblk * self.L
        for pps in PPS_BUCKETS:
            for with_pre in (False, True):
                T = S + (PREFILL_TOKENS if with_pre else 0)
                groups = [(S, 1, pps)] + ([(1, PREFILL_TOKENS, pps)] if with_pre else [])
                o0, m0 = build_group_metadata([([], -1, 1)] * S, 1, pps, oob)
                extra = {}
                if with_pre:
                    o1, m1 = build_group_metadata([([], -1, PREFILL_TOKENS)], PREFILL_TOKENS, pps, oob)
                    extra = dict(offs1=o1.to(d), mask1=m1.to(d))
                cos, sin = rope_tables(torch.zeros(T, dtype=torch.long), self.theta)
                talker_step(torch.zeros(2048, T, dtype=torch.bfloat16).to(d), cos.to(d), sin.to(d), **self.tw,
                            k_cache=self.k_cache, v_cache=self.v_cache, rows_hbm=kv_write_indices([-1] * T, [0] * T, oob).to(d),
                            offs0=o0.to(d), mask0=m0.to(d), **extra, groups=tuple(groups), eps=self.eps)
                torch.neuron.synchronize()
                self.log(f"  warmup talker pps={pps} prefill={with_pre}: {time.perf_counter() - t0:.0f}s")
        self.submit("Warm up.", "ryan", "english", sampling=Sampling(max_new_tokens=4))
        while self.step():
            pass
        self.stats = {"steps": 0, "step_ms": []}
        self.phase_ms = {}
        self.log(f"warmup done in {time.perf_counter() - t0:.0f}s")

    def active(self):
        return [s for s in self.slots if s is not None]

    def _mark(self, name, t0):
        """Profiling only: sync the device and attribute elapsed wall time to a phase."""
        if not self.profile:
            return t0
        torch.neuron.synchronize()
        now = time.perf_counter()
        self.phase_ms.setdefault(name, []).append((now - t0) * 1e3)
        return now

    # ----------------------------------------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self):
        t_step = time.perf_counter()
        tp = t_step
        decoders = [s for s in self.active() if not s.done]
        pre = None
        if self.waiting and any(x is None for x in self.slots):
            pre = self.waiting.popleft()
            pre.slot = self.slots.index(None)
            self.slots[pre.slot] = pre
            self._reset_codec_row(pre.slot)
            if not self.mix_prefill:
                decoders = []                                   # debug: prefill-only step (no mixed batch)
        if not decoders and pre is None and not self.active():
            return False

        # ---- talker token batch: decode slots (static S rows) + optional prefill chunk
        S = self.S
        xs = torch.zeros(S, 2048, dtype=torch.bfloat16)
        pos = torch.zeros(S, dtype=torch.long)
        wblk, woff = [-1] * S, [0] * S
        dec_meta = [([], -1, 1)] * S                    # padding segments: q_pos -1 -> fully masked
        for s in decoders:
            tt = s.prompt.trailing[s.step] if s.step < len(s.prompt.trailing) else TTS_PAD
            xs[s.slot] = s.codec_sum + self.P([tt])[0]
            pos[s.slot] = s.ctx
            self._alloc(s, s.ctx + 1)
            wblk[s.slot], woff[s.slot] = s.blocks[s.ctx // PAGE], s.ctx % PAGE
            dec_meta[s.slot] = (s.blocks, s.ctx, 1)
        groups = []
        max_ctx = max([s.ctx + 1 for s in decoders] + [1])
        if pre is not None:
            L_ = len(pre.prompt.text_ids)
            self._alloc(pre, L_)
            emb = self.P(pre.prompt.text_ids).clone()
            m = torch.tensor(pre.prompt.codec_mask)
            emb[m] += self.codec_emb[torch.tensor(pre.prompt.codec_ids)[m]]
            px = torch.zeros(PREFILL_TOKENS, 2048, dtype=torch.bfloat16)
            px[:L_] = emb
            xs = torch.cat([xs, px])
            pos = torch.cat([pos, torch.arange(PREFILL_TOKENS)])
            wblk += [pre.blocks[t // PAGE] for t in range(L_)] + [-1] * (PREFILL_TOKENS - L_)
            woff += [t % PAGE for t in range(L_)] + [0] * (PREFILL_TOKENS - L_)
            max_ctx = max(max_ctx, L_)
        pps = next(b for b in PPS_BUCKETS if b * PAGE >= max_ctx)
        oob = self.nblk * self.L               # out of range for the WHOLE stacked cache, not just one layer's view
        offs0, mask0 = build_group_metadata(dec_meta, 1, pps, oob)
        groups.append((S, 1, pps))
        extra = {}
        if pre is not None:
            # prefill queries j >= L_ are padding: give them no visible keys
            offs1, mask1 = build_group_metadata([(pre.blocks, 0, L_)], PREFILL_TOKENS, pps, oob)
            groups.append((1, PREFILL_TOKENS, pps))
            extra = dict(offs1=offs1.to(self.device), mask1=mask1.to(self.device))
        rows = kv_write_indices(wblk, woff, oob)
        cos, sin = rope_tables(pos, self.theta)
        d = self.device
        tp = self._mark("host: build talker batch", tp)
        hidden = talker_step(xs.t().contiguous().to(d), cos.to(d), sin.to(d), **self.tw, k_cache=self.k_cache,
                             v_cache=self.v_cache, rows_hbm=rows.to(d), offs0=offs0.to(d), mask0=mask0.to(d), **extra,
                             groups=tuple(groups), eps=self.eps)           # [2048, T]
        tp = self._mark("talker step (NKI)", tp)
        hT = hidden.t()                                                       # [T, 2048] device
        sel = [s.slot for s in decoders] + ([S + L_ - 1] if pre is not None else [])
        owners = decoders + ([pre] if pre is not None else [])
        logits = self.codec_head(hT).float().cpu()                            # [T, 3072]
        tp = self._mark("codec_head + D2H", tp)
        for s in decoders:
            s.ctx += 1
            s.step += 1
        if pre is not None:
            pre.ctx = L_

        # ---- talker sampling
        sampled, cp_rows = {}, []
        for s, r in zip(owners, sel):
            lg = logits[r].clone()
            lg[self.suppress] = -torch.inf
            if s.n_sampled < 3:                                               # nari: EOS blocked for 3 tokens
                lg[CODEC_EOS] = -torch.inf
            if s.sampling.repetition_penalty != 1.0 and s.seen.any():
                p = s.sampling.repetition_penalty
                v = lg[s.seen]
                lg[s.seen] = torch.where(v > 0, v / p, v * p)
            tok = int(_sample(lg.unsqueeze(0), s.sampling.temperature, s.sampling.top_k, s.sampling.top_p,
                              s.sampling.greedy, self.gen)[0])
            if s.forced is not None:                                          # teacher forcing: record, then override
                s.pred.append([tok])
                f_i = len(s.frames)
                tok = int(s.forced[f_i, 0]) if f_i < len(s.forced) else CODEC_EOS
            s.seen[tok] = True
            s.n_sampled += 1
            terminal = tok == CODEC_EOS or s.step + 1 >= max(2, s.sampling.max_new_tokens)
            if tok == CODEC_EOS:
                s.done = True
            else:
                sampled[s.slot] = (s, tok, r)
                cp_rows.append(s)
                if terminal:
                    s.done = True

        tp = self._mark("host: talker sampling", tp)
        # ---- code predictor frame for sampled (non-EOS) slots
        if cp_rows:
            if any(s.forced is not None for s in cp_rows) or any(s.sampling.sub_top_k != 50 for s in cp_rows):
                self._code_predictor(cp_rows, sampled, hT)                    # host path (teacher forcing / tests)
            else:
                self._code_predictor_device(cp_rows, sampled, hT)
        tp = self._mark("code predictor (16 NKI steps + host sampling)", tp)

        # ---- codec: one frame for every slot (inactive rows decode silence codes; their output is dropped)
        codes = torch.zeros(S, 16, 1, dtype=torch.long)
        emit = []
        for s in self.active():
            if s.frames and s.audio_frames < len(s.frames):
                codes[s.slot, :, 0] = s.frames[s.audio_frames]
                emit.append(s)
        if emit:
            wav = self.codec(codes.to(d)).float().cpu()                       # [S, 1, 1920]
            tp = self._mark("codec frame (front + NKI)", tp)
            now = time.perf_counter()
            for s in emit:
                s.audio_frames += 1
                if s.t_first_audio is None:
                    s.t_first_audio = now
                if s.sink is not None:
                    s.sink(wav[s.slot, 0].numpy())
        for s in self.active():
            if s.done and s.audio_frames >= len(s.frames):
                s.finished_audio, s.t_done = True, time.perf_counter()
                if s.sink is not None:
                    s.sink(None)
                self._release(s)
        self.stats["steps"] += 1
        self.stats["step_ms"].append((time.perf_counter() - t_step) * 1e3)
        return True

    def _reset_codec_row(self, row):
        """A new stream starts in ``row``: zero that row of every codec state tensor (KV has batch on dim 1)."""
        for k, t in self.codec.state.items():
            if k in ("tf_k", "tf_v"):
                t[:, row] = 0
            else:
                t[row] = 0

    def _code_predictor_device(self, rows, sampled, hT):
        """16 fused NKI depth steps with on-device sampling: one H2D (uniforms) and one D2H (16 tokens) per frame."""
        S, d = self.S, self.device
        tok0 = torch.zeros(S, dtype=torch.long)
        hid_idx = torch.zeros(S, dtype=torch.long)
        inv_t = torch.ones(S, 1)
        greedy = torch.zeros(S, dtype=torch.bool)
        for s in rows:
            _, tok, r = sampled[s.slot]
            tok0[s.slot], hid_idx[s.slot] = tok, r
            inv_t[s.slot] = 1.0 / s.sampling.sub_temperature
            greedy[s.slot] = s.sampling.greedy
        u = torch.rand(CP_STEPS, S, generator=self.gen).to(d)
        inv_t, greedy, tok0_d = inv_t.to(d), greedy.to(d), tok0.to(d)
        x = self.small_proj(hT.index_select(0, hid_idx.to(d)))
        toks = [tok0_d]
        for st in range(CP_STEPS):
            logitsT, _ = cp_step(x.t().contiguous(), self.cp_kv, **self.cpw, lm_head=self.cp_heads[max(st - 1, 0)],
                                 rope=self.cp_rope[st], s=st, eps=self.cp_eps, with_head=True)
            if st == 0:
                x = F.embedding(tok0_d, self.proj_l0_dev)
                continue
            tok, x = self.sampler(logitsT, u[st], inv_t, greedy, self.proj_res_dev[st - 1])
            toks.append(tok)
        frame = torch.stack(toks, 1).cpu()                                      # [S, 16]
        self._commit_frames(rows, frame)

    def _commit_frames(self, rows, frame):
        slots = torch.tensor([s.slot for s in rows])
        f = frame[slots]                                                        # [n, 16]
        cs = self.codec_emb[f[:, 0]].float()
        for g in range(1, 16):
            cs += self.res_emb[g - 1][f[:, g]].float()
        cs = cs.to(torch.bfloat16)
        for i, s in enumerate(rows):
            s.frames.append(f[i].clone())
            s.codec_sum = cs[i]

    def _code_predictor(self, rows, sampled, hT):
        S, d = self.S, self.device
        tok0 = torch.zeros(S, dtype=torch.long)
        live = torch.zeros(S, dtype=torch.bool)
        hid_idx = torch.zeros(S, dtype=torch.long)
        for s in rows:
            _, tok, r = sampled[s.slot]
            tok0[s.slot], live[s.slot], hid_idx[s.slot] = tok, True, r
        past = hT.index_select(0, hid_idx.to(d))                                # [S, 2048] device
        x = self.small_proj(past)                                               # step-0 input [S, 1024]
        toks = [tok0]
        for st in range(CP_STEPS):
            head = max(st - 1, 0)
            logitsT, _ = cp_step(x.t().contiguous(), self.cp_kv, **self.cpw, lm_head=self.cp_heads[head],
                                 rope=self.cp_rope[st], s=st, eps=self.cp_eps, with_head=True)
            if st == 0:
                nxt = toks[0]
            else:
                lg = logitsT.t().float().cpu()                                  # [S, 2048]
                nxt = torch.zeros(S, dtype=torch.long)
                for s in rows:
                    nxt[s.slot] = _sample(lg[s.slot:s.slot + 1], s.sampling.sub_temperature, s.sampling.sub_top_k, 1.0,
                                          s.sampling.greedy, self.gen)[0]
                    if s.forced is not None:
                        s.pred[-1].append(int(nxt[s.slot]))
                        nxt[s.slot] = s.forced[len(s.frames), st]
                toks.append(nxt)
            if st < CP_STEPS - 1:
                table = self.proj_l0 if st == 0 else self.proj_res[st - 1]
                x = table[nxt].to(d)
        frame = torch.stack(toks, 1)                                            # [S, 16]
        self._commit_frames(rows, frame)
