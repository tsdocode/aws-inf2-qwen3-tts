"""Lockstep continuous-batching Qwen3-TTS engine on one inf2 NeuronCore (v1).

Every ``step()``:
  1. ONE fused talker step (kernels/talker_kernels.talker_step): a decode token for each active slot (paged KV) +
     waiting prompts packed back to back into up to 3 tiles of 128 tokens (varlen prefill), in the same launch.
  2. codec_head + talker sampling on host (suppress mask, repetition penalty, temperature, top-k, top-p).
  3. Code Predictor frame for slots that did not emit EOS: ONE kernel launch (cp_kernels.cp_frame: 16 depth steps,
     on-chip top-k sampling with host uniforms, next-input lookups). Teacher forcing / non-default sub top-k use the
     per-step host path (cp_step + host sampling).
  4. Every ``codec_every`` steps, K streaming codec frames for every slot (codec_stream.StreamingCodecDecoder,
     batch = slots), PCM for active slots is handed to their sinks.
Protocol mirrors nari (see engine/prompt.py): EOS blocked for the first 3 sampled tokens, decode input =
codec_sum(previous frame) + P(trailing text or tts_pad), positions = prefill 0..L-1 then L + t.
"""

import copy
import json
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
from cp_kernels import SLOTS as CP_STEPS, cp_frame, cp_step, cp_step_params  # noqa: E402
from prompt import CODEC_EOS, TTS_PAD, build_prompt  # noqa: E402
from talker_kernels import (PAGE, build_group_metadata, kv_write_indices, packed_prefill_mask, rope_tables,  # noqa: E402
                            talker_params, talker_step)

PREFILL_TILE_BUCKETS = (1, 2, 3)  # 128-token tiles of packed fresh prompts per step; T = 32 + 3*128 = 416 <= 512: one
#   token chunk per weight block (T = 544 made every block run a second, 32-token matmul: +11 ms of fixed costs)
PPS_BUCKETS = (1, 2, 4, 8)         # pages per segment (128 / 256 / 512 / 1024 tokens of context); short utterances
#   stay under 128 tokens, and every padded page still costs its matmuls
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
    replay: int = 0                                # frames to re-decode from a reset codec row (first-audio path)
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
    if top_k and top_k < logits.shape[-1]:
        # sample among the top-k only (same distribution as masking the rest to -inf; ~10x cheaper on the host)
        vals, idx = torch.topk(logits / temperature, top_k, dim=-1)              # sorted descending
        if top_p < 1.0:
            p = torch.softmax(vals, -1)
            vals = vals.masked_fill(p.cumsum(-1) - p > top_p, -torch.inf)
        pick = torch.multinomial(torch.softmax(vals, -1), 1, generator=gen)
        return idx.gather(-1, pick).squeeze(-1)
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


class HostAssets:
    """CPU-side model assets, loaded once and shared by every engine (one engine per NeuronCore).

    Two engines each loading their own CPU copies need ~20 GB of host RAM; inf2.xlarge has 16 GB.
    """

    def __init__(self, model_dir, log=print):
        from nari_qwen3_tts.config import ModelAssetConfig
        from nari_qwen3_tts.model.checkpoint import CheckpointLoader
        from transformers import Qwen2TokenizerFast

        t0 = time.perf_counter()
        self.cfg = json.load(open(Path(model_dir) / "config.json"))["talker_config"]
        self.tok = Qwen2TokenizerFast.from_pretrained(model_dir, fix_mistral_regex=True)
        loader = CheckpointLoader(ModelAssetConfig(model_id=str(model_dir), device="cpu", require_h100=False,
                                                   local_files_only=True))
        self.talker = loader.load_talker(device=torch.device("cpu"), dtype=torch.bfloat16).eval()
        self.predictor, layer0 = loader.load_code_predictor(device=torch.device("cpu"), dtype=torch.bfloat16)
        self.codec = loader.load_codec(device=torch.device("cpu"), dtype=torch.float32).float().decoder.eval()
        self.text_emb = self.talker.model.text_embedding.weight.detach()
        self.text_proj = self.talker.text_projection.eval()
        self.codec_emb = self.talker.model.codec_embedding.weight.detach()            # talker codec emb (group 0)
        with torch.no_grad():
            proj = self.predictor.small_to_mtp_projection
            self.res_emb = torch.stack([e.weight.detach() for e in self.predictor.model.codec_embedding])  # [15,2048,2048]
            self.proj_l0 = proj(layer0.weight).detach()                                 # [3072, 1024]
            self.proj_res = torch.stack([proj(e.weight) for e in self.predictor.model.codec_embedding[:-1]]).detach()
        self.talker_params = talker_params(self.talker)
        self.cp_params = cp_step_params(self.predictor)
        self.p_cache = {}
        log(f"host assets loaded in {time.perf_counter() - t0:.0f}s")

    def release_device_copies(self):
        """After every engine has uploaded its weights: drop the CPU copies only needed for that (tiled talker/CP
        weights, talker decoder layers, codec module) so one process per NeuronCore fits in host RAM. Keeps the
        host tables used per step (text embedding + projection, codec/CP embedding tables)."""
        import gc

        self.talker_params = self.cp_params = None
        self.talker.model.layers = torch.nn.ModuleList()
        self.codec = None
        gc.collect()


class NeuronTTSEngine:
    def __init__(self, model_dir=None, slots=32, device="neuron", seed=0, log=print, mix_prefill=True, profile=False,
                 codec_every=2, assets=None):
        self.log, self.device, self.S = log, device, slots
        self.mix_prefill = mix_prefill
        self.profile = profile
        self.codec_every = codec_every
        self.slow_step_ms = 300.0
        self._step_phases = []
        self.phase_ms = {}
        self.gen = torch.Generator().manual_seed(seed)
        t0 = time.perf_counter()
        A = assets or HostAssets(model_dir, log)
        self.cfg, self.tok = A.cfg, A.tok

        # ---- host tables (shared)
        self.text_emb, self.text_proj, self._p_cache = A.text_emb, A.text_proj, A.p_cache
        self.codec_emb, self.res_emb, self.proj_l0, self.proj_res = A.codec_emb, A.res_emb, A.proj_l0, A.proj_res
        # ---- small modules: per-engine device copies (``.to(device)`` moves modules in place)
        self.small_proj = copy.deepcopy(A.predictor.small_to_mtp_projection).to(device)
        self.proj_l0_dev = self.proj_l0.to(device)
        self.proj_res_rows = self.proj_res.reshape(-1, self.proj_res.shape[-1]).contiguous().to(device)  # [14*2048, 1024]
        self.ident_dev = torch.eye(128, dtype=torch.bfloat16).to(device)
        self.codec_head = copy.deepcopy(A.talker.codec_head).to(device)
        suppress = torch.zeros(self.cfg["vocab_size"], dtype=torch.bool)
        suppress[2048:] = True
        suppress[CODEC_EOS] = False
        self.suppress = suppress

        # ---- device weights / kernels
        tp = dict(A.talker_params)
        self.eps = tp.pop("eps")
        self.theta = A.talker.model.layers[0].self_attn.rope_theta
        self.tw = {k: v.to(device) for k, v in tp.items()}
        cp = dict(A.cp_params)
        self.cp_eps = cp.pop("eps")
        self.cp_rope = cp.pop("rope").to(device)
        self.cp_heads = cp.pop("lm_heads").to(device)
        self.cpw = {k: v.to(device) for k, v in cp.items()}
        self.L = len(A.talker.model.layers)
        self.nblk = slots * PPS_BUCKETS[-1] + 8
        self.k_cache = torch.zeros(self.L, self.nblk, PAGE, 8, 128, dtype=torch.bfloat16).to(device)
        self.v_cache = torch.zeros(self.L, self.nblk, PAGE, 8, 128, dtype=torch.bfloat16).to(device)
        self.free_blocks = deque(range(self.nblk))
        self.cp_kv = torch.zeros(len(A.predictor.model.layers), 2, 128, CP_STEPS, 8, slots, dtype=torch.bfloat16).to(device)
        self.codec = StreamingCodecDecoder(copy.deepcopy(A.codec), batch=slots, device=device)

        self.waiting, self.slots = deque(), [None] * slots
        self.next_rid = 0
        self.stats = {"steps": 0, "step_ms": [], "phases": deque(maxlen=4000)}
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
        if len(prompt.text_ids) > PAGE:
            raise ValueError(f"prompt of {len(prompt.text_ids)} tokens > {PAGE} (chunked prefill TODO)")
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
            for npb in (0,) + PREFILL_TILE_BUCKETS:
                T = S + npb * PAGE
                groups = [(S, 1, pps)] + ([(npb, PAGE, 0)] if npb else [])
                o0, m0 = build_group_metadata([([], -1, 1)] * S, 1, pps, oob)
                extra = {}
                if npb:
                    extra = dict(offs1=None, mask1=packed_prefill_mask([-1] * (npb * PAGE)).to(d))
                cos, sin = rope_tables(torch.zeros(T, dtype=torch.long), self.theta)
                talker_step(torch.zeros(2048, T, dtype=torch.bfloat16).to(d), cos.to(d), sin.to(d), **self.tw,
                            k_cache=self.k_cache, v_cache=self.v_cache, rows_hbm=kv_write_indices([-1] * T, [0] * T, oob).to(d),
                            offs0=o0.to(d), mask0=m0.to(d), **extra, groups=tuple(groups), eps=self.eps)
                torch.neuron.synchronize()
                self.log(f"  warmup talker pps={pps} prefill tiles={npb}: {time.perf_counter() - t0:.0f}s")
        # End to end with bursts of simultaneous requests: eager ops (codec_head, index_select, ...) compile per
        # distinct token count T = S + n_prefill_tiles * 128, and an uncompiled shape stalls every stream for
        # seconds the first time it appears under traffic.
        # (bursts of 12 / 20 short prompts fill 2 / 3 packed prefill tiles)
        for burst in (1, 2, 4, 8, 12, 20):
            for _ in range(burst):
                self.submit("Warm up the engine.", "ryan", "english", sampling=Sampling(max_new_tokens=6))
            while self.step():
                pass
        # ...and once at full scale (every slot busy, staggered lengths), so every first-use path — codec chunks with
        # all rows live, fresh-row resets, long-context page buckets — runs before the first real request.
        for i in range(self.S):
            self.submit("This is a full scale warm up of every slot in the engine.", "ryan", "english",
                        sampling=Sampling(max_new_tokens=8 + (i % 8)))
        while self.step():
            pass
        self.stats = {"steps": 0, "step_ms": [], "phases": deque(maxlen=4000)}
        self.phase_ms = {}
        self.log(f"warmup done in {time.perf_counter() - t0:.0f}s")

    def active(self):
        return [s for s in self.slots if s is not None]

    def _mark(self, name, t0):
        """Attribute elapsed wall time to a phase (device-synced only when profiling; otherwise host-side time, which
        includes any blocking D2H). Kept per step so slow steps can be explained in the log."""
        if self.profile:
            torch.neuron.synchronize()
        now = time.perf_counter()
        ms = (now - t0) * 1e3
        self._step_phases.append((name, ms))
        if self.profile:
            self.phase_ms.setdefault(name, []).append(ms)
        return now

    # ----------------------------------------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self):
        t_step = time.perf_counter()
        tp = t_step
        self._step_phases = []
        decoders = [s for s in self.active() if not s.done]
        # admit waiting prompts, packed first-fit into 128-token tiles (varlen prefill, FIFO: stop at the first
        # prompt that fits nowhere)
        pres, tiles = [], []                                    # tiles: tokens used per tile
        while self.waiting and any(x is None for x in self.slots):
            n = len(self.waiting[0].prompt.text_ids)
            ti = next((i for i, u in enumerate(tiles) if u + n <= PAGE), None)
            if ti is None:
                if len(tiles) == PREFILL_TILE_BUCKETS[-1]:
                    break
                tiles.append(0)
                ti = len(tiles) - 1
            pre = self.waiting.popleft()
            pre.slot = self.slots.index(None)
            self.slots[pre.slot] = pre
            self._reset_codec_row(pre.slot)
            pre.tile, pre.tile_off = ti, tiles[ti]
            tiles[ti] += n
            pres.append(pre)
        if pres and not self.mix_prefill:
            decoders = []                                       # debug: prefill-only step (no mixed batch)
        if not decoders and not pres and not self.active():
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
        pre_rows = []
        if pres:
            nt = next(b for b in PREFILL_TILE_BUCKETS if b >= len(tiles))
            px = torch.zeros(nt * PAGE, 2048, dtype=torch.bfloat16)
            pwblk, pwoff = [-1] * (nt * PAGE), [0] * (nt * PAGE)
            ppos = torch.zeros(nt * PAGE, dtype=torch.long)
            seg = [-1] * (nt * PAGE)
            for i, pre in enumerate(pres):
                L_ = len(pre.prompt.text_ids)
                base = pre.tile * PAGE + pre.tile_off
                self._alloc(pre, L_)
                emb = self.P(pre.prompt.text_ids).clone()
                m = torch.tensor(pre.prompt.codec_mask)
                emb[m] += self.codec_emb[torch.tensor(pre.prompt.codec_ids)[m]]
                px[base:base + L_] = emb
                pwblk[base:base + L_] = [pre.blocks[t // PAGE] for t in range(L_)]
                pwoff[base:base + L_] = [t % PAGE for t in range(L_)]
                ppos[base:base + L_] = torch.arange(L_)
                seg[base:base + L_] = [i] * L_
                pre_rows.append(S + base + L_ - 1)
            xs = torch.cat([xs, px])
            pos = torch.cat([pos, ppos])
            wblk += pwblk
            woff += pwoff
        pps = next(b for b in PPS_BUCKETS if b * PAGE >= max_ctx)
        oob = self.nblk * self.L               # out of range for the WHOLE stacked cache, not just one layer's view
        offs0, mask0 = build_group_metadata(dec_meta, 1, pps, oob)
        groups.append((S, 1, pps))
        extra = {}
        if pres:
            groups.append((nt, PAGE, 0))                  # packed fresh prompts: causal within each prompt
            extra = dict(offs1=None, mask1=packed_prefill_mask(seg).to(self.device))
        rows = kv_write_indices(wblk, woff, oob)
        cos, sin = rope_tables(pos, self.theta)
        d = self.device
        tp = self._mark("host: build talker batch", tp)
        hidden = talker_step(xs.t().contiguous().to(d), cos.to(d), sin.to(d), **self.tw, k_cache=self.k_cache,
                             v_cache=self.v_cache, rows_hbm=rows.to(d), offs0=offs0.to(d), mask0=mask0.to(d), **extra,
                             groups=tuple(groups), eps=self.eps)           # [2048, T]
        tp = self._mark("talker step (NKI)", tp)
        hT = hidden.t()                                                       # [T, 2048] device
        sel = [s.slot for s in decoders] + pre_rows
        owners = decoders + pres
        logits = self.codec_head(hT).float().cpu()                            # [T, 3072]
        tp = self._mark("codec_head + D2H", tp)
        for s in decoders:
            s.ctx += 1
            s.step += 1
        for pre in pres:
            pre.ctx = len(pre.prompt.text_ids)

        # ---- talker sampling
        sampled, cp_rows = {}, []
        toks = self._sample_talker(owners, logits[sel]) if owners else []
        for s, r, tok in zip(owners, sel, toks):
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
        tp = self._mark("code predictor (cp_frame NKI)", tp)

        # ---- codec: every ``codec_every`` steps decode K frames for each row with >= K pending (the compiled codec
        # front costs ~30 ms per call regardless of K, so batching frames amortizes it). Rows with fewer pending
        # frames (just started) ride along with zero codes and get their state reset; finished rows flush with
        # zero padding (their padded audio is dropped).
        K = self.codec_every
        if self.stats["steps"] % K == 0:
            # First audio fast path: a NEW stream with 0 < pending < K frames is decoded zero-padded (the codec is
            # causal, so its first frames' audio is exact), emitted, and its codec row reset; the next call re-decodes
            # those ``replay`` frames from the fresh state together with the new ones and emits only the new audio.
            codes = torch.zeros(S, 16, K, dtype=torch.long)
            emit, reset = [], []                                               # emit: (stream, skip, n)
            for s in self.active():
                pending = len(s.frames) - s.audio_frames
                r = s.replay
                if r and pending > 0:                                          # replay + new frames, fresh state
                    n = min(K - r, pending)
                    codes[s.slot, :, :r + n] = torch.stack(s.frames[s.audio_frames - r:s.audio_frames + n], 1)
                    emit.append((s, r, n))
                    s.replay = 0
                elif pending >= K or (s.done and pending > 0):
                    n = min(K, pending)
                    codes[s.slot, :, :n] = torch.stack(s.frames[s.audio_frames:s.audio_frames + n], 1)
                    emit.append((s, 0, n))
                elif s.audio_frames == 0:
                    if pending > 0:                                            # fresh, short: emit now, replay later
                        codes[s.slot, :, :pending] = torch.stack(s.frames[:pending], 1)
                        emit.append((s, 0, pending))
                        s.replay = pending
                    reset.append(s)                                            # undo the zero-code ride-along
            if emit:
                wav = self.codec(codes.to(d)).float().cpu()                   # [S, 1, K*1920]
                tp = self._mark("codec chunk (front + NKI)", tp)
                for s in reset:
                    self._reset_codec_row(s.slot)
                now = time.perf_counter()
                for s, skip, n in emit:
                    s.audio_frames += n
                    if s.t_first_audio is None:
                        s.t_first_audio = now
                    if s.sink is not None:
                        s.sink(wav[s.slot, 0, skip * 1920:(skip + n) * 1920].numpy())
        for s in self.active():
            if s.done and s.audio_frames >= len(s.frames):
                s.finished_audio, s.t_done = True, time.perf_counter()
                if s.sink is not None:
                    s.sink(None)
                self._release(s)
        tp = self._mark("host: emit audio + finish", tp)
        self.stats["steps"] += 1
        step_ms = (time.perf_counter() - t_step) * 1e3
        self.stats["step_ms"].append(step_ms)
        self.stats["phases"].append((step_ms, len(pres), dict(self._step_phases)))
        if step_ms > self.slow_step_ms:
            n_dec = len([s for s in self.active() if not s.done])
            self.log(f"[{self.device}] SLOW STEP {step_ms:.0f} ms (step {self.stats['steps']}, active {n_dec}, "
                     f"prefills {len(pres)}): " + ", ".join(f"{n} {m:.0f}" for n, m in self._step_phases))
        return True

    def _sample_talker(self, owners, lg):
        """Talker token for every owner row at once (lg [n, V] fp32 host logits): suppress mask, EOS block for the
        first 3 tokens, repetition penalty, then temperature / top-k / top-p sampling batched over rows that share
        (top_k, top_p, greedy) -- normally all of them."""
        lg = lg.clone()
        lg[:, self.suppress] = -torch.inf
        early = torch.tensor([s.n_sampled < 3 for s in owners])                      # nari: EOS blocked for 3 tokens
        lg[early, CODEC_EOS] = -torch.inf
        pen = torch.tensor([s.sampling.repetition_penalty for s in owners]).unsqueeze(1)
        if (pen != 1.0).any():
            seen = torch.stack([s.seen for s in owners])
            lg = torch.where(seen, torch.where(lg > 0, lg / pen, lg * pen), lg)
        out = [0] * len(owners)
        groups = {}
        for i, s in enumerate(owners):
            groups.setdefault((s.sampling.top_k, s.sampling.top_p, s.sampling.greedy), []).append(i)
        for (top_k, top_p, greedy), idx in groups.items():
            rows = lg[idx]
            if greedy:
                tok = rows.argmax(-1)
            else:
                temp = torch.tensor([owners[i].sampling.temperature for i in idx]).unsqueeze(1)
                tok = _sample(rows / temp, 1.0, top_k, top_p, False, self.gen)
            for i, t in zip(idx, tok.tolist()):
                out[i] = t
        return out

    def _reset_codec_row(self, row):
        """A new stream starts in ``row``: zero that row of every codec state tensor (KV has batch on dim 1).

        Done as a multiply by a keep-mask *tensor*: an eager ``t[row] = 0`` bakes the Python int into the compiled op,
        so every distinct row value triggered fresh Neuron compiles mid-traffic.
        """
        keep = torch.ones(self.S)
        keep[row] = 0.0
        keep_d = keep.to(self.device)
        for k, t in self.codec.state.items():
            if t.device.type == "cpu":
                t[row] = 0
                continue
            shape = [1] * t.dim()
            bdim = 3 if k in ("fk", "fv") else 1 if k in ("tf_k", "tf_v") else 0
            shape[bdim] = self.S
            t.mul_(keep_d.to(t.dtype).view(shape))

    def _code_predictor_device(self, rows, sampled, hT):
        """Whole CP frame in ONE NKI launch (16 depth steps, on-chip top-k sampling + next-input lookups): one H2D
        (uniforms) and one D2H (15 tokens) per frame."""
        S, d = self.S, self.device
        tok0 = torch.zeros(S, 1)
        hid_idx = torch.zeros(S, dtype=torch.long)
        inv_t = torch.ones(S, 1)
        greedy = torch.zeros(S, 1)
        for s in rows:
            _, tok, r = sampled[s.slot]
            tok0[s.slot], hid_idx[s.slot] = tok, r
            inv_t[s.slot] = 1.0 / s.sampling.sub_temperature
            greedy[s.slot] = float(s.sampling.greedy)
        u = torch.rand(S, CP_STEPS - 1, generator=self.gen)
        x = self.small_proj(hT.index_select(0, hid_idx.to(d)))
        toks = cp_frame(x.t().contiguous(), tok0.to(d), self.cp_kv, **self.cpw, lm_heads=self.cp_heads,
                        rope=self.cp_rope, proj_l0=self.proj_l0_dev, proj_res=self.proj_res_rows, u=u.to(d),
                        inv_t=inv_t.to(d), greedy=greedy.to(d), ident=self.ident_dev, eps=self.cp_eps)
        frame = torch.cat([tok0.long(), toks.cpu().round().long()], 1)            # [S, 16]
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
