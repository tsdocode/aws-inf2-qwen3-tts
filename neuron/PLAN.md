# Qwen3-TTS on Inferentia2 — port plan

Goal: beat the internal L4 result (**64 concurrent streams, TTFA < 500 ms**, real-time playback)
on **one inf2.xlarge** (1 Inferentia2 chip, 2 NeuronCore-v2, 32 GB HBM, 4 vCPU, 16 GB RAM).

## Model (Qwen3-TTS-12Hz-1.7B-CustomVoice)

| Stage | Shape | Per-frame work (1 frame = 80 ms audio) |
|---|---|---|
| Talker | 28 L, d=2048, 16 Q / 8 KV heads, hd=128, ffn 6144, codec vocab 3072 | 1 decode step, paged KV |
| Code Predictor | 5 L, d=1024, 16/8 heads, hd=128, ffn 3072, 15 lm_heads × 2048 | **16 sequential steps**, dense 16-slot KV, sampling between steps |
| Codec decoder | 8 L sliding-window (72) transformer d=512 + ConvNeXt/transconv upsampler ×1920 | chunked, stateful (conv history, transconv overlap, KV) |

Positions: repo uses one scalar position per token (mrope collapses to 1-D RoPE for text-only TTS).

## Hardware budget (per NeuronCore ≈ 95 TF bf16, ≈ 410 GB/s)

Plan = **2 independent replicas, one per NeuronCore** (no TP → no collective latency), 32 streams each.

Per frame-batch of B=32 on one core, memory-bound floor:
- Talker: 1.41 B params = 2.8 GB → **~6.9 ms** + KV reads (~114 KB/token/stream; 32×300 tok ≈ 1.1 GB → ~2.7 ms)
- Code Predictor: ~110 M params incl. heads = 220 MB × 16 steps = 3.5 GB → **~8.6 ms**
- Codec: TBD (conv-heavy, measure)
- Budget: 80 ms per frame for real time → ~**60 ms of headroom** before codec. Compute is not the limit
  (talker+CP ≈ 0.2 TFLOP per frame-batch).

## Measured on this box (2026-10-21)

| | |
|---|---|
| Per compiled-graph launch overhead | **~140 µs** |
| 15 separate graph calls vs 15 steps fused in one graph | **2150 µs vs 277 µs** |
| Eager single op | 60–90 µs |

→ Every stage must be **one NEFF per (stage, batch bucket)**. Eager PyTorch on Neuron is unusable
for serving. (Correction: 16 separate step launches cost ~2.4 ms/frame-batch ≈ 3% of budget, so
fusing CP's whole loop is an optimization, not a requirement.)

### Code Predictor prototypes (B=32, bf16, torch.compile backend="neuron")

| Variant | Compile | ms / frame-batch | Notes |
|---|---|---|---|
| whole frame, 16 steps unrolled (`proto/cp_graph.py`) | 567 s | 73.5 | 79% of HBM traffic = 320 `scatter`s from in-place `kv_cache[...].copy_()` (12.2 GB/frame) |
| one reusable step graph, step idx as tensor (`proto/cp_step_graph.py`) | **26 s** | **32.1** | 1.79 ms device time per step |

Step-graph profile (`neuron/profiles/cp_step`): instruction-overhead bound, not bandwidth bound.
- ScalarE ACTIVATE ×3249 = 1.06 ms (PSUM evacuation + tiny rope/softmax/norm ops)
- TensorE MATMUL ×19121 + LDWEIGHTS ×19040: weights stationary, only 32 moving columns per tile
- index_select of lm_head / embedding tables: 12.6 MB copied per step
- Bandwidth floor for 194 MB/step ≈ 0.47 ms → target fused NKI step kernel ~0.5–0.6 ms (~10 ms/frame)

Rules learned:
1. No in-place indexed writes into big buffers inside a graph (→ full-buffer scatter). Use
   `torch.where` masks, or NKI kernels for KV writes (esp. talker paged cache).
2. Select per-step weights by passing them as graph inputs, not `index_select` inside the graph.
3. Load weights with sequential reads: mmap'd safetensors on this EBS volume ran at 4.6 MB/s.

### Talker decode prototype (`proto/talker_decode.py`, B=32, dense KV ctx 512, read-only in graph)
- 29.6 ms / step, compile 20 s. Bandwidth floor ≈ 10 ms.
- Per-component numerics (`proto/numerics_probe.py`): rmsnorm/q_norm exact, linears ~1e-5,
  rope 8e-5, masked attention 2.5e-3, MLP 4.5e-3 rel L2 — all bf16-level.
- Per-layer drift (`proto/layer_drift.py`): with random KV, CPU bf16 vs fp32 drifts to 55% at
  layer 27 and Neuron to 56% → synthetic inputs are ill-conditioned, not a Neuron bug.
  Open item: Neuron error jumps 3× vs CPU at layer 2 — recheck with real prefill KV.

### Baseline per frame-batch (B=32 per core, compiler codegen, no NKI)
Talker 29.6 + CP 32.1 = 61.7 ms + codec (TBD) vs 80 ms real-time budget → kernels are required.

4. Disable dynamo `automatic_dynamic_shapes` — a second batch bucket for the same module otherwise
   recompiles with dynamic dims, which neuronx-cc cannot lower.
5. Accuracy must be validated on real states (prefill KV from golden prompts), not random tensors.

Accuracy note: greedy token agreement with random `past_hidden` is 82% on Neuron, but even CPU
bf16 with reordered math is only 98.2% — near-tie logits on random inputs. Needs teacher-forced
logit comparison on real talker hidden states (golden prompts) vs fp32.

## Gotchas already found
- NKI-Lib gen2 (inf2/trn1) code paths are explicitly untested upstream ("no pre-trn2 HW in fleet").
  Treat NKI-Lib as a starting point; validate every kernel on this box.
- `torch.rand(device="neuron")` recursion bug (no on-device RNG on inf2) → sampling needs our own
  RNG (counter-based hash in-kernel, or host-supplied uniforms per step).
- Compile time per NEFF is seconds–minutes → shrink the CUDA bucket lists (ttfa profile has ~100
  graph keys). Use few batch buckets (e.g. 1, 4, 8, 16, 32) + persistent compile cache.
- No torchaudio for torch 2.14 → stubbed (only unused 25 Hz tokenizer imported it).

## Phases

### 0. Reference + harness (day 1)
- [x] Env: `~/venvs/qtts` (layered on torch-neuronx 2.14 venv), weights in `~/models`.
- [x] CPU fp32 greedy golden codes/audio (`neuron/reference/make_golden.py`): short 22 frames,
      medium 74 frames.
- [ ] Load generator matching the internal L4 benchmark (need: text-length distribution,
      arrival pattern, streaming vs not, TTFA definition, sampling params).

### 1. Functional Neuron backend, plain PyTorch compiled graphs (days 1–3)
Seams (from code map): `CaptureDriver` (executor/cuda_graph.py), `TalkerKVBackend` /
`TalkerAttentionContext` (executor/talker_kv.py), sampler injection, `build_neuron_execution`
mirroring executor/build.py, host-copy PCM pool.
- Talker decode: paged KV via gather on block table (static shapes), one graph per batch bucket.
- Talker prefill: token buckets [32, 64, 128, 256, 640].
- Code Predictor: **one graph = full 16-step loop + on-device top-k/top-p sampling + codec_sum**.
- Codec: warm-chunk graphs for the few chunk sizes we keep.
- Correctness: greedy tokens match golden; codec audio close to upstream whole-sequence decoder.

### 2. Serving on two NeuronCores (day 3–4)
- Two engine replicas pinned to NC0 / NC1 (`NEURON_RT_VISIBLE_CORES`), shared HTTP front.
- Host CPU: only 4 vCPU — profile the Python planner/staging; move per-step metadata building
  (KV page tables, pinned allocs) to preallocated buffers.

### 3. NKI kernels where the profile says so (days 3–6)
Candidates, in expected order of payoff:
1. **Talker paged decode attention** + fused QK-RMSNorm + RoPE + KV-cache write
   (start from NKI-Lib `attention_tkg` w/ `active_blocks_table` + fused RoPE; validate on gen2).
2. **Code Predictor fused step** (short-cache ≤16 attention, QK-norm/RoPE, write) — and possibly a
   persistent multi-step kernel keeping activations in SBUF.
3. **On-device sampling** (top-k 50 / top-p / temperature over 3072 & 2048 vocab, own RNG).
4. Fused RMSNorm+residual, SiLU-mul MLP (NKI-Lib `mlp` tkg path, validate gen2).
5. Codec convs / transposed convs if the codec dominates.

### 4. Load test + tuning (days 6–7)
Batch buckets, chunk schedule, admission policy, TTFA p50/p95/p99 at 64 CCU, RTF per stream.
