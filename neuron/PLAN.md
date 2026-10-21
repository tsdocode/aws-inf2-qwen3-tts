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

### Codec prototype (`proto/codec_chunk.py`, upstream whole-sequence decoder, B=32, 12 frames)
- fp32 (repo default): 449.5 ms / chunk = **37.5 ms per frame-batch**, SNR 111.7 dB vs CPU fp32,
  compile 876 s. fp32 TensorE is ~4x slower than bf16 → bf16 run pending.

- bf16: 29.2 ms per frame-batch (only 22% faster) and **SNR 24.9 dB** (worst row 28% rel err) →
  plain bf16 unacceptable. Profile (`profiles/codec-bf16-b32`): 293 ms device / 12-frame chunk,
  **9.8 GB HBM read + 4.4 GB written**, TensorE 16%; ScalarE 250k ACTIVATE (96 ms), GpSimd 117 ms,
  VectorE casts/elementwise ~130 ms. Cause: SnakeBeta `x + 1/b*sin^2(a*x)` after every conv at audio
  rate, each op a separate pass spilled to HBM; bf16 sin() also explains the quality loss.
  Compute floor ≈ 2–3 GFLOP/frame/row → ~1–4 ms per frame-batch: **largest headroom of any stage**.

### Baseline per frame-batch (B=32 per core, compiler codegen, no NKI)
Talker 29.6 + CP 32.1 + codec 37.5 (fp32) = **99.2 ms vs 80 ms real-time budget** → kernels required.

### CORRECTION (sustained bandwidth, `kernels/dma_both_cores.py`): 67 MB streams looped back-to-back
- **257 GB/s per NeuronCore sustained; both cores at once 251 + 257 ≈ 508 GB/s per chip** (no contention).
  NVIDIA L4: 300 GB/s spec (~240–270 achievable) → inf2 chip ≈ 1.9x L4 *if both cores stream*.
- The "175–180 GB/s ceiling" below was a single short 8 MB call dominated by ~10 µs fixed start cost.
  Lesson: per-kernel fixed overhead matters → fewer, longer fused kernels approach 257 GB/s.
- Revised talker floor: 2.8 GB bf16 weights / 257 GB/s ≈ 11 ms (+ KV) per step at B=32.

### Measured DMA ceiling (`kernels/dma_ceiling.py`) — superseded by the correction above
Pure HBM→SBUF stream on one inf2 NeuronCore: **~175–180 GB/s end-to-end** (~220 GB/s while active),
independent of 4–64 KiB/partition payloads. Not 410 (chip HBM/2) nor 272 (doc DMA peak).
Revised floors @ ~200 GB/s, B=32/core: talker 17–23 ms (weights 2.8 GB + KV), CP 15.5 ms, codec
compute-bound ~1–4 ms if fused. → bytes must drop, not just kernels improve:
1. FP8 weights (TensorE supports fp8 on v2; H100 path already uses fp8 talker gate/up): ~-8 ms talker,
   ~-7 ms CP. Needs audio-quality check.
2. Fewer weight reads per stream: TP=2 across cores (halves weight bytes/stream, +~160 all-reduces per
   frame) or stage split (core0 talker+CP for all 64 streams, core1 codec).
3. Realistic context lengths from the benchmark (ctx 512 is pessimistic for KV).

### First NKI kernel (`kernels/decode_linear.py`: fused RMSNorm + linear, transposed layout)
- Correct first try (1.7e-4 rel L2), NKI compile 3 s. Device 70 µs for 8.46 MB (qkv 1024x4096,
  B=32) vs 47 µs pure stream: weight DMA starts only after the norm preamble (11 µs) then ~157 GB/s.

### CP step profile deep-dive
- ~5,100 tiny TensorE transposes (16x16 / x2) per step from attention reshapes (GQA
  repeat_interleave, head layout) — layout overhead, not math.
- On NeuronCore-v2 MM cost = max(N, 64 cycles): with 32 tokens, weights-stationary is already the
  cheaper mapping; the win is removing transposes + fusing ACTIVATE-level small ops.

### NKI-Lib on inf2
- `nkilib.core.mlp.mlp` (TKG, fused add+RMSNorm+SwiGLU, 32x1024x3072): **fails to compile on gen2**
  — SBUF allocation out of bounds (`SB<0,204288>` > 192 KiB/partition). Sized for trn2+.
  → Write our own inf2 kernels; use NKI-Lib as algorithm reference only.
- torch.compile MLP alone at that shape: 183 µs/call (46 µs bandwidth floor + launch overhead).

6. One process grabs BOTH NeuronCores by default → always set `NEURON_RT_VISIBLE_CORES`.

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

### Codec NKI kernels (`kernels/codec_kernels.py`, `kernels/codec_decoder.py`) — DONE (whole-sequence)
Hybrid decoder: compiled frame-rate front + NKI audio-rate blocks. B=32, 16-frame chunks, golden codes:

| Decoder | ms / frame-batch | audio SNR vs upstream fp32 |
|---|---|---|
| compiled fp32 (baseline) | 37.5 | 111.7 dB |
| compiled bf16 | 29.2 | 24.9 dB |
| **NKI hybrid, fp32 front** | **9.4** | **52.6 dB** (worst row 50.0) |
| NKI hybrid, bf16 front (`front_bf16`) | 8.4 | 41.5 dB (worst row 39.1) — needs listening test |

Kernels (all validated per block on real activations, 51–59 dB vs upstream fp32):
- `residual_unit`: SnakeBeta → dilated K=7 conv → SnakeBeta → 1x1 conv → +res, one pass. Channels on
  partitions, conv = K shifted accumulating matmuls (free-axis offsets). C=768: 383 µs device,
  TensorE 79% (~50 TF/s). For C ≤ 192 SnakeBeta's two terms go to TensorE as separate matmuls
  (conv is linear) — takes the add off the VectorE bottleneck (block3 2.73 → 1.88 ms device).
- `snake_transconv`: SnakeBeta + causal ConvTranspose1d via phase decomposition
  `y[n*r+m] = W_m^T x[n] + W_{m+r}^T x[n-1]`; weights in SBUF in output-channel groups (block0 37.7 MB).
- `snake_conv_out`: tail SnakeBeta(96) + conv 96→1 + clamp: 46 → 8 ms per chunk.
inf2 gotchas: ScalarE sin valid only on [-π, π] and |αx| reaches 142 → range-reduce with the fp32
round trick `(q + 1.5·2^23) − 1.5·2^23` (VectorE has no fmod); Copy activation takes only immediate
bias (use VectorE tensor_scalar for per-channel bias); GpSimd rejects fp32 tensor_tensor (NCC_IXCG965)
but does casts fine; `torch.neuron.synchronize()` for timing (D2H of block outputs is hundreds of MB).
TODO: streaming mode — feed conv histories / transconv overlap as kernel inputs instead of zeros.

### FP8 weight-only decode linear (`kernels/decode_linear.py: rmsnorm_linear_fp8_t`)
- v2 TensorE accepts fp8 stationary x bf16 moving (fp32 PSUM). Only *legacy* float8_e4m3 (IEEE-style,
  max 240) on v2; torch lacks it → encode with `ml_dtypes.float8_e4m3`, ship as uint8, `.view()` in
  kernel. Verified on device: kernel vs host math on dequantized W = 2.4e-3 rel L2.
- Per-output-channel quantization error: 2.6% rel L2 per matrix (end-to-end audio check TODO).
- Talker MLP-shaped 2048x6144, B=32: **bf16 147.0 µs (172 GB/s) → fp8 104.6 µs** (12.75 MB). Only
  1.4x: ~11 µs norm prologue before the weight stream + tail. A fused layer kernel keeps the weight
  stream continuous across ops, which should get closer to 2x.

### Fused Code Predictor step kernel (`kernels/cp_kernels.py: cp_step`) — DONE (bf16)
One NKI kernel per depth step: 5 layers + final norm + lm_head, KV slot written in place by DMA.
- q/k-norm weight + RoPE folded into one 128x128 matrix per (layer, step, q|k): 2 fp32 matmuls per
  layer instead of rotate-half shuffles (the compiler's ~5.1k transposes).
- Attention over s+1 slots: elementwise q*K, ones-matmul (sums over head dim + broadcasts), softmax
  on the free axis, p*V reduced over slots → lands in o_proj's input layout. Specialized per s (16
  kernels, ~4.5 s compile each).
- Accuracy vs fp32 oracle (teacher-forced, 8 steps, B=32): kernel 0.9–1.9e-2 logits rel L2, CPU
  bf16 1.0–2.1e-2 → as accurate as torch bf16.
- **12.6 ms per frame-batch (B=32) vs 32.1 compiled** (2.5x). Step device time 848 µs at 203 GB/s,
  DMA 83% → at the measured bandwidth ceiling.
- FP8 (per-output-channel e4m3): 9.37 ms/frame but logits error 7–17% (all groups) / 3–8% (MLP only)
  vs 1–2% bf16 → too lossy. Block scales via `scalar_tensor_tensor` partial-sum accumulation cost ~0.35 ms
  /step at 128-wide blocks (≈ the fp8 saving). Kept as opt-in per group (`cp_step_params(fp8=[...])`);
  revisit with audio-level evaluation or K=512 blocks.

### Talker fused step — dense path (`kernels/talker_kernels.py: talker_dense_step`, attention stubbed)
- One launch for all 28 layers over a MIXED token batch of T tokens (decode + prefill chunks).
- Per-token RoPE: rotate_half is a fixed partition permutation → `rotate_half(w*x) = (Perm diag(w)) x`
  as one fp32 matmul for all heads/tokens; host passes per-token cos/sin [128, T]. 1-layer check vs
  torch: 3.0e-3 rel L2.
- 28-layer weight stream (2.82 GB bf16): **T=32 11.65 ms (242 GB/s, 94% of peak)**, T=64 11.82 ms,
  T=128 12.90 ms. → adding 96 prefill tokens to a 32-stream decode step costs ~1.25 ms: fused mixed
  prefill+decode is nearly free on the dense side; only attention scales with tokens x context.
- Target with paged attention: ~11.7 + KV read (1.9 GB @ ctx 512 ≈ 7.3 ms) ≈ 19 ms vs 29.6 compiled.

### Paged KV on gen2 (`kernels/test_paged_io.py`) — idioms verified bit-exact on inf2
- K cache D-major `[blocks, D, BL, KVH]`, V cache token-major `[blocks, BL, KVH, D]` (BL=128): each page
  gather is one `vector_select` DMA with 2 KiB contiguous per partition, landing directly in the
  stationary layouts for QK^T (K: [D, keys]) and PV (V: [keys, D]).
- Writes at runtime (block, offset): `.ap(..., scalar_offset=idx, indirect_dim=0)` per token; V is
  transposed to token-major on TensorE first.
- NKI-Lib attention kernels are NOT usable on gen2 as-is (indirect dma_transpose TRN2+, tensor
  indirection v4+, hwdge v3+, range_select v3+). Design follows attention_tkg's orientation (K stationary,
  Q moving; V stationary, P moving → no P transpose) with gen2-safe pieces. Score bound for a constant
  softmax max is 187–5600 → real max pass needed (batched: free-axis reduce + one PE transpose).
- Mixed batch plan: write all new K/V into the cache first, then attend from the cache only with one
  predicate `key_pos <= q_pos` (covers decode, causal prefill chunks and padding).

### Running total per frame-batch (B=32 per core)
talker 29.6 (compiled) + CP 32.1 (compiled) + codec 9.4 (NKI) = 71.1 ms →
talker 29.6 (compiled) + **CP 12.6 (NKI)** + codec 9.4 (NKI) = **51.6 ms vs 80 ms budget**.

### Revised priority (2026-10-21, after baselines)
1. ~~Codec fused conv + SnakeBeta kernel~~ done: 37.5 → 9.4 ms (fp32 front) / 8.4 ms (bf16 front).
2. **FP8 decode linear** (talker + CP): measure fp8 TensorE speed + quality on inf2.
3. Placement: TP=2 vs stage split vs DP — measure inf2 all-reduce latency first.
4. CP fused step kernel (attention ≤16, in-kernel KV write, no transposes); talker paged attention.

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
