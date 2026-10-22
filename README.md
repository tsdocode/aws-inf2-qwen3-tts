# Qwen3-TTS on AWS Inferentia2

A streaming serving engine for
[Qwen3-TTS 1.7B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) on a single
**inf2.xlarge** (one Inferentia2 chip, 2 NeuronCore-v2), with the hot path written as hand-tuned
[NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/) kernels.

It is a port of [nari-labs/nari-qwen3-tts](https://github.com/nari-labs/nari-qwen3-tts), Nari Labs'
high-throughput H100 engine: the upstream package (model loading, prompt protocol, codec) is kept as is, and
everything Neuron-specific lives in [`neuron/`](neuron/).

## Results

64 concurrent streaming clients on one inf2.xlarge (2 NeuronCores, one engine process per core), closed loop
(each client sends its next request as soon as the previous one finishes), 90 s, 0 errors:

| | |
|---|---|
| Real-time factor per stream (p50) | **0.79** (faster than real time) |
| Aggregate audio throughput | **77x real time** |
| TTFA, steady state (requests after the first 10 s) | p50 **245 ms**, p95 267 ms, p99 **279 ms** |
| TTFA, all requests | p50 246 ms, p95 279 ms, p99 669 ms (the 64-request burst at t = 0) |
| Requests with a playback stall > 50 ms | 0.9% |

Mean engine step 60 ms per core for 32 streams (budget: 80 ms of audio per step). Accuracy vs the CPU fp32
upstream model (teacher-forced golden prompts): talker codebook 0 95-100%, code predictor 96-97%, end of speech
on the golden frame; streamed audio vs a whole-sequence fp32 decode of the same codes: 46-48 dB SNR.

## How it works

Per NeuronCore, a lockstep continuous-batching engine ([`neuron/engine/neuron_engine.py`](neuron/engine/neuron_engine.py))
runs one step per 80 ms audio frame for up to 32 streams:

1. **Talker** (28 layers) — one NKI kernel per step for a *mixed* batch: one decode token per active stream
   with paged KV attention, plus waiting prompts packed back to back into 128-token tiles (varlen prefill with a
   block-diagonal causal mask, attending this step's K/V directly from SBUF).
   [`neuron/kernels/talker_kernels.py`](neuron/kernels/talker_kernels.py)
2. **Code predictor** — the whole 16-step depth loop in one launch, with on-chip top-k sampling and next-input
   embedding gathers. [`neuron/kernels/cp_kernels.py`](neuron/kernels/cp_kernels.py)
3. **Codec** — streaming, stateful decoder every 2 steps: NKI kernels for the sliding-window front transformer,
   the fused SnakeBeta + conv residual units, transposed convs and the output tail.
   [`neuron/kernels/codec_kernels.py`](neuron/kernels/codec_kernels.py),
   [`codec_front_kernels.py`](neuron/kernels/codec_front_kernels.py), [`codec_stream.py`](neuron/kernels/codec_stream.py)

An OpenAI-style streaming HTTP server ([`neuron/engine/server.py`](neuron/engine/server.py)) runs one process
per NeuronCore behind a small streaming router ([`router.py`](neuron/engine/router.py)).
[`neuron/PLAN.md`](neuron/PLAN.md) is the full engineering log: every measurement, profile and design decision,
including the inf2/NKI gotchas found along the way.

## Quick start (inf2.xlarge, Neuron SDK 2.33 / torch-neuronx 2.14)

```bash
# Python env layered on the Neuron DLAMI's torch-neuronx venv
./neuron/setup_env.sh ~/venvs/qtts && source ~/venvs/qtts/bin/activate
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ~/models/Qwen3-TTS-12Hz-1.7B-CustomVoice

# one backend per NeuronCore (:8001, :8002) + router on :8000
./neuron/engine/start_server.sh

curl -N http://127.0.0.1:8000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"input": "Hello from Inferentia.", "voice": "ryan", "language": "english", "response_format": "pcm", "stream": true}' \
  > out.pcm   # 24 kHz mono PCM16

# load test: 64 concurrent streaming clients
python neuron/engine/loadtest.py --url http://127.0.0.1:8000 --concurrency 64 --duration 90
```

The first start compiles and warms up every kernel variant (~15 min per core); compiled NEFFs and traced NKI
kernels are cached under `~/.cache/qtts-neuron/`, keyed by the kernel sources.

## Tests and benchmarks

- `neuron/engine/test_engine_golden.py --teacher --audio-check` — end to end vs the CPU fp32 golden codes and audio
- `neuron/kernels/test_*.py` — each kernel vs a PyTorch / fp32 reference
- `neuron/kernels/bench_*.py`, `neuron/engine/bench_engine.py --profile` — per-kernel and per-phase timings
- `neuron/tools/profile_neff.py` — device profile (neuron-explorer) of one kernel

## Layout

```
neuron/
  kernels/    NKI kernels + their tests and benchmarks
  engine/     engine, HTTP server, router, load test, golden test
  reference/  CPU fp32 golden generation
  golden/     golden codes / audio for the accuracy tests
  proto/      early compiled-graph prototypes (baselines in PLAN.md)
  tools/      profiling helpers
  PLAN.md     engineering log
src/nari_qwen3_tts/   upstream package (unchanged)
docs/README.nari.md   upstream README
```

## Credits and citation

- **Qwen3-TTS** by the Qwen team, Alibaba — [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
  [model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice),
  [technical report](https://arxiv.org/abs/2601.15621).
- **nari-qwen3-tts** by [Nari Labs](https://nari-labs.com) —
  [nari-labs/nari-qwen3-tts](https://github.com/nari-labs/nari-qwen3-tts): the serving engine this port is based
  on (Apache-2.0); its original README is in [`docs/README.nari.md`](docs/README.nari.md).

If you use Qwen3-TTS, please cite:

```bibtex
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)), as upstream. The Qwen3-TTS model weights are subject
to their own license on the model card.
