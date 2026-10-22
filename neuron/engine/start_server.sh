#!/usr/bin/env bash
# One TTS server process per NeuronCore (separate GILs) + a streaming router on :8000.
# Backends start one after the other (host RAM peaks while loading; NEFFs are shared through the compile cache).
set -euo pipefail
cd "$(dirname "$0")/../.."
source ~/venvs/qtts/bin/activate
LOG=${LOG_DIR:-/tmp}
# Persistent caches (not /tmp: wiped on reboot). Without the NKI trace cache every new process re-traces every kernel
# variant in Python just to compute the NEFF cache key (~100 s per talker variant, ~15 min of warm-up per core); with
# it a cached variant loads in ~1 s. Our kernel dir is added to the fingerprinted roots (it is imported via sys.path,
# not as a package, so helper edits were not seen: stale kernels), and the cache dir is keyed by a hash of the kernel
# sources as a second guard.
KHASH=$(cat neuron/kernels/*.py | sha1sum | cut -c1-12)
export NKI_ENABLE_TRACE_CACHE=1
export NKI_TRACE_CACHE_ROOTS="$PWD/neuron/kernels"
export NKI_TRACE_CACHE_URL="$HOME/.cache/qtts-neuron/nki-trace/$KHASH"
export TORCH_NEURONX_NEFF_CACHE_DIR="$HOME/.cache/qtts-neuron/neff"
mkdir -p "$NKI_TRACE_CACHE_URL" "$TORCH_NEURONX_NEFF_CACHE_DIR"
SLOTS=${SLOTS:-32}
CODEC_EVERY=${CODEC_EVERY:-2}
for core in 0 1; do
  port=$((8001 + core))
  NEURON_RT_VISIBLE_CORES=$core nohup python -u neuron/engine/server.py --cores 1 --slots "$SLOTS" \
      --codec-every "$CODEC_EVERY" --port "$port" --seed $((1234 + core)) > "$LOG/tts_core$core.log" 2>&1 &
  echo "core $core -> :$port (pid $!)"
  until curl -sf "localhost:$port/ready" > /dev/null; do
    if ! kill -0 $! 2>/dev/null; then echo "backend on core $core died, see $LOG/tts_core$core.log"; exit 1; fi
    sleep 5
  done
done
nohup python -u neuron/engine/router.py --backends http://127.0.0.1:8001 http://127.0.0.1:8002 --slots "$SLOTS" \
    --port 8000 > "$LOG/tts_router.log" 2>&1 &
echo "router :8000 (pid $!)"
