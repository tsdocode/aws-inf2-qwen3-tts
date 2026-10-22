"""Closed-loop streaming load test: C concurrent clients, each sending requests back to back.

Measures TTFA (request sent -> first PCM byte), per-request real-time factor and stalls (gaps where the client
would have run out of audio given a playback start at first byte), plus aggregate audio throughput.
Usage: python neuron/engine/loadtest.py --url http://127.0.0.1:8000 --concurrency 64 --duration 120
"""
import argparse
import asyncio
import random
import time

import httpx
import numpy as np

TEXTS = [
    "The quick brown fox jumps over the lazy dog, and then it runs back home before the rain starts.",
    "Please confirm your appointment for tomorrow at three in the afternoon.",
    "Streaming speech synthesis needs low latency and steady real time playback.",
    "Hello from Nari Labs.",
    "Our support team is available every day from nine in the morning until six in the evening.",
    "Thanks for calling. Your order has shipped and should arrive within two business days.",
]
BYTES_PER_S = 24000 * 2


async def one(client, url, voice, lang, text, results):
    t0 = time.perf_counter()
    first, got, underrun = None, 0, 0.0
    async with client.stream("POST", f"{url}/v1/audio/speech", json={
            "input": text, "voice": voice, "language": lang, "response_format": "pcm", "stream": True}) as r:
        if r.status_code != 200:
            results.append({"error": r.status_code})
            return
        async for chunk in r.aiter_bytes():
            now = time.perf_counter()
            if first is None:
                first = now
            else:  # playback started at first byte: did we run dry before this chunk arrived?
                played_until = first + got / BYTES_PER_S
                if now > played_until:
                    underrun += now - played_until
            got += len(chunk)
    t1 = time.perf_counter()
    audio_s = got / BYTES_PER_S
    results.append({"t_start": t0, "ttfa": (first - t0) * 1e3 if first else None, "audio_s": audio_s, "wall_s": t1 - t0,
                    "rtf": (t1 - t0) / max(audio_s, 1e-6), "underrun_s": underrun})


async def client_loop(client, url, stop_at, results, rng):
    while time.perf_counter() < stop_at:
        await one(client, url, rng.choice(["ryan", "aiden"]), "english", rng.choice(TEXTS), results)


async def main_async(a):
    results = []
    limits = httpx.Limits(max_connections=a.concurrency * 2, max_keepalive_connections=a.concurrency * 2)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        t0 = time.perf_counter()
        stop_at = t0 + a.duration
        await asyncio.gather(*[client_loop(client, a.url, stop_at, results, random.Random(i))
                               for i in range(a.concurrency)])
        wall = time.perf_counter() - t0
    ok = [r for r in results if "error" not in r and r["ttfa"] is not None]
    errs = len(results) - len(ok)
    ttfa = np.array([r["ttfa"] for r in ok])
    under = np.array([r["underrun_s"] for r in ok])
    audio = sum(r["audio_s"] for r in ok)
    print(f"concurrency {a.concurrency}, {wall:.0f}s: {len(ok)} requests ok, {errs} errors | "
          f"audio {audio:.0f}s -> {audio / wall:.1f}x real time aggregate")
    print(f"TTFA ms: p50 {np.percentile(ttfa, 50):.0f} | p90 {np.percentile(ttfa, 90):.0f} | "
          f"p95 {np.percentile(ttfa, 95):.0f} | p99 {np.percentile(ttfa, 99):.0f} | max {ttfa.max():.0f}")
    t_first_wave_end = min(r["t_start"] for r in ok) + a.warm_window
    steady = np.array([r["ttfa"] for r in ok if r["t_start"] > t_first_wave_end])
    if len(steady):
        print(f"steady-state TTFA ms (requests sent after the first {a.warm_window:.0f}s, n={len(steady)}): "
              f"p50 {np.percentile(steady, 50):.0f} | p90 {np.percentile(steady, 90):.0f} | p95 {np.percentile(steady, 95):.0f} | "
              f"p99 {np.percentile(steady, 99):.0f} | max {steady.max():.0f}")
    print(f"playback stalls: {np.mean(under > 0.05) * 100:.1f}% of requests stalled > 50 ms; "
          f"mean stall {under.mean() * 1e3:.0f} ms; RTF p50 {np.median([r['rtf'] for r in ok]):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--warm-window", type=float, default=10, help="seconds counted as the start-up wave")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
