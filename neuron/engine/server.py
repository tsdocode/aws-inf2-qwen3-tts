"""OpenAI-style streaming TTS server on inf2: one NeuronTTSEngine per NeuronCore, least-loaded routing.

POST /v1/audio/speech  {"input", "voice", "language", "instruct", "response_format": "pcm"|"wav", "stream",
                        "temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens",
                        "subtalker_temperature", "subtalker_top_k", "do_sample"}
  stream=true  -> chunked PCM16 (24 kHz mono) as frames are decoded (for wav: a streaming header first).
  stream=false -> whole file.
GET /health, /ready

Usage: NEURON_RT_VISIBLE_CORES=0,1 python neuron/engine/server.py --cores 2 --slots 32 --port 8000
"""

import argparse
import asyncio
import io
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

sys.path.insert(0, str(Path(__file__).parent))
from neuron_engine import HostAssets, NeuronTTSEngine, Sampling  # noqa: E402

SR = 24000
MODEL = str(Path.home() / "models/Qwen3-TTS-12Hz-1.7B-CustomVoice")


def wav_header(n_bytes=0xFFFFFFFF - 36):
    return (b"RIFF" + struct.pack("<I", n_bytes + 36) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16)
            + b"data" + struct.pack("<I", n_bytes))


def to_pcm16(x):
    return (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


class EngineWorker:
    """Runs one engine's step loop in a thread; requests are handed over through the engine's waiting queue."""

    def __init__(self, engine):
        self.engine = engine
        self.cv = threading.Condition()
        self.load = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, **kw):
        with self.cv:
            self.load += 1
            s = self.engine.submit(**kw)
            self.cv.notify()
        return s

    def done(self):
        with self.cv:
            self.load -= 1

    def _run(self):
        # NOTE: no torch.set_num_threads here. Dynamo guards on that global: changing it in the serving thread made
        # every compiled graph recompile on its first serving step (2.5 s stall for all streams in flight).
        while True:
            with self.cv:
                while not self.engine.waiting and not self.engine.active():
                    self.cv.wait()
            with torch.no_grad():
                self.engine.step()


def build_app(workers):
    app = FastAPI()
    state = {"ready": False}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if not state["ready"]:
            raise HTTPException(503, "warming up")
        return {"ready": True, "engines": len(workers), "load": [w.load for w in workers]}

    @app.get("/stats")
    def stats():
        out = []
        for w in workers:
            ms = np.array(w.engine.stats["step_ms"][-2000:] or [0.0])
            out.append({"steps": w.engine.stats["steps"], "step_ms_p50": float(np.median(ms)),
                        "step_ms_p90": float(np.percentile(ms, 90)), "active": len(w.engine.active()),
                        "waiting": len(w.engine.waiting)})
        return out

    @app.post("/v1/audio/speech")
    async def speech(req: dict):
        text = req.get("input")
        if not text:
            raise HTTPException(400, "input is required")
        fmt = req.get("response_format", "pcm")
        if fmt not in ("pcm", "wav"):
            raise HTTPException(400, "response_format must be pcm or wav")
        sampling = Sampling(
            temperature=float(req.get("temperature", 0.9)), top_k=int(req.get("top_k", 50)),
            top_p=float(req.get("top_p", 1.0)), repetition_penalty=float(req.get("repetition_penalty", 1.05)),
            sub_temperature=float(req.get("subtalker_temperature", 0.9)),
            sub_top_k=int(req.get("subtalker_top_k", 50)),
            max_new_tokens=int(req.get("max_new_tokens", 2048)), greedy=not bool(req.get("do_sample", True)))
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def sink(pcm):                                  # called on the engine thread
            loop.call_soon_threadsafe(q.put_nowait, None if pcm is None else to_pcm16(pcm))

        # route to the engine with the most FREE slots: a request routed to a full engine would wait for a whole
        # utterance to finish before it can even prefill
        worker = max(workers, key=lambda w: w.engine.S - w.load)
        try:
            worker.submit(text=text, speaker=req.get("voice", "aiden"), language=req.get("language", "auto"),
                          instruct=req.get("instruct", req.get("instructions", "")) or "", sampling=sampling, sink=sink)
        except (ValueError, KeyError) as e:
            worker.done()
            raise HTTPException(400, str(e))

        async def chunks():
            try:
                if fmt == "wav":
                    yield wav_header()
                while True:
                    c = await q.get()
                    if c is None:
                        return
                    yield c
            finally:
                worker.done()

        if req.get("stream", False):
            return StreamingResponse(chunks(), media_type="audio/pcm" if fmt == "pcm" else "audio/wav")
        body = io.BytesIO()
        async for c in chunks():
            body.write(c)
        data = body.getvalue()
        if fmt == "wav":
            pcm = data[44:]
            data = wav_header(len(pcm)) + pcm
        return Response(data, media_type="audio/pcm" if fmt == "pcm" else "audio/wav")

    return app, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--codec-every", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch.set_num_threads(1)            # once, BEFORE warm-up: warm-up and serving must see the same global state
    assets = HostAssets(MODEL)
    engines = [NeuronTTSEngine(slots=a.slots, device=f"neuron:{i}", codec_every=a.codec_every, assets=assets,
                               seed=1234 + i) for i in range(a.cores)]
    del assets
    # warm up one engine at a time: NKI kernel tracing is not thread-safe (concurrent traces fail with
    # "No backend set" / "duplicate op name"); afterwards every shape is cached and serving does not trace.
    for e in engines:
        e.warmup()
    workers = [EngineWorker(e) for e in engines]
    app, state = build_app(workers)
    state["ready"] = True
    print(f"ready: {a.cores} engines x {a.slots} slots on port {a.port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
