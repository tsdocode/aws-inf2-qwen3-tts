"""Streaming reverse proxy in front of one TTS server process per NeuronCore (no shared GIL between engines).

Routes each request to the backend with the most free slots (local counters), streams the body through unchanged.
Usage: python neuron/engine/router.py --backends http://127.0.0.1:8001 http://127.0.0.1:8002 --slots 32 --port 8000
"""
import argparse

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


def build(backends, slots):
    app = FastAPI()
    load = {b: 0 for b in backends}
    client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=4096, max_keepalive_connections=1024))

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        states = []
        for b in backends:
            try:
                r = await client.get(f"{b}/ready")
                states.append(r.status_code == 200)
            except httpx.HTTPError:
                states.append(False)
        if not all(states):
            raise HTTPException(503, f"backends ready: {states}")
        return {"ready": True, "backends": backends, "load": load}

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        body = await request.body()
        backend = max(backends, key=lambda b: slots - load[b])
        load[backend] += 1
        req = client.build_request("POST", f"{backend}/v1/audio/speech", content=body,
                                   headers={"content-type": "application/json"})
        try:
            resp = await client.send(req, stream=True)
        except httpx.HTTPError as e:
            load[backend] -= 1
            raise HTTPException(502, str(e))

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                load[backend] -= 1

        return StreamingResponse(relay(), status_code=resp.status_code,
                                 media_type=resp.headers.get("content-type", "application/octet-stream"))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", required=True)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(build(a.backends, a.slots), host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
