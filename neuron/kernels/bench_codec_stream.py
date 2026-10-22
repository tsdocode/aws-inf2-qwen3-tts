"""Per-stage time of one streaming codec chunk (B rows x F frames) on inf2.
Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/kernels/bench_codec_stream.py --rows 32 --frames 1
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from codec_kernels import residual_unit, snake_conv_out, snake_transconv  # noqa: E402
from codec_front_kernels import front_attention_mask, front_rope_tables, front_transformer  # noqa: E402
from codec_stream import StreamingCodecDecoder  # noqa: E402
from cp_graph import MODEL  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--frames", type=int, nargs="+", default=[1])
    ap.add_argument("--front-bf16", action="store_true")
    a = ap.parse_args()
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    dec = loader.load_codec(device=torch.device("cpu"), dtype=torch.float32).float().decoder.eval()
    neu = StreamingCodecDecoder(dec, batch=a.rows, front_dtype=torch.bfloat16 if a.front_bf16 else torch.float32)
    for F in a.frames:
        codes = torch.randint(0, 2048, (a.rows, 16, F)).to("neuron")
        for _ in range(3):
            neu(codes)
        torch.neuron.synchronize()
        st = neu.state
        rows, n = [], 10

        def timed(name, fn):
            out = fn()
            torch.neuron.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                out = fn()
            torch.neuron.synchronize()
            rows.append((name, (time.perf_counter() - t0) / n * 1e3))
            return out

        if neu.nki_front:
            d = neu.device
            hT = timed("front pre (compiled)", lambda: neu.pre(codes, st["pre"]))[0]
            tabs = timed("front host tables + H2D", lambda: [
                t.to(d) for t in (*front_rope_tables(st["pos"], F, neu.tf_inv_freq),
                                  front_attention_mask(st["ctx"], F, retained=neu.front_mod.retained,
                                                       window=neu.front_mod.window))])
            yT = timed("front transformer (NKI)", lambda: front_transformer(
                hT, *tabs, st["fk"], st["fv"], **neu.tf_params, eps=neu.tf_eps, qk_norm=neu.tf_qk_norm))[0]
            h = timed("front post (compiled)", lambda: neu.post(yT, st["cnx0"], st["cnx1"], st["dec0"]))[0]
        else:
            h = timed("front (compiled)", lambda: neu.front(codes, st["pos"], st["ctx"], st["pre"], st["tf_k"],
                                                            st["tf_v"], st["cnx0"], st["cnx1"], st["dec0"]))[0]
        for i, ((tc, tcs, _), units) in enumerate(neu.blocks):
            h = timed(f"block{i} transconv", lambda h=h: snake_transconv(h, **tc, **tcs, hist=st[f"tc{i}"]))[0]
            for j, (prm, sts, _) in enumerate(units):
                h = timed(f"block{i} ru{j}", lambda h=h: residual_unit(h, **prm, **sts, hist=st[f"ru{i}{j}"]))[0]
        timed("tail", lambda: snake_conv_out(h, **neu.tail_params, hist=st["tail"]))
        tot = sum(ms for _, ms in rows)
        t0 = time.perf_counter()
        for _ in range(n):
            neu(codes)
        torch.neuron.synchronize()
        full = (time.perf_counter() - t0) / n * 1e3
        print(f"B={a.rows} F={F}: full chunk {full:.1f} ms ({full / F:.1f} ms per frame); per-stage sum {tot:.1f} ms")
        for name, ms in rows:
            print(f"    {name:20s} {ms:6.2f} ms")


if __name__ == "__main__":
    main()
