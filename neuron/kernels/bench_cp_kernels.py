"""Time a full CP frame (16 fused step-kernel calls) at batch B; compare with the compiled step graph (32.1 ms).

Usage: NEURON_RT_VISIBLE_CORES=1 python neuron/kernels/bench_cp_kernels.py --batch 32
"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "proto"))
from cp_graph import MODEL  # noqa: E402
from cp_kernels import SLOTS, cp_step, cp_step_params  # noqa: E402
from nari_qwen3_tts.config import ModelAssetConfig  # noqa: E402
from nari_qwen3_tts.model.checkpoint import CheckpointLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fp8", default="", help="comma list of FP8 weight groups: qkv,o,gate_up,down,lm_heads")
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    loader = CheckpointLoader(ModelAssetConfig(model_id=MODEL, device="cpu", require_h100=False, local_files_only=True))
    predictor, _ = loader.load_code_predictor(device=torch.device("cpu"), dtype=torch.bfloat16)
    fp8 = [g for g in a.fp8.split(",") if g]
    prm = cp_step_params(predictor, fp8=fp8)
    eps = prm.pop("eps")
    rope, heads = prm.pop("rope").to("neuron"), prm.pop("lm_heads").to("neuron")
    heads_s = prm.pop("lm_heads_s").to("neuron") if "lm_heads_s" in prm else None
    dev = {k: v.to("neuron") for k, v in prm.items()}
    B, L = a.batch, len(predictor.model.layers)
    kv = torch.zeros(L, 2, 128, SLOTS, 8, B, dtype=torch.bfloat16).to("neuron")
    x = (torch.randn(1024, B) * 0.3).to(torch.bfloat16).to("neuron")

    def frame():
        out = None
        for s in range(SLOTS):
            extra = dict(lm_head_s=heads_s[max(s - 1, 0)]) if heads_s is not None else {}
            out = cp_step(x, kv, **dev, lm_head=heads[max(s - 1, 0)], rope=rope[s], s=s, eps=eps, with_head=s > 0, **extra)
        return out

    t0 = time.perf_counter()
    frame()
    torch.neuron.synchronize()
    print(f"first frame (16 kernel specializations compiled/loaded): {time.perf_counter() - t0:.1f} s")
    for _ in range(2):
        frame()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        frame()
    torch.neuron.synchronize()
    ms = (time.perf_counter() - t0) / a.iters * 1e3
    print(f"B={B}: {ms:.2f} ms per CP frame-batch (16 launches, wall)  [compiled step graph: 32.1 ms]")


if __name__ == "__main__":
    main()
