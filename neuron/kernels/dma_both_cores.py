"""Achieved HBM->SBUF bandwidth from wall time: one stream kernel looped, on whichever core is visible.
Run two copies (NEURON_RT_VISIBLE_CORES=0 and =1) at once to see whether the cores share bandwidth.
"""
import sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dma_ceiling import stream  # noqa: E402

x = torch.randn(16, 128, 16384, dtype=torch.bfloat16).to("neuron")   # 67 MB per call
stream(x)
torch.neuron.synchronize()
n = 200
t0 = time.perf_counter()
for _ in range(n):
    stream(x)
torch.neuron.synchronize()
dt = time.perf_counter() - t0
print(f"core {sys.argv[1]}: {x.numel() * 2 * n / dt / 1e9:.0f} GB/s ({dt / n * 1e3:.2f} ms per 67 MB)", flush=True)
