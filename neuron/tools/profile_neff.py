"""Device-profile one NEFF on a single NeuronCore and print the key summary.

Usage: python neuron/tools/profile_neff.py <file.neff | --latest-since EPOCH> --name NAME [--core 1]
Writes NEFF/NTFF under neuron/profiles/NAME and parquet tables under ~/.local/share/neuron-profile.
"""
import argparse, glob, os, shutil, subprocess, sys
from pathlib import Path

import pandas as pd

NE = "/opt/aws/neuron/bin/neuron-explorer"
DATA = os.path.expanduser("~/.local/share/neuron-profile")
ROOT = Path(__file__).resolve().parents[1] / "profiles"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("neff", nargs="?")
    ap.add_argument("--latest-since", type=float, help="pick the newest /tmp/neff_cache NEFF modified after this epoch")
    ap.add_argument("--name", required=True)
    ap.add_argument("--core", default="1")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    neff = a.neff
    if a.latest_since:
        cands = [(os.path.getmtime(f), f) for f in glob.glob("/tmp/neff_cache/**/*.neff", recursive=True)
                 if os.path.getmtime(f) > a.latest_since]
        if not cands:
            sys.exit("no NEFF newer than --latest-since")
        neff = max(cands)[1]
    out = ROOT / a.name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(neff, out / "k.neff")
    env = dict(os.environ, NEURON_RT_VISIBLE_CORES=a.core)
    subprocess.run([NE, "capture", "-n", "k.neff", "-s", "k.ntff", "--profile-nth-exec=3"], cwd=out, env=env,
                   stdout=open(out / "cap.log", "w"), stderr=subprocess.STDOUT, check=False)
    shutil.rmtree(f"{DATA}/profiles/global/{a.name}@latest", ignore_errors=True)
    subprocess.run([NE, "view", "-n", "k.neff", "-s", "k_exec_3.ntff", "--data-path", DATA, "--display-name", a.name,
                    "--ingest-only"], cwd=out, stdout=open(out / "ingest.log", "w"), stderr=subprocess.STDOUT, check=False)
    d = f"{DATA}/profiles/global/{a.name}@latest"
    s = pd.read_parquet(f"{d}/Summary.parquet").iloc[0]
    t = s["total_time"]
    print(f"[{a.name}] device {t * 1e6:.1f} us | HBM rd {s['hbm_read_bytes'] / 1e6:.2f} MB wr {s['hbm_write_bytes'] / 1e6:.2f} MB"
          f" | eff {s['hbm_read_bytes'] / t / 1e9:.0f} GB/s read | DMA {s['dma_active_time_percent'] * 100:.0f}%"
          f" TensorE {s['tensor_engine_active_time_percent'] * 100:.0f}% VectorE {s['vector_engine_active_time_percent'] * 100:.0f}%"
          f" ScalarE {s['scalar_engine_active_time_percent'] * 100:.0f}% GpSimd {s['gpsimd_engine_active_time_percent'] * 100:.0f}%")
    o = pd.read_parquet(f"{d}/OpcodeSummary.parquet")
    o = o[~o.opcode.str.contains("SEMAPHORE")].sort_values("active_time_ns", ascending=False).head(a.top)
    for r in o.itertuples():
        print(f"    {r.engine:7s} {r.opcode:22s} n={r.instruction_count:6d} {r.active_time_ns / 1e3:8.1f} us")


if __name__ == "__main__":
    main()
