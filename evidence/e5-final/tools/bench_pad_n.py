#!/usr/bin/env python3
"""bench_pad_n.py [--m 256,1024,2048] [--n 12576,12544,12672,12800] [--json OUT]: the kda_in GEMM (K 4096, per rank)
at several N with random weights, through tools/bench_fp8_marlin.py's own bench() (BF16 / FP8 / NVFP4 / INT8 / INT4,
CUDA-graph replay). Tests whether kda_in's large-M Marlin slowdown comes from its N (12576 is not a multiple of 128).
Run it like the microbench, in glm53-sm121-v13 with the repo at /work."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import bench_fp8_marlin as b  # noqa: E402

ints = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
p = argparse.ArgumentParser()
p.add_argument("--m", type=ints, default=[256, 1024, 2048])
p.add_argument("--n", type=ints, default=[12576, 12544, 12672, 12800])
p.add_argument("--k", type=int, default=4096)
p.add_argument("--json")
a = p.parse_args()
gemms = [dict(group=f"n{n}", name="kda_in_like", n=n, k=a.k, count=34, parts=[], tp=2, rank=0) for n in a.n]
args = argparse.Namespace(
    groups=[g["group"] for g in gemms], m=a.m, gate_m=[0], iters=50, rotate_mb=256, random=True,
    no_nvfp4=False, no_int=False, int_group_size=128, max_ratio=0.6, max_fro_err=0.1, max_ratio_nvfp4=0.8,
    max_fro_err_nvfp4=0.02, max_ratio_int8=1.1, max_fro_err_int=0.02, json=a.json)
sys.exit(b.bench(gemms, args))
