"""Relative reconstruction error of each weight-only format on real checkpoint
weights, with the patch's own quantizers (docker/patch_v13_fp8.py MODULE_SRC).

Reads 512 rows (256 at the start and 256 at the middle) of one to three
tensors per group from the nvidia pack and the DFlash2 drafter with pread, then drops them
from the page cache (posix_fadvise DONTNEED); ~80 MiB in total. CPU only.

  python3 evidence/e3-int-lane-cpu/weight_error.py      # needs torch + numpy

Columns: ||W_hat - W||_F / ||W||_F per tensor. "out" rows are the same ratio
for Y = X W^T with X ~ N(0, 1) (64 rows), which matches the weight error in
expectation for isotropic inputs. int8 bf16* is INT8 g128 with W_hat rounded to
BF16, the product GPTQ-Marlin forms before the MMA when scales are grouped.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = load("bench_fp8_marlin", REPO / "tools/bench_fp8_marlin.py")
Q = {}
exec(compile(load("patch_v13_fp8", REPO / "docker/patch_v13_fp8.py").MODULE_SRC, "glm53", "exec"), Q)
CK = bench.hub_dir() / f"models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/{bench.recipe_pin('revision')}"
DR = bench.hub_dir() / f"models--incoai--GLM-5.3-Flash-DFlash2/snapshots/{bench.recipe_pin('draft_revision')}"
T = "model.language_model.layers."
TENSORS = [  # (group, snapshot, tensor)
    ("kda_in", CK, T + "5.self_attn.q_proj.weight"),
    ("kda_in", CK, T + "40.self_attn.v_proj.weight"),
    ("kda_o", CK, T + "5.self_attn.o_proj.weight"),
    ("kda_o", CK, T + "40.self_attn.o_proj.weight"),
    ("mla", CK, T + "3.self_attn.q_b_proj.weight"),
    ("mla", CK, T + "39.self_attn.o_proj.weight"),
    ("shared", CK, T + "3.mlp.shared_experts.down_proj.weight"),
    ("shared", CK, T + "40.mlp.shared_experts.gate_proj.weight"),
    ("lm_head", CK, "lm_head.weight"),
    ("draft", DR, "layers.0.mlp.gate_proj.weight"),
    ("draft", DR, "layers.4.self_attn.o_proj.weight"),
    ("draft", DR, "fc.weight"),
]
ROWS = 256  # per position
READ = [0]


def read_rows(headers, name, frac):
    path, start, dtype, (r, c) = headers[name]
    assert dtype == "BF16", (name, dtype)
    rows = min(ROWS, r)
    off = start + int((r - rows) * frac) * c * 2
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = os.pread(fd, rows * c * 2, off)
        os.posix_fadvise(fd, off, rows * c * 2, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    READ[0] += rows * c * 2
    u = np.frombuffer(buf, np.uint16).reshape(rows, c).astype(np.uint32) << 16
    return torch.from_numpy(u.view(np.float32)).to(torch.bfloat16)


def formats(w):
    clip = Q["INT_CLIP_RATIOS"]["int4"]
    yield "fp8", Q["dequantize_per_channel"](*Q["quantize_per_channel"](w))
    yield "nvfp4", Q["dequantize_nvfp4"](*Q["quantize_nvfp4"](w))
    int8 = Q["dequantize_int"](*Q["quantize_int"](w, 8, 128), 8)
    yield "int8", int8
    yield "int8 bf16*", int8.to(torch.bfloat16).float()
    yield "int8 g64", Q["dequantize_int"](*Q["quantize_int"](w, 8, 64), 8)
    yield "int4 amax", Q["dequantize_int"](*Q["quantize_int"](w, 4, 128), 4)
    yield "int4", Q["dequantize_int"](*Q["quantize_int"](w, 4, 128, clip), 4)
    yield "int4 g64", Q["dequantize_int"](*Q["quantize_int"](w, 4, 64, clip), 4)


def main():
    torch.manual_seed(0)
    heads = {CK: bench.read_headers(CK), DR: bench.read_headers(DR)}
    cols, per_group = None, {}
    for group, snap, name in TENSORS:
        w = torch.cat([read_rows(heads[snap], name, f) for f in (0.0, 0.5)])
        wf, x = w.float(), torch.randn(64, w.shape[1])
        y = x @ wf.T
        res = {c: ((d - wf).norm() / wf.norm(), (x @ d.T - y).norm() / y.norm())
               for c, d in formats(w)}
        if cols is None:
            cols = list(res)
            print(f"{'group':8} {'tensor (512 rows)':40} {'K':>6} " + "".join(f"{c:>11}" for c in cols))
        print(f"{group:8} {name[-40:]:40} {w.shape[1]:>6} "
              + "".join(f"{float(res[c][0]):11.4f}" for c in cols))
        print(f"{'':8} {'  out, Gaussian X':40} {'':>6} "
              + "".join(f"{float(res[c][1]):11.4f}" for c in cols))
        for c in cols:
            per_group.setdefault(group, {}).setdefault(c, []).append(float(res[c][0]))
    print("\nper group (mean weight error over its tensors)")
    print(f"{'group':8} " + "".join(f"{c:>11}" for c in cols))
    for group, d in per_group.items():
        print(f"{group:8} " + "".join(f"{np.mean(d[c]):11.4f}" for c in cols))
    print(f"\nread {READ[0] / 2**20:.1f} MiB from safetensors")


if __name__ == "__main__":
    main()
