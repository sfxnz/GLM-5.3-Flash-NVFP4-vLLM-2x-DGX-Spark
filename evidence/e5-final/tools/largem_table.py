#!/usr/bin/env python3
"""largem_table.py LARGEM_JSON[,MORE_JSON...] [E4_JSON]: prefill-size GEMM table from tools/bench_fp8_marlin.py runs
(--m 1,256,1024,2048 and --m 1152; rows of all given JSONs are merged).

Per GEMM (one rank, TP=2 shard): ms at each M >= 256 for BF16 / INT8 g128 / FP8 / NVFP4 and INT8/BF16. Then the extra
ms per prefill chunk of M rows that INT8 costs over BF16 on the target groups the recipe puts in INT8, with per-chunk
counts: shared (gate_up + down) x42, mla (q_b + o) x11, kda_o x34, kda_in x34. lm_head only sees the rows vLLM samples
from: the target's logits_indices (the last row of the chunk, M=1, model_runner.sample) and the DFlash2 drafter's query
block (M=8 at c=1; it shares the lm_head); M=8 comes from E4_JSON (evidence/e4-microbench/bench.json).

The serve's prefill chunks are 1152 tokens (the F3 profile's step annotations read execute_context_1(1152), last
chunk 1267), not the 2048 of max_num_batched_tokens, so the comparison with the measured slowdown is per prefill
token: extra us/token = extra ms per chunk / M. Measured: bench_decode cell E server prefill tok/s. Stdlib only."""
import json
import sys

PREFILL_COUNT = {"shared": 42, "mla": 11, "kda_o": 34, "kda_in": 34}
TARGET = ("shared", "mla", "kda_o", "kda_in")
FMT = (("bf16", "bf16_us"), ("int8", "int8_us"), ("fp8", "fp8_us"), ("nvfp4", "nvfp4_us"))

rows_all = []
for p in sys.argv[1].split(","):
    rows_all += json.load(open(p))["rows"]
e4 = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
rows = sorted((r for r in rows_all if r["group"] in TARGET + ("lm_head",)),
              key=lambda r: (TARGET.index(r["group"]) if r["group"] in TARGET else 9, r["gemm"], r["M"]))
Ms = sorted({r["M"] for r in rows if r["M"] >= 256})

print("ms per GEMM call, one rank (CUDA-graph replay, checkpoint weights, rotated copies)")
print(f"{'group':8} {'gemm':16} {'n x k':>12} {'cnt':>4} {'M':>5}" + "".join(f"{f:>8}" for f, _ in FMT)
      + f"{'int8/bf16':>10}")
for r in rows:
    if r["M"] not in Ms:
        continue
    cnt = PREFILL_COUNT.get(r["group"], 1)
    v = [r[k] / 1e3 for _, k in FMT]
    print(f"{r['group']:8} {r['gemm']:16} {r['n']:>6}x{r['k']:<5} {cnt:>4} {r['M']:>5}"
          + "".join(f"{x:8.3f}" for x in v) + f"{v[1] / v[0]:10.2f}")

print("\nper prefill chunk of M rows, target INT8 groups, one rank, count-weighted (ms)")
print(f"{'group':8} {'M':>5}" + "".join(f"{f:>9}" for f, _ in FMT) + f"{'int8-bf16':>11}{'int8/bf16':>10}")
tot = {}
for M in Ms:
    t = dict.fromkeys([f for f, _ in FMT], 0.0)
    for g in TARGET:
        s = dict.fromkeys(t, 0.0)
        for r in rows:
            if r["group"] == g and r["M"] == M:
                for f, k in FMT:
                    s[f] += r[k] / 1e3 * PREFILL_COUNT[g]
        for f in t:
            t[f] += s[f]
        print(f"{g:8} {M:>5}" + "".join(f"{s[f]:9.2f}" for f in s)
              + f"{s['int8'] - s['bf16']:11.2f}{s['int8'] / s['bf16']:10.2f}")
    print(f"{'4 groups':8} {M:>5}" + "".join(f"{t[f]:9.2f}" for f in t)
          + f"{t['int8'] - t['bf16']:11.2f}{t['int8'] / t['bf16']:10.2f}")
    tot[M] = t

lm = {r["M"]: r for r in rows_all if r["group"] == "lm_head"}
lm1 = (lm[1]["int8_us"] - lm[1]["bf16_us"]) / 1e3
lm8 = None
if e4:
    r8 = [r for r in e4["rows"] if r["group"] == "lm_head" and r["M"] == 8]
    lm8 = (r8[0]["int8_us"] - r8[0]["bf16_us"]) / 1e3 if r8 else None
print(f"\nlm_head per chunk: target logits M=1 int8-bf16 {lm1:+.2f} ms"
      + ("" if lm8 is None else f"; drafter query block M=8 (E4 microbench) {lm8:+.2f} ms"))
lm_tot = lm1 + (lm8 or 0.0)
pred = {}
for M in Ms:
    extra = tot[M]["int8"] - tot[M]["bf16"] + lm_tot
    pred[M] = extra / M * 1e3
    print(f"implied INT8 extra per {M:>4}-row chunk: {tot[M]['int8'] - tot[M]['bf16']:+7.1f} (4 groups) {lm_tot:+.1f}"
          f" (lm_head) = {extra:+7.1f} ms per chunk = {pred[M]:+6.1f} us per prefill token")

print("\nmeasured (bench_decode cell E, server prefill tok/s at 32k / 128k; us per prefill token):")
meas = {"E0": ("v11, BF16 target + BF16 draft", [(1331, 1329)]),
        "E3a": ("v13, BF16 target + NVFP4 draft", [(1355, 1353)]),
        "F": ("v13 defaults, INT8 target (F1 bench-1, F2 bench-E1, bench-E2)", [(1193, 1191), (1199, 1194), (1168, 1164)])}
us = {}
for k, (desc, panels) in meas.items():
    a = sum(p[0] for p in panels) / len(panels)
    b = sum(p[1] for p in panels) / len(panels)
    us[k] = (1e6 / a, 1e6 / b)
    print(f"  {k:4} {desc:62} {a:7.1f} / {b:7.1f} tok/s = {us[k][0]:6.1f} / {us[k][1]:6.1f} us/token")
for k in ("E0", "E3a"):
    d32, d128 = us["F"][0] - us[k][0], us["F"][1] - us[k][1]
    print(f"  F - {k:3}: {d32:+6.1f} / {d128:+6.1f} us/token = {d32 * 1.152:+6.1f} / {d128 * 1.152:+6.1f} ms per 1152-token"
          f" chunk ({100 * (us[k][0] / us['F'][0] - 1):+.1f}% / {100 * (us[k][1] / us['F'][1] - 1):+.1f}% tok/s)")
m = 1152 if 1152 in pred else min(Ms, key=lambda x: abs(x - 1152))
print(f"  predicted at M={m}: {pred[m]:+.1f} us/token = {pred[m] * m / 1e3:+.1f} ms per chunk")
