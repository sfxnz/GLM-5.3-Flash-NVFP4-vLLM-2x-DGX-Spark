#!/usr/bin/env python3
"""microbench_table.py BENCH_JSON: per-group count-weighted ms per step (one rank) for BF16 / FP8 / NVFP4 / INT8 /
INT4 at M = 8 and 16, the INT8/FP8 ratio, the worst INT output error vs its own dequantized weights, and the E4a mix
(NVFP4 draft + INT8 on the five target groups). Stdlib only."""
import json
import sys

d = json.load(open(sys.argv[1]))
S = {(s["group"], s["M"]): s for s in d["summary"]}
groups = ["draft", "shared", "mla", "kda_o", "kda_in", "lm_head"]
fmt = ["bf16", "fp8", "nvfp4", "int8", "int4"]
print(f"passed={d['passed']} int_group_size={d['int_group_size']} max_ratio_int8={d['max_ratio_int8']}")
for M in (8, 16):
    print(f"\nM={M}  ms per step per rank (count-weighted)")
    print(f"{'group':8}" + "".join(f"{f:>8}" for f in fmt) + f"{'int8/fp8':>10}{'gate':>6}")
    tot = dict.fromkeys(fmt, 0.0)
    for g in groups:
        s = S[(g, M)]
        vals = {f: s[f"{f}_ms"] for f in fmt}
        for f in fmt:
            tot[f] += vals[f]
        r = vals["int8"] / vals["fp8"]
        print(f"{g:8}" + "".join(f"{vals[f]:8.2f}" for f in fmt) + f"{r:10.3f}{'PASS' if s['int8_passed'] else 'FAIL':>6}")
    print(f"{'all':8}" + "".join(f"{tot[f]:8.2f}" for f in fmt) + f"{tot['int8'] / tot['fp8']:10.3f}")
    mix = S[("draft", M)]["nvfp4_ms"] + sum(S[(g, M)]["int8_ms"] for g in groups[1:])
    print(f"E4a mix (draft nvfp4 + int8 x5): {mix:.2f} ms vs BF16 {tot['bf16']:.2f} (saves {tot['bf16'] - mix:.2f});"
          f" E2e (draft nvfp4 only): {S[('draft', M)]['nvfp4_ms'] + sum(S[(g, M)]['bf16_ms'] for g in groups[1:]):.2f}")
worst = max(d["rows"], key=lambda r: r["int8_fro_err"])
worst4 = max(d["rows"], key=lambda r: r["int4_fro_err"])
print(f"\nworst INT8 fro vs own dequant: {worst['int8_fro_err']:.2e} ({worst['group']} {worst['gemm']} M={worst['M']});"
      f" INT4: {worst4['int4_fro_err']:.2e} ({worst4['group']} {worst4['gemm']} M={worst4['M']}); limit 0.02")
e8 = [r["int8_vs_bf16_fro_err"] for r in d["rows"]]
e4 = [r["int4_vs_bf16_fro_err"] for r in d["rows"]]
ef = [r["fro_rel_err"] for r in d["rows"]]
en = [r["nvfp4_vs_bf16_fro_err"] for r in d["rows"]]
print(f"output error vs BF16 weights (range over rows): fp8 {min(ef):.4f}-{max(ef):.4f}  nvfp4 {min(en):.4f}-{max(en):.4f}"
      f"  int8 {min(e8):.4f}-{max(e8):.4f}  int4 {min(e4):.4f}-{max(e4):.4f}")
