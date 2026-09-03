#!/usr/bin/env python3
# vendored from sfxnz/forge kit @ 6285f70
"""Compare two bench.json files row by row. Usage: kit/bench_compare.py <reference.json> <new.json> [--tol 0.05]

Accepts both row schemas: the kit's (decode, aggregate, ttft_p50) and the pre-kit SUMMARY names
(median_decode_tok_s, median_agg_tok_s, median_ttft_s). Prints one line per (phase, concurrency)
and exits 1 if any decode or aggregate value is outside +-tol of the reference. TTFT is printed, not gated.
"""
from __future__ import annotations

import argparse
import json
import sys

ALIASES = {
    "decode": ("decode", "median_decode_tok_s"),
    "aggregate": ("aggregate", "median_agg_tok_s"),
    "ttft_p50": ("ttft_p50", "median_ttft_s"),
}


def pick(row: dict, field: str) -> float:
    for name in ALIASES[field]:
        if name in row:
            return float(row[name])
    raise KeyError(f"row {row.get('phase')} c={row.get('concurrency')} has no {field}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("new")
    p.add_argument("--tol", type=float, default=0.05)
    args = p.parse_args()
    ref = {(r["phase"], int(r["concurrency"])): r for r in json.load(open(args.reference))}
    new = {(r["phase"], int(r["concurrency"])): r for r in json.load(open(args.new))}
    bad = 0
    for key in sorted(ref):
        if key not in new:
            print(f"phase={key[0]} c={key[1]} MISSING in new")
            bad += 1
            continue
        parts = []
        row_ok = True
        for field in ("decode", "aggregate"):
            a, b = pick(ref[key], field), pick(new[key], field)
            delta = (b - a) / a if a else 0.0
            ok = abs(delta) <= args.tol
            row_ok &= ok
            parts.append(f"{field}={a:.1f}->{b:.1f} ({delta:+.1%})")
        parts.append(f"ttft_p50={pick(ref[key], 'ttft_p50'):.2f}->{pick(new[key], 'ttft_p50'):.2f}")
        print(f"phase={key[0]} c={key[1]} {' '.join(parts)} {'PASS' if row_ok else 'FAIL'}")
        bad += not row_ok
    print(f"result={'pass' if bad == 0 else 'fail'} tol={args.tol:.0%}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
