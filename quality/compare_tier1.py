#!/usr/bin/env python3
"""Compare two Tier-1 runs item by item (PLAN section 6 rule).

  python3 quality/compare_tier1.py A.jsonl B.jsonl [--json out.json]

A is the baseline, B the candidate. Items are paired on (task, id); the last
row per item wins (resumed runs). Pairs where either side has an error are
dropped and counted.

  pooled   mean of (B - A) over all paired items, in points, with its paired
           standard error. PASS needs the point estimate >= -2.0.
  groups   IFEval, GSM8K, MMLU-Pro, BFCL, Vision. A group FAILS only if the
           exact two-sided McNemar p < 0.01 AND its accuracy drops >= 5 points.
  invalid  more than 2% of pairs dropped for errors: no verdict, exit 2.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

POOLED_FLOOR = -2.0
GROUP_P = 0.01
GROUP_DROP = 5.0
MAX_ERROR_FRAC = 0.02


def load(path: Path) -> dict:
    last = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "task" in r:
                last[(r["task"], str(r["id"]))] = r
    return last


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: binomial(b + c, 0.5) on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_stats(pairs: list[tuple[int, int]]) -> dict:
    n = len(pairs)
    d = [bb - aa for aa, bb in pairs]
    mean = sum(d) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n else 0.0
    b = sum(1 for aa, bb in pairs if aa and not bb)
    c = sum(1 for aa, bb in pairs if bb and not aa)
    return {"n": n, "acc_a": round(100 * sum(a for a, _ in pairs) / max(1, n), 2),
            "acc_b": round(100 * sum(b_ for _, b_ in pairs) / max(1, n), 2),
            "diff": round(100 * mean, 3), "se": round(100 * se, 3),
            "ci95": [round(100 * (mean - 1.96 * se), 3), round(100 * (mean + 1.96 * se), 3)],
            "a_only": b, "b_only": c, "p_mcnemar": mcnemar_p(b, c)}


def compare(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    errors = [k for k in keys if "error" in a[k] or "error" in b[k]]
    ok = [k for k in keys if k not in set(errors)]
    pairs = {k: (int(a[k]["correct"]), int(b[k]["correct"])) for k in ok}
    pooled = paired_stats(list(pairs.values()))
    pooled["pass"] = pooled["diff"] >= POOLED_FLOOR
    groups, tasks = {}, {}
    for k in ok:
        groups.setdefault(a[k]["group"], []).append(pairs[k])
        tasks.setdefault(k[0], []).append(pairs[k])
    group_rows = {}
    for g, ps in sorted(groups.items()):
        st = paired_stats(ps)
        drop = st["acc_a"] - st["acc_b"]
        st["fail"] = st["p_mcnemar"] < GROUP_P and drop >= GROUP_DROP
        group_rows[g] = st
    missing = {"only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a))}
    invalid = len(errors) > MAX_ERROR_FRAC * max(1, len(keys))
    passed = pooled["pass"] and not any(s["fail"] for s in group_rows.values())
    verdict = "INVALID" if invalid else ("PASS" if passed else "FAIL")
    return {"verdict": verdict, "paired": len(ok), "error_pairs": len(errors), "unpaired": missing,
            "pooled": pooled, "groups": group_rows,
            "tasks": {t: paired_stats(ps) for t, ps in sorted(tasks.items())}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a", type=Path, help="baseline per-item JSONL")
    ap.add_argument("b", type=Path, help="candidate per-item JSONL")
    ap.add_argument("--json", type=Path, help="write the full comparison here")
    args = ap.parse_args(argv)
    res = compare(load(args.a), load(args.b))
    p = res["pooled"]
    print(f"pooled n={p['n']} A={p['acc_a']} B={p['acc_b']} diff={p['diff']:+.2f} pts "
          f"se={p['se']:.2f} ci95={p['ci95']} floor={POOLED_FLOOR} -> {'PASS' if p['pass'] else 'FAIL'}")
    for g, s in res["groups"].items():
        print(f"group {g:9s} n={s['n']:4d} A={s['acc_a']:6.2f} B={s['acc_b']:6.2f} diff={s['diff']:+6.2f} "
              f"A-only={s['a_only']} B-only={s['b_only']} p={s['p_mcnemar']:.4f} -> {'FAIL' if s['fail'] else 'ok'}")
    for t, s in res["tasks"].items():
        print(f"task  {t:9s} n={s['n']:4d} A={s['acc_a']:6.2f} B={s['acc_b']:6.2f} diff={s['diff']:+6.2f}")
    print(f"errors dropped={res['error_pairs']} unpaired={res['unpaired']}")
    print(f"VERDICT {res['verdict']}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1) + "\n")
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[res["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
