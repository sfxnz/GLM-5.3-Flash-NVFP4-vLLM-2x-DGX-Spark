#!/usr/bin/env python3
"""Compare two arms of ruler-v2 bench.json files (one file per boot).

    python3 kit/compare.py --a base-boot1/bench.json base-boot2/bench.json \
                           --b arm-boot1/bench.json  arm-boot2/bench.json

Keep rule (evidence/review-20260925/PLAN.md section 6), per cell, boot as unit:
relative change of mean tok_s B vs A with a z-interval (delta method on the
ratio of boot means, 95%).
    KEEP          lower bound > 0 and point gain >= +3%
    REVERT        upper bound < +1%
    INCONCLUSIVE  otherwise, or fewer than 2 valid boots in either arm
The z-interval follows the PLAN. At 2-3 boots per arm a Welch t-interval
(df ~2-4, t ~2.8-4.3) would be 1.4-2.2x wider, so this 95% interval is
anti-conservative and false KEEPs are more likely than 5%. Treat a KEEP whose
lower bound sits near 0 as needing another boot per arm.
step_ms and acceptance_len get the same interval for reading, not for the verdict.
For acceptance-changing experiments it also prints a paired per-prompt bootstrap
of acceptance_len (c=1 cells, prompts matched by prompt_id). Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys

Z95 = 1.96
KEEP_MIN_GAIN = 0.03
REVERT_MAX_UPPER = 0.01
METRICS = ("tok_s", "step_ms", "acceptance_len")


def rel_change(a: list[float], b: list[float]) -> dict | None:
    """B/A - 1 with a 95% z-interval; None unless both arms have >= 2 boots."""
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    ratio = mb / ma
    se = ratio * (vb / mb**2 + va / ma**2) ** 0.5
    return {"rel": ratio - 1, "lo": ratio - 1 - Z95 * se, "hi": ratio - 1 + Z95 * se}


def verdict(change: dict | None) -> str:
    if change is None:
        return "INCONCLUSIVE"
    if change["lo"] > 0 and change["rel"] >= KEEP_MIN_GAIN:
        return "KEEP"
    if change["hi"] < REVERT_MAX_UPPER:
        return "REVERT"
    return "INCONCLUSIVE"


def paired_bootstrap(a: dict[str, float], b: dict[str, float], reps: int = 10000, seed: int = 0) -> dict | None:
    """Relative change of mean acceptance_len over matched prompts, percentile CI."""
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return None

    def stat(sample: list[str]) -> float:
        return sum(b[i] for i in sample) / sum(a[i] for i in sample) - 1

    rng = random.Random(seed)
    boots = sorted(stat([rng.choice(ids) for _ in ids]) for _ in range(reps))
    return {"n_prompts": len(ids), "rel": stat(ids),
            "lo": boots[int(0.025 * reps)], "hi": boots[int(0.975 * reps) - 1]}


def load(paths: list[str]) -> list[dict]:
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            rep = json.load(fh)
        rep["_path"] = path
        out.append(rep)
    return out


def boot_values(arm: list[dict], group: str) -> tuple[dict[str, list[float]], list[str]]:
    """Per-metric list of boot means for one cell group, skipping invalid boots."""
    vals: dict[str, list[float]] = {m: [] for m in METRICS}
    skipped = []
    for rep in arm:
        s = next((x for x in rep["summary"] if x["group"] == group), None)
        if s is None:
            continue
        if not s["valid"]:
            skipped.append(f"{rep['_path']} ({s['invalid_reason']})")
            continue
        for m in METRICS:
            if s.get(m):
                vals[m].append(s[m]["mean"])
    return vals, skipped


def per_prompt_acceptance(arm: list[dict], group: str) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for rep in arm:
        s = next((x for x in rep["summary"] if x["group"] == group), None)
        if s is None or not s["valid"]:
            continue
        for w in rep["waves"]:
            if w["group"] == group and w["c"] == 1 and w["ok"] and w.get("acceptance_len"):
                acc.setdefault(w["requests"][0]["prompt_id"], []).append(w["acceptance_len"])
    return {k: statistics.fmean(v) for k, v in acc.items()}


def compare(arm_a: list[dict], arm_b: list[dict]) -> list[dict]:
    groups = [s["group"] for s in arm_a[0]["summary"]]
    rows = []
    for group in groups:
        va, skip_a = boot_values(arm_a, group)
        vb, skip_b = boot_values(arm_b, group)
        if not va["tok_s"] or not vb["tok_s"]:
            continue
        row = {"group": group, "n_a": len(va["tok_s"]), "n_b": len(vb["tok_s"]),
               "skipped": skip_a + skip_b}
        for m in METRICS:
            row[m] = {
                "a": statistics.fmean(va[m]) if va[m] else None,
                "b": statistics.fmean(vb[m]) if vb[m] else None,
                "change": rel_change(va[m], vb[m]),
            }
        row["verdict"] = verdict(row["tok_s"]["change"])
        row["acceptance_paired"] = paired_bootstrap(
            per_prompt_acceptance(arm_a, group), per_prompt_acceptance(arm_b, group)
        )
        rows.append(row)
    return rows


def pct(change: dict | None) -> str:
    if change is None:
        return "-"
    return f"{100 * change['rel']:+.1f}% [{100 * change['lo']:+.1f},{100 * change['hi']:+.1f}]"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", nargs="+", required=True, help="baseline arm: one bench.json per boot")
    p.add_argument("--b", nargs="+", required=True, help="candidate arm: one bench.json per boot")
    p.add_argument("--json", default=None, help="also write the comparison as JSON")
    args = p.parse_args(argv)

    arm_a, arm_b = load(args.a), load(args.b)
    for rep in arm_a + arm_b:
        if rep.get("ruler_version") != "v2":
            print(f"warning: {rep['_path']} ruler_version={rep.get('ruler_version')!r}", file=sys.stderr)
    models = {r.get("model") for r in arm_a + arm_b}
    if len(models) > 1:
        print(f"note: arms served different model ids {sorted(map(str, models))}", file=sys.stderr)

    rows = compare(arm_a, arm_b)
    print(f"{'cell':<8}{'boots':>6}{'tok/s A':>9}{'tok/s B':>9}  {'tok/s change [95% CI]':<26}"
          f"{'step_ms change':<26}{'acc_len change':<26}{'acc paired bootstrap':<26}verdict")
    for r in rows:
        ap = r["acceptance_paired"]
        print(f"{r['group']:<8}{r['n_a']:>3}/{r['n_b']:<2}{r['tok_s']['a']:>9.2f}{r['tok_s']['b']:>9.2f}  "
              f"{pct(r['tok_s']['change']):<26}{pct(r['step_ms']['change']):<26}"
              f"{pct(r['acceptance_len']['change']):<26}{pct(ap):<26}{r['verdict']}")
        for s in r["skipped"]:
            print(f"  skipped invalid boot: {s}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
