#!/usr/bin/env python3
"""Compare two arms of ruler-v2 bench.json files, with the serve boot as the unit.

    python3 kit/compare.py --a base/bench-1/bench.json base/bench-2/bench.json \
                           --b arm/bench-1/bench.json  arm/bench-2/bench.json

Each bench.json is one panel. Panels from the same boot are averaged first, and
each boot is one sample. bench_decode.py records the boot as boot_id (the API
server's process_start_time_seconds). Older files without it fall back to where
they sit: a bench-* directory is a panel of the boot directory above it
(evidence/<boot>/bench-N/bench.json), and any other directory is its own boot.

Keep rule (evidence/review-20260925/PLAN.md section 6), per cell, with >= 2 valid
boots in each arm: relative change of mean tok_s B vs A with a z-interval (delta
method on the ratio of boot means, 95%).
    KEEP          lower bound > 0 and point gain >= +3%
    REVERT        upper bound < +1%
    INCONCLUSIVE  otherwise
The z-interval follows the PLAN. At 2-3 boots per arm a Welch t-interval
(df ~2-4, t ~2.8-4.3) would be 1.4-2.2x wider, so this 95% interval is
anti-conservative and false KEEPs are more likely than 5%. Treat a KEEP whose
lower bound sits near 0 as needing another boot per arm.

With fewer than 2 boots in either arm the interval comes from the panels (at
least 2 per arm, else point estimate only). It is within-boot and cannot see
boot-to-boot variance, so the verdict is INCONCLUSIVE(single-boot) unless the
whole interval clears the cross-boot band (--boot-band, default BOOT_BAND):
    KEEP          lower bound > band and point gain >= +3%
    REVERT        upper bound < -band
BOOT_BAND is the largest same-code shift seen between two boots, rounded up.
E1a (v13, every switch off) vs E0 (v11): tok/s -1.6..-3.7% and step_ms
+1.6..+3.7% on A, B, J@c1 and H. E1b vs E0: step_ms +0.3..+1.8%
(evidence/e1a-v13-off/notes.txt, evidence/e1b-v13-warmcache-draft-bf582e4/notes.txt).

step_ms and acceptance_len get the same interval for reading, not for the verdict.
For acceptance-changing experiments it also prints a paired per-prompt bootstrap
of acceptance_len (c=1 cells, prompts matched by prompt_id). Stdlib only.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
import time

Z95 = 1.96
KEEP_MIN_GAIN = 0.03
REVERT_MAX_UPPER = 0.01
BOOT_BAND = 0.04
METRICS = ("tok_s", "step_ms", "acceptance_len")


def rel_change(a: list[float], b: list[float]) -> dict | None:
    """B/A - 1 with a 95% z-interval; lo/hi are None unless both arms have >= 2 values."""
    if not a or not b:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    ratio = mb / ma
    if len(a) < 2 or len(b) < 2:
        return {"rel": ratio - 1, "lo": None, "hi": None}
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    se = ratio * (vb / mb**2 + va / ma**2) ** 0.5
    return {"rel": ratio - 1, "lo": ratio - 1 - Z95 * se, "hi": ratio - 1 + Z95 * se}


def verdict(change: dict | None, single_boot: bool = False, band: float = BOOT_BAND) -> str:
    if change is None or change["lo"] is None:
        return "INCONCLUSIVE(single-boot)" if single_boot else "INCONCLUSIVE"
    if single_boot:
        if change["lo"] > band and change["rel"] >= KEEP_MIN_GAIN:
            return "KEEP"
        return "REVERT" if change["hi"] < -band else "INCONCLUSIVE(single-boot)"
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


def boot_of(rep: dict) -> str:
    """The boot a panel came from: boot_id, else its evidence directory (module doc)."""
    if rep.get("boot_id") is not None:
        return "serve started " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(rep["boot_id"]))
    panel = os.path.dirname(os.path.abspath(rep["_path"]))
    return os.path.relpath(os.path.dirname(panel) if os.path.basename(panel).startswith("bench-") else panel)


def boot_values(arm: list[dict], group: str) -> tuple[dict[str, list[float]], dict[str, list[float]], list[str]]:
    """Per-metric boot means (same-boot panels averaged) and panel means for one
    cell group, skipping invalid panels."""
    by_boot: dict[str, dict[str, list[float]]] = {}
    skipped = []
    for rep in arm:
        s = next((x for x in rep["summary"] if x["group"] == group), None)
        if s is None:
            continue
        if not s["valid"]:
            skipped.append(f"{rep['_path']} ({s['invalid_reason']})")
            continue
        panels = by_boot.setdefault(boot_of(rep), {m: [] for m in METRICS})
        for m in METRICS:
            if s.get(m):
                panels[m].append(s[m]["mean"])
    boots = {m: [statistics.fmean(p[m]) for p in by_boot.values() if p[m]] for m in METRICS}
    panels = {m: [v for p in by_boot.values() for v in p[m]] for m in METRICS}
    return boots, panels, skipped


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


def compare(arm_a: list[dict], arm_b: list[dict], band: float = BOOT_BAND) -> list[dict]:
    groups = [s["group"] for s in arm_a[0]["summary"]]
    rows = []
    for group in groups:
        boots_a, panels_a, skip_a = boot_values(arm_a, group)
        boots_b, panels_b, skip_b = boot_values(arm_b, group)
        if not boots_a["tok_s"] or not boots_b["tok_s"]:
            continue
        single = len(boots_a["tok_s"]) < 2 or len(boots_b["tok_s"]) < 2
        va, vb = (panels_a, panels_b) if single else (boots_a, boots_b)
        row = {"group": group, "n_a": len(boots_a["tok_s"]), "n_b": len(boots_b["tok_s"]),
               "panels_a": len(panels_a["tok_s"]), "panels_b": len(panels_b["tok_s"]),
               "interval": "within-boot" if single else "boot", "skipped": skip_a + skip_b}
        for m in METRICS:
            row[m] = {
                "a": statistics.fmean(boots_a[m]) if boots_a[m] else None,
                "b": statistics.fmean(boots_b[m]) if boots_b[m] else None,
                "change": rel_change(va[m], vb[m]),
            }
        row["verdict"] = verdict(row["tok_s"]["change"], single, band)
        row["acceptance_paired"] = paired_bootstrap(
            per_prompt_acceptance(arm_a, group), per_prompt_acceptance(arm_b, group)
        )
        rows.append(row)
    return rows


def pct(change: dict | None) -> str:
    if change is None:
        return "-"
    if change["lo"] is None:
        return f"{100 * change['rel']:+.1f}%"
    return f"{100 * change['rel']:+.1f}% [{100 * change['lo']:+.1f},{100 * change['hi']:+.1f}]"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", nargs="+", required=True, help="baseline arm: bench.json panels, one or more per boot")
    p.add_argument("--b", nargs="+", required=True, help="candidate arm: bench.json panels, one or more per boot")
    p.add_argument("--boot-band", type=float, default=BOOT_BAND,
                   help=f"cross-boot band a single-boot interval must clear for a verdict (default {BOOT_BAND})")
    p.add_argument("--json", default=None, help="also write the comparison as JSON")
    args = p.parse_args(argv)

    arm_a, arm_b = load(args.a), load(args.b)
    for rep in arm_a + arm_b:
        if rep.get("ruler_version") != "v2":
            print(f"warning: {rep['_path']} ruler_version={rep.get('ruler_version')!r}", file=sys.stderr)
    models = {r.get("model") for r in arm_a + arm_b}
    if len(models) > 1:
        print(f"note: arms served different model ids {sorted(map(str, models))}", file=sys.stderr)
    for name, arm in (("A", arm_a), ("B", arm_b)):
        boots = collections.Counter(boot_of(r) for r in arm)
        print(f"arm {name} boots: " + "; ".join(f"{b} ({n} panel(s))" for b, n in boots.items()))

    rows = compare(arm_a, arm_b, args.boot_band)
    print(f"{'cell':<8}{'boots':>6}{'tok/s A':>9}{'tok/s B':>9}  {'tok/s change [95% CI]':<26}"
          f"{'step_ms change':<26}{'acc_len change':<26}{'acc paired bootstrap':<26}verdict")
    for r in rows:
        ap = r["acceptance_paired"]
        print(f"{r['group']:<8}{r['n_a']:>3}/{r['n_b']:<2}{r['tok_s']['a']:>9.2f}{r['tok_s']['b']:>9.2f}  "
              f"{pct(r['tok_s']['change']):<26}{pct(r['step_ms']['change']):<26}"
              f"{pct(r['acceptance_len']['change']):<26}{pct(ap):<26}{r['verdict']}")
        for s in r["skipped"]:
            print(f"  skipped invalid panel: {s}")
    if any(r["interval"] == "within-boot" for r in rows):
        print(f"single-boot: an arm has < 2 boots, so the intervals are within-boot (same-boot panels) and a "
              f"verdict needs one to clear +-{100 * args.boot_band:.1f}% (--boot-band). Run 2 boots per arm, ABAB.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
