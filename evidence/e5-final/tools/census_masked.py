#!/usr/bin/env python3
"""census_masked.py DIR [--rank 0] [--classes CLASSES_JSON] [--json OUT]: adaptive-verify masked rows from a
GLM53_EXPERT_CENSUS directory (tools/census_report.py's loader). CPU, numpy.

With GLM53_ADAPTIVE_VERIFY=1 a masked verify row takes its anchor row's top-8 ids in every MoE layer
(docker/README-v13.md, "Census: it records the ids after the remap"). A draft row p >= 1 whose ids equal the
anchor's in all layers is counted as masked; the verify width m is the number of draft rows before the first
masked one. Reports, per request class (CLASSES_JSON maps census request uid -> class; uids count requests in
arrival order) and overall:
  - verify blocks, drafts scheduled, width m histogram, masked rows per block and share of draft rows;
  - whether masking is a suffix (no live row after a masked one) and accepted <= m (nsampled - 1 <= m);
  - distinct experts per layer over all rows of a block vs over its live rows only, against D(n_live);
  - per-step distinct experts per layer (union over the step's requests) and rows per step.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from census_report import SEG, distinct_per_layer, independent_distinct, load_census  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--classes", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    meta, topk, seg = load_census(a.dir, a.rank)
    ids = topk.astype(np.int64)
    cls_of = {}
    if a.classes:
        cls_of = {int(k): v for k, v in json.loads(a.classes.read_text()).items()}
    is_verify = (seg[:, SEG["ndraft"]] > 0) & (seg[:, SEG["prefill"]] == 0)
    stats = collections.defaultdict(lambda: {"blocks": 0, "drafts": 0, "masked": 0, "m_hist": collections.Counter(),
                                             "suffix_ok": 0, "accept_le_m": 0, "d_all": [], "d_live": [],
                                             "d_ind_live": [], "n_live": []})
    per_step = collections.defaultdict(list)
    for s in seg[is_verify]:
        r0, nr = int(s[SEG["row0"]]), int(s[SEG["nrows"]])
        rows = ids[r0:r0 + nr]
        same = np.all(rows[1:] == rows[0], axis=(1, 2))  # [nr-1] draft row equals anchor everywhere
        m = int(np.argmax(same)) if same.any() else nr - 1
        n_masked = int(same.sum())
        suffix = bool(same[m:].all()) if same.any() else True
        accepted = int(s[SEG["nsampled"]]) - 1
        live = rows[: 1 + m]
        d_all = float(distinct_per_layer(rows).mean())
        d_live = float(distinct_per_layer(live).mean())
        for key in ("all", cls_of.get(int(s[SEG["req"]]), "unmapped")):
            st = stats[key]
            st["blocks"] += 1
            st["drafts"] += nr - 1
            st["masked"] += n_masked
            st["m_hist"][m] += 1
            st["suffix_ok"] += suffix
            st["accept_le_m"] += accepted <= m
            st["d_all"].append(d_all)
            st["d_live"].append(d_live)
            st["d_ind_live"].append(independent_distinct(1 + m))
            st["n_live"].append(1 + m)
        per_step[int(s[SEG["step"]])].append((r0, nr, m, cls_of.get(int(s[SEG["req"]]), "unmapped")))
    out = {"rank": a.rank, "steps_recorded": int(len(np.unique(seg[:, SEG["step"]]))), "classes": {}}
    for key, st in stats.items():
        b = st["blocks"]
        out["classes"][key] = {
            "verify_blocks": b, "drafts_per_block": round(st["drafts"] / b, 3),
            "masked_rows_per_block": round(st["masked"] / b, 3),
            "masked_share_of_draft_rows": round(st["masked"] / max(1, st["drafts"]), 4),
            "width_m_hist": dict(sorted(st["m_hist"].items())),
            "mean_live_rows": round(float(np.mean(st["n_live"])), 3),
            "suffix_masking": f"{st['suffix_ok']}/{b}", "accepted_le_m": f"{st['accept_le_m']}/{b}",
            "distinct_per_layer_all_rows": round(float(np.mean(st["d_all"])), 3),
            "distinct_per_layer_live_rows": round(float(np.mean(st["d_live"])), 3),
            "D_of_live_rows_independent": round(float(np.mean(st["d_ind_live"])), 3),
        }
    # Per verify step (every request a verify block): union of experts over all rows vs live rows, by step shape.
    shapes = collections.defaultdict(lambda: {"steps": 0, "rows": [], "live": [], "d_all": [], "d_live": []})
    for step, blocks in per_step.items():
        sel = seg[seg[:, SEG["step"]] == step]
        if not is_verify[np.nonzero(seg[:, SEG["step"]] == step)[0]].all():
            continue
        kind = "+".join(sorted(b[3] for b in blocks)) if len(blocks) > 1 else blocks[0][3]
        all_rows = np.concatenate([ids[r0:r0 + nr] for r0, nr, _, _ in blocks])
        live_rows = np.concatenate([ids[r0:r0 + 1 + m] for r0, _, m, _ in blocks])
        sh = shapes[f"c={len(sel)} {kind}"]
        sh["steps"] += 1
        sh["rows"].append(len(all_rows))
        sh["live"].append(len(live_rows))
        sh["d_all"].append(float(distinct_per_layer(all_rows).mean()))
        sh["d_live"].append(float(distinct_per_layer(live_rows).mean()))
    out["steps"] = {k: {"steps": v["steps"], "rows": round(float(np.mean(v["rows"])), 2),
                        "live_rows": round(float(np.mean(v["live"])), 2),
                        "distinct_per_layer": round(float(np.mean(v["d_all"])), 2),
                        "D_rows_independent": round(float(np.mean([independent_distinct(n) for n in v["rows"]])), 2),
                        "D_live_independent": round(float(np.mean([independent_distinct(n) for n in v["live"]])), 2)}
                    for k, v in sorted(shapes.items())}
    print(f"rank {a.rank}: {out['steps_recorded']} steps recorded")
    print(f"{'class':<14}{'blocks':>7}{'drafts':>7}{'masked':>7}{'mask%':>7}{'live':>6}{'D_all':>7}{'D_live':>7}"
          f"{'D(n_live)':>10}  suffix  acc<=m  width m histogram")
    for key, c in sorted(out["classes"].items(), key=lambda kv: kv[0] != "all"):
        print(f"{key:<14}{c['verify_blocks']:>7}{c['drafts_per_block']:>7.2f}{c['masked_rows_per_block']:>7.2f}"
              f"{100 * c['masked_share_of_draft_rows']:>6.1f}%{c['mean_live_rows']:>6.2f}"
              f"{c['distinct_per_layer_all_rows']:>7.2f}{c['distinct_per_layer_live_rows']:>7.2f}"
              f"{c['D_of_live_rows_independent']:>10.2f}  {c['suffix_masking']:>7} {c['accepted_le_m']:>7}  {c['width_m_hist']}")
    print("\nper verify step (union over the step's requests; D = independent routing)")
    print(f"{'shape':<28}{'steps':>6}{'rows':>6}{'live':>6}{'distinct':>9}{'D(rows)':>8}{'D(live)':>8}")
    for k, v in out["steps"].items():
        print(f"{k:<28}{v['steps']:>6}{v['rows']:>6.2f}{v['live_rows']:>6.2f}{v['distinct_per_layer']:>9.2f}"
              f"{v['D_rows_independent']:>8.2f}{v['D_live_independent']:>8.2f}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
