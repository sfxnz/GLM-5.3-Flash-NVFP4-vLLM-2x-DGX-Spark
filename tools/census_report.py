#!/usr/bin/env python3
"""Expert-census report for GLM53_EXPERT_CENSUS files (CPU only, numpy).

  python3 tools/census_report.py DIR [--rank 0] [--gbps 230] [--json OUT]

DIR holds census-rank{R}.json and census-rank{R}-NNNN.npz written by the v13
census patch (docker/patch_v13_census.py). A verify block is one request's
rows in one step when it has draft tokens: row 0 is the anchor (the last
sampled token), rows 1..k are the drafts. Reported:

  prefix curve   distinct experts per layer over the first n rows of each
                 verify block (n = 1..8) vs independent routing
                 D(n) = E * (1 - (1 - K/E)^n), E = 288, K = 8
  duplicates     1 - distinct / (n*K), measured vs the D(n) baseline
  positions      share of row p's K experts also in the anchor row, new
                 experts row p adds over rows < p, per-position acceptance
  step bytes     routed-MoE bytes per rank per verify step: distinct experts
                 over all verify rows of the step (c=2: union of both
                 requests), summed over MoE layers, x 7.08 MB per expert per
                 layer per rank (14,155,800 B nvidia-pack expert / TP 2)
  SD-1           bytes saved if dead rows read no new experts. Oracle: a
                 block keeps its first nsampled rows (anchor + accepted drafts
                 produce every emitted token's logits). Fixed cut c: every
                 block keeps its first c rows; tokens lost = nsampled beyond c.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

N_EXPERTS = 288
TOP_K = 8
EXPERT_BYTES_PER_RANK = 7_077_900  # 14,155,800 B per expert per layer / TP 2
SEG = {c: i for i, c in enumerate(
    ("step", "req", "row0", "nrows", "ndraft", "nsampled", "ncomputed", "prefill"))}
MAX_N = 8


def independent_distinct(n: float, experts: int = N_EXPERTS, top_k: int = TOP_K) -> float:
    """Expected distinct experts per layer over n independently routed rows."""
    return experts * (1.0 - (1.0 - top_k / experts) ** n)


def distinct_per_layer(rows: np.ndarray) -> np.ndarray:
    """rows [n, L, K] expert ids -> [L] distinct experts per layer."""
    n, layers, k = rows.shape
    flat = np.sort(rows.transpose(1, 0, 2).reshape(layers, n * k), axis=1)
    return 1 + (np.diff(flat, axis=1) != 0).sum(axis=1)


def position_stats(rows: np.ndarray, experts: int = N_EXPERTS) -> tuple[np.ndarray, np.ndarray]:
    """rows [n, L, K] -> (overlap with anchor [n], new experts [n]), layer means."""
    n, layers, k = rows.shape
    li = np.arange(layers)[:, None]
    seen = np.zeros((layers, experts), bool)
    anchor = np.zeros((layers, experts), bool)
    anchor[li, rows[0]] = True
    overlap, new = np.empty(n), np.empty(n)
    for p in range(n):
        m = np.zeros((layers, experts), bool)
        m[li, rows[p]] = True
        overlap[p] = (m & anchor).sum(axis=1).mean() / k
        new[p] = (m & ~seen).sum(axis=1).mean()
        seen |= m
    return overlap, new


def load_census(directory: Path, rank: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Concatenate one rank's chunks. seg row0 becomes a global row index."""
    meta = json.loads((directory / f"census-rank{rank}.json").read_text())
    pat = re.compile(rf"census-rank{rank}-(\d+)\.npz$")
    chunks = sorted((int(m.group(1)), p) for p in directory.iterdir()
                    if (m := pat.search(p.name)))
    if not chunks:
        raise SystemExit(f"no census-rank{rank}-*.npz in {directory}")
    topks, segs, base = [], [], 0
    for _, path in chunks:
        with np.load(path, allow_pickle=False) as z:
            topk, seg = z["topk"], z["seg"].copy()
        # Each step's rows are contiguous and in step order within a chunk.
        step_rows: dict[int, int] = {}
        for s, end in zip(seg[:, SEG["step"]], seg[:, SEG["row0"]] + seg[:, SEG["nrows"]]):
            step_rows[int(s)] = max(step_rows.get(int(s), 0), int(end))
        offset, start = base, {}
        for s in sorted(step_rows):
            start[s] = offset
            offset += step_rows[s]
        if offset - base != len(topk):
            raise SystemExit(f"{path}: segments cover {offset - base} rows, topk has {len(topk)}")
        seg[:, SEG["row0"]] += [start[int(s)] for s in seg[:, SEG["step"]]]
        topks.append(topk)
        segs.append(seg)
        base = offset
    return meta, np.concatenate(topks), np.concatenate(segs)


def analyze(meta: dict, topk: np.ndarray, seg: np.ndarray, gbps: float = 230.0) -> dict:
    experts = int(meta.get("num_experts", N_EXPERTS))
    k = int(meta.get("top_k", TOP_K))
    layers = topk.shape[1]
    ids = topk.astype(np.int64)
    bytes_per_expert_slot = EXPERT_BYTES_PER_RANK

    def rows_of(s: np.ndarray, n: int | None = None) -> np.ndarray:
        r0, nr = int(s[SEG["row0"]]), int(s[SEG["nrows"]])
        return ids[r0:r0 + (nr if n is None else min(n, nr))]

    is_verify = (seg[:, SEG["ndraft"]] > 0) & (seg[:, SEG["prefill"]] == 0)
    steps = np.unique(seg[:, SEG["step"]])
    kinds = {"verify": 0, "prefill": 0, "mixed": 0, "decode": 0}
    for st in steps:
        sel = seg[:, SEG["step"]] == st
        v, p = is_verify[sel].any(), (seg[sel, SEG["prefill"]] > 0).any()
        kinds["mixed" if v and p else "verify" if v else "prefill" if p else "decode"] += 1

    blocks = seg[is_verify]
    max_n = min(MAX_N, int(blocks[:, SEG["nrows"]].max())) if len(blocks) else 0
    prefix = []
    for n in range(1, max_n + 1):
        full = blocks[blocks[:, SEG["nrows"]] >= n]
        if not len(full):
            continue
        d = float(np.mean([distinct_per_layer(rows_of(b, n)).mean() for b in full]))
        dn = independent_distinct(n, experts, k)
        prefix.append({
            "n": n, "blocks": len(full), "distinct": round(d, 3),
            "independent": round(dn, 3), "ratio": round(d / dn, 4),
            "dup": round(1 - d / (n * k), 4), "dup_independent": round(max(0.0, 1 - dn / (n * k)), 4),
        })

    npos = int(blocks[:, SEG["nrows"]].max()) if len(blocks) else 0
    ov_sum, new_sum, cnt = np.zeros(npos), np.zeros(npos), np.zeros(npos)
    for b in blocks:
        ov, nw = position_stats(rows_of(b), experts)
        ov_sum[:len(ov)] += ov
        new_sum[:len(nw)] += nw
        cnt[:len(ov)] += 1
    accepted = blocks[:, SEG["nsampled"]] - 1 if len(blocks) else np.zeros(0)
    positions = [{
        "pos": p, "blocks": int(cnt[p]),
        "overlap_anchor": round(float(ov_sum[p] / cnt[p]), 4),
        "new_experts": round(float(new_sum[p] / cnt[p]), 3),
        "accept_rate": None if p == 0 else round(float((accepted >= p).mean()), 4),
    } for p in range(npos) if cnt[p]]

    # Step-level bytes and SD-1, over steps whose every request is a verify block.
    per_step, cut_rows = [], range(1, npos + 1)
    for st in steps:
        sel = seg[:, SEG["step"]] == st
        if not is_verify[sel].all():
            continue
        bl = seg[sel]
        all_rows = np.concatenate([rows_of(b) for b in bl])
        live = np.concatenate([rows_of(b, max(1, int(b[SEG["nsampled"]]))) for b in bl])
        d_all = distinct_per_layer(all_rows)
        rec = {
            "rows": len(all_rows), "reqs": len(bl),
            "tokens": int(bl[:, SEG["nsampled"]].sum()),
            "experts": int(d_all.sum()),
            "experts_per_req_sum": int(sum(distinct_per_layer(rows_of(b)).sum() for b in bl)),
            "experts_oracle": int(distinct_per_layer(live).sum()),
            "experts_cut": [int(distinct_per_layer(np.concatenate(
                [rows_of(b, c) for b in bl])).sum()) for c in cut_rows],
            "tokens_cut": [int(np.minimum(bl[:, SEG["nsampled"]], c).sum()) for c in cut_rows],
        }
        per_step.append(rec)

    def gb(experts_sum: float) -> float:
        return experts_sum * bytes_per_expert_slot / 1e9

    def ms(experts_sum: float) -> float:
        return gb(experts_sum) / gbps * 1e3

    out = {
        "rank": meta.get("rank"), "layers": layers, "experts": experts, "top_k": k,
        "steps": int(len(steps)), "step_kinds": kinds, "verify_blocks": int(len(blocks)),
        "prefix": prefix, "positions": positions, "gbps": gbps,
    }
    if per_step:
        mean = {key: float(np.mean([r[key] for r in per_step]))
                for key in ("rows", "reqs", "tokens", "experts", "experts_per_req_sum", "experts_oracle")}
        ind = float(np.mean([independent_distinct(r["rows"], experts, k) for r in per_step])) * layers
        out["step"] = {
            "verify_steps": len(per_step),
            "rows": round(mean["rows"], 3), "reqs": round(mean["reqs"], 3),
            "tokens": round(mean["tokens"], 3),
            "distinct_per_layer": round(mean["experts"] / layers, 3),
            "distinct_per_layer_independent": round(ind / layers, 3),
            "moe_gb_per_rank": round(gb(mean["experts"]), 4),
            "moe_gb_per_rank_independent": round(gb(ind), 4),
            "moe_ms_at_gbps": round(ms(mean["experts"]), 3),
            "cross_request_shared_gb": round(gb(mean["experts_per_req_sum"] - mean["experts"]), 4),
        }
        oracle_saved = mean["experts"] - mean["experts_oracle"]
        out["sd1"] = {
            "oracle": {"saved_gb": round(gb(oracle_saved), 4), "saved_ms": round(ms(oracle_saved), 3),
                       "saved_frac_moe": round(oracle_saved / mean["experts"], 4)},
            "cut": [{
                "keep_rows": c,
                "saved_gb": round(gb(mean["experts"] - float(np.mean([r["experts_cut"][i] for r in per_step]))), 4),
                "saved_ms": round(ms(mean["experts"] - float(np.mean([r["experts_cut"][i] for r in per_step]))), 3),
                "tokens_kept_frac": round(float(np.sum([r["tokens_cut"][i] for r in per_step]))
                                          / max(1, sum(r["tokens"] for r in per_step)), 4),
            } for i, c in enumerate(cut_rows)],
        }
    return out


def ranks_agree(directory: Path) -> dict | None:
    """Routing is replicated across TP ranks; compare rank 0 and rank 1."""
    if not (directory / "census-rank1.json").is_file() or not (directory / "census-rank0.json").is_file():
        return None
    _, t0, s0 = load_census(directory, 0)
    _, t1, s1 = load_census(directory, 1)
    same_shape = t0.shape == t1.shape and s0.shape == s1.shape
    return {"same_shape": same_shape,
            "identical": bool(same_shape and np.array_equal(t0, t1) and np.array_equal(s0, s1))}


def render(rep: dict) -> str:
    lines = [f"rank {rep['rank']}: {rep['steps']} steps {rep['step_kinds']}, "
             f"{rep['verify_blocks']} verify blocks, {rep['layers']} MoE layers, "
             f"E={rep['experts']} K={rep['top_k']}",
             "", "prefix curve: distinct experts per layer over the first n rows of a verify block",
             f"{'n':>2} {'blocks':>7} {'distinct':>9} {'D(n)':>7} {'ratio':>6} {'dup':>6} {'dup_ind':>7}"]
    for r in rep["prefix"]:
        lines.append(f"{r['n']:>2} {r['blocks']:>7} {r['distinct']:>9.2f} {r['independent']:>7.2f} "
                     f"{r['ratio']:>6.3f} {r['dup']:>6.3f} {r['dup_independent']:>7.3f}")
    lines += ["", "positions (0 = anchor): overlap with anchor, new experts vs rows < p, acceptance",
              f"{'pos':>3} {'blocks':>7} {'overlap':>8} {'new':>6} {'accept':>7}"]
    for r in rep["positions"]:
        acc = "-" if r["accept_rate"] is None else f"{r['accept_rate']:.3f}"
        lines.append(f"{r['pos']:>3} {r['blocks']:>7} {r['overlap_anchor']:>8.3f} "
                     f"{r['new_experts']:>6.2f} {acc:>7}")
    if "step" in rep:
        s = rep["step"]
        lines += ["", f"verify steps: {s['verify_steps']}, rows/step {s['rows']}, reqs/step {s['reqs']}, "
                  f"tokens/step {s['tokens']}",
                  f"distinct/layer {s['distinct_per_layer']} (independent {s['distinct_per_layer_independent']})",
                  f"routed-MoE per rank per step: {s['moe_gb_per_rank']} GB "
                  f"(independent {s['moe_gb_per_rank_independent']} GB) = {s['moe_ms_at_gbps']} ms "
                  f"at {rep['gbps']} GB/s; cross-request sharing saves {s['cross_request_shared_gb']} GB",
                  "", f"SD-1 oracle (keep nsampled rows): saves {rep['sd1']['oracle']['saved_gb']} GB = "
                  f"{rep['sd1']['oracle']['saved_ms']} ms/step ({rep['sd1']['oracle']['saved_frac_moe']:.1%} "
                  "of routed-MoE bytes), no tokens lost",
                  f"{'keep':>4} {'saved_GB':>9} {'saved_ms':>9} {'tokens_kept':>11}"]
        for r in rep["sd1"]["cut"]:
            lines.append(f"{r['keep_rows']:>4} {r['saved_gb']:>9.4f} {r['saved_ms']:>9.3f} "
                         f"{r['tokens_kept_frac']:>11.3f}")
    if rep.get("ranks") is not None:
        lines += ["", f"rank0 vs rank1 routing identical: {rep['ranks']['identical']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--gbps", type=float, default=230.0,
                    help="achieved bandwidth for the ms columns (default 230)")
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    args = ap.parse_args(argv)
    meta, topk, seg = load_census(args.dir, args.rank)
    rep = analyze(meta, topk, seg, args.gbps)
    rep["ranks"] = ranks_agree(args.dir)
    print(render(rep))
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
