#!/usr/bin/env python3
"""Prefill one fixed token sequence K times and count large run-to-run swings of the teacher-forced logprob per
row bucket. A swing is a row whose K logprobs span more than --swing nats while the best run is above --likely
(a likely token that some run scores as very unlikely). Rounding noise and unpredictable filler tokens stay far
below that. Buckets of --bucket rows show where in the context the swings occur.

    python3 evidence/e3-tools/prefill_buckets.py --tokens 12288 --k 4 --out DIR/prefill-buckets.json
    python3 evidence/e3-tools/prefill_buckets.py --seq evidence/e3-tools/prefill-seq-1500.json --k 6 --out ...

Without --seq the sequence is the tier0 needle filler with a fixed seed and no salt (the same ids on every boot;
each request still gets its own cache_salt). The ids are saved in the output. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quality.tier0 import filler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_prefill_probe import model_id, post  # noqa: E402


def fixed_ids(model: str, tokens: int) -> list[int]:
    paras = filler(777, max(8, tokens // 60))
    text = "Archive glm53-e3-fixed. Field notes follow.\n\n" + "\n\n".join(paras)
    ids = post("/tokenize", {"model": model, "prompt": text})["tokens"]
    return ids[:tokens]


def prefill(model: str, ids: list[int]) -> list[float]:
    r = post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                 "prompt_logprobs": 0, "cache_salt": uuid.uuid4().hex})
    plp = r["choices"][0]["prompt_logprobs"]
    return [plp[i][str(ids[i])]["logprob"] for i in range(1, len(ids))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq")
    ap.add_argument("--tokens", type=int, default=12288)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bucket", type=int, default=1024)
    ap.add_argument("--swing", type=float, default=5.0)
    ap.add_argument("--likely", type=float, default=-3.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = model_id()
    if a.seq:
        s = json.loads(Path(a.seq).read_text())
        ids = s["p_ids"] + s["g_ids"]
    else:
        ids = fixed_ids(model, a.tokens)
    t0 = time.time()
    runs = [prefill(model, ids) for _ in range(a.k)]
    rows = []
    for i, col in enumerate(zip(*runs)):
        span = max(col) - min(col)
        if span > a.swing and max(col) > a.likely:
            rows.append({"pos": i + 1, "span": round(span, 3), "runs": [round(x, 3) for x in col]})
    buckets = {}
    for b in range(0, len(ids), a.bucket):
        spans = sorted(max(c) - min(c) for c in list(zip(*runs))[max(0, b - 1): b - 1 + a.bucket])
        n = sum(1 for r in rows if b <= r["pos"] < b + a.bucket)
        buckets[f"{b}-{min(b + a.bucket, len(ids))}"] = {"swings": n, "median_span": spans[len(spans) // 2] if spans else 0}
    res = {"tokens": len(ids), "k": a.k, "swing": a.swing, "likely": a.likely, "s": round(time.time() - t0, 1),
           "n_swings": len(rows), "buckets": buckets, "swings": rows, "ids": ids}
    Path(a.out).write_text(json.dumps(res))
    print(f"{len(ids)} tokens x {a.k} prefills ({res['s']} s): {len(rows)} swings (span > {a.swing}, best > {a.likely})")
    for k, v in buckets.items():
        print(f"  rows {k:>13}: swings {v['swings']:>3}  median span {v['median_span']:.4f}")
    for r in rows[:10]:
        print(f"  pos {r['pos']}: span {r['span']} runs {r['runs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
