#!/usr/bin/env python3
"""Prefill the same token ids K times (prompt_logprobs=0, new cache_salt each time) and measure the per-position
spread of the teacher-forced logprobs. Separates a prefill-side defect (large, randomly placed outliers) from
bf16 rounding noise, and the single-chunk regime (<= 2048 rows, no indexer top-k) from the chunked one.

    python3 evidence/e3-tools/prefill_repeat.py --make SEQ.json            # build a 1.5k prompt + 2000 greedy tokens
    python3 evidence/e3-tools/prefill_repeat.py --seq SEQ.json --k 6 --lengths 2040,0 --out DIR/prefill-repeat.json

--lengths: prefix lengths in tokens (0 = the whole sequence). Stdlib only; talks to the running serve.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_prefill_probe import build_prompt, generate, model_id, post  # noqa: E402

import uuid  # noqa: E402


def prefill(model: str, ids: list[int], start: int) -> list[float]:
    r = post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                 "prompt_logprobs": 0, "cache_salt": uuid.uuid4().hex})
    plp = r["choices"][0]["prompt_logprobs"]
    return [plp[i][str(ids[i])]["logprob"] for i in range(start, len(ids))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make")
    ap.add_argument("--seq")
    ap.add_argument("--ctx", type=int, default=1500)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--lengths", default="2040,0")
    ap.add_argument("--out")
    a = ap.parse_args()
    model = model_id()
    if a.make:
        p_ids, salt = build_prompt(model, a.ctx, 4242)
        g = {}
        generate(model, p_ids, a.n, salt, g)
        Path(a.make).write_text(json.dumps({"p_ids": p_ids, "g_ids": g["g_ids"], "decode_lp": g["lp"], "salt": salt}))
        print(f"sequence: prompt {len(p_ids)} + generated {len(g['g_ids'])} -> {a.make}")
        return 0
    seq = json.loads(Path(a.seq).read_text())
    ids = seq["p_ids"] + seq["g_ids"]
    start = 1
    res = {"seq": a.seq, "k": a.k, "total": len(ids), "prompt": len(seq["p_ids"]), "lengths": {}}
    for L in (int(x) for x in a.lengths.split(",")):
        L = L or len(ids)
        t0 = time.time()
        runs = [prefill(model, ids[:L], start) for _ in range(a.k)]
        spread = [max(col) - min(col) for col in zip(*runs)]
        big = [(start + i, round(s, 3), [round(r[i], 3) for r in runs]) for i, s in enumerate(spread) if s > 1.0]
        srt = sorted(spread)
        gen = spread[len(seq["p_ids"]) - start:]
        gsrt = sorted(gen) or [0.0]
        row = {"L": L, "positions": len(spread), "median_spread": srt[len(srt) // 2],
               "gen_positions": len(gen), "gen_median_spread": gsrt[len(gsrt) // 2],
               "gen_p99_spread": gsrt[int(0.99 * len(gsrt))], "gen_max_spread": gsrt[-1],
               "gen_n_spread_gt_1": sum(s > 1 for s in gen), "gen_n_spread_gt_5": sum(s > 5 for s in gen),
               "runs": runs,
               "p99_spread": srt[int(0.99 * len(srt))], "max_spread": srt[-1],
               "n_spread_gt_0.5": sum(s > 0.5 for s in spread), "n_spread_gt_1": len(big),
               "n_spread_gt_5": sum(s > 5 for s in spread), "n_lp_lt_-10": sum(sum(x < -10 for x in r) for r in runs),
               "outliers": big[:40], "s": round(time.time() - t0, 1)}
        res["lengths"][str(L)] = row
        print(f"L={L:>6} k={a.k}: positions {row['positions']} median spread {row['median_spread']:.4f} "
              f"p99 {row['p99_spread']:.3f} max {row['max_spread']:.2f} | >0.5: {row['n_spread_gt_0.5']} >1: "
              f"{row['n_spread_gt_1']} >5: {row['n_spread_gt_5']} | lp<-10 over all runs: {row['n_lp_lt_-10']} "
              f"({row['s']}s)", flush=True)
        print(f"    generated positions {row['gen_positions']}: median spread {row['gen_median_spread']:.4f} "
              f"p99 {row['gen_p99_spread']:.3f} max {row['gen_max_spread']:.2f} | >1: {row['gen_n_spread_gt_1']} "
              f">5: {row['gen_n_spread_gt_5']}", flush=True)
        gbig = [b for b in big if b[0] >= len(seq["p_ids"])]
        for pos, s, vals in sorted(gbig, key=lambda b: -b[1])[:6]:
            print(f"    pos {pos} (gen {pos - len(seq['p_ids'])}): spread {s} runs {vals}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
