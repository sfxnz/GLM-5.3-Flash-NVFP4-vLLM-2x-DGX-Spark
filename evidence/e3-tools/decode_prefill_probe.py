#!/usr/bin/env python3
"""Decode against prefill on decode-built kpool pools (README-v13, GLM53_KPOOL_TAIL_FIX, GPU validation 4).

For each (context, concurrency) scenario:
  1. Build a unique-salt filler prompt of ~CTX tokens (quality/tier0.py filler), chat template, thinking off.
  2. Generate N greedy tokens through /v1/completions from the prompt ids (min_tokens=N, logprobs=0,
     return_token_ids). The decode path builds the pools past the prompt through the speculative verify steps.
     At c=2 two streams with different fillers run at once.
  3. Teacher-force prompt+generated ids twice through /v1/completions (prompt_logprobs=0, max_tokens=1),
     each with a new cache_salt. Prefill compresses pools straight from the batch and never reads the ring.
  4. Per generated token i >= SKIP: d_i = decode logprob - prefill-1 logprob (decode vs prefill), and
     a_i = prefill-2 - prefill-1 (the prefill-vs-prefill A/A at the same length). Report mean, p99 and max |.|,
     top-1 flips (prefill rank of the greedy token != 1) and the same per position bucket.

    python3 evidence/e3-tools/decode_prefill_probe.py --ctx 1500,32768 --c 1,2 --n 2000 --out DIR/probe.json

Stdlib only. Talks to the running serve; never starts or stops it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quality.tier0 import filler  # noqa: E402

URL = "http://127.0.0.1:8000"
ASK = ("\n\nRetell all of the field notes above as one long, detailed chronicle in your own words, "
       "going through them in order. Do not stop early; write at least 2500 words.")
BUCKETS = ((64, 500), (500, 1000), (1000, 1500), (1500, 10**9))


def post(path: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def model_id() -> str:
    with urllib.request.urlopen(URL + "/v1/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def build_prompt(model: str, target: int, seed: int) -> tuple[list[int], str]:
    salt = uuid.uuid4().hex
    paras = filler(seed, max(8, target // 60))

    def ids_for(n: int) -> list[int]:
        text = f"Archive {salt}. Field notes follow.\n\n" + "\n\n".join(paras[:n]) + ASK
        return post("/tokenize", {"model": model, "messages": [{"role": "user", "content": text}],
                                  "add_generation_prompt": True,
                                  "chat_template_kwargs": {"enable_thinking": False}})["tokens"]

    n = len(paras)
    ids = ids_for(n)
    while len(ids) < target:  # grow
        paras += filler(seed + len(paras), len(paras) // 2 + 8)
        n = len(paras)
        ids = ids_for(n)
    per = len(ids) / n
    n = max(4, int(target / per))
    ids = ids_for(n)
    for _ in range(4):  # adjust to within 2%
        if abs(len(ids) - target) <= 0.02 * target:
            break
        n = max(4, int(n * target / len(ids)))
        ids = ids_for(n)
    return ids, salt


def generate(model: str, p_ids: list[int], n: int, salt: str, out: dict) -> None:
    t0 = time.time()
    try:
        g = post("/v1/completions", {"model": model, "prompt": p_ids, "max_tokens": n, "min_tokens": n,
                                     "temperature": 0, "logprobs": 0, "return_token_ids": True,
                                     "skip_special_tokens": False, "cache_salt": salt})
        ch = g["choices"][0]
        out.update(g_ids=ch["token_ids"], lp=ch["logprobs"]["token_logprobs"], finish=ch["finish_reason"],
                   usage=g.get("usage"), gen_s=time.time() - t0)
    except Exception as e:  # noqa: BLE001
        out.update(error=repr(e), gen_s=time.time() - t0)


def teacher_force(model: str, ids: list[int], n_prompt: int, g_ids: list[int]) -> dict:
    salt = uuid.uuid4().hex
    t0 = time.time()
    r = post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                 "prompt_logprobs": 0, "cache_salt": salt})
    plp = r["choices"][0]["prompt_logprobs"]
    lp, rank = [], []
    for i, tid in enumerate(g_ids):
        e = plp[n_prompt + i][str(tid)]
        lp.append(e["logprob"])
        rank.append(e.get("rank"))
    usage = r.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {"lp": lp, "rank": rank, "salt": salt, "tf_s": time.time() - t0,
            "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": details.get("cached_tokens")}


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = sorted(abs(x) for x in xs)
    return {"n": len(xs), "mean_abs": statistics.fmean(a), "p99_abs": a[min(len(a) - 1, int(0.99 * len(a)))],
            "max_abs": a[-1], "mean": statistics.fmean(xs), "frac_gt_0.05": sum(x > 0.05 for x in a) / len(a),
            "frac_gt_0.5": sum(x > 0.5 for x in a) / len(a)}


def analyse(dec: list[float], p1: dict, p2: dict, skip: int) -> dict:
    d = [x - y for x, y in zip(dec, p1["lp"])]
    a = [x - y for x, y in zip(p2["lp"], p1["lp"])]
    res = {"decode_vs_prefill": stats(d[skip:]), "prefill_aa": stats(a[skip:]),
           "flips_prefill1": sum(1 for r in p1["rank"][skip:] if r != 1),
           "flips_prefill2": sum(1 for r in p2["rank"][skip:] if r != 1), "buckets": {}}
    for lo, hi in BUCKETS:
        if lo < len(d):
            res["buckets"][f"{lo}-{min(hi, len(d))}"] = {"d": stats(d[lo:hi]), "aa": stats(a[lo:hi])}
    return res


def fmt(s: dict) -> str:
    if not s.get("n"):
        return "n=0"
    return (f"mean|.|={s['mean_abs']:.4f} p99={s['p99_abs']:.4f} max={s['max_abs']:.3f} "
            f"mean={s['mean']:+.4f} >0.05={100 * s['frac_gt_0.05']:.1f}% >0.5={100 * s['frac_gt_0.5']:.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctx", default="1500,32768,131072")
    ap.add_argument("--c", default="1,2")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--skip", type=int, default=64)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = model_id()
    results = []
    out_path = Path(a.out)
    for ctx in (int(x) for x in a.ctx.split(",")):
        for c in (int(x) for x in a.c.split(",")):
            t0 = time.time()
            prompts = [build_prompt(model, ctx, a.seed + 1000 * ctx + 100 * c + i) for i in range(c)]
            gens = [{} for _ in range(c)]
            th = [threading.Thread(target=generate, args=(model, p, a.n, s, g)) for (p, s), g in zip(prompts, gens)]
            for t in th:
                t.start()
            for t in th:
                t.join()
            for i, ((p_ids, salt), g) in enumerate(zip(prompts, gens)):
                row = {"ctx": ctx, "c": c, "stream": i, "prompt_tokens": len(p_ids), "salt": salt}
                if "error" in g:
                    row["error"] = g["error"]
                    print(f"ctx={ctx} c={c} stream={i}: generate error {g['error']}", flush=True)
                    results.append(row)
                    continue
                g_ids = g["g_ids"]
                p1 = teacher_force(model, p_ids + g_ids, len(p_ids), g_ids)
                p2 = teacher_force(model, p_ids + g_ids, len(p_ids), g_ids)
                an = analyse(g["lp"], p1, p2, a.skip)
                row.update(gen_tokens=len(g_ids), finish=g["finish"], gen_s=round(g["gen_s"], 1),
                           tf_s=[round(p1["tf_s"], 1), round(p2["tf_s"], 1)],
                           cached_tokens=[p1["cached_tokens"], p2["cached_tokens"]], **an,
                           raw={"decode": g["lp"], "prefill1": p1["lp"], "prefill2": p2["lp"],
                                "rank1": p1["rank"], "g_ids": g_ids})
                results.append(row)
                print(f"ctx={ctx} c={c} stream={i} prompt={len(p_ids)} gen={len(g_ids)} ({g['finish']}, "
                      f"{g['gen_s']:.0f}s) cached={row['cached_tokens']} flips p1/p2={an['flips_prefill1']}/"
                      f"{an['flips_prefill2']}\n   decode-prefill: {fmt(an['decode_vs_prefill'])}\n"
                      f"   prefill A/A:    {fmt(an['prefill_aa'])}", flush=True)
                for k, v in an["buckets"].items():
                    print(f"     pos {k:>10}: d {fmt(v['d'])} | aa mean|.|={v['aa'].get('mean_abs', 0):.4f}"
                          f" p99={v['aa'].get('p99_abs', 0):.4f}", flush=True)
            print(f"   scenario ctx={ctx} c={c} took {time.time() - t0:.0f}s", flush=True)
            out_path.write_text(json.dumps({"model": model, "n": a.n, "skip": a.skip, "results": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
