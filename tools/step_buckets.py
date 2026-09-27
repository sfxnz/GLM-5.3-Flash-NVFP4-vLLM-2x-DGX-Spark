#!/usr/bin/env python3
"""Per-verify-step GPU kernel-time buckets from one rank's profile (CPU only).

  python3 tools/step_buckets.py TRACE.pt.trace.json.gz [--json OUT] [--only RE]
  python3 tools/step_buckets.py --nsys-kern-sum KERN_SUM.csv --steps N

Torch-profiler trace (tools/nsys_step.sh writes one per rank): streamed with
ijson under an RLIMIT_AS cap, never json.load, so it is safe next to a serve.
Steps are cut at the worker's execute_model annotations ("execute_context_
C(..)_generation_G(..)"); a kernel belongs to the step whose annotation
precedes the CUDA call that launched it (graph kernels carry the correlation
id of their cudaGraphLaunch). Kernels launched inside the annotation are the
target forward. Kernels launched after it, up to the next step, are
sample_tokens: target lm_head + logits until the first sampler kernel, the
sampler / rejection kernels, and everything else is counted as drafter
(DFlash2 propose plus postprocess). The last, possibly cut, step is dropped.

Buckets: routed_moe (Marlin MoE + routing glue), linear_marlin (dense Marlin:
layers 0-2 NVFP4 MLP, GLM53_FP8_W8A16), bf16_gemm (with a per-signature
table whose module is guessed from launches per step: 34 KDA, 11 MLA, 42
shared-expert, 3 dense layers), kda, mla_indexer, mhc, nccl, other (target),
lm_head_logits, sampler_rejection, drafter (post). Kernel-name regexes are
first guesses; the top "other" kernels are printed so they can be refined.
Buckets sum to busy time only when streams do not overlap; wall - busy is
GPU idle.

nsys fallback: `nsys stats -r cuda_gpu_kern_sum -f csv` totals, bucketed by
name only (no phases, so drafter kernels land in the name buckets or other),
divided by --steps.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import gzip
import json
import re
import resource
import statistics
import sys

TARGET_BUCKETS = [
    ("nccl", r"nccl|cross_device_reduce|[Aa]ll[_]?[Rr]educe"),
    ("routed_moe", r"marlin_moe|moe_wna16|moe_align|moe_sum|count_and_sort|grouped_topk|noaux|"
                   r"fused_topk|topkGating|_moe_|^moe"),
    ("linear_marlin", r"[Mm]arlin"),
    ("mhc", r"mhc|hc_pre|hc_post|hc_prenorm|hc_head|sinkhorn"),
    ("kda", r"kda|fused_recurrent|gated_delta|chunk_|l2norm|layer_norm_gated|conv1d|solve_tril|"
            r"recompute_w_u|merge_16x16|cumsum|_fused_post_conv"),
    ("mla_indexer", r"mla|MLA|flashinfer|BatchPrefill|BatchDecode|xqa|fmha|flash_attn|indexer|mqa|"
                    r"kpool|hadamard|fwht|deep_gemm|paged|[Tt]op[Kk]|radix|sparse|concat_and_cache"),
    ("bf16_gemm", r"gemm|gemv|Gemm|Gemv|cutlass|nvjet|cublas|xmma|splitK|Kernel2"),
]
SAMPLER = r"sampl|reject|gumbel|argmax|argMax|softmax|topk_topp|penalt|logprob|bitmask|exponential"
_TARGET_RE = [(b, re.compile(p)) for b, p in TARGET_BUCKETS]
_SAMPLER_RE = re.compile(SAMPLER)
ORDER = [b for b, _ in TARGET_BUCKETS] + ["other", "lm_head_logits", "sampler_rejection", "drafter"]
MODULE_COUNTS = {34: "kda", 11: "mla", 42: "shared_expert", 45: "kda+mla", 3: "dense_0-2"}


def classify(name: str) -> str:
    for bucket, rx in _TARGET_RE:
        if rx.search(name):
            return bucket
    return "other"


def module_guess(per_step: float) -> str:
    n = round(per_step)
    if abs(per_step - n) > 0.01 or n == 0:
        return "?"
    for m in (1, 2, 3, 4):
        for base, label in MODULE_COUNTS.items():
            if n == base * m:
                return label if m == 1 else f"{label} x{m}"
    return "lm_head" if n in (1, 2) else "?"


def extract(path: str) -> dict:
    """Stream the trace: device events, CUDA launch calls, execute_ annotations."""
    import ijson

    dev, launch, ann = [], {}, []
    with gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb") as fh:
        for e in ijson.items(fh, "traceEvents.item", use_float=True):
            if e.get("ph") != "X":
                continue
            cat, a = e.get("cat"), e.get("args") or {}
            if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
                dev.append((float(e["ts"]), float(e.get("dur", 0.0)), e["name"][:200],
                            a.get("correlation"), tuple(a.get("grid") or ())))
            elif cat in ("cuda_runtime", "cuda_driver") and a.get("correlation") is not None:
                launch[a["correlation"]] = float(e["ts"])
            elif cat == "user_annotation" and e.get("name", "").startswith("execute_"):
                ann.append((float(e["ts"]), float(e.get("dur", 0.0)), e["name"]))
    return {"dev": dev, "launch": launch, "ann": sorted(ann)}


def analyze(ex: dict, only: str | None = r"_context_0\(") -> dict:
    ann = ex["ann"]
    starts = [a[0] for a in ann]
    steps = [{"name": a[2], "ends": a[0] + a[1], "k": [], "first_sampler": None} for a in ann]
    for ts, dur, name, corr, grid in sorted(ex["dev"], key=lambda d: d[0]):
        lts = ex["launch"].get(corr, ts)
        i = bisect.bisect_right(starts, lts) - 1
        if i < 0:
            continue
        st = steps[i]
        if lts <= st["ends"]:
            bucket = classify(name)
        elif _SAMPLER_RE.search(name):
            bucket = "sampler_rejection"
            st["first_sampler"] = st["first_sampler"] or lts
        elif st["first_sampler"] is None:
            bucket = "lm_head_logits"
        else:
            bucket = "drafter"
        st["k"].append((ts, dur, name, grid, bucket))
    rx = re.compile(only) if only else None
    # The last step may be cut by the stop; it only marks where the one before ends.
    kept = [i for i, s in enumerate(steps[:-1]) if s["k"] and (rx is None or rx.search(s["name"]))]
    if not kept:
        raise SystemExit("no complete steps matched; check --only and the trace window")

    per_bucket = collections.defaultdict(list)
    per_bucket_n = collections.defaultdict(list)
    walls, busys, gemm, other = [], [], collections.defaultdict(lambda: [0.0, 0]), collections.Counter()
    for i in kept:
        ks = steps[i]["k"]
        t0 = ks[0][0]
        nxt = steps[i + 1]["k"]
        t1 = nxt[0][0] if nxt else max(k[0] + k[1] for k in ks)
        busy, cur_s, cur_e = 0.0, None, None
        for ts, dur, *_ in ks:
            if cur_e is None or ts > cur_e:
                busy += 0 if cur_e is None else cur_e - cur_s
                cur_s, cur_e = ts, ts + dur
            else:
                cur_e = max(cur_e, ts + dur)
        busy += cur_e - cur_s
        walls.append(t1 - t0)
        busys.append(busy)
        sums, counts = collections.Counter(), collections.Counter()
        for ts, dur, name, grid, bucket in ks:
            sums[bucket] += dur
            counts[bucket] += 1
            if bucket == "bf16_gemm":
                g = gemm[(name[:90], grid)]
                g[0] += dur
                g[1] += 1
            elif bucket == "other":
                other[name[:90]] += dur
        for b in ORDER:
            per_bucket[b].append(sums[b])
            per_bucket_n[b].append(counts[b])

    n = len(kept)
    wall = statistics.mean(walls) / 1e3
    rows = [{"bucket": b, "ms": round(statistics.mean(per_bucket[b]) / 1e3, 3),
             "pct_wall": round(100 * statistics.mean(per_bucket[b]) / 1e3 / wall, 1),
             "launches": round(statistics.mean(per_bucket_n[b]), 1)} for b in ORDER]
    return {
        "steps": n, "step_names": dict(collections.Counter(steps[i]["name"] for i in kept)),
        "wall_ms": round(wall, 3), "wall_ms_stdev": round(statistics.pstdev(walls) / 1e3, 3),
        "busy_ms": round(statistics.mean(busys) / 1e3, 3),
        "idle_ms": round((statistics.mean(walls) - statistics.mean(busys)) / 1e3, 3),
        "bucket_sum_ms": round(sum(r["ms"] for r in rows), 3),
        "buckets": rows,
        "bf16_gemm_signatures": [
            {"kernel": k[0], "grid": list(k[1]), "per_step": round(v[1] / n, 2),
             "ms": round(v[0] / n / 1e3, 3), "module_guess": module_guess(v[1] / n)}
            for k, v in sorted(gemm.items(), key=lambda kv: -kv[1][0])[:20]],
        "top_other": [{"kernel": k, "ms": round(v / n / 1e3, 3)} for k, v in other.most_common(15)],
    }


def nsys_kern_sum(path: str, steps: int) -> dict:
    sums = collections.Counter()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Name", "")
            b = "sampler_rejection" if _SAMPLER_RE.search(name) else classify(name)
            sums[b] += float(row["Total Time (ns)"]) / 1e6 / steps
    total = sum(sums.values())
    return {"steps": steps, "bucket_sum_ms": round(total, 3),
            "buckets": [{"bucket": b, "ms": round(sums[b], 3), "pct_kernels": round(100 * sums[b] / total, 1)}
                        for b in ORDER if b in sums]}


def render(rep: dict) -> str:
    out = []
    if "wall_ms" in rep:
        out += [f"{rep['steps']} steps {rep['step_names']}",
                f"wall {rep['wall_ms']} ms (sd {rep['wall_ms_stdev']}), busy {rep['busy_ms']}, "
                f"idle {rep['idle_ms']}, bucket sum {rep['bucket_sum_ms']}", ""]
    out.append(f"{'bucket':<18} {'ms/step':>8} {'%':>6} {'launches':>9}")
    for r in rep["buckets"]:
        pct = r.get("pct_wall", r.get("pct_kernels"))
        out.append(f"{r['bucket']:<18} {r['ms']:>8.3f} {pct:>6.1f} {r.get('launches', ''):>9}")
    if rep.get("bf16_gemm_signatures"):
        out += ["", "bf16_gemm by signature (module guessed from launches/step)",
                f"{'ms':>7} {'n/step':>6} {'module':<16} kernel [grid]"]
        for g in rep["bf16_gemm_signatures"]:
            out.append(f"{g['ms']:>7.3f} {g['per_step']:>6} {g['module_guess']:<16} {g['kernel'][:70]} {g['grid']}")
    if rep.get("top_other"):
        out += ["", "top 'other' kernels (refine TARGET_BUCKETS with these)"]
        out += [f"{o['ms']:>7.3f} {o['kernel']}" for o in rep["top_other"]]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace", nargs="?", help="*.pt.trace.json.gz from the torch profiler")
    ap.add_argument("--only", default=r"_context_0\(",
                    help="regex on the step annotation; default keeps pure decode/verify steps; '' keeps all")
    ap.add_argument("--nsys-kern-sum", help="CSV from nsys stats -r cuda_gpu_kern_sum -f csv")
    ap.add_argument("--steps", type=int, help="steps in the nsys capture window")
    ap.add_argument("--json", help="also write the report as JSON")
    ap.add_argument("--rlimit-gib", type=int, default=3, help="RLIMIT_AS cap (default 3)")
    args = ap.parse_args(argv)
    cap = args.rlimit_gib << 30
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    if args.nsys_kern_sum:
        if not args.steps:
            ap.error("--nsys-kern-sum needs --steps")
        rep = nsys_kern_sum(args.nsys_kern_sum, args.steps)
    elif args.trace:
        rep = analyze(extract(args.trace), args.only or None)
    else:
        ap.error("give a trace or --nsys-kern-sum")
    print(render(rep))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
