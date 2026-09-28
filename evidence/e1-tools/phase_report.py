#!/usr/bin/env python3
"""Summarise one kit/uma_watch.py evidence dir: memory per phase, phase spans,
activity windows and the compiler census.

    python3 evidence/e1-tools/phase_report.py evidence/e1a-v13-off > evidence/e1a-v13-off/memory-by-phase.txt

Reads uma.tsv, compilers.tsv and (optional) activity.tsv (label, start, end in
UTC ISO). Stdlib only.
"""
import collections
import csv
import datetime as dt
import os
import re
import sys

PHASES = ["LOAD", "PROFILE", "KV_READY", "SERVING"]


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def hms(t):
    return t.strftime("%H:%M:%S")


def kind(args):
    if "deep_gemm" in args:
        return "deep_gemm"
    m = re.search(r"cached_ops/([A-Za-z0-9_]+)", args)
    if m:
        return "flashinfer:" + m.group(1)[:60]
    if "flashinfer" in args:
        return "flashinfer:other"
    if "tilelang" in args:
        return "tilelang"
    if re.search(r"--gpu-name sm_\S+ /tmp/tmp\w+\.ptx", args):
        return "triton"
    m = re.search(r"tmpxft_[0-9a-f_-]+?_([A-Za-z]\w*)\.(?:ptx|cubin|cpp|ii|c)", args)
    if m:
        return "nvcc-child:" + m.group(1)[:40]
    return "other"


def main(d):
    rows = list(csv.DictReader(open(os.path.join(d, "uma.tsv")), delimiter="\t"))
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["node"], r["phase"])].append(r)
    print("== MemAvailable per phase (GiB): min @ time, first, last, swap first/max (UTC)")
    nodes = sorted({r["node"] for r in rows})
    for n in nodes:
        for p in PHASES:
            s = by.get((n, p))
            if not s:
                continue
            mn = min(s, key=lambda r: float(r["memavail_gib"]))
            print(f"{n} {p:<8} n={len(s):>6} min={float(mn['memavail_gib']):6.2f} @ {hms(ts(mn['ts']))}"
                  f"  first={float(s[0]['memavail_gib']):6.2f} last={float(s[-1]['memavail_gib']):6.2f}"
                  f"  swap_first={float(s[0]['swap_used_gib']):.2f} swap_max={max(float(r['swap_used_gib']) for r in s):.2f}"
                  f"  span={hms(ts(s[0]['ts']))}..{hms(ts(s[-1]['ts']))}")
    print("== phase spans (first sample in phase, UTC)")
    first = {}
    for r in rows:
        first.setdefault(r["phase"], r["ts"])
    for p in PHASES:
        if p in first:
            print(f"{p:<8} from {first[p]}")
    act = os.path.join(d, "activity.tsv")
    if os.path.exists(act):
        print("== activity windows: min MemAvailable GiB per node (and swap growth)")
        for a in csv.DictReader(open(act), delimiter="\t"):
            t0, t1 = ts(a["start"]), ts(a["end"])
            parts = []
            for n in nodes:
                s = [r for r in rows if r["node"] == n and t0 <= ts(r["ts"]) <= t1]
                if not s:
                    continue
                mn = min(s, key=lambda r: float(r["memavail_gib"]))
                parts.append(f"{n} min={float(mn['memavail_gib']):.2f}@{hms(ts(mn['ts']))} "
                             f"swap {float(s[0]['swap_used_gib']):.2f}->{float(s[-1]['swap_used_gib']):.2f}")
            print(f"{a['label']:<34} {hms(t0)}..{hms(t1)}  " + " | ".join(parts))
    comp = os.path.join(d, "compilers.tsv")
    if not os.path.exists(comp):
        return
    crow = list(csv.DictReader(open(comp), delimiter="\t"))
    print("== compilers: per node/kind: distinct pids, first..last, phases, peak single-proc RSS MiB, peak summed RSS MiB per sample")
    g = collections.defaultdict(list)
    for r in crow:
        g[(r["node"], kind(r["args"]))].append(r)
    for (n, k), s in sorted(g.items()):
        per_ts = collections.defaultdict(int)
        for r in s:
            per_ts[r["ts"]] += int(r["rss_kib"])
        print(f"{n} {k:<66} pids={len({r['pid'] for r in s}):>4} rows={len(s):>5} "
              f"{hms(ts(s[0]['ts']))}..{hms(ts(s[-1]['ts']))} phases={sorted({r['phase'] for r in s}, key=PHASES.index)} "
              f"peak_proc={max(int(r['rss_kib']) for r in s) // 1024:>5} peak_sum={max(per_ts.values()) // 1024:>5}")
    print("== compiler-busy sample-seconds per node/phase (sampler lists compilers once a second) and peak summed RSS MiB")
    busy = collections.defaultdict(set)
    rss = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in crow:
        busy[(r["node"], r["phase"])].add(r["ts"][:19])
        rss[(r["node"], r["phase"])][r["ts"]] += int(r["rss_kib"])
    for n in nodes:
        for p in PHASES:
            if (n, p) in busy:
                print(f"{n} {p:<8} busy_s={len(busy[(n, p)]):>5} peak_sum_rss={max(rss[(n, p)].values()) // 1024:>5}")


if __name__ == "__main__":
    main(sys.argv[1])
