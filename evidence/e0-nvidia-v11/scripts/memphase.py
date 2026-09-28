#!/usr/bin/env python3
"""Summarize uma.tsv (+ compilers.tsv, activity.tsv) from kit/uma_watch.py.

usage: memphase.py EVIDENCE_DIR > memory-by-phase.txt
activity.tsv (optional): start_iso<TAB>end_iso<TAB>label, one serving activity per row.
"""
import collections
import datetime as dt
import os
import re
import sys

E = sys.argv[1]


def ts(s):
    return dt.datetime.strptime(s.rstrip("Z")[:23], "%Y-%m-%dT%H:%M:%S.%f")


def hms(t):
    return t.strftime("%H:%M:%S")


rows = []
with open(os.path.join(E, "uma.tsv")) as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) == 5:
            rows.append((ts(p[0]), p[1], p[2], float(p[3]), float(p[4])))

print("== MemAvailable per phase (GiB): min @ time, first, last, max swap (UTC)")
by = collections.OrderedDict()
for r in rows:
    by.setdefault((r[1], r[2]), []).append(r)
order = ["LOAD", "PROFILE", "KV_READY", "SERVING"]
for node in sorted({r[1] for r in rows}):
    for ph in order:
        rs = by.get((node, ph))
        if not rs:
            continue
        m = min(rs, key=lambda r: r[3])
        print(f"{node} {ph:<8} n={len(rs):>6} min={m[3]:>6.2f} @ {hms(m[0])}  first={rs[0][3]:>6.2f} "
              f"last={rs[-1][3]:>6.2f}  swap_first={rs[0][4]:.2f} swap_max={max(r[4] for r in rs):.2f} "
              f"span={hms(rs[0][0])}..{hms(rs[-1][0])}")

# phase transition times
print("== phase spans (first sample in phase, UTC)")
first = {}
for r in rows:
    first.setdefault(r[2], r[0])
for ph in order:
    if ph in first:
        print(f"{ph:<8} from {first[ph].isoformat()}Z")

act = os.path.join(E, "activity.tsv")
if os.path.exists(act):
    print("== SERVING activity windows: min MemAvailable GiB per node (and swap growth)")
    for line in open(act):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3 or p[0] == "start":
            continue
        a, b = ts(p[0]), ts(p[1]) if p[1] else dt.datetime.max
        out = []
        for node in sorted({r[1] for r in rows}):
            rs = [r for r in rows if r[1] == node and a <= r[0] <= b]
            if rs:
                m = min(rs, key=lambda r: r[3])
                out.append(f"{node} min={m[3]:.2f}@{hms(m[0])} swap {rs[0][4]:.2f}->{rs[-1][4]:.2f}")
        print(f"{p[2]:<34} {hms(a)}..{hms(b) if p[1] else 'open'}  " + " | ".join(out))

prof = [r for r in rows if r[2] == "PROFILE"]
if prof:
    print("== PROFILE trajectory, 5 s buckets (min GiB per node)")
    t0 = prof[0][0]
    buckets = collections.OrderedDict()
    for r in prof:
        k = int((r[0] - t0).total_seconds() // 5)
        buckets.setdefault(k, {}).setdefault(r[1], []).append(r[3])
    for k, d in buckets.items():
        t = t0 + dt.timedelta(seconds=5 * k)
        print(hms(t) + " " + " ".join(f"{n}={min(v):6.2f}" for n, v in sorted(d.items())))

comp = os.path.join(E, "compilers.tsv")
if os.path.exists(comp):
    print("== compilers: groups by node/kind (first..last, phases, peak single-proc RSS MiB, peak summed RSS MiB per sample-second)")
    groups = collections.OrderedDict()
    fp4 = 0
    with open(comp) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            t, node, ph, pid, rss, _, args = p
            if "fp4_gemm" in args:
                fp4 += 1
            m = re.search(r"flashinfer/[^ ]*?/cached_ops/([^/ ]+)", args) or re.search(r"generated/([^/ ]+)", args)
            if "deep_gemm" in args:
                kind = "deep_gemm"
            elif "tilelang" in args.lower():
                kind = "tilelang"
            elif m:
                kind = "flashinfer:" + m.group(1)[:60]
            else:
                kind = "other"
            g = groups.setdefault((node, kind), {"n": 0, "first": t, "last": t, "phases": set(), "peak": 0, "sums": {}})
            g["n"] += 1
            g["last"] = t
            g["phases"].add(ph)
            g["peak"] = max(g["peak"], int(rss) // 1024)
            sec = t[:19]
            g["sums"][sec] = g["sums"].get(sec, 0) + int(rss) // 1024
    for (node, kind), g in sorted(groups.items()):
        print(f"{node} {kind:<62} n={g['n']:>5} {g['first'][11:19]}..{g['last'][11:19]} "
              f"phases={sorted(g['phases'])} peak_proc={g['peak']:>6} peak_sum={max(g['sums'].values()):>6}")
    print(f"fp4_gemm lines: {fp4}")
