#!/usr/bin/env python3
"""inspect_diff.py A_DIR B_DIR: diff two boots' docker inspect captures (evidence/e1-tools/capture_serve.sh output:
docker-args-{head,worker}.json, docker-env-{head,worker}.json, docker-mounts-*.json, docker-image-*.txt), per rank.
Prints args equal or the differing positions, env added/removed, mounts, image IDs, and the GLM53_* values. Stdlib."""
import json
import os
import sys

a_dir, b_dir = sys.argv[1], sys.argv[2]


def load(d, kind, node):
    p = os.path.join(d, f"docker-{kind}-{node}.json")
    return json.load(open(p)) if os.path.exists(p) else None


print(f"A = {a_dir}\nB = {b_dir}")
for node in ("head", "worker"):
    print(f"== {node}")
    aa, ba = load(a_dir, "args", node), load(b_dir, "args", node)
    if aa == ba:
        print(f"args: identical ({len(aa)} words)")
    else:
        print(f"args: DIFF (A {len(aa)} words, B {len(ba)} words)")
        for i in range(max(len(aa), len(ba))):
            x = aa[i] if i < len(aa) else None
            y = ba[i] if i < len(ba) else None
            if x != y:
                print(f"  [{i}] A={x!r} B={y!r}")
    ae, be = set(load(a_dir, "env", node)), set(load(b_dir, "env", node))
    print(f"env: {len(ae)} vs {len(be)} entries; only in A: {sorted(ae - be)}; only in B: {sorted(be - ae)}")
    g = lambda s: sorted(x for x in s if x.startswith("GLM53_"))  # noqa: E731
    print(f"GLM53_* A: {g(ae)}")
    print(f"GLM53_* B: {g(be)}")
    print(f"GLM53_* equal: {g(ae) == g(be)} (count {len(g(be))})")
    am = [m["Source"] + ":" + m["Destination"] for m in load(a_dir, "mounts", node) or []]
    bm = [m["Source"] + ":" + m["Destination"] for m in load(b_dir, "mounts", node) or []]
    print(f"mounts: {'identical' if am == bm else f'DIFF A={am} B={bm}'}")
    ai = open(os.path.join(a_dir, f"docker-image-{node}.txt")).read().split()
    bi = open(os.path.join(b_dir, f"docker-image-{node}.txt")).read().split()
    print(f"image: {'identical ' + bi[0] if ai[:2] == bi[:2] else f'DIFF A={ai[:2]} B={bi[:2]}'}")
