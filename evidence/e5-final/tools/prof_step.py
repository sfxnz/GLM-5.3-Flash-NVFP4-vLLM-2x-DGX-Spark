#!/usr/bin/env python3
"""prof_step.py --kind KIND --out DIR: one torch-profiler capture on a serve booted with
EXTRA_ARGS="--profiler-config $(tools/nsys_step.sh --print-config)", like tools/nsys_step.sh but with a choice of
prompt (nsys_step.sh has prose only). Warmup (profiler off), /start_profile, the streams (512 forced tokens,
greedy, thinking off), /stop_profile, then copies the new traces from both nodes' $HF_CACHE/glm53-prof into
DIR/rank0 and DIR/rank1. Also records the /metrics delta over the profiled wave (acceptance_len, step_ms,
per-position acceptance) with bench_decode.py's own helpers.

  KIND  prose       c=1, bench_decode.py PROSE[0]
        code        c=1, bench_decode.py CODE[0]
        structured  c=1, "Count from 1 to 1000 ..." (the J prompt runs out after ~25 steps; 512 forced tokens of a
                    1..1000 count stay inside its natural length and give > DELAY+STEPS verify steps)
        prose2      c=2, PROSE[1] and PROSE[2] (distinct prompts)
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import bench_decode as bd  # noqa: E402

API = "http://127.0.0.1:8000"
URL = API + "/v1/chat/completions"
STRUCT_1000 = "Count from 1 to 1000. Output only the numbers, separated by commas, with no other text."
KINDS = {"prose": [bd.PROSE[0]], "code": [bd.CODE[0]], "structured": [STRUCT_1000], "prose2": [bd.PROSE[1], bd.PROSE[2]]}


def post(path, timeout):
    req = urllib.request.Request(API + path, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()[:300].decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=KINDS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--worker", default="spark2")
    ap.add_argument("--flush-s", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof_dir = os.path.expanduser("~/.cache/huggingface/glm53-prof")
    model = json.load(urllib.request.urlopen(API + "/v1/models", timeout=10))["data"][0]["id"]
    prompts = KINDS[a.kind]
    rep = {"kind": a.kind, "prompts": prompts, "tokens": a.tokens, "model": model}

    warm = bd.stream_one(URL, model, bd.spec("warm", prompts[0], 128, True, temperature=0.0), time.perf_counter())
    rep["warmup"] = {k: warm.get(k) for k in ("completion_tokens", "tok_s", "error")}
    before_files = set(glob.glob(f"{prof_dir}/**/*.pt.trace.json*", recursive=True))
    start_epoch = int(time.time()) - 1
    m0 = bd.scrape(API + "/metrics")
    code, body = post("/start_profile", 60)
    rep["start_profile"] = [code, body]
    print(f"start_profile HTTP {code}", flush=True)
    t_zero = time.perf_counter()
    rows = [None] * len(prompts)

    def run(i):
        rows[i] = bd.stream_one(URL, model, bd.spec(f"{a.kind}/{i}", prompts[i], a.tokens, True, temperature=0.0), t_zero)

    th = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    m1 = bd.scrape(API + "/metrics")
    t_stop = time.time()
    try:
        code, body = post("/stop_profile", 900)
    except Exception as e:  # noqa: BLE001
        code, body = None, f"{type(e).__name__}: {e}"
    rep["stop_profile"] = [code, body, round(time.time() - t_stop, 1)]
    print(f"stop_profile {rep['stop_profile']}", flush=True)
    time.sleep(a.flush_s)
    rep["streams"] = [{k: r.get(k) for k in ("prompt_id", "completion_tokens", "decode_s", "tok_s", "ttft_s", "error")}
                      for r in rows]
    dec_tokens = sum(r.get("decode_tokens", 0) for r in rows)
    dec_s = max((r.get("decode_s") or 0) for r in rows)
    # Same factorization as bench_decode.run_wave: decode_s summed over the streams.
    rep["wave"] = bd.factorize(dec_tokens, sum((r.get("decode_s") or 0) for r in rows), bd.metrics_delta(m0, m1))
    rep["wave"]["decode_tokens"] = dec_tokens
    rep["wave"]["wall_decode_s"] = dec_s

    r0 = out / "rank0"
    r0.mkdir(exist_ok=True)
    new0 = [f for f in glob.glob(f"{prof_dir}/**/*.pt.trace.json*", recursive=True)
            if f not in before_files and os.path.getmtime(f) >= start_epoch]
    for f in new0:
        subprocess.run(["cp", f, str(r0)], check=False)
    r1 = out / "rank1"
    r1.mkdir(exist_ok=True)
    remote = subprocess.run(["ssh", a.worker, f"find {prof_dir} -name '*.pt.trace.json*' -newermt @{start_epoch}"],
                            capture_output=True, text=True).stdout.split()
    for f in remote:
        subprocess.run(["scp", "-q", f"{a.worker}:{f}", str(r1)], check=False)
    rep["traces"] = {"rank0": [(os.path.basename(f), os.path.getsize(f)) for f in new0], "rank1_remote": remote}
    (out / "prof_step.json").write_text(json.dumps(rep, indent=1) + "\n")
    w = rep["wave"]
    print(f"kind={a.kind} streams={[(s['completion_tokens'], round(s['tok_s'] or 0, 2), s['error']) for s in rep['streams']]}"
          f" acc_len={w.get('acceptance_len')} step_ms={w.get('step_ms')} per_pos={[round(x, 3) for x in w.get('per_pos', [])]}")
    print(f"traces rank0={rep['traces']['rank0']} rank1={remote}")
    return 0 if code == 200 and all(not s["error"] for s in rep["streams"]) and new0 else 1


if __name__ == "__main__":
    sys.exit(main())
