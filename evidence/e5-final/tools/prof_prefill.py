#!/usr/bin/env python3
"""prof_prefill.py --out DIR [--words 11000] [--tokens 32]: torch-profiler capture of prefill chunks on a serve booted
with EXTRA_ARGS="--profiler-config $(tools/nsys_step.sh --print-config)" (delay 8, 30 steps, fixed at boot).

One c=1 request with bench_decode.long_prompt (unique salt, so no prefix-cache hit) of ~2 tokens per word: 11000 words
is ~22k tokens = 11 chunks of max_num_batched_tokens 2048, so the profiler (which skips the first 8 steps after
/start_profile) records chunks 8..10 and then the decode steps. Warmup (profiler off), /start_profile, the request,
/stop_profile, then copies the new traces from both nodes' $HF_CACHE/glm53-prof into DIR/rank0 and DIR/rank1, like
prof_step.py. Parse with tools/step_buckets.py --only '_context_1\\(' (prefill-chunk steps)."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import bench_decode as bd  # noqa: E402

API = "http://127.0.0.1:8000"
URL = API + "/v1/chat/completions"


def post(path, timeout):
    req = urllib.request.Request(API + path, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()[:300].decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", type=int, default=11000)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--worker", default="spark2")
    ap.add_argument("--flush-s", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prof_dir = os.path.expanduser("~/.cache/huggingface/glm53-prof")
    model = json.load(urllib.request.urlopen(API + "/v1/models", timeout=10))["data"][0]["id"]
    rep = {"kind": "prefill", "words": a.words, "tokens": a.tokens, "model": model}
    warm = bd.stream_one(URL, model, bd.spec("warm", bd.PROSE[0], 32, True, temperature=0.0), time.perf_counter())
    rep["warmup"] = {k: warm.get(k) for k in ("completion_tokens", "tok_s", "error")}
    text, salt = bd.long_prompt(a.words)
    s = bd.spec("prefill/0", text, a.tokens, True, temperature=0.0)
    s["salt"] = salt
    before_files = set(glob.glob(f"{prof_dir}/**/*.pt.trace.json*", recursive=True))
    start_epoch = int(time.time()) - 1
    m0 = bd.scrape(API + "/metrics")
    code, body = post("/start_profile", 60)
    rep["start_profile"] = [code, body]
    print(f"start_profile HTTP {code}", flush=True)
    row = bd.stream_one(URL, model, s, time.perf_counter())
    m1 = bd.scrape(API + "/metrics")
    t_stop = time.time()
    try:
        code, body = post("/stop_profile", 900)
    except Exception as e:  # noqa: BLE001
        code, body = None, f"{type(e).__name__}: {e}"
    rep["stop_profile"] = [code, body, round(time.time() - t_stop, 1)]
    print(f"stop_profile {rep['stop_profile']}", flush=True)
    time.sleep(a.flush_s)
    rep["stream"] = {k: row.get(k) for k in ("prompt_id", "prompt_tokens", "completion_tokens", "ttft_s", "decode_s",
                                             "tok_s", "client_prefill_tok_s", "error")}
    d = bd.metrics_delta(m0, m1)
    rep["metrics_delta"] = d
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
    (out / "prof_prefill.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    st = rep["stream"]
    print(f"prompt_tokens={st['prompt_tokens']} ttft={st['ttft_s']} completion={st['completion_tokens']} "
          f"error={st['error']} chunks~{(st['prompt_tokens'] or 0) / 2048:.1f}")
    print(f"traces rank0={rep['traces']['rank0']} rank1={remote}")
    return 0 if code == 200 and not st["error"] and new0 else 1


if __name__ == "__main__":
    sys.exit(main())
