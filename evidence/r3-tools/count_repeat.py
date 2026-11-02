#!/usr/bin/env python3
"""Repeat Tier 0's greedy count-200 probe N times and keep each text. Usage: count_repeat.py N OUT.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quality.common import Client  # noqa: E402
from quality.tier0 import OFF, count_ok  # noqa: E402

PROMPT = "Count from 1 to 200. Output only the numbers, separated by commas, with no other text."
n, out_path = int(sys.argv[1]), sys.argv[2]
c = Client("http://127.0.0.1:8000")
runs = []
for i in range(n):
    out = c.chat(PROMPT, max_tokens=1200, temperature=0, **OFF)
    res = {**count_ok(out["content"]), "finish_reason": out["finish_reason"], "content": out["content"]}
    runs.append(res)
    bad = res["first_bad"]
    nums = [x.strip() for x in out["content"].split(",")]
    around = nums[max(0, (bad or 0) - 3):(bad or 0) + 3] if bad is not None else []
    print(f"count run {i}: pass={res['pass']} n={res['n_numbers']} first_bad={bad} around={around}", flush=True)
Path(out_path).write_text(json.dumps(runs, indent=1) + "\n")
print(f"COUNT {sum(r['pass'] for r in runs)}/{n} pass")
