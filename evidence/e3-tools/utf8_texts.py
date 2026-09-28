#!/usr/bin/env python3
"""Run Tier 0's utf8 probe N times (greedy, thinking off, streamed, as tier0.py does) and keep the text, which
tier0.json does not store, plus the judge verdict and a hash per run.

    python3 evidence/e3-tools/utf8_texts.py --n 3 --out DIR/utf8-texts.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quality.common import Client  # noqa: E402
from quality.tier0 import OFF, UTF8_PROMPT, judge_utf8  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()
c = Client("http://127.0.0.1:8000")
runs = []
for i in range(a.n):
    out = c.chat(UTF8_PROMPT, stream=True, temperature=0, max_tokens=3000, **OFF)
    text = out.get("content") or ""
    try:
        j = judge_utf8(out)
    except Exception as e:  # noqa: BLE001
        j = {"error": f"{type(e).__name__}: {e}"}
    j = {k: v for k, v in j.items() if not isinstance(v, (list, dict))}
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    runs.append({"sha": h, "judge": j, "finish": out.get("finish_reason"), "text": text})
    first_bad = next((ln for ln in text.splitlines()[2:] if ln.count("|") < 5), None)
    print(f"run {i}: sha {h} chars {len(text)} finish {out.get('finish_reason')} judge {j}")
    print(f"   head {text[:160]!r}")
    print(f"   first odd line {first_bad!r}")
Path(a.out).write_text(json.dumps(runs, ensure_ascii=False, indent=1))
