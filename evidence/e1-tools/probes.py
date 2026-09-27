#!/usr/bin/env python3
"""Fast post-ready probes: greedy count-200 (thinking off) and a thinking-off smoke.

    python3 evidence/e1-tools/probes.py > DIR/probe-count.txt

Uses quality/tier0.py's own count prompt and CoT judge, so the result matches
the Tier-0 `count` criterion. Exit 1 on any failure.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quality.common import Client  # noqa: E402
from quality.tier0 import OFF, looks_like_cot, run_count  # noqa: E402

c = Client("http://127.0.0.1:8000")
cnt = run_count(c)
print(f"count {json.dumps(cnt)}")
out = c.chat("Reply with exactly the word PING and nothing else.", max_tokens=64, temperature=0, **OFF)
content, reasoning = (out.get("content") or ""), (out.get("reasoning") or "")
smoke = {"content": content[:80], "reasoning_chars": len(reasoning), "cot": looks_like_cot(content),
         "finish_reason": out.get("finish_reason")}
print(f"thinking_off {json.dumps(smoke)}")
ok = cnt["pass"] and not smoke["cot"] and not reasoning and "PING" in content
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
