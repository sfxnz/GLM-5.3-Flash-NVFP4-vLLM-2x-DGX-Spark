#!/usr/bin/env python3
"""QUAL-2 live kwarg check: 4 chat_template_kwargs shapes x (block, stream), c=1.

Judged with quality/tier0.py judge_kwarg_cell (content says Paris, no <think>
tags, reasoning non-empty exactly when thinking is expected, finish stop, no
chain-of-thought opening content when thinking is off).
"""
import json
import sys
from pathlib import Path

REPO = Path("/home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia")
sys.path.insert(0, str(REPO / "quality"))
from common import Client  # noqa: E402
from tier0 import KWARG_PROMPT, judge_kwarg_cell  # noqa: E402

# serve default is --default-chat-template-kwargs '{"enable_thinking": false}'
SHAPES = [
    ("thinking_true", {"thinking": True}, True),
    ("enable_thinking_true", {"enable_thinking": True}, True),
    ("thinking_false", {"thinking": False}, False),
    ("empty", None, False),
]

c = Client("http://127.0.0.1:8000")
cells = []
for name, kw, think in SHAPES:
    for stream in (False, True):
        extra = {} if kw is None else {"chat_template_kwargs": kw}
        out = c.chat(KWARG_PROMPT, stream=stream, temperature=0, max_tokens=2048 if think else 128, **extra)
        j = judge_kwarg_cell(out, think)
        cell = {"cell": f"{name}.{'stream' if stream else 'block'}", "kwargs": kw, "thinking_expected": think,
                **j, "reasoning_head": (out.get("reasoning") or "")[:160],
                "completion_tokens": out["usage"].get("completion_tokens"), "s": out["s"]}
        cells.append(cell)
        print(f"{'PASS' if j['pass'] else 'FAIL'} {cell['cell']:<28} content={j['content']!r} "
              f"reasoning_chars={j['reasoning_chars']} finish={j['finish_reason']}"
              + ("" if j["pass"] else f" checks={j['checks']}"), flush=True)
ok = all(x["pass"] for x in cells)
print(f"QUAL2 {'PASS' if ok else 'FAIL'} {sum(x['pass'] for x in cells)}/{len(cells)}")
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps({"pass": ok, "cells": cells}, indent=1, ensure_ascii=False) + "\n")
sys.exit(0 if ok else 1)
