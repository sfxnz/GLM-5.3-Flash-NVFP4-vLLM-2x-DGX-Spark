#!/usr/bin/env python3
"""Render the effort_low / enable_thinking_true prompts and show the raw greedy completion (special tokens kept)."""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "nvidia/GLM-5.3-Flash-NVFP4"
Q = "What is the capital of France? Answer with one word."


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps({"model": MODEL, **body}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


cases = [("effort_low (as vLLM builds it)", {"enable_thinking": True, "reasoning_effort": "low"}),
         ("enable_thinking_true", {"enable_thinking": True}),
         ("effort_high", {"enable_thinking": True, "reasoning_effort": "high"})]
for name, kw in cases:
    tok = post("/tokenize", {"messages": [{"role": "user", "content": Q}], "add_generation_prompt": True,
                             "chat_template_kwargs": kw})
    prompt = post("/detokenize", {"tokens": tok["tokens"]})["prompt"]
    out = post("/v1/completions", {"prompt": prompt, "max_tokens": 200, "temperature": 0,
                                   "skip_special_tokens": False})
    print(f"== {name} kwargs={kw}")
    print("prompt tail:", repr(prompt[-160:]))
    print("raw output:", repr(out["choices"][0]["text"][:400]))
chat = post("/v1/chat/completions", {"messages": [{"role": "user", "content": Q}], "reasoning_effort": "low",
                                      "temperature": 0, "max_tokens": 200})
m = chat["choices"][0]["message"]
print("== chat reasoning_effort=low: content", repr(m.get("content")), "reasoning", repr((m.get("reasoning") or m.get("reasoning_content") or "")[:200]))
