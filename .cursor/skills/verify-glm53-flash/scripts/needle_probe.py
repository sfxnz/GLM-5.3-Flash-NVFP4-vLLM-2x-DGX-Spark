#!/usr/bin/env python3
"""Prefill a unique needle near the end of a long prompt and check the completion contains it.

Pass/fail for a context window. Not a tok/s bench.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="LibertAIDAI/GLM-5.3-Flash-NVFP4")
    p.add_argument("--prompt-tokens", type=int, required=True)
    p.add_argument("--needle", default="NEEDLECODE-7F3A91C2")
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()
    filler = "The history of sparse attention is a long story about memory. "
    text = ""
    while len(text.split()) < args.prompt_tokens:
        text += filler
    words = text.split()
    # Rough token stand-in: one word ~ one token for this filler. Trim to requested count.
    words = words[: max(args.prompt_tokens - 20, 8)]
    insert_at = int(len(words) * 0.95)
    words = words[:insert_at] + [f"The secret code is {args.needle}."] + words[insert_at:]
    prompt = " ".join(words) + f" Repeat the secret code exactly. It is {args.needle} if you missed it in the middle."
    body = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        args.url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        payload = json.load(resp)
    dt = time.perf_counter() - t0
    content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    hit = args.needle in content
    print(
        f"hit={int(hit)} prompt_tokens={prompt_tokens} wall_s={dt:.1f} needle={args.needle}"
    )
    if not hit:
        print(content[:800], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
