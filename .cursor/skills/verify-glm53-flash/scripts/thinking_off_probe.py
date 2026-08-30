#!/usr/bin/env python3
"""Thinking-off completion. Fail if content is empty or <think> leaks."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="LibertAIDAI/GLM-5.3-Flash-NVFP4")
    p.add_argument("--max-tokens", type=int, default=64)
    args = p.parse_args()
    body = json.dumps(
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly the word PING and nothing else.",
                }
            ],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        args.url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    message = ((payload.get("choices") or [{}])[0].get("message") or {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    leaked = "<think>" in content or "</think>" in content
    reasoning_leak = "<think>" in reasoning or "</think>" in reasoning
    print(
        f"content_chars={len(content.strip())} leaked_think={int(leaked)} "
        f"reasoning_chars={len(reasoning)} reasoning_leak={int(reasoning_leak)}"
    )
    print("SUMMARY", json.dumps({"content": content, "reasoning_content": reasoning}))
    if not content.strip():
        print("empty message content", file=sys.stderr)
        return 1
    if leaked or reasoning_leak:
        print(content[:500], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
