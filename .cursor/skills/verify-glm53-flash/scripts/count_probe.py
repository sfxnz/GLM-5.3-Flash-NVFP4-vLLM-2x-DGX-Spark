#!/usr/bin/env python3
"""Greedy count probe. Fail unless the completion contains a long consecutive run of integers."""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="LibertAIDAI/GLM-5.3-Flash-NVFP4")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--need", type=int, default=80)
    args = p.parse_args()
    body = json.dumps(
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Count from 1 to 200. Output only the numbers, separated by commas, with no other text.",
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
    text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    nums = [int(x) for x in re.findall(r"\d+", text)]
    best = cur = 0
    prev = None
    for n in nums:
        if prev is not None and n == prev + 1:
            cur += 1
        else:
            cur = 1
        best = max(best, cur)
        prev = n
    print(f"consecutive={best} need={args.need} nums={len(nums)}")
    if best < args.need:
        print(text[:500], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
