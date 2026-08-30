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
    p.add_argument(
        "--salt",
        default="",
        help="Unique string mixed into the filler so prefix cache cannot fake prefill",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    salt = args.salt or f"S{time.time_ns()}"
    filler = f"The history of sparse attention is a long story about memory {salt}. "
    words: list[str] = []
    while len(words) < args.prompt_tokens:
        words.extend(filler.split())
    words = words[: max(args.prompt_tokens - 20, 8)]
    insert_at = int(len(words) * 0.95)
    words = words[:insert_at] + [f"The secret code is {args.needle}."] + words[insert_at:]
    prompt = " ".join(words) + f" Repeat the secret code exactly. It is {args.needle} if you missed it in the middle."
    if args.dry_run:
        print(f"dry_run=1 words={len(prompt.split())} salt={salt}")
        return 0
    body = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        args.url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    first = None
    content_parts: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta and first is None:
                first = time.perf_counter()
            if delta:
                content_parts.append(delta)
    t1 = time.perf_counter()
    if first is None:
        print("no streamed content tokens", file=sys.stderr)
        return 1
    content = "".join(content_parts)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    hit = args.needle in content
    ttft_s = first - t0
    wall_s = t1 - t0
    prefill_tok_s = (prompt_tokens / ttft_s) if ttft_s > 0 else 0.0
    print(
        f"hit={int(hit)} prompt_tokens={prompt_tokens} wall_s={wall_s:.3f} "
        f"ttft_s={ttft_s:.3f} prefill_tok_s={prefill_tok_s:.1f} salt={salt} needle={args.needle}"
    )
    print(
        "SUMMARY",
        json.dumps(
            {
                "hit": int(hit),
                "prompt_tokens": prompt_tokens,
                "wall_s": round(wall_s, 3),
                "ttft_s": round(ttft_s, 3),
                "prefill_tok_s": round(prefill_tok_s, 1),
                "salt": salt,
            }
        ),
    )
    if not hit:
        print(content[:800], file=sys.stderr)
        return 1
    if prompt_tokens <= 0:
        print("usage.prompt_tokens missing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
