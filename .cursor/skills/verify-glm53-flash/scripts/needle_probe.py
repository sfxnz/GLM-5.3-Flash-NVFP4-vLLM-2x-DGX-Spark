#!/usr/bin/env python3
"""Prefill a unique needle near the end of a long prompt and check the completion contains it.

Pass/fail for a context window. Not a tok/s bench. --concurrency N fires N unique-salt
streams at once (advertised occupancy is 2) and fails if the serve dies.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_prompt(prompt_tokens: int, salt: str, needle: str) -> str:
    filler = f"The history of sparse attention is a long story about memory {salt}. "
    words: list[str] = []
    while len(words) < prompt_tokens:
        words.extend(filler.split())
    words = words[: max(prompt_tokens - 20, 8)]
    insert_at = int(len(words) * 0.95)
    words = words[:insert_at] + [f"The secret code is {needle}."] + words[insert_at:]
    return " ".join(words) + f" Repeat the secret code exactly. It is {needle} if you missed it in the middle."


def models_url(completions_url: str) -> str:
    if completions_url.endswith("/chat/completions"):
        return completions_url[: -len("/chat/completions")] + "/models"
    return completions_url.rsplit("/", 1)[0] + "/models"


def serve_alive(url: str) -> bool:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def run_one(url: str, model: str, prompt: str, needle: str, salt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
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
        raise RuntimeError("no streamed content tokens")
    content = "".join(content_parts)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    ttft_s = first - t0
    wall_s = t1 - t0
    return {
        "hit": int(needle in content),
        "prompt_tokens": prompt_tokens,
        "wall_s": round(wall_s, 3),
        "ttft_s": round(ttft_s, 3),
        "prefill_tok_s": round((prompt_tokens / ttft_s) if ttft_s > 0 else 0.0, 1),
        "salt": salt,
        "content": content,
    }


def print_row(row: dict, needle: str) -> None:
    print(
        f"hit={row['hit']} prompt_tokens={row['prompt_tokens']} wall_s={row['wall_s']:.3f} "
        f"ttft_s={row['ttft_s']:.3f} prefill_tok_s={row['prefill_tok_s']:.1f} "
        f"salt={row['salt']} needle={needle}"
    )
    print(
        "SUMMARY",
        json.dumps(
            {
                "hit": row["hit"],
                "prompt_tokens": row["prompt_tokens"],
                "wall_s": row["wall_s"],
                "ttft_s": row["ttft_s"],
                "prefill_tok_s": row["prefill_tok_s"],
                "salt": row["salt"],
            }
        ),
    )


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
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel unique-salt streams. 2 is advertised max-num-seqs.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.concurrency < 1:
        print("concurrency must be >= 1", file=sys.stderr)
        return 1
    base_salt = args.salt or f"S{time.time_ns()}"
    salts = [base_salt if args.concurrency == 1 else f"{base_salt}-{i}" for i in range(args.concurrency)]
    prompts = [build_prompt(args.prompt_tokens, salt, args.needle) for salt in salts]
    if args.dry_run:
        print(f"dry_run=1 n={args.concurrency} words={len(prompts[0].split())} salt={salts[0]}")
        return 0

    rows: list[dict] = [{} for _ in salts]
    errors: list[str] = []
    if args.concurrency == 1:
        try:
            rows[0] = run_one(args.url, args.model, prompts[0], args.needle, salts[0], args.max_tokens)
        except Exception as exc:
            errors.append(str(exc))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {
                pool.submit(
                    run_one, args.url, args.model, prompts[i], args.needle, salts[i], args.max_tokens
                ): i
                for i in range(args.concurrency)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    rows[i] = fut.result()
                except Exception as exc:
                    errors.append(f"stream {i}: {exc}")

    alive = serve_alive(models_url(args.url))
    for row in rows:
        if row:
            print_row(row, args.needle)
    failed = sum(1 for row in rows if not row or row.get("hit") != 1 or row.get("prompt_tokens", 0) <= 0)
    if args.concurrency > 1:
        print(
            "SUMMARY",
            json.dumps(
                {
                    "n": args.concurrency,
                    "failed": failed,
                    "serve_alive": int(alive),
                    "errors": errors,
                }
            ),
        )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if not alive:
        print("serve_alive=0 after prefill", file=sys.stderr)
        return 1
    for row in rows:
        if row.get("hit") != 1:
            print((row.get("content") or "")[:800], file=sys.stderr)
            return 1
        if row.get("prompt_tokens", 0) <= 0:
            print("usage.prompt_tokens missing", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
