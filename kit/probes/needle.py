#!/usr/bin/env python3
# vendored from sfxnz/forge kit @ 6f40808
"""Prefill a unique needle near the end of a long prompt and check the completion contains it.

Pass/fail for a context window. Not a tok/s bench. --concurrency N fires N unique-salt
streams at once and fails if the serve dies. 1M prefill can take tens of minutes.

    kit/probes/needle.py <recipe-dir> <evidence-dir>                       # every entry in recipe.yaml probes: needle:
    kit/probes/needle.py <recipe-dir> <evidence-dir> --prompt-tokens 8192 [--concurrency 2] [--salt S] [--dry-run]
    recipe.yaml:  needle: [8192, {tokens: 20480, concurrency: 2}]

Writes needle-<tokens>.txt, or needle-<tokens>-c<N>.txt when concurrency > 1.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from _probe import Probe


def build_prompt(prompt_tokens: int, salt: str, needle: str) -> str:
    filler = f"The history of sparse attention is a long story about memory {salt}. "
    words: list[str] = []
    while len(words) < prompt_tokens:
        words.extend(filler.split())
    words = words[: max(prompt_tokens - 20, 8)]
    insert_at = int(len(words) * 0.95)
    words = words[:insert_at] + [f"The secret code is {needle}."] + words[insert_at:]
    return " ".join(words) + f" Repeat the secret code exactly. It is {needle} if you missed it in the middle."


def serve_alive(url: str) -> bool:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def run_one(url: str, model: str, prompt: str, needle: str, salt: str, max_tokens: int, kwargs: dict) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": kwargs,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    first = None
    content_parts: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=7200) as resp:
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


def row_line(row: dict, needle: str) -> str:
    return (
        f"hit={row['hit']} prompt_tokens={row['prompt_tokens']} wall_s={row['wall_s']:.3f} "
        f"ttft_s={row['ttft_s']:.3f} prefill_tok_s={row['prefill_tok_s']:.1f} "
        f"salt={row['salt']} needle={needle}"
    )


def run_case(probe: Probe, prompt_tokens: int, concurrency: int, salt_arg: str, needle: str, max_tokens: int, dry_run: bool) -> int:
    probe.lines = []
    filename = f"needle-{prompt_tokens}.txt" if concurrency == 1 else f"needle-{prompt_tokens}-c{concurrency}.txt"
    url = probe.recipe.completions_url
    model = probe.recipe.served_name
    kwargs = probe.recipe.chat_template_kwargs
    base_salt = salt_arg or f"S{time.time_ns()}"
    salts = [base_salt if concurrency == 1 else f"{base_salt}-{i}" for i in range(concurrency)]
    prompts = [build_prompt(prompt_tokens, salt, needle) for salt in salts]
    if dry_run:
        print(f"dry_run=1 n={concurrency} words={len(prompts[0].split())} salt={salts[0]}")
        return 0
    probe.record(
        "request",
        {
            "url": url,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "concurrency": concurrency,
            "needle": needle,
            "salts": salts,
            "max_tokens": max_tokens,
            "chat_template_kwargs": kwargs,
            "prompt_head": prompts[0][:200],
        },
    )

    rows: list[dict] = [{} for _ in salts]
    errors: list[str] = []
    if concurrency == 1:
        try:
            rows[0] = run_one(url, model, prompts[0], needle, salts[0], max_tokens, kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {
                pool.submit(run_one, url, model, prompts[i], needle, salts[i], max_tokens, kwargs): i
                for i in range(concurrency)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    rows[i] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"stream {i}: {exc}")

    alive = serve_alive(probe.recipe.models_url)
    for row in rows:
        if row:
            probe.say(row_line(row, needle))
            probe.say("SUMMARY " + json.dumps({k: v for k, v in row.items() if k != "content"}))
            probe.record("response content", row["content"])
    probe.lines.append("http_status=200" if not errors else "http_status=error")
    failed = sum(1 for row in rows if not row or row.get("hit") != 1 or row.get("prompt_tokens", 0) <= 0)
    if concurrency > 1:
        probe.say("SUMMARY " + json.dumps({"n": concurrency, "failed": failed, "serve_alive": int(alive), "errors": errors}))
    if errors:
        return probe.finish(False, "; ".join(errors), filename)
    if not alive:
        return probe.finish(False, "serve_alive=0 after prefill", filename)
    for row in rows:
        if row.get("hit") != 1:
            return probe.finish(False, "needle_miss", filename)
        if row.get("prompt_tokens", 0) <= 0:
            return probe.finish(False, "usage.prompt_tokens missing", filename)
    return probe.finish(True, filename=filename)


def main() -> int:
    probe = Probe("needle", __doc__)
    probe.parser.add_argument("--prompt-tokens", type=int, default=None, help="one case instead of the recipe.yaml list")
    probe.parser.add_argument("--needle", default="NEEDLECODE-7F3A91C2")
    probe.parser.add_argument("--max-tokens", type=int, default=64)
    probe.parser.add_argument("--salt", default="", help="Unique string mixed into the filler so prefix cache cannot fake prefill")
    probe.parser.add_argument("--concurrency", type=int, default=1, help="Parallel unique-salt streams. Fail if the serve dies.")
    probe.parser.add_argument("--dry-run", action="store_true")
    args = probe.parse()
    if args.concurrency < 1:
        print("concurrency must be >= 1", file=sys.stderr)
        return 1
    if args.prompt_tokens is not None:
        cases = [(args.prompt_tokens, args.concurrency)]
    else:
        cases = []
        for entry in probe.recipe.probes().get("needle") or []:
            if isinstance(entry, dict):
                cases.append((int(entry["tokens"]), int(entry.get("concurrency", 1))))
            else:
                cases.append((int(entry), 1))
        if not cases:
            print("no needle cases: pass --prompt-tokens or list them under probes: needle:", file=sys.stderr)
            return 1
    rc = 0
    for tokens, conc in cases:
        rc |= run_case(probe, tokens, conc, args.salt, args.needle, args.max_tokens, args.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
