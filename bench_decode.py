#!/usr/bin/env python3
"""Streamed decode bench against a live OpenAI-compatible /v1/chat/completions."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


# Two decode regimes: prose is the low-acceptance regime (the drafter guesses
# free text), structured is the high-acceptance regime (counting is nearly
# deterministic, so most draft positions verify).
PHASES = {
    "prose": (
        "Write a short paragraph about why sparse attention helps long-context "
        "language models. Keep it around eighty words. No bullet points."
    ),
    "structured": (
        "Count from 1 to 200. Output only the numbers, separated by commas, "
        "with no other text."
    ),
}


def stream_one(url: str, model: str, prompt: str, max_tokens: int) -> dict:
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
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
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
                chunks += 1
    t1 = time.perf_counter()
    if first is None:
        raise RuntimeError("no streamed content tokens")
    completion = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    decode_tokens = max(completion - 1, 0)
    decode_s = t1 - first
    return {
        "ttft_s": first - t0,
        "total_s": t1 - t0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "decode_tok_s": (decode_tokens / decode_s) if decode_s > 0 else 0.0,
        "chunks": chunks,
    }


def wave(
    url: str, model: str, prompt: str, max_tokens: int, concurrency: int
) -> tuple[list[dict], float, float]:
    t0 = time.perf_counter()
    out = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(stream_one, url, model, prompt, max_tokens) for _ in range(concurrency)]
        for fut in as_completed(futs):
            try:
                out.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - keep the wave's other streams
                print(f"stream failed: {exc}", file=sys.stderr, flush=True)
    if not out:
        raise RuntimeError("every stream in the wave failed")
    wall = time.perf_counter() - t0
    decode_tokens = sum(max(r["completion_tokens"] - 1, 0) for r in out)
    # Shared wall clock after the first token of the slowest-to-start stream is messy.
    # Aggregate is total decode tokens over the wave's wall time minus median TTFT.
    ttfts = [r["ttft_s"] for r in out]
    adj = wall - statistics.median(ttfts)
    agg = (decode_tokens / adj) if adj > 0 else 0.0
    return out, wall, agg


def median_key(rows: list[dict], key: str) -> float:
    return statistics.median(r[key] for r in rows)


SPEC_COUNTERS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")


def spec_counters(metrics_url: str) -> dict[str, float] | None:
    """Sum vLLM's spec-decode counters across label sets. None when absent."""
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (OSError, urllib.error.URLError):
        return None
    out = dict.fromkeys(SPEC_COUNTERS, 0.0)
    seen = False
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:spec_decode_"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        for counter in SPEC_COUNTERS:
            # prometheus_client >= 0.4 exposes Counters with a _total suffix;
            # accept the bare name too. The two forms never coexist.
            if name in (
                f"vllm:spec_decode_{counter}_total",
                f"vllm:spec_decode_{counter}",
            ):
                out[counter] += float(line.rsplit(" ", 1)[1])
                seen = True
    return out if seen else None


def acceptance(before: dict[str, float] | None, after: dict[str, float] | None) -> dict:
    if before is None or after is None:
        return {}
    drafts = after["num_drafts"] - before["num_drafts"]
    draft_tokens = after["num_draft_tokens"] - before["num_draft_tokens"]
    accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]
    if drafts <= 0:
        return {}
    return {
        # Emitted tokens per verification step: accepted draft tokens plus the
        # verifier's own token, the "acceptance length" from spec-decode papers.
        "acceptance_len": 1.0 + accepted / drafts,
        "draft_acceptance_rate": (accepted / draft_tokens) if draft_tokens > 0 else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="LibertAIDAI/GLM-5.3-Flash-NVFP4")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    p.add_argument("--phase", choices=[*PHASES, "both"], default="both")
    args = p.parse_args()

    phases = list(PHASES) if args.phase == "both" else [args.phase]
    print(
        f"url={args.url} model={args.model} max_tokens={args.max_tokens} "
        f"runs={args.runs} concurrency={args.concurrency} phases={phases}",
        flush=True,
    )
    metrics_url = args.url.split("/v1/", 1)[0] + "/metrics"
    summary = []
    for phase in phases:
        prompt = PHASES[phase]
        for c in args.concurrency:
            per_stream = []
            aggs = []
            counters_before = spec_counters(metrics_url)
            for i in range(args.runs):
                rows, wall, agg = wave(args.url, args.model, prompt, args.max_tokens, c)
                per_stream.extend(rows)
                aggs.append(agg)
                dec = ",".join(f"{r['decode_tok_s']:.2f}" for r in rows)
                ttft = ",".join(f"{r['ttft_s']:.3f}" for r in rows)
                print(
                    f"phase={phase} c={c} run={i+1} wall={wall:.2f}s agg={agg:.2f} tok/s "
                    f"per_stream=[{dec}] ttft=[{ttft}]",
                    flush=True,
                )
            summary.append(
                {
                    "phase": phase,
                    "concurrency": c,
                    "median_decode_tok_s": median_key(per_stream, "decode_tok_s"),
                    "median_ttft_s": median_key(per_stream, "ttft_s"),
                    "median_agg_tok_s": statistics.median(aggs),
                    "median_completion_tokens": median_key(per_stream, "completion_tokens"),
                    "n": len(per_stream),
                    **acceptance(counters_before, spec_counters(metrics_url)),
                }
            )
    print("SUMMARY", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
