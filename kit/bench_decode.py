#!/usr/bin/env python3
# vendored from sfxnz/forge kit @ 6285f70
"""Streamed decode bench against a live OpenAI-compatible /v1/chat/completions. The frozen ruler.

    kit/bench_decode.py --recipe <recipe-dir> --phase prose|structured|both --out <evidence-dir>

Semantics (do not change; the README decode tables are comparable only while these hold):
streamed greedy (temperature 0), thinking off (recipe.yaml bench.chat_template_kwargs),
200 completion tokens, 3 runs per (phase, concurrency), concurrency 1 and 2, per-stream median
decode tok/s, aggregate tok/s per wave, TTFT p50. A failed stream or a completion_tokens==0
stream fails the wave.

Writes <out>/bench.txt (stdout copy, ends with the SUMMARY JSON) and <out>/bench.json: one object
per phase x concurrency row using the recipe.yaml measured.decode.rows field names
(phase, concurrency, decode, aggregate, ttft_p50) plus n, completion_tokens and, when the serve
exposes vllm:spec_decode_* counters, acceptance_len and draft_acceptance_rate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitlib import Recipe  # noqa: E402

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


def stream_one(url: str, model: str, prompt: str, max_tokens: int, chat_template_kwargs: dict) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": chat_template_kwargs,
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
    url: str, model: str, prompt: str, max_tokens: int, concurrency: int, chat_template_kwargs: dict | None = None
) -> tuple[list[dict], float, float]:
    kwargs = chat_template_kwargs or {}
    t0 = time.perf_counter()
    out = []
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(stream_one, url, model, prompt, max_tokens, kwargs) for _ in range(concurrency)]
        for fut in as_completed(futs):
            try:
                out.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"stream failed: {exc}", file=sys.stderr, flush=True)
                errors.append(exc)
    if errors:
        raise RuntimeError(f"{len(errors)} stream(s) in the wave failed")
    if not out:
        raise RuntimeError("every stream in the wave failed")
    if any(int(r["completion_tokens"]) == 0 for r in out):
        raise RuntimeError("a stream returned completion_tokens==0")
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


class Tee:
    """Print to stdout and collect the same text for bench.txt."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str) -> None:
        print(text, flush=True)
        self.lines.append(text)


def to_row(summary: dict) -> dict:
    row = {
        "phase": summary["phase"],
        "concurrency": summary["concurrency"],
        "decode": summary["median_decode_tok_s"],
        "aggregate": summary["median_agg_tok_s"],
        "ttft_p50": summary["median_ttft_s"],
        "completion_tokens": summary["median_completion_tokens"],
        "n": summary["n"],
    }
    for k in ("acceptance_len", "draft_acceptance_rate"):
        if k in summary:
            row[k] = summary[k]
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", required=True, help="recipe directory (reads recipe.yaml)")
    p.add_argument("--out", required=True, help="evidence directory for bench.txt and bench.json")
    p.add_argument("--phase", choices=[*PHASES, "both"], default="both")
    p.add_argument("--url", default=None, help="override the /v1/chat/completions URL")
    p.add_argument("--model", default=None, help="override the served model name")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    args = p.parse_args()

    recipe = Recipe(args.recipe)
    url = args.url or recipe.completions_url
    model = args.model or recipe.served_name
    kwargs = recipe.chat_template_kwargs
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = Tee()

    phases = list(PHASES) if args.phase == "both" else [args.phase]
    say(
        f"url={url} model={model} max_tokens={args.max_tokens} "
        f"runs={args.runs} concurrency={args.concurrency} phases={phases} "
        f"chat_template_kwargs={json.dumps(kwargs, sort_keys=True)}"
    )
    metrics_url = url.split("/v1/", 1)[0] + "/metrics"
    summary = []
    for phase in phases:
        prompt = PHASES[phase]
        for c in args.concurrency:
            per_stream = []
            aggs = []
            counters_before = spec_counters(metrics_url)
            for i in range(args.runs):
                rows, wall, agg = wave(url, model, prompt, args.max_tokens, c, kwargs)
                per_stream.extend(rows)
                aggs.append(agg)
                dec = ",".join(f"{r['decode_tok_s']:.2f}" for r in rows)
                ttft = ",".join(f"{r['ttft_s']:.3f}" for r in rows)
                say(
                    f"phase={phase} c={c} run={i+1} wall={wall:.2f}s agg={agg:.2f} tok/s "
                    f"per_stream=[{dec}] ttft=[{ttft}]"
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
    say("SUMMARY " + json.dumps(summary, indent=2))
    (out_dir / "bench.txt").write_text("\n".join(say.lines) + "\n")
    (out_dir / "bench.json").write_text(json.dumps([to_row(s) for s in summary], indent=2) + "\n")
    print(f"evidence={out_dir}/bench.txt {out_dir}/bench.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
