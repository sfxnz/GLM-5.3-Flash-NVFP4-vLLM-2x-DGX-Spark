#!/usr/bin/env python3
"""Ruler v2: streamed decode bench against a live OpenAI-compatible server.

Published decode score is cell A (prose, 8 distinct prompts x 512 forced
tokens, greedy, thinking off). Every cell is factorized from Prometheus
/metrics deltas taken around each wave (one request at c=1):

    acceptance_len = 1 + accepted / drafts      (tokens per verify step)
    step_ms        = sum(decode_s) * 1000 / drafts
    tok_s          = (completion_tokens - 1) / decode_s

decode_s runs from the first streamed token (content, reasoning or tool call)
to the end of the stream, so the prefill token is excluded from both sides.
TTFT ends at that same first token, so a thinking-on cell (T) whose first
tokens are all reasoning is timed like A. T's forced 512 tokens count
reasoning + content (usage.completion_tokens).
Sanity: tok_s ~= acceptance_len * 1000 / step_ms. Stdlib only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import random
import re
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor

RULER_VERSION = "v2"
TIMEOUT_S = 1800
SWAP_LIMIT_MIB = 64
SANITY_TOL = 0.03
SEED = 1234
E_TOKENS = 128
F_TOKENS = 256
E_SIZES = (("32k", 32768), ("128k", 131072))
WARMUP_TOKENS = 64

# Long-form prompts: natural answers run well past 600 tokens, so forcing
# 512 tokens (min_tokens = max_tokens) stays inside the natural distribution.
PROSE = [
    "Write a detailed essay of about 1000 words on how the printing press changed "
    "European society between 1450 and 1650. Use several paragraphs and no bullet points.",
    "Write a short story of about 1000 words about a lighthouse keeper who finds a "
    "message in a bottle during a winter storm. Use vivid description and dialogue.",
    "Write an explanatory essay of about 1000 words on how the water cycle shapes "
    "regional climates, for a curious general reader. No bullet points or headings.",
    "Write a long, warm letter of about 1000 words to your younger self about what "
    "learning the piano as an adult taught you about patience and practice.",
    "Write an essay of about 1000 words on the economics of public transport in large "
    "cities: fares, subsidies, congestion and land use. Use flowing paragraphs.",
    "Write a travelogue of about 1000 words describing a week spent walking a remote "
    "coastal trail, day by day, with the weather, people and food along the way.",
    "Explain to a curious high-school student, in an essay of about 1000 words, how "
    "vaccines train the immune system. Use analogies and no bullet points.",
    "Write an essay of about 1000 words comparing Stoicism and Epicureanism: their "
    "founders, their views of pleasure and virtue, and what each offers a modern reader.",
]
CODE = [
    "Write a complete, thread-safe Python LRU cache class with per-entry TTL, full "
    "docstrings, type hints, and a unittest test suite covering eviction and expiry.",
    "Write a JavaScript module implementing debounce, throttle and memoize utilities "
    "with JSDoc comments, edge-case handling and usage examples for each.",
    "Write a Rust program with a tokenizer and a recursive-descent parser that "
    "evaluates arithmetic expressions with + - * / parentheses and unary minus, plus tests.",
    "Write a SQL schema for a public library (books, authors, members, loans, "
    "reservations) with constraints and indexes, then eight example queries with comments.",
    "Write a Go HTTP server exposing an in-memory key-value store with GET, PUT and "
    "DELETE handlers, a mutex, JSON errors, and table-driven tests.",
    "Write a C implementation of a growable int vector with push, pop, insert, remove "
    "and bounds checks, plus a main() that exercises every function.",
    "Write a Bash backup script with argument parsing, logging, dry-run mode, "
    "rotation of old archives and clear error handling. Comment every section.",
    "Write a Python CLI using argparse and dataclasses that reads a CSV of bank "
    "transactions and prints monthly totals per category, with unit tests.",
]
WARM_PROSE = "Write an essay of about 800 words on the history of tea."
WARM_CODE = "Write a Python function that merges overlapping intervals, with tests."
# Legacy prompts, byte-identical to the pre-v2 bench (cells K and J).
LEGACY_PROSE = (
    "Write a short paragraph about why sparse attention helps long-context "
    "language models. Keep it around eighty words. No bullet points."
)
STRUCTURED = (
    "Count from 1 to 200. Output only the numbers, separated by commas, "
    "with no other text."
)
IMAGE_PROMPT = (
    "Describe this image in as much detail as you can: every shape, colour, position "
    "and what the composition might represent. Write at least 400 words."
)
WORDS = (
    "river stone market lantern harbor copper winter garden signal meadow engine "
    "orchard ledger canyon violet thunder bridge compass pepper saddle marble "
    "whistle forest window candle anchor feather quarry velvet island tunnel "
    "blanket mirror ribbon falcon cellar beacon glacier pillow timber cotton"
).split()

CELLS = {
    "A": "prose 8x512 forced, greedy, c=1 (published score)",
    "B": "code 8x512 forced, greedy, c=1",
    "J": "structured count 1..200, max 200, c=1 and c=2 (legacy structured phase; "
         "sanity_err > 3% expected: max_tokens cuts the last verify step)",
    "H": "prose distinct prompts, 512 forced, c=2 (A's prompts: prefix-cache hit, TTFT not comparable to A)",
    "I": "prose distinct prompts, 512 forced, c=4 (capacity; queues at MAX_NUM_SEQS=2)",
    "E": "long-context TTFT + prefill at 32k and 128k, unique salt, 128 forced",
    "F": "image prompt (generated PNG), 256 forced, c=1",
    "G": "prose 8x512 forced, sampled T=1.0 top_p=0.95 seed fixed, c=1",
    "K": "legacy prose (~98 natural tokens, max 200), c=1 and c=2 (continuity)",
    "T": "prose 8x512 forced (reasoning + content), greedy, thinking on (Max effort), c=1",
}
DEFAULT_CELLS = "A,B,J,H,K"

# ----------------------------------------------------------------- metrics

SPEC_NAMES = {
    "vllm:spec_decode_num_drafts": "drafts",
    "vllm:spec_decode_num_draft_tokens": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "accepted",
    "vllm:inter_token_latency_seconds_sum": "itl_sum",
    "vllm:inter_token_latency_seconds_count": "itl_count",
    "vllm:request_prefill_time_seconds_sum": "prefill_sum",
    "vllm:request_prefill_time_seconds_count": "prefill_count",
}
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"
# The API server's start time: fixed for one serve boot, new on the next.
BOOT_METRIC = "process_start_time_seconds"
_POS_RE = re.compile(r'position="(\d+)"')


def parse_metrics(text: str) -> dict:
    """Sum the counters we need across label sets. per_pos is keyed by position.
    boot_id is BOOT_METRIC when the server exposes it."""
    out: dict = {"per_pos": {}}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, val = line.rpartition(" ")
        name = head.split("{", 1)[0].strip()
        # prometheus_client exposes Counters with a _total suffix.
        if name.endswith("_total"):
            name = name[: -len("_total")]
        try:
            value = float(val)
        except ValueError:
            continue
        if name == PER_POS:
            m = _POS_RE.search(head)
            if m:
                pos = int(m.group(1))
                out["per_pos"][pos] = out["per_pos"].get(pos, 0.0) + value
        elif name == BOOT_METRIC:
            out["boot_id"] = value
        elif name in SPEC_NAMES:
            key = SPEC_NAMES[name]
            out[key] = out.get(key, 0.0) + value
    return out


def scrape(metrics_url: str) -> dict | None:
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as resp:
            return parse_metrics(resp.read().decode("utf-8", "replace"))
    except (OSError, urllib.error.URLError):
        return None


def metrics_delta(before: dict | None, after: dict | None) -> dict:
    if before is None or after is None:
        return {}
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in SPEC_NAMES.values()}
    positions = sorted(after["per_pos"])
    d["per_pos"] = [after["per_pos"][p] - before["per_pos"].get(p, 0.0) for p in positions]
    return d


def factorize(decode_tokens: int, decode_s: float, delta: dict) -> dict:
    """acceptance_len, step_ms and the tok_s identity check for one wave."""
    out: dict = {}
    drafts = delta.get("drafts", 0.0)
    if delta.get("itl_count"):
        out["itl_ms"] = 1000.0 * delta["itl_sum"] / delta["itl_count"]
        # One ITL sample per verify step; the count may lead drafts by 1 per request.
        out["itl_minus_drafts"] = delta["itl_count"] - drafts
    if delta.get("prefill_count"):
        out["server_prefill_s"] = delta["prefill_sum"] / delta["prefill_count"]
    if drafts <= 0 or decode_s <= 0:
        return out
    accepted = delta.get("accepted", 0.0)
    out["drafts"] = drafts
    out["acceptance_len"] = 1.0 + accepted / drafts
    out["step_ms"] = 1000.0 * decode_s / drafts
    if delta.get("draft_tokens"):
        out["draft_acceptance_rate"] = accepted / delta["draft_tokens"]
    out["per_pos"] = [a / drafts for a in delta.get("per_pos", [])]
    # tok_s / (acceptance_len / step) = decode_tokens / (drafts + accepted).
    out["sanity_err"] = decode_tokens / (drafts + accepted) - 1.0 if drafts + accepted else None
    return out


# ----------------------------------------------------------------- hygiene


def parse_meminfo(text: str) -> dict:
    kb = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            kb[key.strip()] = int(parts[0])
    return {
        "swap_used_mib": (kb.get("SwapTotal", 0) - kb.get("SwapFree", 0)) / 1024.0,
        "mem_available_mib": kb.get("MemAvailable", 0) / 1024.0,
    }


def read_meminfo(remote: str | None = None) -> dict | None:
    try:
        if remote is None:
            with open("/proc/meminfo", encoding="ascii") as fh:
                return parse_meminfo(fh.read())
        res = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", remote, "cat", "/proc/meminfo"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return parse_meminfo(res.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def snapshot_hosts(remote: str | None) -> dict:
    snap = {"local": read_meminfo()}
    if remote:
        snap[remote] = read_meminfo(remote)
    return snap


def swap_verdict(before: dict, after: dict, limit_mib: float = SWAP_LIMIT_MIB) -> str | None:
    """Reason string if any host's swap use grew more than limit_mib, else None."""
    reasons = []
    for host, b in before.items():
        a = after.get(host)
        if not b or not a:
            continue
        grew = a["swap_used_mib"] - b["swap_used_mib"]
        if grew > limit_mib:
            reasons.append(f"{host} swap +{grew:.0f} MiB")
    return "; ".join(reasons) or None


# ----------------------------------------------------------------- prompts


def make_png(seed: int, size: int = 256) -> bytes:
    """Four coloured quadrants and a white disc, encoded as RGB PNG with zlib."""
    rng = random.Random(seed)
    cols = [bytes(rng.randrange(256) for _ in range(3)) for _ in range(4)]
    cx, cy, r2 = rng.randrange(size), rng.randrange(size), (size // 4) ** 2
    raw = bytearray()
    for y in range(size):
        raw.append(0)  # filter: none
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 < r2:
                raw += b"\xff\xff\xff"
            else:
                raw += cols[(2 * y // size) * 2 + (2 * x // size)]

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")
    )


def image_content(seed: int) -> list:
    url = "data:image/png;base64," + base64.b64encode(make_png(seed)).decode()
    return [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": IMAGE_PROMPT},
    ]


def long_prompt(n_words: int) -> tuple[str, str]:
    """Unique salt first so no prefix-cache block can hit; filler is seeded."""
    salt = uuid.uuid4().hex
    rng = random.Random(SEED)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    body = " ".join(
        " ".join(words[i : i + 12]) + "." for i in range(0, n_words, 12)
    )
    text = (
        f"Document id {salt}.\n{body}\n\nAbove is a long document of notes. In about "
        "200 words, describe which words appear most often and any patterns you notice."
    )
    return text, salt


def spec(pid: str, content, max_tokens: int, forced: bool, **sampling) -> dict:
    return {"prompt_id": pid, "content": content, "max_tokens": max_tokens,
            "forced": forced, **sampling}


def forced_set(cell: str, prompts: list[str], n: int, **sampling) -> list[dict]:
    return [spec(f"{cell}/{i:02d}", p, n, True, **sampling) for i, p in enumerate(prompts)]


def plan_cell(cell: str, runs: int, n: int) -> tuple[list[dict], list[tuple[str, int, list[dict]]]]:
    """(warm-up specs, waves). A wave is (group, concurrency, specs). E is planned later."""
    waves: list[tuple[str, int, list[dict]]] = []
    greedy = {"temperature": 0.0}
    if cell in ("A", "B", "G", "T"):
        prompts = CODE if cell == "B" else PROSE
        sampling = {"temperature": 1.0, "top_p": 0.95, "seed": SEED} if cell == "G" else greedy
        if cell == "T":
            sampling = {**greedy, "thinking": True}
        warm = [spec("warm", WARM_CODE if cell == "B" else WARM_PROSE, WARMUP_TOKENS, True, **sampling)]
        for _ in range(runs):
            waves += [(cell, 1, [s]) for s in forced_set(cell, prompts, n, **sampling)]
    elif cell in ("H", "I"):
        c = 2 if cell == "H" else 4
        warm = [spec(f"warm{i}", f"{WARM_PROSE} Variation {i}.", WARMUP_TOKENS, True, **greedy)
                for i in range(c)]
        specs = forced_set(cell, PROSE, n, **greedy)
        for _ in range(runs):
            waves += [(cell, c, specs[i : i + c]) for i in range(0, len(specs), c)]
    elif cell in ("J", "K"):
        prompt = STRUCTURED if cell == "J" else LEGACY_PROSE
        warm = [spec("warm", prompt, 200, False, **greedy)]
        for c in (1, 2):
            waves += [(f"{cell}@c{c}", c, [spec(f"{cell}/00", prompt, 200, False, **greedy)] * c)
                      for _ in range(3 * runs)]
    elif cell == "F":
        warm = [spec("warm", image_content(SEED - 1), WARMUP_TOKENS, True, **greedy)]
        for r in range(runs):
            waves += [(cell, 1, [spec(f"F/{i:02d}", image_content(SEED + 100 * r + i), F_TOKENS, True, **greedy)])
                      for i in range(4)]
    elif cell == "E":
        warm = [spec("warm", long_prompt(6000)[0], 16, False, **greedy)]
    else:
        raise ValueError(f"unknown cell {cell!r}")
    return warm, waves


def plan_long(runs: int, tokens_per_word: float) -> list[tuple[str, int, list[dict]]]:
    waves = []
    for _ in range(runs):
        for label, target in E_SIZES:
            text, salt = long_prompt(int(target / tokens_per_word))
            s = spec(f"E/{label}", text, E_TOKENS, True, temperature=0.0)
            s["salt"] = salt
            waves.append((f"E@{label}", 1, [s]))
    return waves


# ----------------------------------------------------------------- streaming


def request_body(model: str, s: dict) -> bytes:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": s["content"]}],
        "max_tokens": s["max_tokens"],
        "temperature": s.get("temperature", 0.0),
        "stream": True,
        "stream_options": {"include_usage": True},
        # Thinking off except cell T; reasoning_effort is deliberately not sent (Max).
        "chat_template_kwargs": {"enable_thinking": bool(s.get("thinking"))},
    }
    if s["forced"]:
        body["min_tokens"] = s["max_tokens"]
    for key in ("top_p", "seed"):
        if key in s:
            body[key] = s[key]
    return json.dumps(body).encode()


def stream_one(url: str, model: str, s: dict, t_zero: float) -> dict:
    row = {"prompt_id": s["prompt_id"], "max_tokens": s["max_tokens"], "forced": s["forced"]}
    if "salt" in s:
        row["salt"] = s["salt"]
    req = urllib.request.Request(
        url, data=request_body(model, s), headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    first = None
    usage: dict = {}
    finish = None
    text = []
    done = False
    server_error = None
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # vLLM streams failures as data: {"error": ...} and then [DONE].
                if ev.get("error"):
                    server_error = server_error or ev["error"]
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = (delta.get("content") or "") + (
                        delta.get("reasoning") or delta.get("reasoning_content") or ""
                    )
                    if (piece or delta.get("tool_calls")) and first is None:
                        first = time.perf_counter()
                    text.append(piece)
                    finish = ch.get("finish_reason") or finish
    except urllib.error.HTTPError as exc:
        row["error"] = f"{exc}: {exc.read()[:500].decode('utf-8', 'replace')}"
        return row
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # IncompleteRead (server died mid-chunk) is an HTTPException, not an OSError.
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    t1 = time.perf_counter()
    completion = int(usage.get("completion_tokens") or 0)
    if server_error is not None:
        row["error"] = f"server error event: {json.dumps(server_error)[:500]}"
    elif not done:
        row["error"] = "stream ended without [DONE]"
    elif first is None:
        row["error"] = "no streamed tokens"
    elif not usage:
        row["error"] = "no usage chunk"
    elif completion == 0:
        row["error"] = "completion_tokens == 0"
    if "error" in row:
        return row
    decode_tokens = max(completion - 1, 0)
    decode_s = t1 - first
    row.update(
        t_start=t0 - t_zero, t_first=first - t_zero, t_end=t1 - t_zero,
        ttft_s=first - t0, decode_s=decode_s,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=completion, decode_tokens=decode_tokens,
        tok_s=decode_tokens / decode_s if decode_s > 0 else 0.0,
        finish_reason=finish, sha256=hashlib.sha256("".join(text).encode()).hexdigest(),
        short=bool(s["forced"] and completion < s["max_tokens"]),
    )
    if row["ttft_s"] > 0:
        row["client_prefill_tok_s"] = row["prompt_tokens"] / row["ttft_s"]
    return row


def run_wave(url: str, metrics_url: str, model: str, group: str, c: int, specs: list[dict]) -> dict:
    before = scrape(metrics_url)
    t_zero = time.perf_counter()
    if c == 1:
        rows = [stream_one(url, model, specs[0], t_zero)]
    else:
        with ThreadPoolExecutor(max_workers=c) as pool:
            rows = list(pool.map(lambda s: stream_one(url, model, s, t_zero), specs))
    after = scrape(metrics_url)
    wave: dict = {"group": group, "c": c, "requests": rows}
    good = [r for r in rows if "error" not in r]
    wave["ok"] = len(good) == len(rows)
    if not wave["ok"]:
        return wave
    decode_tokens = sum(r["decode_tokens"] for r in good)
    decode_s = sum(r["decode_s"] for r in good)
    wave["tok_s"] = statistics.fmean(r["tok_s"] for r in good)
    wave["ttft_s"] = statistics.median(r["ttft_s"] for r in good)
    if c > 1:
        span = max(r["t_end"] for r in good) - min(r["t_first"] for r in good)
        wave["agg_tok_s"] = decode_tokens / span if span > 0 else 0.0
    wave.update(factorize(decode_tokens, decode_s, metrics_delta(before, after)))
    if "server_prefill_s" in wave:
        wave["server_prefill_tok_s"] = sum(r["prompt_tokens"] for r in good) / len(good) / wave["server_prefill_s"]
    return wave


# ----------------------------------------------------------------- stats

T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def describe(values: list[float]) -> dict | None:
    """n / median / mean / stdev / 95% t-interval of the mean."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    mean = statistics.fmean(vals)
    out = {"n": len(vals), "median": statistics.median(vals), "mean": mean, "stdev": None, "ci95": None}
    if len(vals) > 1:
        sd = statistics.stdev(vals)
        t = T975[len(vals) - 2] if len(vals) - 1 <= len(T975) else 1.96
        half = t * sd / len(vals) ** 0.5
        out.update(stdev=sd, ci95=[mean - half, mean + half])
    return out


def summarize(waves: list[dict], invalid: dict[str, str]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for w in waves:
        groups.setdefault(w["group"], []).append(w)
    out = []
    for group, ws in groups.items():
        cell = group.split("@", 1)[0]
        ok = [w for w in ws if w["ok"]]
        rows = [r for w in ok for r in w["requests"]]
        drafts = sum(w.get("drafts", 0.0) for w in ok)
        pos_acc: list[float] = []
        for w in ok:
            for i, p in enumerate(w.get("per_pos", [])):
                if i == len(pos_acc):
                    pos_acc.append(0.0)
                pos_acc[i] += p * w["drafts"]
        errs = [abs(w["sanity_err"]) for w in ok if w.get("sanity_err") is not None]
        out.append({
            "group": group, "cell": cell, "c": ws[0]["c"],
            "n_waves": len(ok), "n_failed": len(ws) - len(ok), "n_requests": len(rows),
            "tok_s": describe([w["tok_s"] for w in ok]),
            "stream_tok_s": describe([r["tok_s"] for r in rows]),
            "step_ms": describe([w.get("step_ms") for w in ok]),
            "acceptance_len": describe([w.get("acceptance_len") for w in ok]),
            "agg_tok_s": describe([w.get("agg_tok_s") for w in ok]),
            "ttft_s": describe([r["ttft_s"] for r in rows]),
            "prompt_tokens": describe([r["prompt_tokens"] for r in rows]),
            "client_prefill_tok_s": describe([r.get("client_prefill_tok_s") for r in rows]),
            "server_prefill_tok_s": describe([w.get("server_prefill_tok_s") for w in ok]),
            "itl_ms": describe([w.get("itl_ms") for w in ok]),
            "per_pos": [a / drafts for a in pos_acc] if drafts else [],
            "sanity_max_err": max(errs) if errs else None,
            "sanity_ok": all(e <= SANITY_TOL for e in errs) if errs else None,
            "short_requests": sum(1 for r in rows if r.get("short")),
            "valid": cell not in invalid,
            "invalid_reason": invalid.get(cell),
        })
    return out


def fmt(stat: dict | None, key: str = "mean", digits: int = 2) -> str:
    return "-" if not stat or stat.get(key) is None else f"{stat[key]:.{digits}f}"


def table(summary: list[dict]) -> list[str]:
    head = (f"{'cell':<8}{'c':>2}{'n':>4}{'tok/s':>8}{'95% CI':>18}{'step_ms':>9}"
            f"{'acc_len':>8}{'ttft_s':>8}{'agg':>8}{'prefill':>9}  valid")
    lines = [head, "-" * len(head)]
    for s in summary:
        ci = s["tok_s"]["ci95"] if s["tok_s"] else None
        ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci else "-"
        valid = "ok" if s["valid"] else f"INVALID ({s['invalid_reason']})"
        if s["sanity_ok"] is False:
            valid += f" sanity_err={s['sanity_max_err']:.3f}"
        lines.append(
            f"{s['group']:<8}{s['c']:>2}{s['n_waves']:>4}{fmt(s['tok_s']):>8}{ci_s:>18}"
            f"{fmt(s['step_ms'], digits=1):>9}{fmt(s['acceptance_len'], digits=3):>8}"
            f"{fmt(s['ttft_s'], 'median', 3):>8}{fmt(s['agg_tok_s']):>8}"
            f"{fmt(s['server_prefill_tok_s'] or s['client_prefill_tok_s'], digits=0):>9}  {valid}"
        )
    return lines


# ----------------------------------------------------------------- main


def served_models(base: str) -> list[str]:
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as resp:
        return [m["id"] for m in json.load(resp).get("data", [])]


def git_sha() -> str | None:
    """HEAD sha, suffixed -dirty when tracked files differ from HEAD."""
    try:
        res = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "describe", "--always",
             "--abbrev=40", "--dirty", "--match=NONE"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return res.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Ruler v2 decode bench. Cells: "
        + "; ".join(f"{k}={v}" for k, v in CELLS.items()),
    )
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default=None, help="default: first id from /v1/models")
    p.add_argument("--cells", default=DEFAULT_CELLS, help=f"comma list (default {DEFAULT_CELLS})")
    p.add_argument("--full", action="store_true", help="all cells: " + ",".join(CELLS))
    p.add_argument("--runs", type=int, default=1, help="repeat each cell's prompt set")
    p.add_argument("--tokens", type=int, default=512, help="forced length for A/B/G/H/I/T")
    p.add_argument("--remote-meminfo", nargs="?", const="spark2", default=None, metavar="HOST",
                   help="also sample /proc/meminfo on HOST over ssh (default spark2)")
    p.add_argument("--out", default=None, help="directory for bench.txt and bench.json")
    args = p.parse_args(argv)

    cells = list(CELLS) if args.full else [c.strip().upper() for c in args.cells.split(",") if c.strip()]
    unknown = [c for c in cells if c not in CELLS]
    if unknown or not cells:
        p.error(f"unknown or empty cells {unknown}; known: {','.join(CELLS)}")

    base = args.url.split("/v1/", 1)[0]
    metrics_url = base + "/metrics"
    served = served_models(base)
    model = args.model or served[0]
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    report: dict = {
        "ruler_version": RULER_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha(),
        "url": args.url,
        "model": model,
        "served_models": served,
        # kit/compare.py averages the panels of one boot before comparing boots.
        "boot_id": (scrape(metrics_url) or {}).get("boot_id"),
        "args": vars(args),
        "cells": {},
        "waves": [],
    }
    log(f"ruler={RULER_VERSION} url={args.url} model={model} cells={','.join(cells)} "
        f"runs={args.runs} tokens={args.tokens} git={report['git_sha']}")
    invalid: dict[str, str] = {}
    failed_cells: list[str] = []
    unverified: list[str] = []

    def write_out() -> None:
        # Rewritten after every cell so a crash or engine death keeps what was measured.
        report["summary"] = summarize(report["waves"], invalid)
        report["failed_cells"], report["hygiene_unverified"] = failed_cells, unverified
        if not args.out:
            return
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "bench.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        with open(os.path.join(args.out, "bench.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)

    for cell in cells:
        warm, waves = plan_cell(cell, args.runs, args.tokens)
        warm_wave = run_wave(args.url, metrics_url, model, f"{cell}:warmup", len(warm), warm)
        if cell == "E":
            r = warm_wave["requests"][0]
            if "error" in r or not r["prompt_tokens"]:
                log(f"cell E FAILED: calibration request {r.get('error', 'returned 0 prompt tokens')}")
                failed_cells.append(cell)
                report["cells"][cell] = {"desc": CELLS[cell], "warmup": warm_wave,
                                         "error": "calibration request failed"}
                write_out()
                continue
            waves = plan_long(args.runs, r["prompt_tokens"] / 6000)
        before = snapshot_hosts(args.remote_meminfo)
        cell_waves = []
        for group, c, specs in waves:
            w = run_wave(args.url, metrics_url, model, group, c, specs)
            cell_waves.append(w)
            ids = ",".join(r["prompt_id"] for r in w["requests"])
            if not w["ok"]:
                errs = "; ".join(r.get("error", "") for r in w["requests"] if "error" in r)
                log(f"{group} c={c} [{ids}] FAILED {errs}")
                continue
            toks = ",".join(str(r["completion_tokens"]) for r in w["requests"])
            log(f"{group} c={c} [{ids}] tok/s={w['tok_s']:.2f} step_ms={w.get('step_ms', 0):.1f} "
                f"acc_len={w.get('acceptance_len', 0):.3f} ttft={w['ttft_s']:.3f}s tokens=[{toks}]")
        after = snapshot_hosts(args.remote_meminfo)
        reason = swap_verdict(before, after)
        if reason:
            invalid[cell] = reason
            log(f"cell {cell} INVALID: {reason}")
        blind = sorted(h for h in set(before) | set(after) if not before.get(h) or not after.get(h))
        if blind:
            unverified.append(cell)
            log(f"WARN cell {cell} hygiene_unverified: meminfo unreadable on {','.join(blind)}")
        report["cells"][cell] = {"desc": CELLS[cell], "meminfo_before": before,
                                 "meminfo_after": after, "invalid_reason": reason,
                                 "hygiene_unverified": blind, "warmup": warm_wave}
        report["waves"] += cell_waves
        write_out()

    summary = report["summary"] = summarize(report["waves"], invalid)
    log("")
    for line in table(summary):
        log(line)
    log("published decode score = cell A tok/s (prose, c=1); K is the legacy continuity row")
    # The pre-v2 ruler published the median of per-stream tok/s with no warm-up.
    for s in summary:
        if s["cell"] == "K" and s["stream_tok_s"]:
            log(f"{s['group']} legacy-comparable median per-stream tok/s = {s['stream_tok_s']['median']:.2f} "
                f"(the tok/s column is the mean of wave means)")
    short = sum(s["short_requests"] for s in summary)
    if short:
        log(f"FAIL {short} forced request(s) stopped before max_tokens")
    print("SUMMARY", json.dumps(summary), flush=True)
    write_out()
    failed = invalid or failed_cells or unverified or short or any(s["n_failed"] for s in summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
