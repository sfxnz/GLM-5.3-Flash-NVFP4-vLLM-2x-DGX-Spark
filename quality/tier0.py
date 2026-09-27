#!/usr/bin/env python3
"""Tier-0 quality gate for the GLM-5.3-Flash serve (stdlib only, ~15 min).

Run it on every knob change. Two modes:

  record   capture a reference: teacher-forced top-20 logprobs over the frozen
           corpus (quality/data/corpus.jsonl, ~187k tokens) and greedy outputs,
           each twice, so the reference carries its own rerun noise. Writes
           <evals>/ref/<name>.json, then runs the behavioural probes.
  compare  capture once and compare against a reference, then run the probes.

Components (--only picks a subset):
  nll      /v1/completions, prompt_logprobs=20, max_tokens=1, prompt
           "[gMASK]<sop>" + doc. First 16 doc tokens are not scored. Reports
           mean NLL (nats/token), top-1 agreement and top-20 KL vs the ref.
  greedy   20 fixed prompts x 200 tokens, thinking off; first divergence and
           per-token divergence hazard vs the reference's first run.
  count    count 1..200 with thinking off; the integers must be exactly 1..200.
  kwargs   6 chat-template kwarg shapes x stream on/off = 12 cells; content vs
           reasoning split, no <think> or chain-of-thought in content when off.
  utf8     streamed 40-row numeric table with Chinese numerals: zero U+FFFD,
           all 40 rows, every n^2 right.
  tools    50 tool calls, tool_choice auto, half streamed: JSON-valid rate.
  vision   quality/vision.py suite (images; video when the server takes it).
  needle   fixed-salt passcode at 8k/32k (plus 128k with --long) x depth
           0.1/0.5/0.9; 2 of 3 per length.

Gates (PLAN section 6): see GATES. Output tier0.json lists every criterion
PASS/FAIL and a one-line verdict. Exit 1 on any FAIL. c<=2 always (MAX_NUM_SEQS=2);
nll and greedy run at c=1. Never run next to a bench.

  python3 quality/tier0.py record --name libertai-caca4e6
  python3 quality/tier0.py compare --ref libertai-caca4e6 [--stage fp8|nvfp4]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import sys
import time
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, EVALS_DIR, Client, jsonl, numbers, pmap, utc_stamp  # noqa: E402
import vision  # noqa: E402

BOS_TEXT = "[gMASK]<sop>"
BOS_IDS = [154822, 154824]
NLL_SKIP = 16
TOPK = 20
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
COMPONENTS = ["nll", "greedy", "count", "kwargs", "utf8", "tools", "vision", "needle"]
NEEDLE_LENGTHS = (8192, 32768)
NEEDLE_LONG = 131072
DEPTHS = (0.1, 0.5, 0.9)
STAGES = {"fp8": {"top1": 0.99, "kl": 1e-3}, "nvfp4": {"top1": 0.98, "kl": 3e-3}}
GATES = {
    "nll.delta": "|mean NLL - ref| <= max(3 * sigma, 0.005) nats/token; sigma = ref rerun |delta|",
    "nll.top1_rerun": "top-1 agreement >= ref rerun top-1 agreement - 0.5 points",
    "nll.top1_stage": "top-1 agreement >= stage floor (fp8 99%, nvfp4 98%)",
    "nll.kl_stage": "mean top-20 KL(ref||cand) <= stage ceiling (fp8 1e-3, nvfp4 3e-3)",
    "greedy.hazard": "golden divergence hazard <= 2 * max(ref A/A hazard, 0.005)",
    "count": "thinking-off count is exactly 1..200",
    "kwargs.core": "the 10 kwarg-matrix cells other than thinking:true pass",
    "kwargs.thinking_alias": "both thinking:true cells pass (known FAIL until the QUAL-2 template alias)",
    "utf8": "zero U+FFFD, rows 1..40 present, every n^2 right, finish_reason stop",
    "tools.json_valid": "tool-call JSON-valid rate >= 0.98 over 50 calls (a missing call counts as invalid)",
    "vision": "every vision check passes (video may SKIP)",
    "needle": "each length finds >= 2 of 3 depths",
}

GREEDY_PROMPTS = [
    "Explain why the sky is blue in exactly three sentences.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List the first 20 prime numbers, comma separated.",
    "Translate into French: The quick brown fox jumps over the lazy dog near the river bank.",
    "Summarize the plot of Hamlet in five bullet points.",
    "Solve for x: 3x^2 - 12x + 9 = 0. Show the steps.",
    "Write a haiku about memory bandwidth.",
    "What happens if you divide by zero in IEEE 754 floating point? Answer briefly.",
    "用三句话介绍长城的历史。",
    "Write a SQL query that returns the five customers with the highest total order value.",
    "Continue the story: The lighthouse keeper found a letter that had not been there the night before.",
    "Give a markdown table of the planets with their order from the sun and number of known moons.",
    "Explain the difference between TCP and UDP to a beginner.",
    "Write a bash one-liner that counts the lines in every .py file under the current directory.",
    "What is 17 * 23? Show the multiplication step by step.",
    "Describe the water cycle in one paragraph.",
    "Escribe un correo breve para pedir una reunión el martes a las diez.",
    "Write a JSON object describing a book with title, author, year and a list of three genres.",
    "Name five sorting algorithms and give the average time complexity of each.",
    "Explain what a hash table is and how collisions are handled.",
]
GREEDY_TOKENS = 200

KWARG_PROMPT = "What is the capital of France? Answer with one word."
# (name, extra request fields, thinking expected). "default" follows the serve.
KWARG_SHAPES = [
    ("default", {}, None),
    ("enable_thinking_false", {"chat_template_kwargs": {"enable_thinking": False}}, False),
    ("enable_thinking_true", {"chat_template_kwargs": {"enable_thinking": True}}, True),
    ("thinking_true", {"chat_template_kwargs": {"thinking": True}}, True),
    ("effort_low", {"reasoning_effort": "low"}, True),
    ("effort_none", {"reasoning_effort": "none"}, False),
]
# Below Max effort the template opens <think> but the model may close it at once on
# this one-word question (live 2026-09-27: low and high both gave "</think>Paris").
REASONING_OPTIONAL = {"effort_low"}
COT_STARTS = ("okay", "ok,", "ok so", "let me", "let's", "hmm", "the user", "we need", "i need to",
              "first,", "alright", "wait", "so the question", "thinking")

UTF8_PROMPT = ("Write a markdown table with exactly 40 data rows, for n = 1 to 40. Columns: n | n² | n³ | "
               "n in Chinese numerals (for example 二十三). Output only the table, no other text.")


# ---------------------------------------------------------------- pure math

def kl_topk(p_ids: list[int], p_lp: list[float], q_ids: list[int], q_lp: list[float],
            q_extra: dict | None = None) -> float:
    """KL(p || q) over p's top-k plus one 'rest' bucket.

    p's top-k tokens that are missing from q's returned list get
    min(q's k-th prob, q's unreturned mass / (missing + 1)), so q stays a
    proper distribution over (top-k of p) + rest. Identical inputs give 0.
    """
    qmap = dict(zip(q_ids, q_lp))
    for k, v in (q_extra or {}).items():
        qmap.setdefault(k, v)
    q_top_mass = sum(math.exp(x) for x in q_lp)
    missing = [i for i in p_ids if i not in qmap]
    fill = 0.0
    if missing:
        fill = min(math.exp(min(q_lp)), max(0.0, 1.0 - q_top_mass) / (len(missing) + 1))
    kl, p_sum, q_sum = 0.0, 0.0, 0.0
    for i, lp in zip(p_ids, p_lp):
        p = math.exp(lp)
        q = max(math.exp(qmap[i]) if i in qmap else fill, 1e-12)
        kl += p * (lp - math.log(q))
        p_sum += p
        q_sum += q
    p_rest, q_rest = max(0.0, 1.0 - p_sum), max(1e-12, 1.0 - q_sum)
    if p_rest > 1e-12:
        kl += p_rest * math.log(p_rest / q_rest)
    return max(0.0, kl)


def compare_captures(ref: list[dict], cand: list[dict]) -> dict:
    """Paired NLL / top-1 / KL over docs present in both (same token ids).

    Per doc, lp[i] is the logprob of prompt token start+i; tk_ids / tk_lp hold
    the TOPK alternatives of every scored position back to back (flat arrays)."""
    cmap = {d["id"]: d for d in cand}
    tot = {"n": 0, "nll_ref": 0.0, "nll_cand": 0.0, "agree": 0, "kl": 0.0}
    by_domain, mismatched = {}, []
    for r in ref:
        c = cmap.get(r["id"])
        if c is None or c["ids"] != r["ids"]:
            mismatched.append(r["id"])
            continue
        dom = by_domain.setdefault(r["domain"], {"n": 0, "nll_ref": 0.0, "nll_cand": 0.0, "agree": 0, "kl": 0.0})
        n, k, start = len(r["lp"]), TOPK, r["start"]
        rt, rl, ct, cl = r["tk_ids"], r["tk_lp"], c["tk_ids"], c["tk_lp"]
        agree = sum(rt[i * k] == ct[i * k] for i in range(n))
        kl = sum(kl_topk(rt[i * k:(i + 1) * k], rl[i * k:(i + 1) * k], ct[i * k:(i + 1) * k],
                         cl[i * k:(i + 1) * k], {c["ids"][start + i]: c["lp"][i]}) for i in range(n))
        for acc in (tot, dom):
            acc["n"] += n
            acc["nll_ref"] -= sum(r["lp"][:n])
            acc["nll_cand"] -= sum(c["lp"][:n])
            acc["agree"] += agree
            acc["kl"] += kl

    def fin(a: dict) -> dict:
        n = max(1, a["n"])
        return {"tokens": a["n"], "nll_ref": round(a["nll_ref"] / n, 6), "nll_cand": round(a["nll_cand"] / n, 6),
                "delta": round((a["nll_cand"] - a["nll_ref"]) / n, 6), "top1_agree": round(a["agree"] / n, 6),
                "kl": round(a["kl"] / n, 8)}

    return {**fin(tot), "mismatched_docs": mismatched,
            "by_domain": {k: fin(v) for k, v in sorted(by_domain.items())}}


def mean_nll(capture: list[dict]) -> float:
    n = sum(len(d["lp"]) for d in capture)
    return -sum(sum(d["lp"]) for d in capture) / max(1, n)


def first_divergence(a: list, b: list) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def hazard(pairs: list[tuple[list, list]]) -> dict:
    """Divergence events / tokens at risk (a pair stops being at risk once it diverges)."""
    events, at_risk, divs = 0, 0, []
    for a, b in pairs:
        d = first_divergence(a, b)
        divs.append(d)
        if d is None:
            at_risk += len(a)
        else:
            events += 1
            at_risk += d + 1
    return {"hazard": round(events / at_risk, 6) if at_risk else 0.0, "diverged": events,
            "pairs": len(pairs), "first_div": divs}


def count_ok(text: str) -> dict:
    nums = numbers(text)
    want = list(range(1, 201))
    return {"pass": nums == want, "n_numbers": len(nums), "first_bad": first_divergence(nums, want)}


def looks_like_cot(content: str) -> bool:
    t = content.strip().lower()
    return t.startswith(COT_STARTS) or "</think>" in t or "<think>" in t or len(t) > 300


def judge_kwarg_cell(out: dict, think: bool, reasoning_optional: bool = False) -> dict:
    content, reasoning = (out.get("content") or "").strip(), (out.get("reasoning") or "").strip()
    checks = {
        "content": bool(content),
        "answer": "paris" in content.lower(),
        "no_think_tags": "<think>" not in content and "</think>" not in content,
        "reasoning": (bool(reasoning) or reasoning_optional) if think else not reasoning,
        "no_cot": True if think and reasoning else not looks_like_cot(content),
        "finished": out.get("finish_reason") == "stop",
    }
    return {"pass": all(checks.values()), "checks": checks, "content": content[:120],
            "reasoning_chars": len(reasoning), "finish_reason": out.get("finish_reason")}


CN_DIGITS = "零一二三四五六七八九"


def chinese_numeral(n: int) -> str:
    if n < 10:
        return CN_DIGITS[n]
    tens, ones = divmod(n, 10)
    return (CN_DIGITS[tens] if tens > 1 else "") + "十" + (CN_DIGITS[ones] if ones else "")


def judge_utf8(out: dict) -> dict:
    text = (out.get("content") or "") + (out.get("reasoning") or "")
    rows = {}
    for line in (out.get("content") or "").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0].isdigit():
            rows[int(cells[0])] = cells
    sq_err = cube_err = cn_err = 0
    for n, cells in rows.items():
        sq_err += cells[1].replace(",", "") != str(n * n)
        cube_err += cells[2].replace(",", "") != str(n ** 3)
        cn_err += cells[3] != chinese_numeral(n)
    fffd = text.count("�")
    complete = sorted(rows) == list(range(1, 41))
    return {"pass": fffd == 0 and complete and sq_err == 0 and out.get("finish_reason") == "stop",
            "fffd": fffd, "rows": len(rows), "complete": complete, "square_errors": sq_err,
            "cube_errors": cube_err, "chinese_numeral_errors": cn_err, "finish_reason": out.get("finish_reason")}


def values_equal(got, want) -> bool:
    if isinstance(want, bool) or isinstance(got, bool):
        return got is want
    if isinstance(want, (int, float)):
        return isinstance(got, (int, float)) and abs(got - want) < 1e-9
    if isinstance(want, str):
        return isinstance(got, str) and got.strip().casefold() == want.strip().casefold()
    if isinstance(want, list):
        return isinstance(got, list) and len(got) == len(want) and all(map(values_equal, got, want))
    return got == want


def judge_tool_item(item: dict, out: dict) -> dict:
    calls = out.get("tool_calls") or []
    parsed = []
    for c in calls:
        try:
            parsed.append(json.loads(c["arguments"] or "{}"))
        except ValueError:
            parsed.append(None)
    json_valid = bool(calls) and all(isinstance(p, dict) for p in parsed)
    exp = item["expect"]
    name_ok = bool(calls) and calls[0]["name"] == exp["name"]
    args = parsed[0] if parsed else None
    args_ok = (name_ok and isinstance(args, dict)
               and all(k in args and values_equal(args[k], v) for k, v in exp["args"].items())
               and all(k in args for k in item.get("ignore", [])))
    return {"id": item["id"], "n_calls": len(calls), "json_valid": json_valid, "name_ok": name_ok,
            "args_ok": args_ok, "calls": calls[:2], "content": (out.get("content") or "")[:120]}


# ------------------------------------------------------------ needle filler

_ADJ = ("amber brittle copper dusky eager feral gilded hollow ivory jagged knotted languid mossy narrow "
        "ochre pallid quiet russet silent tawny umber velvet wary woven yellowed zealous ashen bleak").split()
_NOUN = ("lantern orchard ledger kettle bridge harbor quarry meadow chimney anvil cellar compass lighthouse "
         "granary loom mill parcel quill ridge saddle tannery thicket vault wagon well workshop barge").split()
_VERB = ("mended carried weighed painted counted guarded sealed traded repaired measured polished hid copied "
         "lifted buried sketched hauled borrowed wrapped tended").split()
_ADV = ("slowly", "quietly", "twice", "again", "carefully", "briskly", "at dawn", "before supper")
_SYL = "ka lo mir ven tas ob rune fel dra is quo zen hal por ith gam sel bri vo tur ney ash cor".split()


def _name(rng: random.Random) -> str:
    return " ".join("".join(rng.choice(_SYL) for _ in range(rng.randint(2, 3))).capitalize() for _ in range(2))


def _sentence(rng: random.Random) -> str:
    a, n, v = rng.choice(_ADJ), rng.choice(_NOUN), rng.choice(_VERB)
    t = rng.randrange(3)
    if t == 0:
        return f"{_name(rng)} {v} the {a} {n} {rng.choice(_ADV)}."
    if t == 1:
        return f"In {_name(rng).split()[0]}, the {a} {n} was {v} by {rng.randint(2, 97)} workers near the {rng.choice(_NOUN)}."
    return f"The ledger lists {rng.randint(3, 999)} {n}s, each {a}, and {_name(rng)} {v} them {rng.choice(_ADV)}."


def filler(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(_sentence(rng) for _ in range(rng.randint(4, 7))) for _ in range(count)]


def needle_code(seed: int) -> tuple[str, str]:
    rng = random.Random(seed * 7919 + 1)
    letters = "BCDFGHJKLMNPQRSTVWXZ"
    code = (f"{''.join(rng.choice(letters) for _ in range(3))}-{rng.randint(1000, 9999)}-"
            f"{''.join(rng.choice(letters) for _ in range(2))}")
    return _name(rng), code


def needle_doc(paras: list[str], depth: float, name: str, code: str, seed: int) -> str:
    idx = max(1, min(len(paras) - 1, int(len(paras) * depth)))
    note = f"The vault code assigned to {name} is {code}. It was never written down again."
    return "\n\n".join([f"Archive glm53-tier0-{seed}. Field notes follow."] + paras[:idx] + [note] + paras[idx:])


# ------------------------------------------------------------ components

def corpus() -> tuple[list[dict], str]:
    path = DATA / "corpus.jsonl"
    return jsonl(path), hashlib.sha256(path.read_bytes()).hexdigest()


def capture_doc(c: Client, doc: dict) -> dict:
    out = c.post("/v1/completions", {"prompt": BOS_TEXT + doc["text"], "max_tokens": 1, "temperature": 0,
                                     "prompt_logprobs": TOPK, "return_token_ids": True})
    ch = out["choices"][0]
    ids, plp = ch["prompt_token_ids"], ch["prompt_logprobs"]
    if ids[:2] != BOS_IDS:
        raise RuntimeError(f"{doc['id']}: prompt does not start with [gMASK]<sop> ids {BOS_IDS}: {ids[:4]}")
    start = len(BOS_IDS) + NLL_SKIP
    lp, tk_ids, tk_lp = array("f"), array("i"), array("f")
    for i in range(start, len(ids)):
        entry = plp[i]
        lp.append(float(entry[str(ids[i])]["logprob"]))
        top = sorted(((v["rank"], -v["logprob"], int(k)) for k, v in entry.items()))[:TOPK]
        if len(top) < TOPK:
            raise RuntimeError(f"{doc['id']}: {len(top)} prompt logprobs < {TOPK}; raise --max-logprobs")
        tk_ids.extend(t[2] for t in top)
        tk_lp.extend(-t[1] for t in top)
    return {"id": doc["id"], "domain": doc["domain"], "ids": ids, "start": start, "lp": lp,
            "tk_ids": tk_ids, "tk_lp": tk_lp}


# Captures hold float32/int32 arrays (~60 MB for the corpus instead of ~500 MB
# of Python lists: the client may run on the UMA head node). JSON gets base64.
PACKED = ("lp", "tk_ids", "tk_lp")


def pack_capture(capture: list[dict]) -> list[dict]:
    return [{**d, **{k: base64.b64encode(d[k].tobytes()).decode() for k in PACKED}} for d in capture]


def unpack_capture(capture: list[dict]) -> list[dict]:
    out = []
    for d in capture:
        arrs = {}
        for k in PACKED:
            a = array("i" if k == "tk_ids" else "f")
            a.frombytes(base64.b64decode(d[k]))
            arrs[k] = a
        out.append({**d, **arrs})
    return out


def capture_nll(c: Client, docs: list[dict]) -> list[dict]:
    out = []
    for k, d in enumerate(docs):
        out.append(capture_doc(c, d))
        if (k + 1) % 16 == 0:
            print(f"  nll {k + 1}/{len(docs)}", flush=True)
    return out


def capture_greedy(c: Client) -> list[list[int]]:
    seqs = []
    for p in GREEDY_PROMPTS:
        out = c.chat(p, max_tokens=GREEDY_TOKENS, temperature=0, return_token_ids=True, **OFF)
        if not out.get("token_ids"):  # an empty sequence would make the hazard pass vacuously
            raise RuntimeError(f"no token_ids returned for greedy prompt {p[:40]!r}")
        seqs.append(out["token_ids"])
    return seqs


def run_count(c: Client) -> dict:
    out = c.chat("Count from 1 to 200. Output only the numbers, separated by commas, with no other text.",
                 max_tokens=1200, temperature=0, **OFF)
    return {**count_ok(out["content"]), "finish_reason": out["finish_reason"]}


def run_kwargs(c: Client, default_thinking: bool) -> dict:
    jobs = [(name, extra, default_thinking if think is None else think, stream)
            for name, extra, think in KWARG_SHAPES for stream in (False, True)]

    def one(job):
        name, extra, think, stream = job
        out = c.chat(KWARG_PROMPT, stream=stream, temperature=0, max_tokens=2048 if think else 128, **extra)
        return {"cell": f"{name}.{'stream' if stream else 'block'}", "thinking": think,
                **judge_kwarg_cell(out, think, name in REASONING_OPTIONAL)}

    cells = pmap(one, jobs)
    return {"pass": all(x["pass"] for x in cells), "passed": sum(x["pass"] for x in cells),
            "total": len(cells), "cells": cells}


def run_utf8(c: Client) -> dict:
    return judge_utf8(c.chat(UTF8_PROMPT, stream=True, temperature=0, max_tokens=3000, **OFF))


def run_tools(c: Client) -> dict:
    spec = json.loads((DATA / "tools50.json").read_text())

    def one(k_item):
        k, item = k_item
        out = c.chat(item["prompt"], stream=bool(k % 2), temperature=0, max_tokens=512, tool_choice="auto",
                     tools=[spec["tools"][t] for t in item["tools"]], **OFF)
        return judge_tool_item(item, out)

    rows = pmap(one, list(enumerate(spec["items"])))
    n = len(rows)
    return {"n": n, "json_valid": round(sum(r["json_valid"] for r in rows) / n, 4),
            "called": round(sum(r["n_calls"] > 0 for r in rows) / n, 4),
            "name_ok": round(sum(r["name_ok"] for r in rows) / n, 4),
            "args_ok": round(sum(r["args_ok"] for r in rows) / n, 4), "items": rows}


def run_needle(c: Client, lengths) -> dict:
    ratio = None
    cells = {}
    for li, length in enumerate(lengths):
        for di, depth in enumerate(DEPTHS):
            seed = 1000 * (li + 1) + di
            name, code = needle_code(seed)
            if ratio is None:
                sample = "\n\n".join(filler(1, 40))
                ratio = c.tokenize_count(sample) / len(sample)
            budget = length - 96
            n = max(4, int(budget / ratio / 420))
            paras = filler(seed, int(n * 1.5))
            for _ in range(8):  # land in [0.97, 1.0] x budget
                ntok = c.tokenize_count(needle_doc(paras[:n], depth, name, code, seed))
                if 0.97 * budget <= ntok <= budget:
                    break
                n = max(4, int(n * budget / ntok * (0.985 if ntok > budget else 1.0)))
                if n > len(paras):
                    paras = filler(seed, int(n * 1.3))
            q = (needle_doc(paras[:n], depth, name, code, seed)
                 + f"\n\nWhat is the vault code assigned to {name}? Reply with only the code.")
            out = c.chat(q, max_tokens=32, temperature=0, **OFF)
            cells[f"{length}@{depth}"] = {"found": code in out["content"].upper(), "code": code,
                                          "answer": out["content"].strip()[:60],
                                          "prompt_tokens": out["usage"].get("prompt_tokens"), "s": out["s"]}
    per_len = {str(L): sum(cells[f"{L}@{d}"]["found"] for d in DEPTHS) for L in lengths}
    return {"pass": all(v >= 2 for v in per_len.values()), "per_length": per_len, "cells": cells}


# ------------------------------------------------------------------ gates

def _crit(name: str, ok: bool, value, limit) -> dict:
    return {"name": name, "pass": bool(ok), "value": value, "limit": limit, "rule": GATES[name]}


def criteria(comps: dict, ref: dict | None, stage: str | None, skip: set) -> list[dict]:
    rows = []
    for name, comp in comps.items():
        if "error" in comp:
            rows.append({"name": f"{name}.error", "pass": False, "value": comp["error"][:300],
                         "limit": None, "rule": "component ran without an exception"})
    nll = comps.get("nll", {}).get("vs_ref")
    if nll and ref:
        sigma = ref["nll"].get("rerun", {}).get("abs_delta")
        lim = max(3 * (sigma or 0.0), 0.005)
        rows.append(_crit("nll.delta", abs(nll["delta"]) <= lim and not nll["mismatched_docs"],
                          nll["delta"], round(lim, 6)))
        r_top1 = ref["nll"].get("rerun", {}).get("top1_agree")
        if r_top1 is not None:
            rows.append(_crit("nll.top1_rerun", nll["top1_agree"] >= r_top1 - 0.005,
                              nll["top1_agree"], round(r_top1 - 0.005, 6)))
        if stage:
            rows.append(_crit("nll.top1_stage", nll["top1_agree"] >= STAGES[stage]["top1"],
                              nll["top1_agree"], STAGES[stage]["top1"]))
            rows.append(_crit("nll.kl_stage", nll["kl"] <= STAGES[stage]["kl"], nll["kl"], STAGES[stage]["kl"]))
    g = comps.get("greedy", {}).get("golden")
    if g and ref:
        lim = 2 * max(ref["greedy"]["aa"]["hazard"], 0.005)
        rows.append(_crit("greedy.hazard", g["hazard"] <= lim, g["hazard"], round(lim, 6)))
    cells = comps.get("kwargs", {}).get("cells", [])
    if cells:
        alias = [x for x in cells if x["cell"].startswith("thinking_true.")]
        core = [x for x in cells if x not in alias]
        for gate, part in (("kwargs.core", core), ("kwargs.thinking_alias", alias)):
            rows.append(_crit(gate, all(x["pass"] for x in part), sum(x["pass"] for x in part), len(part)))
    simple = {"count": ("count", "n_numbers", 200),
              "utf8": ("utf8", "fffd", 0), "vision": ("vision", "passed", None),
              "needle": ("needle", "per_length", ">=2 each")}
    for comp, (gate, key, limit) in simple.items():
        if "pass" in comps.get(comp, {}):
            rows.append(_crit(gate, comps[comp]["pass"], comps[comp].get(key), limit))
    t = comps.get("tools", {})
    if "json_valid" in t:
        rows.append(_crit("tools.json_valid", t["json_valid"] >= 0.98, t["json_valid"], 0.98))
    for r in rows:
        if r["name"] in skip:
            r["skipped"] = True
    return rows


def verdict(mode: str, ref_name: str | None, rows: list[dict]) -> str:
    gated = [r for r in rows if not r.get("skipped")]
    bad = [r["name"] for r in gated if not r["pass"]]
    where = f" vs {ref_name}" if ref_name else ""
    if bad:
        return f"FAIL tier0 {mode}{where}: {len(gated) - len(bad)}/{len(gated)} criteria; failed {', '.join(bad)}"
    return f"PASS tier0 {mode}{where}: {len(gated)}/{len(gated)} criteria"


# ------------------------------------------------------------------- main

def ref_path(name: str) -> Path:
    p = Path(name)
    return p if p.suffix == ".json" and p.exists() else EVALS_DIR / "ref" / f"{name}.json"


def run(args) -> dict:
    c = Client(args.url, args.model)
    comps: dict = {}
    ref = json.loads(ref_path(args.ref).read_text()) if args.mode == "compare" else None
    new_ref = None
    docs, sha = corpus()
    if ref and ref.get("corpus_sha256") != sha:
        raise SystemExit("reference was recorded on a different corpus.jsonl; re-record it")
    lengths = tuple(args.needle_lengths) if args.needle_lengths else NEEDLE_LENGTHS + ((NEEDLE_LONG,) if args.long else ())
    if args.mode == "record":
        new_ref = {"schema": 1, "name": args.name, "ts": utc_stamp(), "model": c.model, "url": c.url,
                   "corpus_sha256": sha, "topk": TOPK}
    runners = {
        "count": lambda: run_count(c),
        "kwargs": lambda: run_kwargs(c, args.default_thinking == "on"),
        "utf8": lambda: run_utf8(c),
        "tools": lambda: run_tools(c),
        "vision": lambda: vision.run_suite(c, video=not args.no_video),
        "needle": lambda: run_needle(c, lengths),
    }
    for name in args.only:
        t0 = time.time()
        try:
            if name == "nll":
                a = capture_nll(c, docs)
                res = {"docs": len(a), "tokens": sum(len(d["lp"]) for d in a), "mean_nll": round(mean_nll(a), 6)}
                if new_ref is not None:
                    b = capture_nll(c, docs)
                    rr = compare_captures(a, b)
                    new_ref["nll"] = {"capture": pack_capture(a), "mean_nll": res["mean_nll"],
                                      "rerun": {"abs_delta": abs(rr["delta"]), "top1_agree": rr["top1_agree"],
                                                "kl": rr["kl"], "mean_nll": rr["nll_cand"]}}
                    res["rerun"] = new_ref["nll"]["rerun"]
                else:
                    res["vs_ref"] = compare_captures(unpack_capture(ref["nll"]["capture"]), a)
            elif name == "greedy":
                a = capture_greedy(c)
                if new_ref is not None:
                    b = capture_greedy(c)
                    new_ref["greedy"] = {"runs": [a, b], "aa": hazard(list(zip(a, b)))}
                    res = {"aa": new_ref["greedy"]["aa"]}
                else:
                    res = {"golden": hazard(list(zip(a, ref["greedy"]["runs"][0]))), "run": a}
            else:
                res = runners[name]()
        except Exception as exc:  # noqa: BLE001 - reported as a failed criterion
            res = {"error": f"{type(exc).__name__}: {exc}"}
        res["s"] = round(time.time() - t0, 1)
        comps[name] = res
        brief = {k: v for k, v in res.items() if k not in ("cells", "items", "checks", "run", "by_domain")}
        print(f"[{name}] {json.dumps(brief)[:400]}", flush=True)
    if new_ref is not None and "nll" in new_ref and "greedy" in new_ref:
        path = ref_path(args.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(new_ref, separators=(",", ":")))
        print(f"reference written: {path}")
    elif new_ref is not None:
        print("no reference written: record needs both nll and greedy in --only")
    return {"comps": comps, "ref": ref}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["record", "compare"])
    ap.add_argument("--name", help="record: reference name (written to <evals>/ref/<name>.json)")
    ap.add_argument("--ref", help="compare: reference name or path")
    ap.add_argument("--stage", choices=sorted(STAGES), help="compare: apply weight-quant stage floors")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None, help="default: first id in /v1/models")
    ap.add_argument("--only", default=",".join(COMPONENTS), help=f"comma list from {COMPONENTS}")
    ap.add_argument("--skip-gate", default="", help="comma list of criteria reported but not gated")
    ap.add_argument("--long", action="store_true", help="add the 128k needle")
    ap.add_argument("--needle-lengths", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--default-thinking", choices=["off", "on"], default="off",
                    help="what the serve does with no kwargs (run.sh: off)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", type=Path, help="default <evals>/runs/<ts>-tier0-<mode>/tier0.json")
    args = ap.parse_args(argv)
    args.only = [x for x in args.only.split(",") if x]
    if set(args.only) - set(COMPONENTS):
        ap.error(f"unknown components {sorted(set(args.only) - set(COMPONENTS))}")
    if args.mode == "record" and not args.name:
        ap.error("record needs --name")
    if args.mode == "compare" and not args.ref:
        ap.error("compare needs --ref")
    t0 = time.time()
    got = run(args)
    rows = criteria(got["comps"], got["ref"], args.stage, set(filter(None, args.skip_gate.split(","))))
    ref_name = args.ref if args.mode == "compare" else None
    line = verdict(args.mode, ref_name, rows)
    for r in rows:
        tag = "SKIP" if r.get("skipped") else ("PASS" if r["pass"] else "FAIL")
        print(f"{tag} {r['name']} value={r['value']} limit={r['limit']}")
    out = args.out or EVALS_DIR / "runs" / f"{utc_stamp()}-tier0-{args.mode}" / "tier0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": 1, "mode": args.mode, "ts": utc_stamp(), "url": args.url, "ref": ref_name,
              "name": args.name, "stage": args.stage, "elapsed_s": round(time.time() - t0, 1),
              "criteria": rows, "verdict": line, "components": got["comps"]}
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(f"written: {out}")
    print(line)
    return 0 if line.startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
