"""Tier-1 answer extraction and scoring (pure functions, stdlib only)."""
from __future__ import annotations

import ast
import json
import re

from common import strip_box

NUM = r"-?\d[\d,]*(?:\.\d+)?"


# ------------------------------------------------------------------ GSM8K

def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def gsm8k_extract(text: str) -> str | None:
    """Number after the last 'Answer:', else the last number in the text."""
    text = strip_box(text)
    hits = re.findall(r"answer\s*(?:is)?\s*[:：]?\s*\**\s*\$?\s*(" + NUM + ")", text, re.I)
    hits = hits or re.findall(NUM, text)
    return hits[-1].replace(",", "").rstrip(".") if hits else None


def gsm8k_correct(text: str, gold: str) -> bool:
    got = gsm8k_extract(text)
    try:
        return got is not None and abs(float(got) - float(gold)) < 1e-6
    except ValueError:
        return False


# ------------------------------------------------------ letter answers

def mmlu_pro_extract(text: str, n_options: int = 10) -> str | None:
    """MMLU-Pro style: 'The answer is (X)', then 'Answer: X', then a lone letter."""
    text = strip_box(text)
    letters = "ABCDEFGHIJ"[:n_options]
    # a bare letter must be uppercase and whole ("answer is a bit" is not A); "(b)" may be lowercase
    letter = rf"(?:\(([{letters}{letters.lower()}])\)|([{letters}])\b)"
    for pat in (rf"(?i:answer is) {letter}", rf"(?i:answer)\s*[:：]\s*\**{letter}"):
        hits = re.findall(pat, text)
        if hits:
            return "".join(hits[-1]).upper()
    last = text.strip().splitlines()[-1].strip() if text.strip() else ""
    m = re.fullmatch(rf"\**\(?([{letters}])\)?\.?\**", last)
    return m.group(1) if m else None


def mmmu_extract(text: str, options: list[str]) -> str | None:
    """Letter as '(X)', a leading 'X.' / 'X', or 'answer is X'; else the one option text quoted."""
    text = strip_box(text).strip()
    letters = "ABCDEFGHIJ"[:len(options)]
    for pat, flags in ((rf"\(([{letters}])\)", 0), (rf"^([{letters}])(?:[.):\s]|$)", re.M),
                       (rf"answer\s*(?:is)?\s*[:：]?\s*\(?([{letters}])\b", re.I)):
        m = re.search(pat, text, flags)
        if m:
            return m.group(1).upper()
    hits = [letters[i] for i, o in enumerate(options) if o and o.lower() in text.lower()]
    return hits[0] if len(hits) == 1 else None


def mmmu_options(raw) -> list[str]:
    return raw if isinstance(raw, list) else list(ast.literal_eval(raw))


# ------------------------------------------------------------ ChartQA

def _to_float(s: str) -> float | None:
    s = s.strip().rstrip("%").replace(",", "").replace("$", "")
    try:
        return float(s)
    except ValueError:
        return None


def chartqa_correct(pred: str, labels: list[str]) -> bool:
    """Relaxed accuracy: numbers within 5% relative, otherwise exact text (case-insensitive).
    A markdown-bold span on the first line is the answer ("**13**", "**82.5** billion")."""
    pred = strip_box(pred).strip().splitlines()[0].strip() if strip_box(pred).strip() else ""
    bold = re.search(r"\*\*(.+?)\*\*", pred)
    pred = (bold.group(1) if bold else pred).strip().rstrip(".")
    for gold in labels:
        g, p = _to_float(gold), _to_float(pred)
        if g is not None and p is not None:
            if (p == g) if g == 0 else abs(p - g) / abs(g) <= 0.05:
                return True
        elif pred.lower() == gold.strip().lower():
            return True
    return False


# ----------------------------------------------------------- OCRBench

def ocrbench_correct(pred: str, answers: list[str], dataset: str) -> bool:
    """OCRBench rule: a gold answer is a substring of the prediction (lowercased).
    HME100k (handwritten formulas) also ignores whitespace."""
    pred = strip_box(pred)
    if dataset == "HME100k":
        p = re.sub(r"\s", "", pred)
        return any(re.sub(r"\s", "", a) in p for a in answers)
    p = pred.lower().replace("\n", " ").strip()
    return any(a.lower().replace("\n", " ").strip() in p for a in answers)


# --------------------------------------------------------------- BFCL

BFCL_TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string", "integer": "integer",
              "string": "string", "boolean": "boolean", "array": "array", "number": "number", "object": "object"}


def bfcl_name(name: str) -> str:
    """OpenAI tool names allow [a-zA-Z0-9_-]; BFCL maps '.' to '_' the same way."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _schema(p: dict) -> dict:
    out = {k: v for k, v in p.items() if k not in ("type", "properties", "items", "optional")}
    out["type"] = BFCL_TYPES.get(p.get("type", "string"), "string")
    if "properties" in p:
        out["properties"] = {k: _schema(v) for k, v in p["properties"].items()}
    if "items" in p:
        out["items"] = _schema(p["items"])
    if out["type"] == "array" and "items" not in out:
        out["items"] = {"type": "string"}
    return out


def bfcl_tools(functions: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": bfcl_name(f["name"]), "description": f.get("description", ""),
                                              "parameters": _schema(f["parameters"])}} for f in functions]


def _std(s: str) -> str:
    return re.sub(r"[ ,./\-_*^]", "", s).lower().replace("'", '"')


def _value_ok(got, allowed: list) -> bool:
    for want in allowed:
        if want == "":
            continue
        if isinstance(want, bool) or isinstance(got, bool):
            if got is want:
                return True
        elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if isinstance(want, int) and isinstance(got, float) and not got.is_integer():
                continue
            if abs(float(got) - float(want)) < 1e-9:
                return True
        elif isinstance(want, str) and isinstance(got, str):
            if _std(got) == _std(want):
                return True
        elif isinstance(want, list) and isinstance(got, list):
            if len(want) == len(got) and all(_value_ok(g, [w]) for g, w in zip(got, want)):
                return True
        elif isinstance(want, dict) and isinstance(got, dict):
            if _args_ok(got, want):
                return True
    return False


def _args_ok(got: dict, truth: dict) -> bool:
    for k, allowed in truth.items():
        if k not in got:
            if "" not in allowed:
                return False
        elif not _value_ok(got[k], allowed):
            return False
    return set(got) <= set(truth)


def bfcl_correct(calls: list[dict], ground_truth: list[dict]) -> tuple[bool, str]:
    """AST-style check for one expected call: name, required args, allowed values."""
    if len(calls) != len(ground_truth):
        return False, f"expected {len(ground_truth)} call(s), got {len(calls)}"
    (fname, truth), = ground_truth[0].items()
    call = calls[0]
    if call.get("name") != bfcl_name(fname):
        return False, f"wrong function {call.get('name')!r}"
    try:
        args = json.loads(call.get("arguments") or "{}")
    except ValueError:
        return False, "arguments are not JSON"
    if not isinstance(args, dict):
        return False, "arguments are not an object"
    return (True, "ok") if _args_ok(args, truth) else (False, "argument mismatch")
