#!/usr/bin/env python3
"""Tier-1 task suite (stdlib only, ~2.75 h per config at c=2).

Groups and sizes (1,010 items):
  IFEval 150 | GSM8K 200 | MMLU-Pro 280 (20 x 14 categories) | BFCL 120
  (simple 60 + multiple 60) | Vision 260 = ChartQA 100 + OCRBench 100
  (10 per question type) + MMMU 60 (validation, multiple choice, <= 4 images,
  2 per subject, topped up where datasets-server cannot serve a subject)

  python3 quality/tier1.py fetch              # download + cache (CPU/network only)
  python3 quality/tier1.py run --out A.jsonl  # score one config; resumable
  python3 quality/compare_tier1.py A.jsonl B.jsonl

Data is cached under <evals>/datasets/ (never in the repo). The item ids are
pinned in quality/data/tier1_ids.json (seed 20260925); `fetch --resample`
redraws them. Runs are greedy by default so two configs pair item by item.
Every row of the output JSONL is one item (task, id, correct 0/1, ...); the
first row is {"meta": ...}. Infrastructure errors are recorded with "error"
and retried on the next run with the same --out.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ifeval  # noqa: E402
import scorers as S  # noqa: E402
from common import DATA, EVALS_DIR, MAX_CONCURRENCY, Client, jsonl, utc_stamp  # noqa: E402

SEED = 20260925
IDS_FILE = DATA / "tier1_ids.json"
DS_DIR = EVALS_DIR / "datasets"
DS_SERVER = "https://datasets-server.huggingface.co/rows"
IFEVAL_URL = ("https://raw.githubusercontent.com/google-research/google-research/"
              "26d8ccdab6fec61b5c83ad6327ea8bda9e580288/instruction_following_eval/data/input_data.jsonl")
GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl")
BFCL_URL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/"
MMMU_SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory", "Basic_Medical_Science",
    "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science", "Design", "Diagnostics_and_Laboratory_Medicine",
    "Economics", "Electronics", "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology",
    "Public_Health", "Sociology"]
SIZES = {"ifeval": 150, "gsm8k": 200, "mmlu_pro": 280, "bfcl": 120, "chartqa": 100, "ocrbench": 100, "mmmu": 60}
GROUPS = {"ifeval": "IFEval", "gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro", "bfcl": "BFCL",
          "chartqa": "Vision", "ocrbench": "Vision", "mmmu": "Vision"}
TASKS = list(SIZES)


# ------------------------------------------------------------------ fetch

def http_get(url: str, tries: int = 12) -> bytes:
    """GET with backoff; honours Retry-After on 429 (datasets-server rate limit).
    A 5xx is retried only twice: datasets-server returns permanent 500s too."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "glm53-quality-tier1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            code = getattr(exc, "code", None)
            if k == tries - 1 or (code is not None and code != 429 and (code < 500 or k >= 2)):
                raise
            retry_after = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
            time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** k))
    raise AssertionError("unreachable")


def ds_rows(dataset: str, config: str, split: str, offset: int, length: int = 100) -> dict:
    """One datasets-server page. Text-only pages are cached on disk; pages with
    images are not, because their signed image URLs expire."""
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                "offset": offset, "length": length})
    cache = DS_DIR / "_pages" / (hashlib.sha256(q.encode()).hexdigest()[:24] + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    time.sleep(0.5)  # stay under the anonymous rate limit
    body = http_get(f"{DS_SERVER}?{q}")
    page = json.loads(body)
    if not any(f["type"].get("_type") == "Image" for f in page.get("features", [])):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(body)
    return page


def ds_all(dataset: str, config: str, split: str) -> list[dict]:
    first = ds_rows(dataset, config, split, 0)
    rows = [r["row"] for r in first["rows"]]
    while len(rows) < first["num_rows_total"]:
        rows += [r["row"] for r in ds_rows(dataset, config, split, len(rows))["rows"]]
    return rows


def ds_pick(dataset: str, config: str, split: str, indices: list[int]) -> dict[int, dict]:
    """Rows at the given indices, fetching only the 100-row pages that hold them."""
    out = {}
    for page in sorted({i // 100 for i in indices}):
        got = ds_rows(dataset, config, split, page * 100)["rows"]
        for r in got:
            if r["row_idx"] in indices:
                out[r["row_idx"]] = r["row"]
    return out


def save_image(img: dict, dest: Path) -> str:
    """Download a datasets-server image (signed URL, expires) into the cache."""
    src = img["src"]
    ext = Path(urllib.parse.urlparse(src).path).suffix or ".png"
    path = dest.with_suffix(ext)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(http_get(src))
    return str(path.relative_to(DS_DIR))


def sample(pool: list, n: int, salt: str) -> list:
    return sorted(random.Random(f"{SEED}-{salt}").sample(sorted(pool), n))


def fetch_all(resample: bool) -> dict:
    pinned = {} if resample or not IDS_FILE.exists() else json.loads(IDS_FILE.read_text())
    ids = {"seed": SEED}
    items = {}

    rows = [json.loads(x) for x in http_get(IFEVAL_URL).decode().splitlines() if x.strip()]
    by_key = {r["key"]: r for r in rows if not set(r["instruction_id_list"]) & ifeval.UNSUPPORTED}
    ids["ifeval"] = pinned.get("ifeval") or sample(list(by_key), SIZES["ifeval"], "ifeval")
    items["ifeval"] = [{"id": k, "prompt": by_key[k]["prompt"], "instruction_id_list": by_key[k]["instruction_id_list"],
                        "kwargs": by_key[k]["kwargs"]} for k in ids["ifeval"]]

    rows = [json.loads(x) for x in http_get(GSM8K_URL).decode().splitlines() if x.strip()]
    ids["gsm8k"] = pinned.get("gsm8k") or sample(range(len(rows)), SIZES["gsm8k"], "gsm8k")
    items["gsm8k"] = [{"id": i, "question": rows[i]["question"], "answer": S.gsm8k_gold(rows[i]["answer"])}
                      for i in ids["gsm8k"]]

    rows = ds_all("TIGER-Lab/MMLU-Pro", "default", "test")
    by_qid = {r["question_id"]: r for r in rows}
    if "mmlu_pro" in pinned:
        ids["mmlu_pro"] = pinned["mmlu_pro"]
    else:
        cats = sorted({r["category"] for r in rows})
        per = SIZES["mmlu_pro"] // len(cats)
        ids["mmlu_pro"] = sorted(q for c in cats for q in sample(
            [r["question_id"] for r in rows if r["category"] == c], per, f"mmlu_pro-{c}"))
    items["mmlu_pro"] = [{"id": q, "question": by_qid[q]["question"], "options": by_qid[q]["options"],
                          "answer": by_qid[q]["answer"], "category": by_qid[q]["category"]} for q in ids["mmlu_pro"]]

    bfcl = {}
    for cat in ("simple", "multiple"):
        qs = {j["id"]: j for j in map(json.loads, http_get(f"{BFCL_URL}BFCL_v3_{cat}.json").decode().splitlines())}
        gts = {j["id"]: j for j in map(json.loads, http_get(
            f"{BFCL_URL}possible_answer/BFCL_v3_{cat}.json").decode().splitlines())}
        bfcl.update({k: {"id": k, "category": cat, "messages": qs[k]["question"][0], "functions": qs[k]["function"],
                         "ground_truth": gts[k]["ground_truth"]} for k in qs})
    if "bfcl" in pinned:
        ids["bfcl"] = pinned["bfcl"]
    else:
        half = SIZES["bfcl"] // 2
        ids["bfcl"] = sample([k for k in bfcl if k.startswith("simple_")], half, "bfcl-simple") + sample(
            [k for k in bfcl if k.startswith("multiple_")], half, "bfcl-multiple")
    items["bfcl"] = [bfcl[k] for k in ids["bfcl"]]

    total = ds_rows("HuggingFaceM4/ChartQA", "default", "test", 0, 1)["num_rows_total"]
    ids["chartqa"] = pinned.get("chartqa") or sample(range(total), SIZES["chartqa"], "chartqa")
    got = ds_pick("HuggingFaceM4/ChartQA", "default", "test", ids["chartqa"])
    items["chartqa"] = [{"id": i, "query": got[i]["query"], "labels": got[i]["label"],
                         "image": save_image(got[i]["image"], DS_DIR / "chartqa/images" / str(i))}
                        for i in ids["chartqa"]]

    rows = ds_all("echo840/OCRBench", "default", "test")
    if "ocrbench" in pinned:
        ids["ocrbench"] = pinned["ocrbench"]
    else:
        types = sorted({r["question_type"] for r in rows})
        per = SIZES["ocrbench"] // len(types)
        ids["ocrbench"] = sorted(i for t in types for i in sample(
            [i for i, r in enumerate(rows) if r["question_type"] == t], per, f"ocrbench-{t}"))
    items["ocrbench"] = [{"id": i, "question": rows[i]["question"], "answers": rows[i]["answer"],
                          "dataset": rows[i]["dataset"], "question_type": rows[i]["question_type"],
                          "image": save_image(rows[i]["image"], DS_DIR / "ocrbench/images" / str(i))}
                         for i in ids["ocrbench"]]

    mmmu, pool, unavailable = {}, {}, []
    for subj in MMMU_SUBJECTS:
        try:
            rows = ds_all("MMMU/MMMU", subj, "validation")
        except urllib.error.HTTPError as exc:  # some subjects 500 permanently on datasets-server
            unavailable.append(f"{subj} (HTTP {exc.code})")
            continue
        for r in rows:
            imgs = [f"image_{k}" for k in range(1, 8) if r.get(f"image_{k}")]
            if r["question_type"] == "multiple-choice" and 1 <= len(imgs) <= 4:
                mmmu[r["id"]] = (r, imgs)
                pool.setdefault(subj, []).append(r["id"])
    if "mmmu" in pinned:
        ids["mmmu"] = pinned["mmmu"]
        missing = [i for i in ids["mmmu"] if i not in mmmu]
        if missing:
            raise SystemExit(f"pinned MMMU items unavailable now: {missing[:5]}; unavailable subjects {unavailable}")
    else:
        # 2 per subject that datasets-server serves, topped up from the rest.
        per = SIZES["mmmu"] // len(MMMU_SUBJECTS)
        picked = [i for s in sorted(pool) for i in sample(pool[s], min(per, len(pool[s])), f"mmmu-{s}")]
        rest = [i for s in sorted(pool) for i in pool[s] if i not in set(picked)]
        ids["mmmu"] = sorted(picked + sample(rest, SIZES["mmmu"] - len(picked), "mmmu-topup"))
    items["mmmu"] = []
    for i in ids["mmmu"]:
        r, imgs = mmmu[i]
        items["mmmu"].append({"id": i, "question": r["question"], "options": S.mmmu_options(r["options"]),
                              "answer": r["answer"], "subject": i.split("_", 1)[1].rsplit("_", 1)[0],
                              "images": [save_image(r[k], DS_DIR / "mmmu/images" / f"{i}-{k}") for k in imgs]})

    counts = {}
    for task, rows in items.items():
        assert len(rows) == SIZES[task], (task, len(rows))
        d = DS_DIR / task
        d.mkdir(parents=True, exist_ok=True)
        (d / "items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        counts[task] = len(rows)
    if not IDS_FILE.exists() or resample:
        IDS_FILE.write_text(json.dumps(ids, indent=0) + "\n")
    manifest = {"ts": utc_stamp(), "seed": SEED, "counts": counts, "total": sum(counts.values()),
                "mmmu_unavailable_subjects": unavailable,
                "ids_sha256": hashlib.sha256(IDS_FILE.read_bytes()).hexdigest()}
    (DS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


# -------------------------------------------------------------------- run

def img_part(rel: str) -> dict:
    path = DS_DIR / rel
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        path.suffix.lower(), "image/png")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"}}


GSM_SUFFIX = "\n\nSolve the problem step by step. End with a final line of the form 'Answer: <number>'."


def build(task: str, it: dict) -> tuple[list, dict, int]:
    """(messages, extra request fields, max_tokens) for one item."""
    user = lambda content: [{"role": "user", "content": content}]  # noqa: E731
    if task == "ifeval":
        return user(it["prompt"]), {}, 2048
    if task == "gsm8k":
        return user(it["question"] + GSM_SUFFIX), {}, 1024
    if task == "mmlu_pro":
        opts = "\n".join(f"{'ABCDEFGHIJ'[k]}. {o}" for k, o in enumerate(it["options"]))
        q = (f"The following is a multiple choice question about {it['category']}. Think step by step and then "
             f'finish your answer with "The answer is (X)" where X is the correct letter choice.\n\n'
             f"Question: {it['question']}\nOptions:\n{opts}")
        return user(q), {}, 2048
    if task == "bfcl":
        return it["messages"], {"tools": S.bfcl_tools(it["functions"]), "tool_choice": "auto"}, 512
    if task == "chartqa":
        return user([img_part(it["image"]), {"type": "text", "text": it["query"]
                     + "\nAnswer the question with a single word or number."}]), {}, 64
    if task == "ocrbench":
        return user([img_part(it["image"]), {"type": "text", "text": it["question"] + "\nAnswer concisely."}]), {}, 128
    if task == "mmmu":
        opts = "\n".join(f"{'ABCDEFGHIJ'[k]}. {o}" for k, o in enumerate(it["options"]))
        text = (f"{it['question']}\n\n{opts}\n\nAnswer with the option's letter from the given choices directly.")
        return user([img_part(p) for p in it["images"]] + [{"type": "text", "text": text}]), {}, 64
    raise KeyError(task)


def score(task: str, it: dict, out: dict) -> tuple[int, object, object]:
    """(correct, prediction, gold)."""
    text = out["content"]
    if task == "ifeval":
        r = ifeval.score(it, text)
        return r["correct"], r["instructions"], it["instruction_id_list"]
    if task == "gsm8k":
        return int(S.gsm8k_correct(text, it["answer"])), S.gsm8k_extract(text), it["answer"]
    if task == "mmlu_pro":
        got = S.mmlu_pro_extract(text, len(it["options"]))
        return int(got == it["answer"]), got, it["answer"]
    if task == "bfcl":
        ok, why = S.bfcl_correct(out["tool_calls"], it["ground_truth"])
        return int(ok), {"calls": out["tool_calls"], "why": why}, it["ground_truth"]
    if task == "chartqa":
        return int(S.chartqa_correct(text, it["labels"])), text.strip()[:80], it["labels"]
    if task == "ocrbench":
        return int(S.ocrbench_correct(text, it["answers"], it["dataset"])), text.strip()[:120], it["answers"]
    if task == "mmmu":
        got = S.mmmu_extract(text, it["options"])
        return int(got == it["answer"]), got, it["answer"]
    raise KeyError(task)


def load_items(task: str) -> list[dict]:
    path = DS_DIR / task / "items.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing: run `python3 quality/tier1.py fetch` first")
    return jsonl(path)


def run_one(c: Client, task: str, idx: int, it: dict, cfg: dict) -> dict:
    messages, extra, max_tokens = build(task, it)
    body = {**extra, "max_tokens": max_tokens + cfg["think_budget"], "temperature": cfg["temperature"]}
    if cfg["temperature"] > 0:
        body["seed"] = cfg["seed"] + idx
    if cfg["chat_kwargs"] is not None:
        body["chat_template_kwargs"] = cfg["chat_kwargs"]
    if cfg["reasoning_effort"]:
        body["reasoning_effort"] = cfg["reasoning_effort"]
    row = {"task": task, "group": GROUPS[task], "id": it["id"]}
    try:
        out = c.chat(messages, **body)
    except Exception as exc:  # noqa: BLE001 - infrastructure failure, retried on resume
        return {**row, "correct": 0, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    correct, pred, gold = score(task, it, out)
    return {**row, "correct": correct, "pred": pred, "gold": gold, "finish_reason": out["finish_reason"],
            "completion_tokens": out["usage"].get("completion_tokens"),
            "prompt_tokens": out["usage"].get("prompt_tokens"), "reasoning_chars": len(out["reasoning"]),
            "s": out["s"]}


def run(args) -> int:
    c = Client(args.url, args.model)
    cfg = {"temperature": args.temperature, "seed": args.seed, "think_budget": args.think_budget,
           "chat_kwargs": json.loads(args.chat_kwargs) if args.chat_kwargs else None,
           "reasoning_effort": args.reasoning_effort}
    meta = {"meta": {**cfg, "model": c.model, "url": c.url, "ids_sha256": hashlib.sha256(
        IDS_FILE.read_bytes()).hexdigest(), "label": args.label}}
    done = set()
    if args.out.exists():
        old = jsonl(args.out)
        old_meta = next((r["meta"] for r in old if "meta" in r), None)
        keep = [*cfg, "model", "ids_sha256"]
        if old_meta and {k: old_meta.get(k) for k in keep} != {k: meta["meta"][k] for k in keep}:
            raise SystemExit(f"{args.out} was written with a different config {old_meta}; use another --out")
        done = {(r["task"], str(r["id"])) for r in old if "task" in r and "error" not in r}
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(meta) + "\n")
    jobs = []
    for task in args.tasks:
        items = load_items(task)[: args.limit or None]
        jobs += [(task, k, it) for k, it in enumerate(items) if (task, str(it["id"])) not in done]
    print(f"{len(jobs)} items to run ({len(done)} already done)", flush=True)
    lock, t0, n = threading.Lock(), time.time(), 0
    with args.out.open("a") as f, cf.ThreadPoolExecutor(max_workers=min(args.concurrency, MAX_CONCURRENCY)) as ex:
        for fut in cf.as_completed([ex.submit(run_one, c, t, k, it, cfg) for t, k, it in jobs]):
            row = fut.result()
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 25 == 0:
                    print(f"  {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    summary = summarize(jsonl(args.out))
    print(json.dumps(summary, indent=1))
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0 if summary["errors"] == 0 else 1


def summarize(rows: list[dict]) -> dict:
    last = {}
    for r in rows:
        if "task" in r:
            last[(r["task"], str(r["id"]))] = r
    per = {}
    for r in last.values():
        p = per.setdefault(r["task"], {"n": 0, "correct": 0, "errors": 0})
        p["n"] += 1
        p["correct"] += r["correct"]
        p["errors"] += "error" in r
    for p in per.values():
        p["acc"] = round(p["correct"] / p["n"], 4)
    n = sum(p["n"] for p in per.values())
    return {"items": n, "acc": round(sum(p["correct"] for p in per.values()) / max(1, n), 4),
            "errors": sum(p["errors"] for p in per.values()), "tasks": per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download and cache the datasets")
    f.add_argument("--resample", action="store_true", help="redraw ids and rewrite quality/data/tier1_ids.json")
    r = sub.add_parser("run", help="score one serve config")
    r.add_argument("--out", type=Path, required=True, help="per-item JSONL (resumable)")
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--model", default=None)
    r.add_argument("--tasks", default=",".join(TASKS))
    r.add_argument("--limit", type=int, default=0, help="first N items per task (smoke only)")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--seed", type=int, default=SEED, help="per-item seed base when temperature > 0")
    r.add_argument("--chat-kwargs", default=None, help='JSON, e.g. {"enable_thinking": false}; default: serve default')
    r.add_argument("--reasoning-effort", default=None, choices=["none", "low", "high", "max"])
    r.add_argument("--think-budget", type=int, default=0, help="extra max_tokens per item when thinking is on")
    r.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    r.add_argument("--label", default="")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        print(json.dumps(fetch_all(args.resample), indent=1))
        return 0
    args.tasks = [t for t in args.tasks.split(",") if t]
    if set(args.tasks) - set(TASKS):
        ap.error(f"unknown tasks {sorted(set(args.tasks) - set(TASKS))}")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
