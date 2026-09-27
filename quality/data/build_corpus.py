#!/usr/bin/env python3
"""Rebuild quality/data/corpus.jsonl (the frozen Tier-0 NLL corpus).

You never need to run this to use Tier 0. It records exactly how the vendored
corpus was cut, so the file can be audited or regenerated. Sources and
licenses are in SOURCES.md. Needs network and the `tokenizers` wheel (only
here, never at eval time):

  pip install --target /tmp/tok tokenizers
  PYTHONPATH=/tmp/tok python3 quality/data/build_corpus.py \
      --tokenizer ~/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/09b04e5e74bca08ca8549fc736d4cdd8624bfde3/tokenizer.json

Each document is cut to at most MAX_TOKENS GLM tokens (no special tokens) at
a line or word boundary. The eval prepends [gMASK]<sop>.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_TOKENS = 2000
UA = {"User-Agent": "glm53-quality-corpus-builder"}

# (eBook number, language, title, fractions of the book body to start at)
PROSE = [
    (1342, "en", "Pride and Prejudice", (0.25, 0.5, 0.75)),
    (2701, "en", "Moby Dick", (0.25, 0.5, 0.75)),
    (1661, "en", "The Adventures of Sherlock Holmes", (0.25, 0.5, 0.75)),
    (84, "en", "Frankenstein", (0.25, 0.5, 0.75)),
    (98, "en", "A Tale of Two Cities", (0.25, 0.5, 0.75)),
    (345, "en", "Dracula", (0.25, 0.5, 0.75)),
    (3300, "en", "The Wealth of Nations", (0.25, 0.5, 0.75)),
    (2009, "en", "On the Origin of Species", (0.25, 0.5, 0.75)),
    (205, "en", "Walden", (0.25, 0.5, 0.75)),
    (1404, "en", "The Federalist Papers", (0.25, 0.5, 0.75)),
    (145, "en", "Middlemarch", (0.25, 0.5, 0.75)),
    (408, "en", "The Souls of Black Folk", (0.25, 0.5, 0.75)),
]
MULTILINGUAL = [
    (14155, "fr", "Madame Bovary", (0.3, 0.7)),
    (17489, "fr", "Les misérables, Tome I", (0.5,)),
    (22367, "de", "Die Verwandlung", (0.3, 0.7)),
    (2229, "de", "Faust I", (0.5,)),
    (2000, "es", "Don Quijote", (0.25, 0.5, 0.75)),
    (45334, "it", "I promessi sposi", (0.3, 0.7)),
    (55752, "pt", "Dom Casmurro", (0.3, 0.7)),
    (11940, "fi", "Seitsemän veljestä", (0.5,)),
    (11024, "nl", "Max Havelaar", (0.5,)),
    (24264, "zh", "紅樓夢", (0.3, 0.7)),
    (23962, "zh", "西遊記", (0.3, 0.7)),
    (23950, "zh", "三國志演義", (0.5,)),
    (1982, "ja", "羅生門", (0.1,)),
]
# (url, license, start fraction). Pinned tags so the bytes never change.
CODE = [
    ("https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/heapq.py", "PSF-2.0", 0.1),
    ("https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/bisect.py", "PSF-2.0", 0.0),
    ("https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/textwrap.py", "PSF-2.0", 0.1),
    ("https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/fractions.py", "PSF-2.0", 0.3),
    ("https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/statistics.py", "PSF-2.0", 0.4),
    ("https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/functools.py", "PSF-2.0", 0.2),
    ("https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/json/decoder.py", "PSF-2.0", 0.0),
    ("https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/shlex.py", "PSF-2.0", 0.1),
    ("https://raw.githubusercontent.com/golang/go/go1.22.0/src/sort/sort.go", "BSD-3-Clause", 0.1),
    ("https://raw.githubusercontent.com/golang/go/go1.22.0/src/container/heap/heap.go", "BSD-3-Clause", 0.0),
    ("https://raw.githubusercontent.com/golang/go/go1.22.0/src/strings/builder.go", "BSD-3-Clause", 0.0),
    ("https://raw.githubusercontent.com/golang/go/go1.22.0/src/bufio/scan.go", "BSD-3-Clause", 0.1),
    ("https://raw.githubusercontent.com/sqlite/sqlite/version-3.45.0/src/func.c", "Public domain", 0.2),
    ("https://raw.githubusercontent.com/sqlite/sqlite/version-3.45.0/src/func.c", "Public domain", 0.6),
    ("https://raw.githubusercontent.com/sqlite/sqlite/version-3.45.0/src/util.c", "Public domain", 0.3),
    ("https://raw.githubusercontent.com/lodash/lodash/4.17.21/lodash.js", "MIT", 0.3),
    ("https://raw.githubusercontent.com/lodash/lodash/4.17.21/lodash.js", "MIT", 0.6),
    ("https://raw.githubusercontent.com/rust-lang/rust/1.77.0/library/core/src/iter/adapters/zip.rs", "MIT OR Apache-2.0", 0.1),
    ("https://raw.githubusercontent.com/rust-lang/rust/1.77.0/library/alloc/src/collections/binary_heap/mod.rs", "MIT OR Apache-2.0", 0.3),
    ("https://raw.githubusercontent.com/rust-lang/rust/1.77.0/library/core/src/str/pattern.rs", "MIT OR Apache-2.0", 0.2),
]
GSM8K_TRAIN = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl"
MATH_CONFIGS = ("algebra", "number_theory", "counting_and_probability", "geometry",
                "intermediate_algebra", "precalculus", "prealgebra")
SPECIAL = ("[gMASK]", "<sop>", "<|user|>", "<|assistant|>", "<|system|>", "<|observation|>",
           "<think>", "</think>", "<|endoftext|>", "<|image|>")


def fetch(url: str, cache: Path) -> bytes:
    path = cache / urllib.parse.quote(url, safe="")
    if not path.exists():
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def gutenberg_body(raw: str) -> str:
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", raw)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", raw)
    return raw[start.end() if start else 0:end.start() if end else len(raw)]


def unwrap(text: str, cjk: bool) -> str:
    """Join hard-wrapped lines inside paragraphs; keep paragraph breaks."""
    paras = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    joiner = "" if cjk else " "
    out = []
    for p in paras:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if lines:
            out.append(joiner.join(lines))
    return "\n\n".join(out)


class Cutter:
    def __init__(self, tokenizer_json: Path):
        from tokenizers import Tokenizer  # build-time only
        self.tok = Tokenizer.from_file(str(tokenizer_json))

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def cut(self, text: str, boundary: str) -> tuple[str, int]:
        """Largest prefix <= MAX_TOKENS ending at a boundary char."""
        enc = self.tok.encode(text, add_special_tokens=False)
        if len(enc.ids) <= MAX_TOKENS:
            return text, len(enc.ids)
        end = enc.offsets[MAX_TOKENS - 1][1]
        cut = text[:end]
        pos = max(cut.rfind(b) for b in boundary)
        if pos > len(cut) // 2:
            cut = cut[:pos + 1]
        cut = cut.rstrip()
        n = self.count(cut)
        while n > MAX_TOKENS:  # boundary effects: trim a line/word at a time
            cut = cut[:max(cut.rfind(b) for b in boundary)].rstrip()
            n = self.count(cut)
        return cut, n


def passage(body: str, frac: float, chars: int = 16000) -> str:
    """Text from the first paragraph start at or after frac of the body."""
    i = int(len(body) * frac)
    j = body.find("\n\n", i)
    j = i if j < 0 else j + 2
    return body[j:j + chars]


def book_docs(cutter: Cutter, cache: Path, books, domain: str) -> list[dict]:
    docs = []
    for num, lang, title, fracs in books:
        raw = fetch(f"https://www.gutenberg.org/cache/epub/{num}/pg{num}.txt", cache).decode("utf-8")
        cjk = lang in ("zh", "ja")
        body = unwrap(gutenberg_body(raw), cjk)
        for k, frac in enumerate(fracs):
            text, n = cutter.cut(passage(body, frac, 8000 if cjk else 16000),
                                 "。！？\n" if cjk else " \n")
            docs.append({"id": f"{domain}-pg{num}-{k}", "domain": domain, "lang": lang,
                         "source": f"Project Gutenberg eBook #{num}", "title": title,
                         "license": "Public domain (US)", "frac": frac, "tokens": n, "text": text})
    return docs


def code_docs(cutter: Cutter, cache: Path) -> list[dict]:
    docs = []
    for k, (url, lic, frac) in enumerate(CODE):
        src = fetch(url, cache).decode("utf-8")
        i = src.find("\n", int(len(src) * frac)) + 1 if frac else 0
        text, n = cutter.cut(src[i:i + 12000], "\n")
        docs.append({"id": f"code-{k:02d}-{url.rsplit('/', 1)[1]}", "domain": "code", "lang": "code",
                     "source": url, "license": lic, "frac": frac, "tokens": n, "text": text})
    return docs


def gsm8k_rows(cache: Path) -> list[dict]:
    rows = [json.loads(x) for x in fetch(GSM8K_TRAIN, cache).decode().splitlines() if x.strip()]
    for r in rows:
        r["answer"] = re.sub(r"<<[^>]*>>", "", r["answer"])
    return rows


def math_rows(cache: Path) -> list[dict]:
    rows = []
    for cfg in MATH_CONFIGS:
        url = ("https://datasets-server.huggingface.co/rows?dataset=EleutherAI/hendrycks_math"
               f"&config={cfg}&split=train&offset=0&length=40")
        rows += [r["row"] for r in json.loads(fetch(url, cache))["rows"]]
    return rows


def pack(cutter: Cutter, chunks: list[str], prefix: str, meta: dict, n_docs: int) -> list[dict]:
    """Greedily pack consecutive chunks into n_docs documents of <= MAX_TOKENS."""
    docs, cur, it = [], "", iter(chunks)
    for chunk in it:
        trial = cur + chunk
        if cur and cutter.count(trial) > MAX_TOKENS:
            docs.append(cur.rstrip())
            if len(docs) == n_docs:
                break
            cur = chunk
        else:
            cur = trial
    out = []
    for k, text in enumerate(docs):
        out.append({"id": f"{prefix}-{k}", **meta, "tokens": cutter.count(text), "text": text})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path.home() / "projects/data/glm53-evals/corpus-src")
    ap.add_argument("--out", type=Path, default=HERE / "corpus.jsonl")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    cutter = Cutter(args.tokenizer)

    docs = book_docs(cutter, args.cache, PROSE, "prose")
    docs += code_docs(cutter, args.cache)
    gsm = gsm8k_rows(args.cache)
    docs += pack(cutter, [f"Problem: {r['question']}\nSolution: {r['answer'].replace('####', 'Answer:')}\n\n"
                          for r in gsm[:400]], "math-gsm8k-train",
                 {"domain": "math", "lang": "en", "source": GSM8K_TRAIN, "license": "MIT"}, 5)
    docs += pack(cutter, [f"Problem: {r['problem']}\nSolution: {r['solution']}\n\n" for r in math_rows(args.cache)],
                 "math-hendrycks-train",
                 {"domain": "math", "lang": "en", "source": "EleutherAI/hendrycks_math (train)", "license": "MIT"}, 8)
    # Chat transcripts in the GLM template shape, reasoning inside <think>.
    turns = [f"<|user|>{r['question']}<|assistant|><think>{r['answer'].split('####')[0].strip()}</think>"
             f"The answer is {r['answer'].split('####')[1].strip()}." for r in gsm[1000:1400]]
    docs += pack(cutter, turns, "chat-gsm8k-train",
                 {"domain": "chat", "lang": "en", "source": GSM8K_TRAIN, "license": "MIT"}, 6)
    docs += book_docs(cutter, args.cache, MULTILINGUAL, "multilingual")

    for d in docs:
        bad = [s for s in SPECIAL if s in d["text"]]
        if bad and not (d["domain"] == "chat" and set(bad) <= {"<|user|>", "<|assistant|>", "<think>", "</think>"}):
            raise SystemExit(f"{d['id']}: special token text {bad}")
    with args.out.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    by = {}
    for d in docs:
        by.setdefault(d["domain"], [0, 0])
        by[d["domain"]][0] += 1
        by[d["domain"]][1] += d["tokens"]
    print(json.dumps({"docs": len(docs), "tokens": sum(d["tokens"] for d in docs), "by_domain": by,
                      "bytes": args.out.stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
