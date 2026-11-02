"""Thinking-off table corruption A/B (evidence/r3/research/bugs.md item 1). Usage: utf8_ab.py [texts.json]"""
import json, re, sys
sys.path.insert(0, "quality")
import tier0
from common import Client
c = Client("http://127.0.0.1:8000")
NUM, CJK = re.compile(r"^[\d,]+$"), re.compile(r"^[一-鿿]+$")
def scan(text):
    rows = [l for l in text.splitlines() if l.strip().startswith("|") and "---" not in l][1:]
    prev = breaks = odd = 0
    for l in rows:
        cells = [x.strip() for x in l.strip().strip("|").split("|")]
        n = int(cells[0]) if cells[0].isdecimal() else None
        breaks += n != prev + 1
        prev = n if n is not None else prev
        odd += sum(bool(x) and not NUM.match(x) and not CJK.match(x) for x in cells[1:])
    return len(rows), breaks, odd
ARMS = {"off_max": {"chat_template_kwargs": {"enable_thinking": False}},
        "off_low": {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}},
        "on_low": {"reasoning_effort": "low"},
        "on_max": {"chat_template_kwargs": {"enable_thinking": True}}}
texts = []
for arm, kw in ARMS.items():
    clean = 0
    for i in range(6):
        out = c.chat(tier0.UTF8_PROMPT, stream=True, temperature=0, max_tokens=12000, **kw)
        rows, breaks, odd = scan(out["content"])
        ok = rows == 40 and breaks == 0 and odd == 0 and out["finish_reason"] == "stop"
        clean += ok
        texts.append({"arm": arm, "i": i, "finish": out["finish_reason"], "ok": ok, "content": out["content"], "reasoning_chars": len(out["reasoning"])})
        print(arm, i, out["finish_reason"], rows, breaks, odd, len(out["reasoning"]), "clean" if ok else "CORRUPT", flush=True)
    print("ARM", arm, f"{clean}/6 clean", flush=True)
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(json.dumps(texts, ensure_ascii=False, indent=1) + "\n")
