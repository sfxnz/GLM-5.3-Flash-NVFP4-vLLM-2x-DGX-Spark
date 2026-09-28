#!/usr/bin/env python3
"""lossless_repeat.py [N] [TF_REPEATS]: evidence/e0-nvidia-v11/scripts/lossless_probe.py on the utf8 prompt, with the
teacher-forced prefill repeated. Generates N tokens greedily (DFlash2 on), then teacher-forces prompt + output
TF_REPEATS times through /v1/completions prompt_logprobs=1 and prints, for every position where any pass says the
generated token is not the target's top-1, the generated token's rank and logprob in each pass. A miss that comes
and goes between identical prefills is prefill noise; a miss in every pass is a decode/prefill disagreement."""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "quality"))
from tier0 import UTF8_PROMPT  # noqa: E402

URL, MODEL = "http://127.0.0.1:8000", "nvidia/GLM-5.3-Flash-NVFP4"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
R = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps({"model": MODEL, **body}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def detok(ids):
    return post("/detokenize", {"tokens": ids})["prompt"]


msgs = [{"role": "user", "content": UTF8_PROMPT}]
p_ids = post("/tokenize", {"messages": msgs, "add_generation_prompt": True,
                           "chat_template_kwargs": {"enable_thinking": False}})["tokens"]
gen = post("/v1/completions", {"prompt": p_ids, "max_tokens": N, "temperature": 0, "return_token_ids": True,
                               "skip_special_tokens": False})
g_ids = gen["choices"][0]["token_ids"]
passes = []
for _ in range(R):
    plp = post("/v1/completions", {"prompt": p_ids + g_ids, "max_tokens": 1, "temperature": 0,
                                   "prompt_logprobs": 1})["choices"][0]["prompt_logprobs"]
    passes.append([plp[len(p_ids) + i] for i in range(len(g_ids))])
miss_any = sorted({i for p in passes for i, e in enumerate(p) if e[str(g_ids[i])]["rank"] != 1})
always = [i for i in miss_any if all(p[i][str(g_ids[i])]["rank"] != 1 for p in passes)]
print(f"utf8: prompt {len(p_ids)} tok, generated {len(g_ids)} tok, finish {gen['choices'][0]['finish_reason']}; "
      f"{R} teacher-forced passes; misses per pass {[sum(e[str(g_ids[i])]['rank'] != 1 for i, e in enumerate(p)) for p in passes]}; "
      f"positions missed in any pass {len(miss_any)}, in every pass {len(always)}")
for i in miss_any:
    cells = []
    for p in passes:
        e = p[i]
        top = min(e.items(), key=lambda kv: kv[1]["rank"])
        cells.append(f"r{e[str(g_ids[i])]['rank']} lp {e[str(g_ids[i])]['logprob']:.2f} (top {detok([int(top[0])])!r} "
                     f"{top[1]['logprob']:.2f})")
    print(f"  pos {i:>3} gen {detok([g_ids[i]])!r}: " + " | ".join(cells))
