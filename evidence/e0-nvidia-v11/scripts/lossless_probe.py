#!/usr/bin/env python3
"""Is spec decoding lossless on this serve? Generate greedily (DFlash2 on), then teacher-force
prompt+output through /v1/completions prompt_logprobs and count generated tokens that are not
the target's top-1. usage: lossless_probe.py [N_TOKENS]"""
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path("/home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia")
sys.path.insert(0, str(REPO / "quality"))
from tier0 import UTF8_PROMPT, GREEDY_PROMPTS  # noqa: E402

URL, MODEL = "http://127.0.0.1:8000", "nvidia/GLM-5.3-Flash-NVFP4"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps({"model": MODEL, **body}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def detok(ids):
    return post("/detokenize", {"tokens": ids})["prompt"]


for name, q in (("utf8", UTF8_PROMPT), ("greedy16", GREEDY_PROMPTS[16]), ("greedy0", GREEDY_PROMPTS[0])):
    msgs = [{"role": "user", "content": q}]
    kw = {"enable_thinking": False}
    p_ids = post("/tokenize", {"messages": msgs, "add_generation_prompt": True, "chat_template_kwargs": kw})["tokens"]
    # generate from the same token ids through /v1/completions so both passes see identical prompts
    gen = post("/v1/completions", {"prompt": p_ids, "max_tokens": N, "temperature": 0, "return_token_ids": True,
                                   "skip_special_tokens": False})
    g_ids = gen["choices"][0]["token_ids"]
    tf = post("/v1/completions", {"prompt": p_ids + g_ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})
    plp = tf["choices"][0]["prompt_logprobs"]
    miss = []
    for i, tid in enumerate(g_ids):
        e = plp[len(p_ids) + i]
        rank = e[str(tid)]["rank"]
        if rank != 1:
            top = min(e.items(), key=lambda kv: kv[1]["rank"])
            miss.append((i, rank, round(e[str(tid)]["logprob"], 3), int(top[0]), round(top[1]["logprob"], 3)))
    print(f"== {name}: prompt {len(p_ids)} tok, generated {len(g_ids)} tok, finish {gen['choices'][0]['finish_reason']}; "
          f"not target top-1 under teacher forcing: {len(miss)} ({100 * len(miss) / max(1, len(g_ids)):.1f}%)")
    for i, rank, lp, top_id, top_lp in miss[:12]:
        print(f"   pos {i}: gen {detok([g_ids[i]])!r} rank {rank} lp {lp} | top {detok([top_id])!r} lp {top_lp} "
              f"| context ...{detok(g_ids[max(0, i - 8):i])!r}")
    print("   text head:", repr(detok(g_ids[:120])))
