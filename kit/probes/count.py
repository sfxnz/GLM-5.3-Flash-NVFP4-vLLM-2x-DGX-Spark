#!/usr/bin/env python3
# vendored from sfxnz/forge kit @ 6285f70
"""Greedy count probe. Fail unless the completion contains a long consecutive run of integers.

    kit/probes/count.py <recipe-dir> <evidence-dir>   (params: count: {need: 80, max_tokens: 512})
"""
from __future__ import annotations

import re
import sys

from _probe import Probe


def main() -> int:
    probe = Probe("count", __doc__)
    probe.parse()
    params = probe.recipe.probes().get("count") or {}
    need = int(params.get("need", 80))
    max_tokens = int(params.get("max_tokens", 512))
    code, payload = probe.post(
        {
            "model": probe.recipe.served_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Count from 1 to 200. Output only the numbers, separated by commas, with no other text.",
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": probe.recipe.chat_template_kwargs,
        }
    )
    if code != 200:
        return probe.finish(False, f"http_{code}")
    text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    nums = [int(x) for x in re.findall(r"\d+", text)]
    best = cur = 0
    prev = None
    for n in nums:
        if prev is not None and n == prev + 1:
            cur += 1
        else:
            cur = 1
        best = max(best, cur)
        prev = n
    probe.say(f"consecutive={best} need={need} nums={len(nums)}")
    return probe.finish(best >= need, f"consecutive={best}<{need}")


if __name__ == "__main__":
    sys.exit(main())
