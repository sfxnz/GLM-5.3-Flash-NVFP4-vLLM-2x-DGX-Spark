#!/usr/bin/env python3
# vendored from sfxnz/forge kit @ 6285f70
"""Thinking-off completion. Fail if content is empty or <think> leaks.

    kit/probes/thinking_off.py <recipe-dir> <evidence-dir>
"""
from __future__ import annotations

import json
import sys

from _probe import Probe


def main() -> int:
    probe = Probe("thinking_off", __doc__)
    probe.parse()
    code, payload = probe.post(
        {
            "model": probe.recipe.served_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly the word PING and nothing else.",
                }
            ],
            "max_tokens": 64,
            "temperature": 0,
            "chat_template_kwargs": probe.recipe.chat_template_kwargs,
        }
    )
    if code != 200:
        return probe.finish(False, f"http_{code}")
    message = (payload.get("choices") or [{}])[0].get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    leaked = "<think>" in content or "</think>" in content
    reasoning_leak = "<think>" in reasoning or "</think>" in reasoning
    probe.say(
        f"content_chars={len(content.strip())} leaked_think={int(leaked)} "
        f"reasoning_chars={len(reasoning)} reasoning_leak={int(reasoning_leak)}"
    )
    probe.say("SUMMARY " + json.dumps({"content": content, "reasoning_content": reasoning}))
    if not content.strip():
        return probe.finish(False, "empty_content")
    if leaked or reasoning_leak:
        return probe.finish(False, "think_leak")
    return probe.finish(True)


if __name__ == "__main__":
    sys.exit(main())
