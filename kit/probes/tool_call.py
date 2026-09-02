#!/usr/bin/env python3
# vendored from sfxnz/forge kit @ 6f40808
"""OpenAI tools request. Fail unless the model returns a parsed get_weather tool call.

    kit/probes/tool_call.py <recipe-dir> <evidence-dir>
"""
from __future__ import annotations

import json
import sys

from _probe import Probe

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def main() -> int:
    probe = Probe("tool_call", __doc__)
    probe.parse()
    code, payload = probe.post(
        {
            "model": probe.recipe.served_name,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Wellington right now? Use the get_weather tool.",
                }
            ],
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
            "max_tokens": 256,
            "temperature": 0,
            "chat_template_kwargs": probe.recipe.chat_template_kwargs,
        }
    )
    if code != 200:
        return probe.finish(False, f"http_{code}")
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    names = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        names.append(fn.get("name") or tc.get("name") or "")
    finish = choice.get("finish_reason")
    content = message.get("content") or ""
    probe.say(
        f"n_tool_calls={len(tool_calls)} names={names} finish_reason={finish} "
        f"content_chars={len(content)}"
    )
    probe.say("SUMMARY " + json.dumps({"tool_calls": tool_calls, "finish_reason": finish}))
    if not tool_calls:
        return probe.finish(False, "no_tool_calls")
    if "get_weather" not in names:
        return probe.finish(False, f"unexpected_tool_names={names}")
    return probe.finish(True)


if __name__ == "__main__":
    sys.exit(main())
