#!/usr/bin/env python3
# vendored from sfxnz/forge kit @ 6285f70
"""Hermes-style tools loop: user -> parsed tool_calls -> role=tool -> assistant content, no think leak.

    kit/probes/hermes_two_turn.py <recipe-dir> <evidence-dir>
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
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def main() -> int:
    probe = Probe("hermes_two_turn", __doc__)
    probe.parse()
    model = probe.recipe.served_name
    kwargs = probe.recipe.chat_template_kwargs

    turn1 = {
        "model": model,
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
        "chat_template_kwargs": kwargs,
    }
    code1, payload1 = probe.post(turn1, label="turn1")
    msg1 = (payload1.get("choices") or [{}])[0].get("message") or {}
    tool_calls = msg1.get("tool_calls") or []
    names = [(tc.get("function") or {}).get("name") or tc.get("name") for tc in tool_calls]
    probe.say(f"turn1 http={code1} n_tool_calls={len(tool_calls)} names={names}")
    if code1 != 200 or not tool_calls or "get_weather" not in names:
        return probe.finish(False, "turn1_no_get_weather_call")

    tc0 = tool_calls[0]
    tool_call_id = tc0.get("id") or "call_weather"
    turn2 = {
        "model": model,
        "messages": [
            turn1["messages"][0],
            {
                "role": "assistant",
                "content": msg1.get("content") or "",
                "tool_calls": tool_calls,
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"city": "Wellington", "temp_c": 12, "sky": "overcast"}),
            },
        ],
        "tools": [WEATHER_TOOL],
        "max_tokens": 128,
        "temperature": 0,
        "chat_template_kwargs": kwargs,
    }
    code2, payload2 = probe.post(turn2, label="turn2")
    msg2 = (payload2.get("choices") or [{}])[0].get("message") or {}
    content2 = (msg2.get("content") or "").strip()
    leaked = "<think>" in content2 or "</think>" in content2
    probe.say(f"turn2 http={code2} content_chars={len(content2)} leaked_think={int(leaked)}")
    probe.say("SUMMARY " + json.dumps({"turn1_names": names, "turn2_content": content2, "http": [code1, code2]}))
    if code2 != 200 or not content2 or leaked:
        return probe.finish(False, f"turn2 http={code2} content_chars={len(content2)} leaked_think={int(leaked)}")
    return probe.finish(True)


if __name__ == "__main__":
    sys.exit(main())
