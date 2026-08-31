#!/usr/bin/env python3
"""OpenAI tools request. Fail unless the model returns a parsed tool call."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

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
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="LibertAIDAI/GLM-5.3-Flash-NVFP4")
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()
    body = json.dumps(
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Wellington right now? Use the get_weather tool.",
                }
            ],
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        args.url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    names = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        names.append(fn.get("name") or tc.get("name") or "")
    finish = choice.get("finish_reason")
    content = message.get("content") or ""
    print(
        f"n_tool_calls={len(tool_calls)} names={names} finish_reason={finish} "
        f"content_chars={len(content)}"
    )
    print("SUMMARY", json.dumps({"tool_calls": tool_calls, "finish_reason": finish}))
    if not tool_calls:
        print(json.dumps(message, indent=2)[:1500], file=sys.stderr)
        return 1
    if "get_weather" not in names:
        print(f"unexpected tool names {names}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
