#!/usr/bin/env python3
"""Hermes-style tools loop: user -> parsed tool_calls -> role=tool -> assistant content."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

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


def post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw[:1500]}
        return exc.code, payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--model", default="LibertAIDAI/GLM-5.3-Flash-NVFP4")
    p.add_argument("--dump-dir", default="")
    args = p.parse_args()

    turn1 = {
        "model": args.model,
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
        "chat_template_kwargs": {"enable_thinking": False},
    }
    code1, payload1 = post(args.url, turn1)
    msg1 = ((payload1.get("choices") or [{}])[0].get("message") or {})
    tool_calls = msg1.get("tool_calls") or []
    names = [(tc.get("function") or {}).get("name") or tc.get("name") for tc in tool_calls]
    print(f"turn1 http={code1} n_tool_calls={len(tool_calls)} names={names}")
    if args.dump_dir:
        d = Path(args.dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "turn1-request.json").write_text(json.dumps(turn1, indent=2) + "\n")
        (d / "turn1-response.json").write_text(json.dumps(payload1, indent=2) + "\n")
        (d / "turn1-http.txt").write_text(str(code1) + "\n")
    if code1 != 200 or not tool_calls or "get_weather" not in names:
        print(json.dumps(msg1, indent=2)[:1500], file=sys.stderr)
        return 1

    tc0 = tool_calls[0]
    tool_call_id = tc0.get("id") or "call_weather"
    turn2 = {
        "model": args.model,
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
        "chat_template_kwargs": {"enable_thinking": False},
    }
    code2, payload2 = post(args.url, turn2)
    msg2 = ((payload2.get("choices") or [{}])[0].get("message") or {})
    content2 = (msg2.get("content") or "").strip()
    leaked = "<think>" in content2 or "</think>" in content2
    print(
        f"turn2 http={code2} content_chars={len(content2)} leaked_think={int(leaked)}"
    )
    if args.dump_dir:
        d = Path(args.dump_dir)
        (d / "turn2-request.json").write_text(json.dumps(turn2, indent=2) + "\n")
        (d / "turn2-response.json").write_text(json.dumps(payload2, indent=2) + "\n")
        (d / "turn2-http.txt").write_text(str(code2) + "\n")
    print("SUMMARY", json.dumps({"turn1_names": names, "turn2_content": content2, "http": [code1, code2]}))
    if code2 != 200 or not content2 or leaked:
        print(json.dumps(msg2, indent=2)[:1500], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
