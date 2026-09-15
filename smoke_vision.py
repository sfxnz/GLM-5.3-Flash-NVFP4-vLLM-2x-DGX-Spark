#!/usr/bin/env python3
"""Live OpenAI-compat vision smoke against a running serve.

Hardcoded 1×1 JPEG so clone-shape CI does not need PIL. The gate is
HTTP not 400 "is not a multimodal model" plus non-empty content, not
a color label.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# Minimal valid 1×1 JFIF. Color is not asserted.
_RED_JPEG_B64 = (
    "/9j/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwsN"
    "DQ0MDQwMDAwMDAwMDAwMDAwMDAz/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAA"
    "AAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AKp//2Q=="
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="nvidia/GLM-5.3-Flash-NVFP4")
    args = parser.parse_args()
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What color is this image? Reply with one word only.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_RED_JPEG_B64}"},
                    },
                ],
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        args.url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        err = exc.read().decode()
        print(err, file=sys.stderr)
        if "is not a multimodal model" in err:
            print("result=fail reason=not_multimodal", file=sys.stderr)
        return 1
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or "") + (msg.get("reasoning") or "")
    print(
        json.dumps(
            {
                "content": msg.get("content"),
                "finish_reason": data["choices"][0].get("finish_reason"),
            },
            indent=2,
        )
    )
    if "is not a multimodal model" in text:
        print("result=fail reason=not_multimodal", file=sys.stderr)
        return 1
    if not (msg.get("content") or "").strip():
        print("result=fail reason=empty_content", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
