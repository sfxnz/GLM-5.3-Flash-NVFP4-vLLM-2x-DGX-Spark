#!/usr/bin/env python3
# vendored from sfxnz/forge kit @ 6285f70
"""Vision smoke: one image_url request. Fail on HTTP error, `is not a multimodal model`, or a
missing expected word.

    kit/probes/vision.py <recipe-dir> <evidence-dir>
    recipe.yaml:  vision: {image: solid-red, expect: red}   # image: solid-red (64x64 JPEG generated
                  with PIL, the smoke_vision.py payload) or a path relative to the recipe dir.
                  expect: case-insensitive substring of the reply; omit to only require a 200.
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import sys

from _probe import Probe


def red_jpeg() -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("PIL is required for the solid-red vision image") from exc
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 20, 60)).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def image_data_url(probe: Probe, image: str) -> str:
    if image == "solid-red":
        return f"data:image/jpeg;base64,{red_jpeg()}"
    path = probe.recipe.dir / image
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def main() -> int:
    probe = Probe("vision", __doc__)
    probe.parse()
    params = probe.recipe.probes().get("vision") or {}
    image = str(params.get("image", "solid-red"))
    expect = params.get("expect")
    code, payload = probe.post(
        {
            "model": probe.recipe.served_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Reply with one word only."},
                        {"type": "image_url", "image_url": {"url": image_data_url(probe, image)}},
                    ],
                }
            ],
            "max_tokens": 64,
            "temperature": 0,
            "chat_template_kwargs": probe.recipe.chat_template_kwargs,
        },
        timeout=180,
    )
    err = json.dumps(payload)
    if code != 200:
        reason = "not_multimodal" if "is not a multimodal model" in err else f"http_{code}"
        return probe.finish(False, reason)
    msg = (payload.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    text = content + (msg.get("reasoning") or "")
    probe.say(json.dumps({"content": content, "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason")}))
    if "is not a multimodal model" in text:
        return probe.finish(False, "not_multimodal")
    if expect and str(expect).lower() not in content.lower():
        return probe.finish(False, f"expected={expect!r}")
    return probe.finish(True)


if __name__ == "__main__":
    sys.exit(main())
