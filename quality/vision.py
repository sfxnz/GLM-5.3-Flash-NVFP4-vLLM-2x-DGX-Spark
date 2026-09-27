#!/usr/bin/env python3
"""Vision correctness suite (stdlib only): synthetic images with known answers.

Images are drawn here and written as PNG with zlib + struct (no PIL). Every
check is greedy, thinking off, and asserts the answer:

  tokens     prompt_tokens(image + text) - prompt_tokens(text) equals the
             processor's placeholder count + 2 (begin/end image tokens)
  solid      four solid colours, one word each
  quadrants  red / green / blue / yellow quadrants, all four named right
  count      3 and 5 black circles
  ocr        two strings drawn with a 5x7 bitmap font, exact match
  order      two images (red, blue) then (blue, red): which one is red
  video      8 PNG frames as data:video/jpeg (vLLM's frame-list form), digits
             in order. Optional: SKIP when the server rejects the video input,
             FAIL on a wrong answer.

HTTP 400 "is not a multimodal model" fails the whole suite loudly.

  python3 quality/vision.py [--url http://127.0.0.1:8000] [--no-video] [--out vision.json]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Client, HTTPFailure, strip_box  # noqa: E402

RGB = {"red": (230, 25, 25), "green": (20, 170, 40), "blue": (25, 60, 230),
       "yellow": (250, 220, 20), "black": (0, 0, 0), "white": (255, 255, 255)}
KWARGS = {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0}
NOT_MM = "is not a multimodal model"

# Processor geometry (nvidia 09b04e5 processor_config.json): patch 14,
# merge 2, temporal patch 2, 16..8000 tokens per image.
FACTOR = 28
MIN_PIXELS = 16 * 2 * FACTOR * FACTOR
MAX_PIXELS = 8000 * 2 * FACTOR * FACTOR

# 5x7 bitmap font, '#' = ink.
FONT = {
    "0": [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
    "1": ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    "2": [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"],
    "3": ["#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "],
    "4": ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
    "5": ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    "6": ["  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "],
    "7": ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],
    "8": [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    "9": [" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "],
    "A": [" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"],
    "F": ["#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "],
    "K": ["#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"],
    "P": ["#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "],
    "X": ["#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"],
}


# ------------------------------------------------------------------ PNG

def png_bytes(width: int, height: int, rgb: bytes) -> bytes:
    """8-bit RGB PNG, filter 0 on every row."""
    if len(rgb) != width * height * 3:
        raise ValueError("pixel buffer size does not match width x height x 3")
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


class Canvas:
    def __init__(self, width: int, height: int, color=RGB["white"]):
        self.w, self.h = width, height
        self.px = bytearray(bytes(color) * (width * height))

    def rect(self, x0: int, y0: int, x1: int, y1: int, color) -> None:
        row = bytes(color) * (x1 - x0)
        for y in range(max(0, y0), min(self.h, y1)):
            i = (y * self.w + x0) * 3
            self.px[i:i + len(row)] = row

    def circle(self, cx: int, cy: int, r: int, color) -> None:
        for y in range(cy - r, cy + r + 1):
            dx = int(math.sqrt(max(0, r * r - (y - cy) ** 2)))
            self.rect(cx - dx, y, cx + dx + 1, y + 1, color)

    def text(self, s: str, x: int, y: int, scale: int, color=RGB["black"]) -> None:
        for k, ch in enumerate(s):
            for row, bits in enumerate(FONT[ch]):
                for col, bit in enumerate(bits):
                    if bit == "#":
                        gx = x + (k * 6 + col) * scale
                        gy = y + row * scale
                        self.rect(gx, gy, gx + scale, gy + scale, color)

    def png(self) -> bytes:
        return png_bytes(self.w, self.h, bytes(self.px))


def solid(color: str, size: int = 224) -> bytes:
    return Canvas(size, size, RGB[color]).png()


def quadrants(size: int = 448) -> bytes:
    c, h = Canvas(size, size), size // 2
    c.rect(0, 0, h, h, RGB["red"])
    c.rect(h, 0, size, h, RGB["green"])
    c.rect(0, h, h, size, RGB["blue"])
    c.rect(h, h, size, size, RGB["yellow"])
    return c.png()


CIRCLE_SPOTS = [(80, 80), (224, 90), (368, 80), (110, 250), (330, 240), (224, 380)]


def circles(n: int, size: int = 448) -> bytes:
    c = Canvas(size, size)
    for cx, cy in CIRCLE_SPOTS[:n]:
        c.circle(cx, cy, 42, RGB["black"])
    return c.png()


def text_image(s: str, scale: int = 8) -> bytes:
    pad = 2 * scale
    w = len(s) * 6 * scale - scale + 2 * pad
    h = 7 * scale + 2 * pad
    w, h = -(-w // FACTOR) * FACTOR, -(-h // FACTOR) * FACTOR
    c = Canvas(w, h)
    c.text(s, pad, pad, scale)
    return c.png()


def digit_frame(d: str, size: int = 224) -> bytes:
    c = Canvas(size, size)
    c.text(d, (size - 5 * 20) // 2, (size - 7 * 20) // 2, 20)
    return c.png()


def expected_image_tokens(width: int, height: int) -> int:
    """Placeholder tokens for a still image inside the pixel budget."""
    h_bar, w_bar = -(-height // FACTOR) * FACTOR, -(-width // FACTOR) * FACTOR
    if not MIN_PIXELS <= 2 * h_bar * w_bar <= MAX_PIXELS:
        raise ValueError("image outside the processor pixel budget; pick another size")
    return (h_bar // FACTOR) * (w_bar // FACTOR)


def png_size(png: bytes) -> tuple[int, int]:
    return struct.unpack(">II", png[16:24])


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def video_url(frames: list[bytes]) -> str:
    # vLLM MediaIO: data:video/jpeg;base64,<frame>,<frame>,... ; each frame is
    # opened with PIL, which reads PNG regardless of the declared type.
    return "data:video/jpeg;base64," + ",".join(base64.b64encode(f).decode() for f in frames)


# --------------------------------------------------------------- scoring

WORD_NUMS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten".split())}


def first_int(text: str) -> int | None:
    m = re.search(r"\d+|\b(" + "|".join(WORD_NUMS) + r")\b", text.lower())
    if not m:
        return None
    return int(m.group(0)) if m.group(0).isdigit() else WORD_NUMS[m.group(0)]


def says_word(text: str, word: str) -> bool:
    return re.search(r"\b" + word + r"\b", text.lower()) is not None


def parse_quadrants(text: str) -> dict:
    out = {}
    for key in ("top-left", "top-right", "bottom-left", "bottom-right"):
        pat = key.replace("-", r"[\s_-]?") + r"\s*[=:]\s*\**\s*([a-z]+)"
        m = re.search(pat, text.lower())
        out[key] = m.group(1) if m else None
    return out


def norm_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def collapse_digits(text: str) -> str:
    ds = re.findall(r"\d", text)
    return "".join(d for i, d in enumerate(ds) if i == 0 or d != ds[i - 1])


# ------------------------------------------------------------------ suite

class NotMultimodal(RuntimeError):
    pass


def _ask(client: Client, parts: list, max_tokens: int = 64) -> dict:
    try:
        return client.chat([{"role": "user", "content": parts}], max_tokens=max_tokens, **KWARGS)
    except HTTPFailure as exc:
        if NOT_MM in exc.body:
            raise NotMultimodal(exc.body[:300]) from exc
        raise


def _img(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(png)}}


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


def check_tokens(client: Client) -> dict:
    png = quadrants(448)
    text = _text("Describe this image.")
    with_img = _ask(client, [_img(png), text], max_tokens=1)["usage"]["prompt_tokens"]
    without = _ask(client, [text], max_tokens=1)["usage"]["prompt_tokens"]
    want = expected_image_tokens(*png_size(png)) + 2
    return {"name": "tokens", "pass": with_img - without == want,
            "answer": with_img - without, "expected": want}


def run_checks(client: Client) -> list[dict]:
    rows = [check_tokens(client)]
    for color in ("red", "green", "blue", "yellow"):
        a = _ask(client, [_img(solid(color)), _text(
            "What single colour fills this image? Answer with one word.")])["content"]
        rows.append({"name": f"solid.{color}", "pass": says_word(a, color), "answer": a, "expected": color})
    a = _ask(client, [_img(quadrants()), _text(
        "This image is split into four equal quadrants, each one solid colour. Reply exactly in the form "
        "top-left=<colour>, top-right=<colour>, bottom-left=<colour>, bottom-right=<colour>")], 96)["content"]
    want = {"top-left": "red", "top-right": "green", "bottom-left": "blue", "bottom-right": "yellow"}
    rows.append({"name": "quadrants", "pass": parse_quadrants(a) == want, "answer": a, "expected": want})
    for n in (3, 5):
        a = _ask(client, [_img(circles(n)), _text(
            "How many black circles are in this image? Answer with a single number.")])["content"]
        rows.append({"name": f"count.{n}", "pass": first_int(a) == n, "answer": a, "expected": n})
    for s in ("7F3A91", "K9P2X"):
        a = _ask(client, [_img(text_image(s)), _text(
            "What text is written in this image? Reply with the text only.")])["content"]
        rows.append({"name": f"ocr.{s}", "pass": norm_text(strip_box(a)) == s, "answer": a, "expected": s})
    q = _text("Which of the two images is red: the first or the second? Answer with one word, first or second.")
    for order, want in ((("red", "blue"), "first"), (("blue", "red"), "second")):
        a = _ask(client, [_img(solid(order[0])), _img(solid(order[1])), q])["content"]
        rows.append({"name": f"order.{order[0]}-{order[1]}", "pass": says_word(a, want) and
                     not says_word(a, "second" if want == "first" else "first"), "answer": a, "expected": want})
    return rows


VIDEO_DIGITS = "3816"


def check_video(client: Client) -> dict:
    frames = [digit_frame(d) for d in VIDEO_DIGITS for _ in range(2)]
    parts = [{"type": "video_url", "video_url": {"url": video_url(frames)}}, _text(
        "This video shows one large digit per scene. List the digits in the order they appear, "
        "as digits only with no spaces.")]
    try:
        a = _ask(client, parts)["content"]
    except HTTPFailure as exc:
        if exc.code == 400:
            return {"name": "video", "pass": None, "status": "SKIP", "answer": exc.body[:300],
                    "expected": VIDEO_DIGITS}
        raise
    ok = collapse_digits(a) == VIDEO_DIGITS
    return {"name": "video", "pass": ok, "status": "PASS" if ok else "FAIL", "answer": a,
            "expected": VIDEO_DIGITS}


def run_suite(client: Client, video: bool = True) -> dict:
    try:
        rows = run_checks(client)
        if video:
            rows.append(check_video(client))
    except NotMultimodal as exc:
        return {"pass": False, "error": f"{NOT_MM}: {exc}", "checks": []}
    gated = [r for r in rows if r["pass"] is not None]
    return {"pass": all(r["pass"] for r in gated), "passed": sum(bool(r["pass"]) for r in gated),
            "total": len(gated), "skipped": [r["name"] for r in rows if r["pass"] is None], "checks": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None, help="default: first id in /v1/models")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = run_suite(Client(args.url, args.model), video=not args.no_video)
    for r in res["checks"]:
        status = r.get("status") or ("PASS" if r["pass"] else "FAIL")
        print(f"{status} {r['name']} expected={r['expected']!r} answer={str(r['answer'])[:120]!r}")
    if "error" in res:
        print(f"FAIL vision: {res['error']}", file=sys.stderr)
    if args.out:
        args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print("VISION", json.dumps({k: res[k] for k in ("pass", "passed", "total", "skipped", "error") if k in res}))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
