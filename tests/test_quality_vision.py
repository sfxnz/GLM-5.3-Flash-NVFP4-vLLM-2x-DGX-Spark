"""PNG writer validity, bitmap font, answer parsing and the suite against a fake serve."""
from __future__ import annotations

import base64
import hashlib
import struct
import sys
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quality"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision  # noqa: E402
from test_quality_fakes import FakeServe  # noqa: E402


def decode_png(data: bytes) -> tuple[int, int, bytes]:
    """Strict stdlib PNG reader for what png_bytes writes: checks every CRC."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "signature"
    pos, chunks = 8, []
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        (crc,) = struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])
        assert zlib.crc32(tag + body) & 0xFFFFFFFF == crc, f"CRC mismatch in {tag}"
        chunks.append((tag, body))
        pos += 12 + n
    assert [t for t, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    w, h, depth, ctype, comp, filt, lace = struct.unpack(">IIBBBBB", chunks[0][1])
    assert (depth, ctype, comp, filt, lace) == (8, 2, 0, 0, 0)
    raw = zlib.decompress(chunks[1][1])
    assert len(raw) == h * (1 + 3 * w), "IDAT size"
    rows = [raw[y * (1 + 3 * w):(y + 1) * (1 + 3 * w)] for y in range(h)]
    assert all(r[0] == 0 for r in rows), "filter byte"
    return w, h, b"".join(r[1:] for r in rows)


def pixel(px: bytes, w: int, x: int, y: int) -> tuple:
    i = (y * w + x) * 3
    return tuple(px[i:i + 3])


class PNGTests(unittest.TestCase):
    def test_roundtrip_and_crc(self):
        w, h, px = decode_png(vision.quadrants(448))
        self.assertEqual((w, h), (448, 448))
        self.assertEqual(pixel(px, w, 10, 10), vision.RGB["red"])
        self.assertEqual(pixel(px, w, 400, 10), vision.RGB["green"])
        self.assertEqual(pixel(px, w, 10, 400), vision.RGB["blue"])
        self.assertEqual(pixel(px, w, 400, 400), vision.RGB["yellow"])
        self.assertEqual(vision.png_size(vision.quadrants(448)), (448, 448))

    def test_corruption_detected(self):
        data = bytearray(vision.solid("red"))
        data[40] ^= 0xFF  # inside IDAT
        with self.assertRaises(Exception):
            decode_png(bytes(data))

    def test_bad_buffer(self):
        with self.assertRaises(ValueError):
            vision.png_bytes(2, 2, b"\x00" * 11)

    def test_pil_can_open(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not installed (optional cross-check)")
        import io
        im = Image.open(io.BytesIO(vision.circles(3)))
        im.load()
        self.assertEqual((im.size, im.mode), ((448, 448), "RGB"))

    def test_font_ink(self):
        for s in ("7F3A91", "K9P2X"):
            w, h, px = decode_png(vision.text_image(s, scale=8))
            self.assertEqual((w % 28, h % 28), (0, 0))
            ink = sum(1 for i in range(0, len(px), 3) if px[i:i + 3] == b"\x00\x00\x00")
            want = sum(row.count("#") for ch in s for row in vision.FONT[ch]) * 64
            self.assertEqual(ink, want)

    def test_circles(self):
        w, h, px = decode_png(vision.circles(5))
        for cx, cy in vision.CIRCLE_SPOTS[:5]:
            self.assertEqual(pixel(px, w, cx, cy), (0, 0, 0))
        cx, cy = vision.CIRCLE_SPOTS[5]
        self.assertEqual(pixel(px, w, cx, cy), (255, 255, 255))

    def test_expected_tokens(self):
        self.assertEqual(vision.expected_image_tokens(448, 448), 256)
        self.assertEqual(vision.expected_image_tokens(336, 112), 48)
        with self.assertRaises(ValueError):
            vision.expected_image_tokens(28, 28)  # below the 16-token floor

    def test_video_url(self):
        frames = [vision.digit_frame(d) for d in "38"]
        url = vision.video_url(frames)
        head, body = url.split(",", 1)
        self.assertEqual(head, "data:video/jpeg;base64")
        parts = body.split(",")
        self.assertEqual(len(parts), 2)
        self.assertEqual(base64.b64decode(parts[1]), frames[1])


class ParseTests(unittest.TestCase):
    def test_parsers(self):
        self.assertEqual(vision.first_int("There are five circles."), 5)
        self.assertEqual(vision.first_int("3"), 3)
        self.assertEqual(vision.parse_quadrants(
            "top-left=Red, top-right=green, bottom left: blue, bottom_right = **yellow**"),
            {"top-left": "red", "top-right": "green", "bottom-left": "blue", "bottom-right": "yellow"})
        self.assertEqual(vision.norm_text(vision.strip_box("<|begin_of_box|>7f3a-91<|end_of_box|>")), "7F3A91")
        self.assertEqual(vision.collapse_digits("3, 3, 8, 1, 6"), "3816")
        self.assertTrue(vision.says_word("It is RED.", "red"))
        self.assertFalse(vision.says_word("reddish", "red"))


def answer_key() -> dict:
    """sha256 of every image the suite sends -> the correct answer."""
    key = {}

    def add(png, ans):
        key[hashlib.sha256(png).hexdigest()] = ans

    for c in ("red", "green", "blue", "yellow"):
        add(vision.solid(c), c.capitalize())
    add(vision.quadrants(), "top-left=red, top-right=green, bottom-left=blue, bottom-right=yellow")
    add(vision.circles(3), "3")
    add(vision.circles(5), "5")
    add(vision.text_image("7F3A91"), "7F3A91")
    add(vision.text_image("K9P2X"), "K9P2X")
    return key


def fake_vlm(key, video_status=200, not_mm=False):
    red = hashlib.sha256(vision.solid("red")).hexdigest()

    def chat(body):
        if not_mm:
            return {"http_error": 400, "message": "fake-glm is not a multimodal model"}
        parts = body["messages"][-1]["content"]
        imgs = [p["image_url"]["url"].split(",", 1)[1] for p in parts if p["type"] == "image_url"]
        shas = [hashlib.sha256(base64.b64decode(u)).hexdigest() for u in imgs]
        if any(p["type"] == "video_url" for p in parts):
            if video_status != 200:
                return {"http_error": video_status, "message": "video not supported"}
            return {"content": "3816"}
        prompt_tokens = 20 + 258 * len(imgs) if body["max_tokens"] == 1 else 10
        if len(shas) == 2:
            return {"content": "First" if shas[0] == red else "Second"}
        return {"content": key.get(shas[0], "") if shas else "", "prompt_tokens": prompt_tokens}

    return chat


class SuiteTests(unittest.TestCase):
    def test_all_pass(self):
        s = FakeServe(fake_vlm(answer_key()))
        try:
            res = vision.run_suite(vision.Client(s.url))
        finally:
            s.close()
        self.assertTrue(res["pass"], [c for c in res["checks"] if not c["pass"]])
        self.assertEqual(res["total"], 13)
        req = [b for p, b in s.requests if p == "/v1/chat/completions"][0]
        self.assertEqual(req["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(req["temperature"], 0)

    def test_wrong_answer_fails(self):
        key = answer_key()
        key[hashlib.sha256(vision.circles(5)).hexdigest()] = "4"
        s = FakeServe(fake_vlm(key))
        try:
            res = vision.run_suite(vision.Client(s.url))
        finally:
            s.close()
        self.assertFalse(res["pass"])
        self.assertEqual([c["name"] for c in res["checks"] if not c["pass"]], ["count.5"])

    def test_video_rejected_is_skip(self):
        for status in (400, 500):  # an optional probe never blocks the smoke
            s = FakeServe(fake_vlm(answer_key(), video_status=status))
            try:
                res = vision.run_suite(vision.Client(s.url))
            finally:
                s.close()
            self.assertTrue(res["pass"])
            self.assertEqual(res["skipped"], ["video"])
            self.assertIn(f"HTTP {status}", res["checks"][-1]["answer"])

    def test_not_multimodal_fails_loudly(self):
        s = FakeServe(fake_vlm({}, not_mm=True))
        try:
            code = vision.main(["--url", s.url + "/v1/chat/completions"])
            res = vision.run_suite(vision.Client(s.url))
        finally:
            s.close()
        self.assertEqual(code, 1)
        self.assertFalse(res["pass"])
        self.assertIn("is not a multimodal model", res["error"])


if __name__ == "__main__":
    unittest.main()
