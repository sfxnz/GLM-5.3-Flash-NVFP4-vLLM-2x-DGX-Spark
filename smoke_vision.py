#!/usr/bin/env python3
"""Live vision gate against a running serve (stdlib only).

Thin wrapper around quality/vision.py: synthetic images with known answers
(token accounting, colours, quadrants, counting, OCR, two-image order, and a
short video when the server accepts it). Exit 0 only when every check passes.
HTTP 400 "is not a multimodal model" fails loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "quality"))
import vision  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(vision.main())
