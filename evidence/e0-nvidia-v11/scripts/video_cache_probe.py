#!/usr/bin/env python3
"""MM processor cache hit vs miss on the tier0 video check: same digits 3816 and the exact tier0
request (vision._ask), frames at the cached 224 px size vs new sizes (fresh hash = cache miss).
Each size is sent twice (first = miss for a new size, second = hit)."""
import sys

sys.path.insert(0, "/home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia/quality")
import vision  # noqa: E402
from common import Client  # noqa: E402

c = Client("http://127.0.0.1:8000")
for size in [int(x) for x in (sys.argv[1:] or ["224", "232", "240", "256"])]:
    frames = [vision.digit_frame(d, size) for d in vision.VIDEO_DIGITS for _ in range(2)]
    parts = [{"type": "video_url", "video_url": {"url": vision.video_url(frames)}}, vision._text(
        "This video shows one large digit per scene. List the digits in the order they appear, "
        "as digits only with no spaces.")]
    answers = [vision._ask(c, parts)["content"].strip() for _ in range(2)]
    print(f"size {size}: first={answers[0]!r} repeat={answers[1]!r}", flush=True)
