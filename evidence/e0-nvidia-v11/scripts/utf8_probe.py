#!/usr/bin/env python3
"""Re-run tier0's utf8 prompt (streamed, greedy, thinking off) and print the rows judge_utf8 flags."""
import sys
from pathlib import Path

REPO = Path("/home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia")
sys.path.insert(0, str(REPO / "quality"))
from common import Client  # noqa: E402
from tier0 import OFF, UTF8_PROMPT, chinese_numeral, judge_utf8  # noqa: E402

c = Client("http://127.0.0.1:8000")
for run in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2):
    out = c.chat(UTF8_PROMPT, stream=True, temperature=0, max_tokens=3000, **OFF)
    j = judge_utf8(out)
    print(f"== run {run}: {j}")
    for line in out["content"].splitlines():
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0].isdigit():
            n = int(cells[0])
            bad = [k for k, ok in (("sq", cells[1].replace(",", "") == str(n * n)),
                                   ("cube", cells[2].replace(",", "") == str(n ** 3)),
                                   ("cn", cells[3] == chinese_numeral(n))) if not ok]
            if bad:
                print(f"  row {n}: {cells} bad={bad} want sq={n*n} cube={n**3} cn={chinese_numeral(n)}")
    if run == 0:
        print("  first 6 lines:", out["content"].splitlines()[:6])
