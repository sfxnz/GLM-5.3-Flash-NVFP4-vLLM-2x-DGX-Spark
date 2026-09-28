#!/usr/bin/env python3
"""utf8_scan.py FILE...: for each utf8-texts.json (evidence/e3-tools/utf8_texts.py), per run: judge verdict, finish,
number of table lines, rows whose n column is not the next integer (repeats / skips), and cells that are neither a
number nor CJK (for example 'numerals', '5 characters'): the leaked-word signature. Stdlib only."""
import json
import re
import sys

NUM = re.compile(r"^[\d,]+$")
CJK = re.compile(r"^[一-鿿]+$")
for path in sys.argv[1:]:
    d = json.load(open(path))
    runs = d["runs"] if isinstance(d, dict) and "runs" in d else d
    print(f"== {path}")
    for i, r in enumerate(runs):
        text = r.get("text", "")
        lines = [ln for ln in text.splitlines() if ln.strip().startswith("|") and "---" not in ln][1:]
        odd, seq_breaks, prev = [], 0, 0
        for ln in lines:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            n = int(cells[0]) if cells and cells[0].isdecimal() else None
            if n is None or n != prev + 1:
                seq_breaks += 1
            if n is not None:
                prev = n
            for c in cells[1:]:
                if c and not NUM.match(c) and not CJK.match(c):
                    odd.append(c)
        j = r.get("judge", {})
        verdict = "error" if "error" in j else ("PASS" if j.get("pass") else f"FAIL sq_err={j.get('square_errors')}")
        print(f"run {i}: {verdict} finish={r.get('finish')} table_lines={len(lines)} n_breaks={seq_breaks}"
              f" odd_cells={len(odd)} {odd[:8]}")
