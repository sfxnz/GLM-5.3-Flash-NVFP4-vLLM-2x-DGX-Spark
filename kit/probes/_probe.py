# vendored from sfxnz/forge kit @ 6f40808
"""Common frame for the Python probes: args, recipe access, the <probe>.txt evidence file.

Each probe writes <evidence-dir>/<name>.txt holding the request, the response, the HTTP status
and a final `verdict=PASS` or `verdict=FAIL reason=...` line, and exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kitlib import Recipe, post_json  # noqa: E402,F401


class Probe:
    def __init__(self, name: str, description: str):
        self.name = name
        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument("recipe_dir")
        self.parser.add_argument("evidence_dir")
        self.lines: list[str] = []

    def parse(self) -> argparse.Namespace:
        self.args = self.parser.parse_args()
        self.recipe = Recipe(self.args.recipe_dir)
        self.out = Path(self.args.evidence_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        return self.args

    def say(self, text: str) -> None:
        print(text, flush=True)
        self.lines.append(text)

    def record(self, label: str, obj: object) -> None:
        body = obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False)
        self.lines.append(f"--- {label}\n{body}")

    def post(self, body: dict, timeout: int = 120, label: str = "") -> tuple[int, dict]:
        suffix = f" {label}" if label else ""
        self.record(f"request{suffix}", body)
        code, payload = post_json(self.recipe.completions_url, body, timeout=timeout)
        self.record(f"response{suffix}", payload)
        self.lines.append(f"http_status{suffix}={code}")
        return code, payload

    def finish(self, ok: bool, reason: str = "", filename: str | None = None) -> int:
        verdict = "verdict=PASS" if ok else f"verdict=FAIL reason={reason}"
        print(verdict, flush=True)
        self.lines.append(verdict)
        path = self.out / (filename or f"{self.name}.txt")
        path.write_text("\n".join(self.lines) + "\n")
        print(f"evidence={path}", flush=True)
        return 0 if ok else 1
