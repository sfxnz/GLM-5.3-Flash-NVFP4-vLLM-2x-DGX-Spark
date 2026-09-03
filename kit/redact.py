#!/usr/bin/env python3
# vendored from sfxnz/forge kit @ 6285f70
"""Redact secrets from evidence, or refuse evidence that still holds one.

    docker inspect C | kit/redact.py > container.json   filter: JSON (or plain text) stdin -> stdout.
                                                         Env entries named *TOKEN*/*SECRET*/*KEY*/... keep
                                                         the name, the value becomes <redacted>; every
                                                         string is scrubbed with the SECRET_VALUES table.
    kit/redact.py --check <path|-> ...                   exit 1 and print file:line:pattern for any match
                                                         of the same table (dirs recurse, - is stdin).
                                                         The matched text is never printed.

build/publish.sh, hillclimb/iterate.sh and kit/recipe_lint.py run --check over evidence/ before a commit.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SECRET_ENV_NAME = re.compile(r"(TOKEN|SECRET|KEY|PASSWORD|PASSWD|CREDENTIAL)", re.I)
SECRET_VALUES = {
    "hf_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "github_token": re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    "slack_token": re.compile(r"xox[abp]-"),
    "bearer": re.compile(r"Bearer [A-Za-z0-9._-]{16,}"),
    "openai_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
}
REDACTED = "<redacted>"


def scrub(text: str) -> str:
    for rx in SECRET_VALUES.values():
        text = rx.sub(REDACTED, text)
    return text


def redact_env(entry: str) -> str:
    name, sep, _ = entry.partition("=")
    if sep and SECRET_ENV_NAME.search(name):
        return f"{name}={REDACTED}"
    return scrub(entry)


def redact(obj: object) -> object:
    if isinstance(obj, dict):
        return {
            k: [redact_env(e) if isinstance(e, str) else redact(e) for e in v]
            if k == "Env" and isinstance(v, list)
            else redact(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return scrub(obj)
    return obj


def filter_stdin() -> int:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        sys.stdout.write(scrub(raw))
        return 0
    json.dump(redact(data), sys.stdout, indent=4)
    sys.stdout.write("\n")
    return 0


def check_lines(label: str, lines: list[str]) -> int:
    hits = 0
    for n, line in enumerate(lines, 1):
        for name, rx in SECRET_VALUES.items():
            if rx.search(line):
                print(f"{label}:{n}:{name}")
                hits += 1
    return hits


def check(paths: list[str]) -> int:
    hits = 0
    for arg in paths:
        if arg == "-":
            hits += check_lines("-", sys.stdin.read().splitlines())
            continue
        p = Path(arg)
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        for f in files:
            hits += check_lines(str(f), f.read_text(errors="replace").splitlines())
    if hits:
        print(f"redact: {hits} secret match(es); refusing", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        return filter_stdin()
    if argv[0] == "--check" and len(argv) > 1:
        return check(argv[1:])
    print(__doc__.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
