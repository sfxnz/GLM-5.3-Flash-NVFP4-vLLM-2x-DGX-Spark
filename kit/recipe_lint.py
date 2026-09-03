#!/usr/bin/env python3
# vendored from sfxnz/forge kit @ 6285f70
"""Clone-shape checks for any recipe repo. No GPU, no serve. Usage: python3 kit/recipe_lint.py <recipe-dir>

Generic checks (every recipe):
  required files, run.sh/stop.sh executable, MIT LICENSE, recipe.yaml has bench:/probes:/lint:,
  `kit/render.py --check` (`--strict` unless lint.strict_evidence is false), the kit's bench phases and
  probe markers, a script for every enabled probe, needle --dry-run at c=1 and c=2,
  `kit/redact.py --check evidence` finds no secret, `VALIDATE_ONLY=1 ./run.sh` passes and prints `validate-only`.
Recipe-specific expectations live in recipe.yaml `lint:`:
  required_files: [path, ...]                    files the clone must ship
  contains: {path: [snippet, ...]}               text each file must contain (README.md, run.sh, ...)
  forbids: {path: [snippet, ...]}                text each file must not contain
  readme_mentions_env: [NAME, ...]               README.md must contain serve.env.NAME's value
  dockerfile_from: {path: "image[:tag][@digest]"} exact FROM line
  expect_cases: [{env: {...}, expect: "text"}]   VALIDATE_ONLY=1 run.sh passes and stdout contains text
  refuse_cases: [{env: {...}, expect: "text"}]   VALIDATE_ONLY=1 run.sh fails and stderr contains text
  commands: ["shell command", ...]               run in the recipe dir, must exit 0
  strict_evidence: true                          false while measured rows have no evidence file yet
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kitlib import Recipe  # noqa: E402

GENERIC_FILES = (
    "README.md",
    "LICENSE",
    "run.sh",
    "stop.sh",
    "recipe.yaml",
    "evidence",
    "kit/render.py",
    "kit/lib.sh",
    "kit/doctor.sh",
    "kit/bench_decode.py",
    "kit/recipe_lint.py",
    "kit/redact.py",
    "kit/probes/run-all.sh",
)

# The vendored kit must keep its methods. sync.sh --check catches drift against forge; these
# markers catch a gutted copy in a clone without forge.
KIT_MARKERS = {
    "kit/bench_decode.py": ['"prose"', '"structured"', "--phase", "chat/completions"],
    "kit/probes/smoke.sh": ['"temperature": 0'],
    "kit/probes/thinking_off.py": ["leaked_think", '"content"'],
    "kit/probes/tool_call.py": ["tool_calls", "get_weather", "tool_choice"],
    "kit/probes/hermes_two_turn.py": ['"role": "tool"', "tool_call_id", "get_weather"],
    "kit/probes/needle.py": ["prefill_tok_s", "--salt", "ttft_s", "--concurrency", "serve_alive"],
}


def from_line(path: Path) -> str | None:
    for line in path.read_text().splitlines():
        if line.startswith("FROM "):
            return line.split(None, 1)[1].strip()
    return None


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(cwd), env=env)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    repo = Path(sys.argv[1]).resolve()
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)

    def text_of(rel: str) -> str:
        p = repo / rel
        return p.read_text() if p.is_file() else ""

    for rel in GENERIC_FILES:
        if not (repo / rel).exists():
            fail(f"missing {rel}")
    for rel in ("run.sh", "stop.sh"):
        if (repo / rel).exists() and not os.access(repo / rel, os.X_OK):
            fail(f"{rel} is not executable")
    if "MIT License" not in text_of("LICENSE"):
        fail("LICENSE is not MIT")

    try:
        recipe = Recipe(repo)
    except Exception as exc:  # noqa: BLE001
        fail(f"recipe.yaml unreadable: {exc}")
        return report(repo, failures)
    data = recipe.data
    lint = data.get("lint") or {}
    if not isinstance(lint, dict):
        fail("recipe.yaml lint: must be a mapping")
        lint = {}
    if not isinstance((data.get("bench") or {}).get("chat_template_kwargs"), dict):
        fail("recipe.yaml bench.chat_template_kwargs must be a mapping (the kwargs that turn thinking off)")
    if not isinstance(data.get("probes"), list):
        fail("recipe.yaml probes: must be a list")
    if not isinstance(lint.get("refuse_cases", []), list):
        fail("recipe.yaml lint.refuse_cases must be a list")

    for rel in lint.get("required_files") or []:
        if not (repo / rel).exists():
            fail(f"missing {rel}")

    render_cmd = [sys.executable, "kit/render.py", "--check"]
    if lint.get("strict_evidence", True):
        render_cmd.append("--strict")
    if (repo / "kit/render.py").exists():
        r = run(render_cmd, repo)
        if r.returncode != 0:
            fail(f"{' '.join(render_cmd[1:])} failed: {(r.stderr or r.stdout).strip().splitlines()[-1]}")

    if (repo / "kit/redact.py").exists() and (repo / "evidence").is_dir():
        r = run([sys.executable, "kit/redact.py", "--check", "evidence"], repo)
        if r.returncode != 0:
            fail(f"kit/redact.py --check evidence: {' '.join(r.stdout.split())}")

    for rel, markers in KIT_MARKERS.items():
        src = text_of(rel)
        if not src:
            fail(f"missing {rel}")
            continue
        for m in markers:
            if m not in src:
                fail(f"{rel} lost {m!r}")

    try:
        probes = recipe.probes()
    except ValueError as exc:
        fail(str(exc))
        probes = {}
    for name in probes:
        if not ((repo / f"kit/probes/{name}.sh").exists() or (repo / f"kit/probes/{name}.py").exists()):
            fail(f"probe {name} enabled but kit/probes/{name}.sh|.py is missing")
    needle = repo / "kit/probes/needle.py"
    if needle.exists():
        for conc in ("1", "2"):
            dry = run(
                [sys.executable, str(needle), str(repo), "/tmp", "--prompt-tokens", "100", "--salt", "lint",
                 "--concurrency", conc, "--dry-run"],
                repo,
            )
            if dry.returncode != 0 or "dry_run=1" not in dry.stdout or f"n={conc}" not in dry.stdout:
                fail(f"needle.py --concurrency {conc} --dry-run failed: {dry.stderr.strip() or dry.stdout.strip()}")

    for rel, snippets in (lint.get("contains") or {}).items():
        src = text_of(rel)
        for s in snippets:
            if s not in src:
                fail(f"{rel} missing {s!r}")
    for rel, snippets in (lint.get("forbids") or {}).items():
        src = text_of(rel)
        for s in snippets:
            if s in src:
                fail(f"{rel} must not contain {s!r}")
    readme = text_of("README.md")
    for name in lint.get("readme_mentions_env") or []:
        want = recipe.env(name)
        if not want or want not in readme:
            fail(f"README does not mention run.sh default {name}={want}")
    for rel, want in (lint.get("dockerfile_from") or {}).items():
        p = repo / rel
        if not p.exists():
            continue
        got = from_line(p)
        if got != want:
            fail(f"{rel} FROM {got!r} want {want!r}")

    run_sh = repo / "run.sh"
    if run_sh.exists():
        base_env = {k: v for k, v in os.environ.items() if k not in ("FORCE_UNSAFE_CTX", "FORCE_UNSAFE_MOE")}
        val = run(["bash", str(run_sh)], repo, env={**base_env, "VALIDATE_ONLY": "1"})
        if val.returncode != 0 or "validate-only" not in val.stdout:
            fail(f"VALIDATE_ONLY=1 ./run.sh failed: rc={val.returncode} {val.stderr.strip() or val.stdout.strip()}")
        for case in lint.get("expect_cases") or []:
            env = {k: str(v) for k, v in (case.get("env") or {}).items()}
            r = run(["bash", str(run_sh)], repo, env={**base_env, "VALIDATE_ONLY": "1", **env})
            if r.returncode != 0 or case["expect"] not in r.stdout:
                fail(f"VALIDATE_ONLY=1 {env} did not print {case['expect']!r}: rc={r.returncode} {r.stderr.strip()}")
        for case in lint.get("refuse_cases") or []:
            env = {k: str(v) for k, v in (case.get("env") or {}).items()}
            r = run(["bash", str(run_sh)], repo, env={**base_env, "VALIDATE_ONLY": "1", **env})
            if r.returncode == 0 or case["expect"] not in r.stderr:
                fail(f"VALIDATE_ONLY=1 {env} did not refuse with {case['expect']!r} (rc={r.returncode})")

    for cmd in lint.get("commands") or []:
        r = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True, cwd=str(repo))
        if r.returncode != 0:
            fail(f"command failed: {cmd}: {(r.stderr or r.stdout).strip()[-300:]}")

    return report(repo, failures)


def report(repo: Path, failures: list[str]) -> int:
    print(f"repo={repo}")
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print(f"result=fail n={len(failures)}")
        return 1
    print("result=pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
