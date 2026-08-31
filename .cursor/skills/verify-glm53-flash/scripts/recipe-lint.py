#!/usr/bin/env python3
"""Clone-shape checks: the GitHub recipe a stranger follows. No GPU, no serve."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
REPO = SKILL_DIR.parent.parent.parent
ARTIFACTS = SKILL_DIR / "artifacts"


def default_of(run_sh: str, var: str) -> str | None:
    m = re.search(rf'^{re.escape(var)}="\$\{{{re.escape(var)}:-(.*)\}}"', run_sh, re.M)
    return m.group(1) if m else None


def from_line(path: Path) -> str | None:
    for line in path.read_text().splitlines():
        if line.startswith("FROM "):
            return line.split(None, 1)[1].strip()
    return None


def shipped_dflash2_sizes(run_sh: str, spec_tokens: int, max_seqs: int) -> str:
    start = run_sh.find("step=$((NUM_SPECULATIVE_TOKENS + 1))")
    if start < 0:
        raise ValueError("run.sh missing dflash2 graph-size loop")
    end = run_sh.find("COMPILATION_CONFIG=", start)
    if end < 0:
        raise ValueError("run.sh graph-size loop is not followed by COMPILATION_CONFIG")
    script = (
        f"NUM_SPECULATIVE_TOKENS={spec_tokens}\n"
        f"MAX_NUM_SEQS={max_seqs}\n"
        f"{run_sh[start:end]}\n"
        "printf '%s\\n' \"$sizes\"\n"
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "graph-size snippet failed")
    return proc.stdout.strip()


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    def need(path: Path) -> None:
        if not path.exists():
            failures.append(f"missing {path.relative_to(REPO)}")

    need(REPO / "README.md")
    need(REPO / "LICENSE")
    need(REPO / "run.sh")
    need(REPO / "stop.sh")
    need(REPO / "bench_decode.py")
    need(REPO / "chat_template.jinja")
    for name in (
        "Dockerfile.sm121-v8",
        "Dockerfile.sm121-v9",
        "Dockerfile.sm121-v10",
        "Dockerfile.sm121-v11",
        "dflash2_backport.diff",
        "patch_v10_dflash_glm5.py",
        "patch_v11_dflash_kv_groups.py",
    ):
        need(REPO / "docker" / name)

    run_sh_path = REPO / "run.sh"
    stop_sh_path = REPO / "stop.sh"
    if run_sh_path.exists() and not os.access(run_sh_path, os.X_OK):
        failures.append("run.sh is not executable")
    if stop_sh_path.exists() and not os.access(stop_sh_path, os.X_OK):
        failures.append("stop.sh is not executable")

    readme = (REPO / "README.md").read_text() if (REPO / "README.md").exists() else ""
    run_sh = run_sh_path.read_text() if run_sh_path.exists() else ""
    license_txt = (REPO / "LICENSE").read_text() if (REPO / "LICENSE").exists() else ""
    bench = (REPO / "bench_decode.py").read_text() if (REPO / "bench_decode.py").exists() else ""

    if "MIT License" not in license_txt:
        failures.append("LICENSE is not MIT")
    if "Recipe scripts are MIT" not in readme:
        failures.append("README missing recipe MIT line")
    if "CC BY-NC-ND" not in readme:
        failures.append("README missing DFlash2 CC BY-NC-ND note")

    for snippet in (
        "docker build -f docker/Dockerfile.sm121-v8",
        "docker build -f docker/Dockerfile.sm121-v9",
        "docker build -f docker/Dockerfile.sm121-v10",
        "docker build -f docker/Dockerfile.sm121-v11",
        "./run.sh",
        "./stop.sh",
        "python3 bench_decode.py",
        "http://127.0.0.1:8000/v1/chat/completions",
        "SPEC=mtp",
        "LibertAIDAI/GLM-5.3-Flash-NVFP4",
        "glm53-sm121-v11",
        "Say hello in one sentence.",
        "enable_thinking",
    ):
        if snippet not in readme:
            failures.append(f"README missing {snippet!r}")

    expected = {
        "IMAGE": "glm53-sm121-v11",
        "PORT": "8000",
        "MAX_MODEL_LEN": "327680",
        "MAX_NUM_SEQS": "2",
        "KV_CACHE_MEMORY": "4445787956",
        "BLOCK_SIZE": "2304",
        "SPEC": "dflash2",
        "SERVED_NAME": "LibertAIDAI/GLM-5.3-Flash-NVFP4",
        "CONTAINER_NAME": "glm53-flash-nvfp4",
        "NUM_SPECULATIVE_TOKENS": "7",
        "KV_CACHE_DTYPE": "fp8_e4m3",
    }
    for var, want in expected.items():
        got = default_of(run_sh, var)
        if got != want:
            failures.append(f"run.sh default {var}={got!r} want {want!r}")
        if want not in readme:
            failures.append(f"README does not mention run.sh default {var}={want}")

    if 'Do not use stock vllm/vllm-openai on sm_121' not in run_sh:
        failures.append("run.sh no longer refuses the stock image")
    if "FORCE_UNSAFE_CTX" not in run_sh or "cannot hold --max-model-len" not in run_sh:
        failures.append("run.sh no longer refuses a 1M window on the fp8 pin")
    tpl = (REPO / "chat_template.jinja").read_text() if (REPO / "chat_template.jinja").exists() else ""
    if "enable_thinking | default(true)" not in tpl:
        failures.append("chat_template.jinja no longer gates <think> on enable_thinking")
    if "{{- '<think></think>' -}}" not in tpl:
        failures.append("chat_template.jinja thinking-off path no longer closes an empty think block")
    if "$NUM_SPECULATIVE_TOKENS" not in run_sh:
        failures.append("run.sh dflash2 spec is not driven by NUM_SPECULATIVE_TOKENS")
    if '"method":"dflash"' not in run_sh:
        failures.append("run.sh missing dflash speculative method")

    chain = {
        REPO / "docker/Dockerfile.sm121-v8": "vllm/vllm-openai:glm53-flash-arm64-cu130",
        REPO / "docker/Dockerfile.sm121-v9": "glm53-sm121-v8",
        REPO / "docker/Dockerfile.sm121-v10": "glm53-sm121-v9",
        REPO / "docker/Dockerfile.sm121-v11": "glm53-sm121-v10",
    }
    for path, want in chain.items():
        if not path.exists():
            continue
        got = from_line(path)
        if got != want:
            failures.append(f"{path.name} FROM {got!r} want {want!r}")

    if '"prose"' not in bench or '"structured"' not in bench:
        failures.append("bench_decode.py missing prose/structured PHASES")
    if "--phase" not in bench or "chat/completions" not in bench:
        failures.append("bench_decode.py missing --phase or completions URL")

    scripts = SKILL_DIR / "scripts"
    thinking = scripts / "thinking_off_probe.py"
    tools = scripts / "tool_call_probe.py"
    needle = scripts / "needle_probe.py"
    hermes = scripts / "hermes_probe.py"
    need(thinking)
    need(tools)
    need(needle)
    need(hermes)
    smoke = scripts / "smoke.sh"
    if smoke.exists() and '"temperature": 0' not in smoke.read_text():
        failures.append("smoke.sh missing temperature 0")
    if thinking.exists():
        src = thinking.read_text()
        if "enable_thinking" not in src or "leaked_think" not in src:
            failures.append("thinking_off_probe.py does not assert a think leak")
        if '"content"' not in src:
            failures.append("thinking_off_probe.py does not read message content")
    if tools.exists():
        src = tools.read_text()
        if "tool_calls" not in src or "get_weather" not in src:
            failures.append("tool_call_probe.py does not assert a parsed get_weather call")
        if "tool_choice" not in src:
            failures.append("tool_call_probe.py missing tool_choice")
    if hermes.exists():
        src = hermes.read_text()
        if '"role": "tool"' not in src or "tool_call_id" not in src:
            failures.append("hermes_probe.py missing role=tool follow-up")
        if "get_weather" not in src:
            failures.append("hermes_probe.py missing get_weather")
    if needle.exists():
        src = needle.read_text()
        if "prefill_tok_s" not in src or "--salt" not in src or "ttft_s" not in src:
            failures.append("needle_probe.py missing prefill_tok_s, ttft_s, or --salt")
        dry = subprocess.run(
            [sys.executable, str(needle), "--prompt-tokens", "100", "--salt", "lint", "--dry-run"],
            check=False,
            capture_output=True,
            text=True,
        )
        if dry.returncode != 0 or "dry_run=1" not in dry.stdout:
            failures.append(f"needle_probe.py --dry-run failed: {dry.stderr.strip() or dry.stdout.strip()}")

    try:
        got_72 = shipped_dflash2_sizes(run_sh, 7, 2)
        if got_72 != "1,2,4,8,16":
            failures.append(f"dflash2 graph sizes spec=7 seqs=2 got {got_72!r} want '1,2,4,8,16'")
        got_54 = shipped_dflash2_sizes(run_sh, 5, 4)
        if got_54 != "1,2,4,6,12,18,24":
            failures.append(f"dflash2 graph sizes spec=5 seqs=4 got {got_54!r} want '1,2,4,6,12,18,24'")
    except (ValueError, RuntimeError) as exc:
        failures.append(f"dflash2 graph-size snippet: {exc}")

    val = subprocess.run(
        ["bash", str(run_sh_path)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "VALIDATE_ONLY": "1"},
    )
    if val.returncode != 0 or "validate-only" not in val.stdout:
        failures.append(
            f"VALIDATE_ONLY=1 ./run.sh failed: rc={val.returncode} {val.stderr.strip() or val.stdout.strip()}"
        )
    refuse = subprocess.run(
        ["bash", str(run_sh_path)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "VALIDATE_ONLY": "1", "MAX_MODEL_LEN": "1048576"},
    )
    if refuse.returncode == 0 or "cannot hold --max-model-len" not in refuse.stderr:
        failures.append("VALIDATE_ONLY=1 MAX_MODEL_LEN=1048576 did not refuse the fp8 1M window")
    tiny_kv = subprocess.run(
        ["bash", str(run_sh_path)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "VALIDATE_ONLY": "1", "KV_CACHE_MEMORY": "3221225472"},
    )
    if tiny_kv.returncode == 0 or "3.62 GiB" not in tiny_kv.stderr:
        failures.append("VALIDATE_ONLY=1 KV_CACHE_MEMORY=3221225472 did not refuse the 3.0 GiB pin")
    min_kv = subprocess.run(
        ["bash", str(run_sh_path)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "VALIDATE_ONLY": "1", "KV_CACHE_MEMORY": "3886945403"},
    )
    if min_kv.returncode == 0 or "327168" not in min_kv.stderr:
        failures.append("VALIDATE_ONLY=1 KV_CACHE_MEMORY=3886945403 did not refuse the 3.62 GiB display pin")

    print(f"repo={REPO}")
    for w in warnings:
        print(f"WARN {w}")
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print(f"result=fail n={len(failures)}")
        return 1
    print("result=pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
