#!/usr/bin/env python3
"""Clone-shape checks: the GitHub recipe a stranger follows. No GPU, no serve."""
from __future__ import annotations

import os
import re
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
    if "enable_thinking | default(true)" not in (REPO / "chat_template.jinja").read_text():
        failures.append("chat_template.jinja no longer gates <think> on enable_thinking")
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
