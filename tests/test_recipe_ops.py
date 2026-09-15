#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NVIDIA_PIN = "09b04e5e74bca08ca8549fc736d4cdd8624bfde3"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("FORCE_UNSAFE_CTX", None)
    env.pop("FORCE_UNSAFE_MOE", None)
    env.pop("FORCE_UNSAFE_VISION", None)
    env.pop("FORCE_UNSAFE_UMA", None)
    env.update(extra)
    return env


def _run_sh(**extra: str) -> subprocess.CompletedProcess[str]:
    env = _env(**extra)
    env["VALIDATE_ONLY"] = "1"
    return subprocess.run(
        [str(ROOT / "run.sh")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )


def _func_body(src: str, name: str) -> str:
    m = re.search(rf"^{name}\(\) \{{(.*?)^\}}", src, re.M | re.S)
    if m is None:
        raise AssertionError(f"missing function {name}")
    return m.group(1)


class RecipeOpsTests(unittest.TestCase):
    def test_nvidia_pin_and_vision_defaults_unchanged(self) -> None:
        run = _read("run.sh")
        recipe = _read("recipe.yaml")
        self.assertIn(f'SNAPSHOT_REV="${{SNAPSHOT_REV:-{NVIDIA_PIN}}}"', run)
        self.assertIn("id: &model nvidia/GLM-5.3-Flash-NVFP4", recipe)
        self.assertIn(NVIDIA_PIN, recipe)
        self.assertIn('LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"', run)
        self.assertIn('MAX_MODEL_LEN="${MAX_MODEL_LEN:-327680}"', run)
        self.assertNotIn("MAX_MODEL_LEN: 1048576", recipe)

    def test_uma_defaults(self) -> None:
        run = _read("run.sh")
        self.assertIn('UMA_RESERVE_GIB="${UMA_RESERVE_GIB:-20}"', run)
        self.assertIn('UMA_ABORT_GIB="${UMA_ABORT_GIB:-16}"', run)
        self.assertIn('FORCE_UNSAFE_UMA="${FORCE_UNSAFE_UMA:-0}"', run)

    def test_refuse_foreign_serve_skips_conduit_and_does_not_rm(self) -> None:
        body = _func_body(_read("run.sh"), "refuse_foreign_serve")
        self.assertIn('"$name" == conduit', body)
        self.assertIn("gpu|nvidia|infiniband", body)
        self.assertIn("Do not docker rm that container from this script", body)
        self.assertNotIn("docker rm", body.replace("Do not docker rm that container from this script", ""))

    def test_abort_load_runs_stop_sh(self) -> None:
        body = _func_body(_read("run.sh"), "abort_load")
        self.assertIn("stop.sh", body)
        self.assertIn("Stopping both ranks", body)

    def test_wait_ready_aborts_on_death_or_low_uma(self) -> None:
        wait = _func_body(_read("run.sh"), "wait_ready")
        self.assertIn("abort_load", wait)
        self.assertIn("wait_uma_or_abort", wait)
        self.assertIn("Container exited early", wait)
        self.assertIn("Worker container", wait)
        self.assertNotIn("exit 1", wait)

    def test_head_preflight_before_worker_scp(self) -> None:
        run = _read("run.sh")
        idx = run.find('ORCHESTRATE" == "auto" && "$ROLE" == "head"')
        self.assertGreater(idx, 0)
        block = run[idx:]
        scp = block.find("scp ")
        self.assertGreater(scp, 0)
        self.assertLess(block.find("refuse_foreign_serve"), scp)
        self.assertLess(block.find("refuse_low_uma"), scp)
        self.assertIn('refuse_foreign_serve "$WORKER_HOST"', block)
        self.assertIn('refuse_low_uma "$WORKER_HOST"', block)
        self.assertIn("FORCE_UNSAFE_UMA='$FORCE_UNSAFE_UMA'", block)
        self.assertIn("UMA_RESERVE_GIB='$UMA_RESERVE_GIB'", block)
        self.assertIn("UMA_ABORT_GIB='$UMA_ABORT_GIB'", block)

    def test_validate_only_defaults_pass(self) -> None:
        proc = _run_sh()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("validate-only", proc.stdout)

    def test_validate_only_still_refuses_1m_and_hidden_vision(self) -> None:
        one_m = _run_sh(MAX_MODEL_LEN="1048576")
        self.assertNotEqual(one_m.returncode, 0)
        self.assertIn("cannot hold --max-model-len", one_m.stderr)
        hidden = _run_sh(LANGUAGE_MODEL_ONLY="1")
        self.assertNotEqual(hidden.returncode, 0)
        self.assertIn("FORCE_UNSAFE_VISION=1", hidden.stderr)


if __name__ == "__main__":
    unittest.main()
