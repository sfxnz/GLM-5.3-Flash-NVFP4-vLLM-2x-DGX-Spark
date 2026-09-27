"""VALIDATE_ONLY checks for run.sh. No GPU, no docker, no network.

Run: python3 -m unittest discover -s tests
Every run sets VALIDATE_ONLY=1 and WORKER_HOST=worker.invalid; ssh and docker
are stubs whenever ORCHESTRATE=auto is exercised.
"""
import os
import re
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_SH = REPO / "run.sh"
LIBERTAI = {"MODEL": "LibertAIDAI/GLM-5.3-Flash-NVFP4", "SNAPSHOT_REV": "caca4e6a4ebbd66f159d3d2fc256683fd6e27177"}


class RunShCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def base_env(self, **extra):
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.home),
            "VALIDATE_ONLY": "1",
            "ORCHESTRATE": "0",
            "ROLE": "head",
            "WORKER_HOST": "worker.invalid",
        }
        env.update(extra)
        return env

    def run_sh(self, **extra):
        return subprocess.run(
            ["bash", str(RUN_SH)], env=self.base_env(**extra), capture_output=True, text=True, timeout=60
        )

    def assertRefused(self, proc, needle):
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn(needle, proc.stderr)
        self.assertNotIn("validate-only", proc.stdout)

    def assertAccepted(self, proc):
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("==> validate-only", proc.stdout)


def worker_command(stdout):
    m = re.search(r"^==> worker command: (.*)$", stdout, re.M)
    assert m, stdout
    return m.group(1)


def shell_words(command):
    """Split a command exactly as the worker's bash would."""
    out = subprocess.run(
        ["bash", "-c", 'eval "set -- $1"; printf "%s\\0" "$@"', "_", command],
        capture_output=True, text=True, check=True,
    ).stdout
    return out.split("\0")[:-1]


class Guards(RunShCase):
    def test_defaults_accepted(self):
        proc = self.run_sh()
        self.assertAccepted(proc)
        for want in ("spec=dflash2", "moe=marlin", "linear=marlin", "served=nvidia/GLM-5.3-Flash-NVFP4",
                     "mm_cache_gb=1", "max_new_tokens=65536", '"cudagraph_capture_sizes":[1,2,4,8,16]'):
            self.assertIn(want, proc.stdout)

    def test_linear_backend_refused(self):
        self.assertRefused(self.run_sh(LINEAR_BACKEND="flashinfer_cutlass"), "FORCE_UNSAFE_LINEAR=1")
        self.assertAccepted(self.run_sh(LINEAR_BACKEND="emulation", FORCE_UNSAFE_LINEAR="1"))

    def test_spec_mtp_refused_on_nvidia(self):
        self.assertRefused(self.run_sh(SPEC="mtp"), "FORCE_UNSAFE_SPEC=1")
        self.assertAccepted(self.run_sh(SPEC="mtp", **LIBERTAI))

    def test_language_model_only_must_be_0_or_1(self):
        self.assertRefused(self.run_sh(LANGUAGE_MODEL_ONLY="2"), "want exactly 0 or 1")
        self.assertRefused(self.run_sh(LANGUAGE_MODEL_ONLY="2", FORCE_UNSAFE_VISION="1"), "want exactly 0 or 1")
        self.assertAccepted(self.run_sh(LANGUAGE_MODEL_ONLY="1", FORCE_UNSAFE_VISION="1"))

    def test_extra_env_secret_refused_without_echoing_value(self):
        proc = self.run_sh(EXTRA_ENV="HF_TOKEN=hf_do_not_print")
        self.assertRefused(proc, "EXTRA_ENV refuses 'HF_TOKEN'")
        self.assertNotIn("hf_do_not_print", proc.stderr + proc.stdout)
        for bad in ("VLLM_API_KEY=x", "NCCL_SECRET_X=1", "LD_PRELOAD=/x.so", "no_equals_sign"):
            with self.subTest(bad=bad):
                self.assertRefused(self.run_sh(EXTRA_ENV=bad), "EXTRA_ENV refuses")

    def test_extra_env_allowlist_accepted(self):
        self.assertAccepted(self.run_sh(EXTRA_ENV="MAX_JOBS=2 FLASHINFER_JIT_VERBOSE=1 NCCL_DEBUG=INFO GLM53_X=1"))

    def test_max_new_tokens_must_be_integer(self):
        self.assertRefused(self.run_sh(MAX_NEW_TOKENS="12x"), "positive integer")
        self.assertRefused(self.run_sh(MAX_NEW_TOKENS="-1"), "positive integer")
        proc = self.run_sh(MAX_NEW_TOKENS="0")
        self.assertAccepted(proc)
        self.assertRegex(proc.stdout, r"max_new_tokens=0\n")

    def test_served_name_follows_model(self):
        proc = self.run_sh(**LIBERTAI)
        self.assertAccepted(proc)
        self.assertIn("served=LibertAIDAI/GLM-5.3-Flash-NVFP4", proc.stdout)


class Forwarding(RunShCase):
    TRICKY = {
        "EXTRA_ARGS": "--load-format dummy --served-model-name 'it'\"s\"",
        "EXTRA_ENV": "MAX_JOBS=2 NCCL_DEBUG=INFO",
        "LIMIT_MM_PER_PROMPT": '{"image":2,"video":0}',
        "SNAPSHOT": "/weights/with space/snap",
        "HF_CACHE": "/cache dir/hf",
        "LINEAR_BACKEND": "emulation",
        "FORCE_UNSAFE_LINEAR": "1",
        "MM_PROCESSOR_CACHE_GB": "0.5",
    }

    def test_every_generated_var_is_forwarded(self):
        block = RUN_SH.read_text().split("# BEGIN generated", 1)[1].split("# END generated", 1)[0]
        generated = re.findall(r'^([A-Z][A-Z0-9_]*)="\$\{\1:-', block, re.M)
        derived = ["SNAPSHOT", "SNAPSHOT_IN_CONTAINER", "LIMIT_MM_PER_PROMPT", "HF_HUB_DISABLE_XET",
                   "SPEC_CONFIG", "ENFORCE_EAGER", "COMPILATION_CONFIG", "SKIP_DOWNLOAD", "EXTRA_ARGS", "EXTRA_ENV"]
        words = shell_words(worker_command(self.run_sh().stdout))
        names = {w.split("=", 1)[0] for w in words if "=" in w}
        self.assertTrue(generated)
        self.assertEqual(sorted(set(generated + derived) - names), [])
        self.assertEqual(words[:3], ["env", "ROLE=worker", "ORCHESTRATE=0"])
        self.assertEqual(words[-2:], ["bash", "/tmp/glm53-run.sh"])

    def test_worker_resolves_the_same_config(self):
        head = self.run_sh(**self.TRICKY)
        self.assertAccepted(head)
        words = shell_words(worker_command(head.stdout))
        worker_env = dict(w.split("=", 1) for w in words[1:-2])
        self.assertEqual(worker_env["EXTRA_ARGS"], self.TRICKY["EXTRA_ARGS"])
        self.assertEqual(worker_env["LIMIT_MM_PER_PROMPT"], self.TRICKY["LIMIT_MM_PER_PROMPT"])
        env = {"PATH": os.environ["PATH"], "HOME": str(self.home), "VALIDATE_ONLY": "1", **worker_env}
        worker = subprocess.run(["bash", str(RUN_SH)], env=env, capture_output=True, text=True, timeout=60)
        self.assertAccepted(worker)
        # Identical validate line and identical forwarded values on both ranks.
        self.assertEqual(head.stdout, worker.stdout)


class JitCache(RunShCase):
    ENVS = ("FLASHINFER_WORKSPACE_BASE=/jit-cache/flashinfer", "VLLM_CACHE_ROOT=/jit-cache/vllm",
            "DG_JIT_CACHE_DIR=/jit-cache/vllm/deep_gemm", "TRITON_CACHE_DIR=/jit-cache/triton",
            "TILELANG_CACHE_DIR=/jit-cache/tilelang")

    def jit_line(self, stdout):
        m = re.search(r"^==> jit_cache=(\S+) args: (.*)$", stdout, re.M)
        self.assertTrue(m, stdout)
        return m.group(1), m.group(2)

    def test_default_mounts_one_dir_per_image(self):
        proc = self.run_sh()
        self.assertAccepted(proc)
        flag, args = self.jit_line(proc.stdout)
        self.assertEqual(flag, "1")
        self.assertIn(f"-v {self.home}/projects/data/glm53-jit-cache/<image-id>:/jit-cache", args)
        for env in self.ENVS:
            self.assertIn(f"-e {env}", args)

    def test_jit_cache_0_drops_mount_and_env(self):
        proc = self.run_sh(JIT_CACHE="0")
        self.assertAccepted(proc)
        self.assertEqual(self.jit_line(proc.stdout), ("0", ""))
        self.assertNotIn("/jit-cache", proc.stdout.split("==> worker command:")[0])

    def test_jit_cache_must_be_0_or_1(self):
        self.assertRefused(self.run_sh(JIT_CACHE="yes"), "JIT_CACHE=yes: want exactly 0 or 1")

    def test_worker_gets_jit_cache_settings(self):
        words = shell_words(worker_command(self.run_sh(JIT_CACHE="0", JIT_CACHE_DIR="/d/jit cache").stdout))
        self.assertIn("JIT_CACHE=0", words)
        self.assertIn("JIT_CACHE_DIR=/d/jit cache", words)


class WarmShards(RunShCase):
    def test_default_is_head_only(self):
        proc = self.run_sh()
        self.assertAccepted(proc)
        self.assertIn(f"==> warm_shards=1 log_dir={self.home}/projects/data/glm53-jit-cache/logs\n", proc.stdout)

    def test_values(self):
        for ok in ("0", "1", "all"):
            with self.subTest(ok=ok):
                self.assertAccepted(self.run_sh(WARM_SHARDS=ok))
        self.assertRefused(self.run_sh(WARM_SHARDS="2"), "WARM_SHARDS=2: want 0, 1 (head only) or all")

    def test_worker_gets_warm_shards(self):
        self.assertIn("WARM_SHARDS=all", shell_words(worker_command(self.run_sh(WARM_SHARDS="all").stdout)))


class StubbedLaunch(RunShCase):
    """The real launch path (no VALIDATE_ONLY) with docker, sudo and curl stubbed.

    IMAGE and CONTAINER_NAME are test-only names, so even an unstubbed docker
    would stop at ensure_image before any `docker run`.
    """
    ID = "sha256:0123456789abcdef0123"

    def stub(self, name, body):
        path = self.home / "bin" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def setUp(self):
        super().setUp()
        self.log = self.home / "stub.log"
        self.stub("docker", 'echo "docker $*" >>"$STUB_LOG"\n'
                            'case "$1" in\n'
                            f'  image) [[ "$3" == -f ]] && echo {self.ID}; exit 0 ;;\n'
                            '  ps|run) exit 0 ;;\n'
                            '  logs) cat "$STUB_DOCKER_LOGS" ;;\n'
                            '  *) exit 1 ;;\n'
                            'esac\n')
        self.stub("sudo", "exit 1\n")
        self.stub("curl", "exit 0\n")
        self.snap = self.home / "snap"
        self.snap.mkdir()
        for i in range(1, 34):
            (self.snap / f"model-{i:05d}-of-00033.safetensors").write_bytes(b"\0" * 4096)

    def launch(self, **extra):
        env = self.base_env(
            PATH=f"{self.home / 'bin'}:{os.environ['PATH']}", STUB_LOG=str(self.log),
            STUB_DOCKER_LOGS=str(REPO / "evidence/iter-nvidia-linear-marlin/head.docker.log"),
            IMAGE="glm53-test-stub-image", CONTAINER_NAME="glm53-test-stub", HF_CACHE=str(self.home / "hf"),
            SNAPSHOT=str(self.snap), SKIP_DOWNLOAD="1", **extra)
        del env["VALIDATE_ONLY"]
        return subprocess.run(["bash", str(RUN_SH)], env=env, capture_output=True, text=True, timeout=60)

    def warm_log(self):
        """Wait for the background warmer to finish; return its only log."""
        for _ in range(200):
            logs = list((self.home / "projects/data/glm53-jit-cache/logs").glob("shard_warm-*.log"))
            if len(logs) == 1 and "exit: advised" in logs[0].read_text():
                break
            time.sleep(0.05)
        self.assertEqual(len(logs), 1, logs)
        return logs[0].read_text()

    def test_head_mounts_jit_cache_and_warms(self):
        proc = self.launch()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        jit = self.home / "projects/data/glm53-jit-cache/0123456789ab"
        self.assertTrue(jit.is_dir())
        run = next(line for line in self.log.read_text().splitlines() if line.startswith("docker run "))
        self.assertIn(f"-v {jit}:/jit-cache -e FLASHINFER_WORKSPACE_BASE=/jit-cache/flashinfer", run)
        self.assertIn("-e TILELANG_CACHE_DIR=/jit-cache/tilelang glm53-test-stub-image ", run)
        self.assertIn("Shard warmer pid", proc.stdout)
        text = self.warm_log()
        self.assertIn("following glm53-test-stub: 33 shards", text)
        self.assertEqual(text.count("WILLNEED "), 32, text)
        self.assertIn("exit: advised 32 shard(s)", text)

    def test_jit_cache_0_and_warm_shards_0(self):
        proc = self.launch(JIT_CACHE="0", WARM_SHARDS="0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("/jit-cache", self.log.read_text())
        self.assertNotIn("Shard warmer", proc.stdout)
        self.assertFalse((self.home / "projects/data/glm53-jit-cache").exists())

    def test_worker_warms_only_with_all(self):
        proc = self.launch(ROLE="worker")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("Shard warmer", proc.stdout)
        proc = self.launch(ROLE="worker", WARM_SHARDS="all")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Shard warmer pid", proc.stdout)
        self.warm_log()


class ImageParity(RunShCase):
    def stub(self, name, body):
        path = self.home / "bin" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def setUp(self):
        super().setUp()
        self.stub("ssh", 'echo "$*" >>"$STUB_LOG"\n'
                         'case "$*" in *"docker image inspect"*) echo "$STUB_REMOTE_ID" ;; esac\n')
        self.stub("docker", 'echo "docker $*" >>"$STUB_LOG"\n'
                            'case "$*" in "image inspect"*) echo "$STUB_LOCAL_ID" ;; *) exit 1 ;; esac\n')
        self.log = self.home / "stub.log"

    def run_auto(self, local_id, remote_id):
        return self.run_sh(
            ORCHESTRATE="auto",
            PATH=f"{self.home / 'bin'}:{os.environ['PATH']}",
            STUB_LOG=str(self.log),
            STUB_LOCAL_ID=local_id,
            STUB_REMOTE_ID=remote_id,
        )

    def test_match_ok(self):
        proc = self.run_auto("sha256:aaa", "sha256:aaa")
        self.assertAccepted(proc)
        self.assertIn("Image parity OK", proc.stdout)
        self.assertNotIn("WARN", proc.stderr)

    def test_mismatch_warns_with_sync_command(self):
        proc = self.run_auto("sha256:aaa", "sha256:bbb")
        self.assertAccepted(proc)
        self.assertIn("WARN image glm53-sm121-v11 differs", proc.stderr)
        self.assertIn("docker save glm53-sm121-v11 | ssh worker.invalid docker load", proc.stderr)
        self.assertNotIn("docker run", self.log.read_text())

    def test_orchestrate_0_never_sshes(self):
        proc = self.run_sh(PATH=f"{self.home / 'bin'}:{os.environ['PATH']}", STUB_LOG=str(self.log))
        self.assertAccepted(proc)
        self.assertFalse(self.log.exists(), self.log.read_text() if self.log.exists() else "")


if __name__ == "__main__":
    unittest.main()
