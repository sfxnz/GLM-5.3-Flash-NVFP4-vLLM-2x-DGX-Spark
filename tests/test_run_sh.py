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
        self.assertAccepted(self.run_sh(EXTRA_ENV="MAX_JOBS=2 FLASHINFER_JIT_VERBOSE=1 NCCL_DEBUG=INFO GLM53_X=1 "
                                                  "FLASHINFER_JIT_DEBUG=0"))

    def test_jit_verbose_needs_jit_debug_0(self):
        # FlashInfer reads VERBOSE=1 as DEBUG=1 when DEBUG is unset: -O0 --device-debug serving kernels.
        for bad in ("FLASHINFER_JIT_VERBOSE=1", "MAX_JOBS=2 FLASHINFER_JIT_VERBOSE=1",
                    "FLASHINFER_JIT_VERBOSE=1 FLASHINFER_JIT_DEBUG=1"):
            with self.subTest(bad=bad):
                self.assertRefused(self.run_sh(EXTRA_ENV=bad), "FLASHINFER_JIT_DEBUG=0")
        for ok in ("FLASHINFER_JIT_DEBUG=0 FLASHINFER_JIT_VERBOSE=1", "FLASHINFER_JIT_VERBOSE=0"):
            with self.subTest(ok=ok):
                self.assertAccepted(self.run_sh(EXTRA_ENV=ok))

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


class DraftRev(RunShCase):
    PIN = "7d74cdd881ed7e32c31175984a67823127b66cfe"
    NEWER = "bf582e4eacc1810f76656d1811693ff6c6737d2a"

    def test_default_is_the_pin(self):
        proc = self.run_sh()
        self.assertAccepted(proc)
        self.assertIn(f" draft_rev={self.PIN} ", proc.stdout)
        words = shell_words(worker_command(proc.stdout))
        self.assertIn(f"DRAFT_REV={self.PIN}", words)
        self.assertIn(f"snapshots/{self.PIN}\"", next(w for w in words if w.startswith("SPEC_CONFIG=")))

    def test_override_drives_both_paths_and_the_worker(self):
        proc = self.run_sh(DRAFT_REV=self.NEWER)
        self.assertAccepted(proc)
        self.assertIn(f" draft_rev={self.NEWER} ", proc.stdout)
        words = shell_words(worker_command(proc.stdout))
        self.assertIn(f"DRAFT_REV={self.NEWER}", words)
        spec = next(w for w in words if w.startswith("SPEC_CONFIG="))
        self.assertIn(f"/cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/{self.NEWER}\"", spec)
        self.assertNotIn(self.PIN, proc.stdout)

    def test_short_or_branch_rev_refused(self):
        for bad in ("bf582e4", "main", self.NEWER.upper()):
            with self.subTest(bad=bad):
                self.assertRefused(self.run_sh(DRAFT_REV=bad), "want a full 40-hex commit sha")


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


class LogitsFp32(RunShCase):
    V13 = {"IMAGE": "glm53-sm121-v13"}

    def test_default_is_0(self):
        proc = self.run_sh()
        self.assertAccepted(proc)
        self.assertIn(" logits_fp32=0 ", proc.stdout)
        self.assertIn("LOGITS_FP32=0", shell_words(worker_command(proc.stdout)))

    def test_1_is_forwarded_and_the_worker_resolves_the_same(self):
        head = self.run_sh(LOGITS_FP32="1", **self.V13)
        self.assertAccepted(head)
        self.assertIn(" logits_fp32=1 ", head.stdout)
        words = shell_words(worker_command(head.stdout))
        self.assertIn("LOGITS_FP32=1", words)
        env = {"PATH": os.environ["PATH"], "HOME": str(self.home), "VALIDATE_ONLY": "1",
               **dict(w.split("=", 1) for w in words[1:-2])}
        worker = subprocess.run(["bash", str(RUN_SH)], env=env, capture_output=True, text=True, timeout=60)
        self.assertAccepted(worker)
        self.assertEqual(head.stdout, worker.stdout)

    def test_only_0_or_1(self):
        for bad in ("2", "yes", "01", "true"):
            with self.subTest(bad=bad):
                self.assertRefused(self.run_sh(LOGITS_FP32=bad, **self.V13), f"LOGITS_FP32={bad}: want exactly 0 or 1")

    def test_1_refused_on_the_v11_image(self):
        self.assertRefused(self.run_sh(LOGITS_FP32="1"), "LOGITS_FP32=1 needs IMAGE=glm53-sm121-v13")
        self.assertAccepted(self.run_sh(LOGITS_FP32="0"))


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
                            '  *) exit 1 ;;\n'
                            'esac\n')
        self.stub("sudo", "exit 1\n")
        self.stub("curl", "exit 0\n")
        self.snap = self.home / "snap"
        self.snap.mkdir()

    def launch(self, **extra):
        env = self.base_env(**{
            "PATH": f"{self.home / 'bin'}:{os.environ['PATH']}", "STUB_LOG": str(self.log),
            "IMAGE": "glm53-test-stub-image", "CONTAINER_NAME": "glm53-test-stub", "HF_CACHE": str(self.home / "hf"),
            "SNAPSHOT": str(self.snap), "SKIP_DOWNLOAD": "1", **extra})
        del env["VALIDATE_ONLY"]
        return subprocess.run(["bash", str(RUN_SH)], env=env, capture_output=True, text=True, timeout=60)

    def test_head_mounts_jit_cache(self):
        proc = self.launch()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        jit = self.home / "projects/data/glm53-jit-cache/0123456789ab"
        self.assertTrue(jit.is_dir())
        run = next(line for line in self.log.read_text().splitlines() if line.startswith("docker run "))
        self.assertIn(f"-v {jit}:/jit-cache -e FLASHINFER_WORKSPACE_BASE=/jit-cache/flashinfer", run)
        self.assertIn("-e TILELANG_CACHE_DIR=/jit-cache/tilelang glm53-test-stub-image ", run)

    def test_jit_cache_0(self):
        proc = self.launch(JIT_CACHE="0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("/jit-cache", self.log.read_text())
        self.assertFalse((self.home / "projects/data/glm53-jit-cache").exists())

    def test_draft_download_pins_draft_rev(self):
        self.stub("hf", 'echo "hf $*" >>"$STUB_LOG"\n')
        rev = DraftRev.NEWER
        proc = self.launch(SKIP_DOWNLOAD="0", JIT_CACHE="0", DRAFT_REV=rev)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(f"Downloading incoai/GLM-5.3-Flash-DFlash2 @ {rev}", proc.stdout)
        calls = [line for line in self.log.read_text().splitlines() if line.startswith("hf ")]
        self.assertEqual(calls, [f"hf download incoai/GLM-5.3-Flash-DFlash2 --revision {rev}"])
        run = next(line for line in self.log.read_text().splitlines() if line.startswith("docker run "))
        self.assertIn(f"/cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/{rev}", run)

    def test_logits_fp32_passes_the_nested_head_dtype_override(self):
        want = '--hf-overrides {"text_config":{"head_dtype":"float32"}} '
        for value, present in (("0", False), ("1", True)):
            with self.subTest(LOGITS_FP32=value):
                self.log.unlink(missing_ok=True)
                proc = self.launch(LOGITS_FP32=value, JIT_CACHE="0")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                run = next(line for line in self.log.read_text().splitlines() if line.startswith("docker run "))
                self.assertEqual(want in run, present, run)
                self.assertEqual("--hf-overrides" in run, present, run)


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
