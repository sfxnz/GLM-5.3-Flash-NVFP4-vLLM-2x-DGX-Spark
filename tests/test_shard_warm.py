"""kit/shard_warm.py against the real engine logs of the first nvidia boot.

Run: python3 -m unittest discover -s tests
No docker: the end-to-end case puts a `docker` stub first on PATH that prints
the recorded log. Shards are empty temp files.
"""
import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVIDENCE = REPO / "evidence" / "iter-nvidia-linear-marlin"
spec = importlib.util.spec_from_file_location("shard_warm", REPO / "kit" / "shard_warm.py")
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)

NAMES = [f"model-{i:05d}-of-00033.safetensors" for i in range(1, 34)]


def log_lines(name):
    # Keep the embedded \r that tqdm writes, as `docker logs` delivers it.
    return (EVIDENCE / name).read_bytes().decode().split("\n")


class Parsing(unittest.TestCase):
    files = [f"/snap/{n}" for n in NAMES]

    def test_natural_order_matches_vllm(self):
        with tempfile.TemporaryDirectory() as d:
            for n in reversed(NAMES + ["model.safetensors.index.json", "config.json"]):
                Path(d, n).touch()
            self.assertEqual([os.path.basename(p) for p in sw.shard_files(d)], NAMES)
        self.assertLess(sw.natural_key("m-2-of-33.safetensors"), sw.natural_key("m-10-of-33.safetensors"))

    def test_real_progress_lines(self):
        lines = log_lines("head.docker.log")
        start = next(line for line in lines if "0/33 [" in line)
        self.assertIn("\r", start)
        self.assertEqual(sw.next_shard(start, self.files), "/snap/model-00002-of-00033.safetensors")
        seven = next(line for line in lines if "| 7/33 [" in line)
        self.assertEqual(sw.next_shard(seven, self.files), "/snap/model-00009-of-00033.safetensors")
        last = [line for line in lines if "33/33 [" in line]
        self.assertEqual(len(last), 2)  # tqdm prints the final line twice
        for line in last + [next(line for line in lines if "32/33 [" in line)]:
            self.assertIsNone(sw.next_shard(line, self.files))

    def test_other_lines_and_other_loads_are_ignored(self):
        lines = log_lines("head.docker.log")
        drafter = [line for line in lines if "/1 [" in line and "checkpoint shards" in line]
        self.assertEqual(len(drafter), 2)
        for line in drafter + [line for line in lines if "Filesystem type" in line]:
            self.assertIsNone(sw.next_shard(line, self.files))
        self.assertIsNone(sw.next_shard("Multi-thread loading shards: 50% Completed | 1/2 [", self.files[:2]))

    def test_follow_head_log_is_one_shard_ahead_and_stops_at_load_end(self):
        said, advised = [], []
        got = sw.follow(log_lines("head.docker.log"), self.files, advised.append, said.append)
        self.assertEqual(got, advised)
        self.assertEqual(advised, self.files[1:])  # shards 2..33, each once, in load order
        self.assertTrue(said[0].startswith("load finished: "), said)
        self.assertIn("Loading weights took 719.26 seconds", said[0])
        self.assertEqual(len(said), 1)

    def test_follow_worker_log_has_nothing_to_follow(self):
        said, advised = [], []
        sw.follow(log_lines("worker.docker.log"), self.files, advised.append, said.append)
        self.assertEqual(advised, [])
        self.assertIn("Loading weights took 214.44 seconds", said[0])
        self.assertIn("global rank 0 only", said[1])

    def test_follow_stops_when_the_stream_ends(self):
        said = []
        sw.follow(["x | 0/33 [", "Loading safetensors checkpoint shards:   3% Completed | 1/33 [00:19<10:11]"],
                  self.files, lambda p: None, said.append)
        self.assertIn("log stream ended", said[0])

    def test_willneed_on_a_real_file(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"\0" * 8192)
            f.flush()
            sw.willneed(f.name)


class EndToEnd(unittest.TestCase):
    def test_main_with_stub_docker(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            snap = d / "snap"
            snap.mkdir()
            for n in NAMES:
                (snap / n).write_bytes(b"\0" * 4096)
            stub = d / "bin" / "docker"
            stub.parent.mkdir()
            stub.write_text(f'#!/bin/sh\necho "$*" >"{d}/args"\ncat "{EVIDENCE}/head.docker.log"\n')
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
            logf = d / "warm.log"
            env = {**os.environ, "PATH": f"{stub.parent}:{os.environ['PATH']}"}
            proc = subprocess.run([sys.executable, str(REPO / "kit" / "shard_warm.py"), "ctr", str(snap),
                                   "--log", str(logf)], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual((d / "args").read_text().strip(), "logs -f ctr")
            text = logf.read_text()
            self.assertIn(f"following ctr: 33 shards in {snap}", text)
            self.assertEqual(text.count("WILLNEED "), 32, text)
            self.assertNotIn("failed", text)
            self.assertIn("WILLNEED model-00002-of-00033.safetensors 0.00 GiB", text)
            self.assertIn("exit: advised 32 shard(s)", text)


if __name__ == "__main__":
    unittest.main()
