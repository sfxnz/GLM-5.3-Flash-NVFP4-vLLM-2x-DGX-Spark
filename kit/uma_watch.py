#!/usr/bin/env python3
"""UMA watcher for one TP=2 boot on 2x DGX Spark (head = this host, worker via ssh).

Samples MemAvailable and swap on both nodes every 0.25 s, follows both ranks'
`docker logs -f` to track the boot phase, and kills both ranks in parallel when
a phase floor is crossed. Also logs compiler processes once a second.

  LOAD      until both ranks log 'Model loading took'        floor 10 GiB
  PROFILE   until 'GPU KV cache size' / '... for KV Cache'    floor 8 GiB + slope rule
  KV_READY  until /v1/models answers 200                      floor 3 GiB
  SERVING   afterwards                                        floor 2 GiB

Kill = `docker kill` on both nodes in parallel (worker over a pre-opened ssh
ControlMaster), then save both ranks' `docker logs`, then `docker rm -f`.

  python3 kit/uma_watch.py --evidence DIR                   # watch a boot
  python3 kit/uma_watch.py --evidence DIR --phase SERVING   # re-attach to a ready serve
  python3 kit/uma_watch.py --evidence DIR --test-kill       # prove kill latency on dummy containers
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

KIB_PER_GIB = 1024 * 1024
PHASES = ["LOAD", "PROFILE", "KV_READY", "SERVING"]
FLOOR_GIB = {"LOAD": 10.0, "PROFILE": 8.0, "KV_READY": 3.0, "SERVING": 2.0}
SLOPE_DROP_GIB = 0.75  # PROFILE: two consecutive samples each dropping this much...
SLOPE_BELOW_GIB = 14.0  # ...while below this level
STALE_S = 5.0  # PROFILE/KV_READY: a silent sampler this long counts as a hang
SERVING_ROW_S = 5.0  # SERVING: write a row this often (or on a new minimum); every sample is still checked
LOADED = "Model loading took"
KV_READY = ("GPU KV cache size", "memory for KV Cache")
NOTABLE = re.compile(
    r"for NVFP4 GEMM|MoE backend|scale_2 must match|Model loading took|Encoder cache|TileLang begins"
    r"|KV cache size|for KV Cache|init engine|startup complete|Traceback|out of memory|NV_ERR|Xid"
)

# Runs on each node as `python3 -u -c`. The bracketed letters keep the pattern
# from matching its own command line (and the ssh line that carries it).
SAMPLER = r"""
import re, subprocess, time
pat = re.compile(r"nv[c]c|ci[c]c|pt[x]as|cu[d]afe|nin[j]a")
n = 0
while True:
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0])
    print("M", m["MemAvailable"], m["SwapTotal"] - m["SwapFree"], flush=True)
    if n % 4 == 0:
        ps = subprocess.run(["ps", "-eo", "pid,rss,etimes,args"], capture_output=True, text=True).stdout
        for l in ps.splitlines()[1:]:
            if pat.search(l):
                print("C", l.strip()[:400], flush=True)
    n += 1
    time.sleep(0.25)
"""


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def open_tsv(path: str, header: str):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", buffering=1)
    if new:
        f.write(header + "\n")
    return f


class Watch:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.ev = a.evidence
        os.makedirs(self.ev, exist_ok=True)
        self.head = socket.gethostname().split(".")[0].lower()
        self.worker = a.worker
        self.cm = os.path.join(a.cm_dir, f"uma-{os.getpid()}-%C")
        self.ssh = ["ssh", "-o", "BatchMode=yes", "-o", f"ControlPath={self.cm}", "-o", "ControlMaster=no", a.worker]
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.phase = a.phase
        self.seen: dict[str, set[str]] = {"head": set(), "worker": set()}
        self.last: dict[str, tuple[float, float]] = {}  # node -> (time, GiB)
        self.fast: dict[str, int] = {}  # node -> consecutive fast drops (PROFILE)
        self.mins: dict[tuple[str, str], float] = {}  # (phase, node) -> GiB
        self.last_row: dict[str, float] = {}
        self.stopping = threading.Event()
        self.killed = threading.Event()
        self.procs: list[subprocess.Popen] = []
        self.uma = open_tsv(os.path.join(self.ev, "uma.tsv"), "ts\tnode\tphase\tmemavail_gib\tswap_used_gib")
        self.comp = open_tsv(os.path.join(self.ev, "compilers.tsv"), "ts\tnode\tphase\tpid\trss_kib\tetimes_s\targs")
        self.wlog = open(os.path.join(self.ev, "watch.log"), "a", buffering=1)

    # ---- plumbing -------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{utc()} {msg}"
        with self.log_lock:
            self.wlog.write(line + "\n")
            print(line, flush=True)

    def remote(self, cmd: str) -> list[str]:
        return self.ssh + [cmd]

    def spawn(self, cmd: list[str]) -> subprocess.Popen:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace")
        self.procs = [q for q in self.procs if q.poll() is None] + [p]
        return p

    def open_master(self) -> None:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=yes", "-o", f"ControlPath={self.cm}",
                        "-o", "ControlPersist=yes", "-o", "ServerAliveInterval=5", "-fN", self.worker],
                       check=True, timeout=20, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self.wlog)
        self.log(f"ssh ControlMaster up for {self.worker}")

    def close(self) -> None:
        self.stopping.set()
        with self.lock:
            mins = " ".join(f"{p}/{n}={v:.2f}" for (p, n), v in sorted(self.mins.items(), key=lambda kv: (PHASES.index(kv[0][0]), kv[0][1])))
        self.log(f"exit; min MemAvailable GiB per phase/node: {mins or 'n/a'}")
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        subprocess.run(["ssh", "-o", f"ControlPath={self.cm}", "-O", "exit", self.worker],
                       capture_output=True, timeout=10)

    # ---- samples --------------------------------------------------------
    def sampler(self, node: str, cmd: list[str]) -> None:
        while not self.stopping.is_set():
            p = self.spawn(cmd)
            for line in p.stdout:
                if line.startswith("M "):
                    _, avail, swap = line.split()
                    self.sample(node, int(avail) / KIB_PER_GIB, int(swap) / KIB_PER_GIB)
                elif line.startswith("C "):
                    self.compiler(node, line[2:].strip())
                else:
                    self.log(f"{node} sampler: {line.rstrip()}")
            p.wait()
            if not self.stopping.is_set():
                self.log(f"{node} sampler exited rc={p.returncode}; restarting")
                time.sleep(1)

    def sample(self, node: str, avail: float, swap: float) -> None:
        reason = None
        with self.lock:
            now, phase = time.time(), self.phase
            key = (phase, node)
            new_min = avail < self.mins.get(key, 1e9)
            if new_min:
                self.mins[key] = avail
            if phase != "SERVING" or new_min or now - self.last_row.get(node, 0) >= SERVING_ROW_S:
                self.uma.write(f"{utc()}\t{node}\t{phase}\t{avail:.3f}\t{swap:.3f}\n")
                self.last_row[node] = now
            prev = self.last.get(node)
            self.last[node] = (now, avail)
            if avail < FLOOR_GIB[phase]:
                reason = f"{node} MemAvailable {avail:.2f} GiB < {phase} floor {FLOOR_GIB[phase]:.0f} GiB"
            elif phase == "PROFILE" and prev is not None:
                fast = prev[1] - avail >= SLOPE_DROP_GIB and avail < SLOPE_BELOW_GIB
                self.fast[node] = self.fast.get(node, 0) + 1 if fast else 0
                if self.fast[node] >= 2:
                    reason = (f"{node} PROFILE slope: {prev[1]:.2f} -> {avail:.2f} GiB, second consecutive "
                              f"drop >= {SLOPE_DROP_GIB} GiB below {SLOPE_BELOW_GIB:.0f} GiB")
        if reason:
            self.kill(reason, f"{utc()} node={node} phase={phase} memavail_gib={avail:.3f} swap_used_gib={swap:.3f}")

    def compiler(self, node: str, line: str) -> None:
        parts = line.split(None, 3)
        if len(parts) < 4:
            return
        pid, rss, etimes, args = parts
        with self.lock:
            self.comp.write(f"{utc()}\t{node}\t{self.phase}\t{pid}\t{rss}\t{etimes}\t{args}\n")
        if self.a.kill_compiler_regex and re.search(self.a.kill_compiler_regex, args):
            self.kill(f"{node} compiler gate /{self.a.kill_compiler_regex}/", f"pid={pid} rss_kib={rss} args={args}")

    # ---- phases ---------------------------------------------------------
    def follower(self, rank: str, cmd: list[str]) -> None:
        with open(os.path.join(self.ev, f"{rank}.stream.log"), "a", buffering=1) as out:
            while not self.stopping.is_set():
                p = self.spawn(cmd)
                for line in p.stdout:
                    if "No such container" in line:
                        continue
                    out.write(line)
                    self.on_log(rank, line)
                p.wait()
                time.sleep(2)

    def on_log(self, rank: str, line: str) -> None:
        text = line.strip()[:300]
        if NOTABLE.search(line):
            self.log(f"{rank}: {text}")
        if self.a.kill_log_regex and re.search(self.a.kill_log_regex, line):
            self.kill(f"{rank} log gate /{self.a.kill_log_regex}/", text)
        if LOADED in line:
            self.seen[rank].add("loaded")
        if any(k in line for k in KV_READY):
            self.seen[rank].add("kv")
        loaded = all("loaded" in s for s in self.seen.values())
        kv = any("kv" in s for s in self.seen.values())
        if loaded:
            self.advance("PROFILE")
        if loaded and kv:
            self.advance("KV_READY")

    def advance(self, phase: str) -> None:
        with self.lock:
            if PHASES.index(phase) <= PHASES.index(self.phase):
                return
            old, self.phase = self.phase, phase
            self.fast.clear()
            mins = " ".join(f"{n}={v:.2f}" for (p, n), v in sorted(self.mins.items()) if p == old)
        self.log(f"PHASE {old} -> {phase} (min GiB in {old}: {mins or 'n/a'})")

    def http_poll(self) -> None:
        url = f"http://127.0.0.1:{self.a.port}/v1/models"
        while not self.stopping.is_set() and self.phase != "SERVING":
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        self.advance("SERVING")
            except Exception:
                pass
            time.sleep(2)

    # ---- kill -----------------------------------------------------------
    def kill(self, reason: str, sample: str) -> None:
        if self.killed.is_set():
            return
        self.killed.set()
        c = self.a.container
        t0 = time.time()
        procs = {"head": subprocess.Popen(["docker", "kill", c], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True),
                 "worker": subprocess.Popen(self.remote(f"docker kill {c}"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)}
        done = {}
        for k, p in procs.items():
            try:
                out = p.communicate(timeout=30)[0].strip()
            except subprocess.TimeoutExpired:
                p.kill()
                out = "TIMEOUT after 30 s"
            done[k] = (out, p.returncode, time.time() - t0)
        latency = max(v[2] for v in done.values())
        running = {"head": subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", c], capture_output=True, text=True).stdout.strip(),
                   "worker": subprocess.run(self.remote(f"docker inspect -f '{{{{.State.Running}}}}' {c}"), capture_output=True, text=True).stdout.strip()}
        confirmed = time.time() - t0
        lines = [f"reason={reason}", f"sample={sample}", f"kill_started={datetime.datetime.fromtimestamp(t0, datetime.timezone.utc).isoformat()}",
                 f"kill_latency_s={latency:.3f} (both docker kill returned, run in parallel)",
                 f"stopped_confirmed_s={confirmed:.3f} running_after head={running['head']} worker={running['worker']}"]
        lines += [f"{k}: rc={v[1]} t={v[2]:.3f}s out={v[0]!r}" for k, v in done.items()]
        with open(os.path.join(self.ev, "abort.reason"), "w") as f:
            f.write("\n".join(lines) + "\n")
        self.log("KILL " + " | ".join(lines))
        with open(os.path.join(self.ev, "head.docker.log"), "w") as f:
            subprocess.run(["docker", "logs", c], stdout=f, stderr=subprocess.STDOUT)
        with open(os.path.join(self.ev, "worker.docker.log"), "w") as f:
            subprocess.run(self.remote(f"docker logs {c}"), stdout=f, stderr=subprocess.STDOUT)
        subprocess.run(["docker", "rm", "-f", c], capture_output=True)
        subprocess.run(self.remote(f"docker rm -f {c}"), capture_output=True)
        with open(os.path.join(self.ev, "abort.snapshot.txt"), "w") as f:
            for name, cmd in (("head", ["bash", "-c", "free -h; swapon --show; docker ps -a"]),
                              ("worker", self.remote("free -h; swapon --show; docker ps -a"))):
                f.write(f"== {name} {utc()}\n")
                f.flush()
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        self.log("logs saved, containers removed; watcher exiting")
        self.stopping.set()

    # ---- modes ----------------------------------------------------------
    def run(self) -> int:
        self.open_master()
        c = self.a.container
        cmd = ["python3", "-u", "-c", SAMPLER]
        jobs = [(self.sampler, (self.head, cmd)),
                (self.sampler, (self.worker, self.remote("python3 -u -c " + shlex.quote(SAMPLER)))),
                (self.follower, ("head", ["docker", "logs", "-f", c])),
                (self.follower, ("worker", self.remote(f"docker logs -f {c}"))),
                (self.http_poll, ())]
        for fn, args in jobs:
            threading.Thread(target=fn, args=args, daemon=True).start()
        self.log(f"watching {c} head={self.head} worker={self.worker} phase={self.phase} floors={FLOOR_GIB} "
                 f"kill_log_regex={self.a.kill_log_regex!r} kill_compiler_regex={self.a.kill_compiler_regex!r}")
        gone = tick = 0
        while not self.stopping.wait(1.0):
            tick += 1
            with self.lock:
                phase, last = self.phase, dict(self.last)
            if phase in ("PROFILE", "KV_READY"):
                for node, (t, _) in last.items():
                    if time.time() - t > STALE_S:
                        self.kill(f"{node} sampler silent for {time.time() - t:.1f} s in {phase}", "stale")
            if phase == "SERVING" and tick % 10 == 0:
                up = subprocess.run(["docker", "ps", "-q", "-f", f"name=^{c}$"], capture_output=True, text=True).stdout.strip()
                gone = 0 if up else gone + 1
                if gone >= 3:
                    self.log(f"{c} gone on the head for 30 s in SERVING; watcher exiting")
                    break
        self.close()
        return 2 if self.killed.is_set() else 0

    def test_kill(self) -> int:
        c, image = self.a.container, self.a.test_image
        self.open_master()
        for name, cmd in (("head", ["docker", "ps", "-aq", "-f", f"name=^{c}$"]), ("worker", self.remote(f"docker ps -aq -f name=^{c}$"))):
            if subprocess.run(cmd, capture_output=True, text=True).stdout.strip():
                self.log(f"refusing --test-kill: a container named {c} already exists on {name}")
                self.close()
                return 1
        run = f"docker run -d --name {c} --entrypoint sleep {image} infinity"
        subprocess.run(shlex.split(run), check=True, capture_output=True)
        subprocess.run(self.remote(run), check=True, capture_output=True)
        self.log(f"dummy {c} running on both nodes ({image}, sleep infinity, no GPUs)")
        self.kill("test-kill", "dummy containers")
        self.close()
        print(open(os.path.join(self.ev, "abort.reason")).read(), end="")
        return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evidence", required=True, help="directory for uma.tsv, compilers.tsv, watch.log, abort.*")
    p.add_argument("--container", default="glm53-flash-nvfp4")
    p.add_argument("--worker", default="spark2")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--phase", choices=PHASES, default="LOAD", help="starting phase (SERVING to re-attach)")
    p.add_argument("--kill-log-regex", default="", help="kill when a log line on either rank matches")
    p.add_argument("--kill-compiler-regex", default="", help="kill when a compiler command line matches")
    p.add_argument("--cm-dir", default=tempfile.gettempdir(), help="directory for the ssh ControlMaster socket")
    p.add_argument("--test-kill", action="store_true", help="kill dummy containers on both nodes and report latency")
    p.add_argument("--test-image", default="glm53-sm121-v11")
    a = p.parse_args()
    w = Watch(a)
    signal.signal(signal.SIGTERM, lambda *_: w.stopping.set())
    signal.signal(signal.SIGINT, lambda *_: w.stopping.set())
    return w.test_kill() if a.test_kill else w.run()


if __name__ == "__main__":
    sys.exit(main())
