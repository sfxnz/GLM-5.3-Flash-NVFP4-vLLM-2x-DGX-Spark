#!/usr/bin/env python3
"""Run a client command and kill it (not the serve) if MemAvailable on either
node drops below --thresh GiB. Logs a line every 30 s and on every new minimum.

    python3 evidence/e1-tools/memguard.py --thresh 3.0 --log DIR/memguard-x.log -- python3 quality/tier0.py ...
"""
import argparse
import datetime
import os
import shlex
import signal
import subprocess
import sys
import threading
import time

LOOP = ("import time\nwhile True:\n m=dict(l.split(':',1) for l in open('/proc/meminfo'))\n"
        " print(int(m['MemAvailable'].split()[0]), int(m['SwapTotal'].split()[0])-int(m['SwapFree'].split()[0]), flush=True)\n"
        " time.sleep(1)\n")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=3.0)
    ap.add_argument("--worker", default="spark2")
    ap.add_argument("--log", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    log = open(a.log, "a", buffering=1)
    child = subprocess.Popen(cmd, start_new_session=True)
    log.write(f"{utc()} guard pid={child.pid} thresh={a.thresh} GiB cmd={cmd}\n")
    mins, lock, last = {}, threading.Lock(), [0.0]

    def sample(node, argv):
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            avail, swap = (int(x) / 2**20 for x in line.split())
            with lock:
                new = avail < mins.get(node, 1e9)
                if new:
                    mins[node] = avail
                if new or time.time() - last[0] > 30:
                    last[0] = time.time()
                    log.write(f"{utc()} {node}={avail:.2f}GiB swap={swap:.2f} mins={ {k: round(v, 2) for k, v in mins.items()} }\n")
            if avail < a.thresh and child.poll() is None:
                log.write(f"{utc()} KILL client: {node} MemAvailable {avail:.2f} < {a.thresh} GiB\n")
                os.killpg(child.pid, signal.SIGKILL)

    for node, argv in (("spark1", ["python3", "-u", "-c", LOOP]),
                       (a.worker, ["ssh", "-o", "BatchMode=yes", a.worker, "python3 -u -c " + shlex.quote(LOOP)])):
        threading.Thread(target=sample, args=(node, argv), daemon=True).start()
    rc = child.wait()
    log.write(f"{utc()} client exit rc={rc} mins={ {k: round(v, 2) for k, v in mins.items()} }\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
