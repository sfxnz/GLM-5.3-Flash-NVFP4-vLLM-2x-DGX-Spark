#!/usr/bin/env python3
"""Pause guard for a client job: SIGTERM PID when MemAvailable on either node
drops below THRESH GiB (default 3.0, 1 GiB above the SERVING floor).

usage: memguard.py PID LOG [THRESH]
Samples spark1 /proc/meminfo and spark2 (one persistent ssh) every 0.5 s.
Logs every 30 s and every new minimum. Exits when PID exits.
"""
import os
import shlex
import signal
import subprocess
import sys
import threading
import time

pid, log = int(sys.argv[1]), sys.argv[2]
thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
REMOTE = ("import time\nwhile True:\n    m={l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}\n"
          "    print(m['MemAvailable'], m['SwapTotal']-m['SwapFree'], flush=True)\n    time.sleep(0.5)\n")
rp = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "spark2", "python3 -u -c " + shlex.quote(REMOTE)],
                      stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
latest = {}


def reader():
    for line in rp.stdout:
        try:
            av, sw = line.split()
            latest["spark2"] = (int(av) / 1048576, int(sw) / 1048576)
        except ValueError:
            pass


threading.Thread(target=reader, daemon=True).start()
f = open(log, "a", buffering=1)
mins = {}
t_log = 0.0


def stamp():
    return time.strftime("%FT%TZ", time.gmtime())


def local():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return m["MemAvailable"] / 1048576, (m["SwapTotal"] - m["SwapFree"]) / 1048576


f.write(f"{stamp()} guard pid={pid} thresh={thresh} GiB\n")
while True:
    try:
        os.kill(pid, 0)
    except OSError:
        f.write(f"{stamp()} pid {pid} gone; mins { {k: round(v, 2) for k, v in mins.items()} }\n")
        break
    vals = {"spark1": local(), **latest}
    newmin = False
    for n, (a, _) in vals.items():
        if a < mins.get(n, 1e9):
            mins[n] = a
            newmin = True
    now = time.time()
    if newmin or now - t_log > 30:
        f.write(f"{stamp()} " + " ".join(f"{n}={a:.2f}GiB swap={s:.2f}" for n, (a, s) in sorted(vals.items()))
                + f" mins={ {k: round(v, 2) for k, v in mins.items()} }\n")
        t_log = now
    low = [n for n, (a, _) in vals.items() if a < thresh]
    if low:
        f.write(f"{stamp()} PAUSE: {low} below {thresh} GiB -> SIGTERM {pid}\n")
        os.kill(pid, signal.SIGTERM)
        break
    time.sleep(0.5)
rp.terminate()
