#!/usr/bin/env python3
"""Warm the next safetensors shard into page cache while vLLM loads. No root.

    python3 kit/shard_warm.py CONTAINER SNAPSHOT_DIR [--log FILE]

Follows `docker logs -f CONTAINER`. vLLM's default loader
(model_loader/weight_utils.py, safetensors_weights_iterator) reads the shards
in natural-sort order. On global rank 0 only, it prints a tqdm line
"Loading safetensors checkpoint shards: ...| n/N [" each time a shard is done.
At n/N it is reading shard n (0-based). The warmer then calls
posix_fadvise(WILLNEED) on shard n+1 and nothing further ahead, so it adds at
most one shard of page cache (8.33 GiB max on 09b04e5). It skips progress lines
whose N is not this snapshot's shard count, such as the drafter's 1-shard load.
It exits at the first "Loading weights took" line or when the log stream ends
(container stopped or gone). Rank 1 prints no progress lines, so on the worker
it only logs that there was nothing to follow.
"""
import argparse
import os
import re
import subprocess
import sys
import time

PROGRESS = re.compile(r"Loading safetensors checkpoint shards(?: \(eager\))?: +\d+% Completed \| (\d+)/(\d+) \[")
DONE = "Loading weights took"


def natural_key(path):
    """weight_utils._natural_sort_key: model-00002-of-00033 sorts before model-00010-of-00033."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", os.path.basename(path))]


def shard_files(snapshot):
    names = (n for n in os.listdir(snapshot) if n.endswith(".safetensors"))
    return sorted((os.path.join(snapshot, n) for n in names), key=natural_key)


def next_shard(line, files):
    """The shard to advise for one log line, or None."""
    m = PROGRESS.search(line)
    if not m:
        return None
    done, total = int(m.group(1)), int(m.group(2))
    if total != len(files) or done + 1 >= total:
        return None
    return files[done + 1]


def follow(lines, files, advise, say):
    """Advise each next shard once, until the load finishes. Returns the advised paths."""
    advised, progress = [], 0
    for line in lines:
        if DONE in line:
            say("load finished: " + line.strip())
            break
        if PROGRESS.search(line):
            progress += 1
        path = next_shard(line, files)
        if path and path not in advised:
            advised.append(path)
            advise(path)
    else:
        say("log stream ended (container stopped or gone)")
    if not progress:
        say("no shard progress lines: vLLM prints them on global rank 0 only")
    return advised


def willneed(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
    finally:
        os.close(fd)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("container")
    ap.add_argument("snapshot")
    ap.add_argument("--log", help="append here instead of stdout")
    args = ap.parse_args(argv)
    out = open(args.log, "a") if args.log else sys.stdout

    def say(msg):
        print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, file=out, flush=True)

    def advise(path):
        t0 = time.monotonic()
        try:
            willneed(path)
        except OSError as e:
            say(f"WILLNEED {os.path.basename(path)} failed: {e}")
            return
        say(f"WILLNEED {os.path.basename(path)} {os.path.getsize(path) / 2**30:.2f} GiB "
            f"({time.monotonic() - t0:.2f} s)")

    files = shard_files(args.snapshot)
    say(f"following {args.container}: {len(files)} shards in {args.snapshot}")
    logs = subprocess.Popen(["docker", "logs", "-f", args.container], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    try:
        advised = follow(logs.stdout, files, advise, say)
    finally:
        logs.kill()
        logs.wait()
    say(f"exit: advised {len(advised)} shard(s)")


if __name__ == "__main__":
    main()
