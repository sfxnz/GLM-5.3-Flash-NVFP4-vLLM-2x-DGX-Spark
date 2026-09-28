#!/usr/bin/env python3
"""e_detail.py DIR [DIR ...]: cell E per request from bench_decode.py bench.json (DIR/bench.json): prompt tokens,
TTFT, server prefill s and tok/s, client prefill tok/s, and MemAvailable / swap before and after the cell. Stdlib."""
import json
import sys

for d in sys.argv[1:]:
    b = json.load(open(f"{d}/bench.json"))
    c = b["cells"]["E"]
    mem = lambda m: " ".join(f"{k} {v['mem_available_mib'] / 1024:.2f}/{v['swap_used_mib'] / 1024:.2f}"  # noqa: E731
                             for k, v in m.items())
    print(f"{d}  boot_id {b.get('boot_id')}  MemAvail/swap GiB before [{mem(c['meminfo_before'])}] "
          f"after [{mem(c['meminfo_after'])}]")
    for w in [c["warmup"]] + b["waves"]:
        for r in w["requests"]:
            print(f"   {w['group']:9} prompt {r['prompt_tokens']:7} TTFT {r['ttft_s']:8.2f} s  server prefill "
                  f"{w.get('server_prefill_s', 0):8.2f} s {w.get('server_prefill_tok_s', 0):7.1f} tok/s  client "
                  f"{r.get('client_prefill_tok_s', 0):7.1f} tok/s  t_start {r['t_start']:.1f}")
