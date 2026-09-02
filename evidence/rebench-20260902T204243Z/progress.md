# GLM-5.3-Flash-NVFP4 re-bench 20260902T204243Z

Head spark1, worker spark2. Fresh boot with `./run.sh`, no env overrides. DeepSeek serve (user's) was stopped by the DeepSeek unit just before this.

- 2026-09-02T20:42:43Z gate 1 ok: `## main...origin/main` clean, HEAD af6bb0d == origin/main
- 2026-09-02T20:42:43Z gate 2 ok: no vllm/sglang containers on spark1 (only conduit) or spark2
- 2026-09-02T20:42:43Z gate 3 ok: no leases; gate 4 ok: image glm53-sm121-v11 (built 2026-08-28) and snapshots caca4e6 (target) + 7d74cdd (DFlash2 draft) present on both nodes
- 2026-09-02T20:42:43Z lease acquired: pair, ttl 3h, unit rebench-glm
- 2026-09-02T20:43:01Z ./run.sh started (fresh boot, defaults)
- 2026-09-02T21:01:18Z ./run.sh exited rc=0
- 2026-09-02T21:05:58Z ready: run.sh rc=0 at 21:01:18Z (boot 18m17s from 20:43:01Z); doctor status=ready; .run-state/started written
- 2026-09-02T21:06:11Z smoke: exit=0 (.cursor/skills/verify-glm53-flash/scripts/smoke.sh)
- 2026-09-02T21:06:13Z count: exit=0 (python3 .cursor/skills/verify-glm53-flash/scripts/count_probe.py)
- 2026-09-02T21:06:19Z thinking-off: exit=0 (python3 .cursor/skills/verify-glm53-flash/scripts/thinking_off_probe.py)
- 2026-09-02T21:06:20Z tool-call: exit=0 (python3 .cursor/skills/verify-glm53-flash/scripts/tool_call_probe.py)
- 2026-09-02T21:06:27Z hermes: exit=0 (python3 .cursor/skills/verify-glm53-flash/scripts/hermes_probe.py)
- 2026-09-02T21:07:02Z bench_decode.py (defaults: both phases, c=1 2, runs=3, max_tokens=200): exit=0, finished 2026-09-02T21:08:03Z
- 2026-09-02T21:12:56Z bench.json written from bench.txt SUMMARY
- 2026-09-02T21:13:09Z needle-8192: exit=0 (python3 .cursor/skills/verify-glm53-flash/scripts/needle_probe.py --prompt-tokens 8192)
- 2026-09-02T21:13:28Z needle-20480-c2: exit=0 (python3 .cursor/skills/verify-glm53-flash/scripts/needle_probe.py --prompt-tokens 20480 --concurrency 2)
- 2026-09-02T21:23:23Z engine log tails + oomkilled.txt saved
- 2026-09-02T21:23:47Z ./stop.sh rc=0; containers-after.txt saved
- 2026-09-02T21:28:30Z docker ps: spark1 only conduit, spark2 empty (an old Exited(137) deepseek-v4-flash-vllm-dspark-1 from 7 days ago is not ours, left alone); lease pair released
- 2026-09-02T21:29:29Z paperwork: branch agent/rebench-2026-09-02, recipe.yaml rows from bench.json (all four -> rebench bench.txt), render --check --strict pass, recipe-lint pass
