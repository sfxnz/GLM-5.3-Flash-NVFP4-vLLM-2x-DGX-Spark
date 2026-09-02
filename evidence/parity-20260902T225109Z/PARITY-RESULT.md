# Kit parity run — GLM-5.3-Flash-NVFP4 — 2026-09-02T22:51Z

Result: **FAIL** (bench rows outside ±5%). Serve booted and every probe passed. Shipped defaults, no env overrides, lease `pair` unit `parity-glm`. Reference: `evidence/rebench-20260902T204243Z/bench.json`.

Boot: `./run.sh` 22:51:26Z → `Ready` 23:10:03Z (18m37s). `doctor.txt`: `status=ready worker=up`. `oomkilled.txt`: both ranks `OOMKilled=false`. `stop.txt`: both ranks stopped; `cleanup.txt`: both nodes clean, lease released.

Probes (`probes.json`, `"failed": 0`): smoke PASS, count PASS (200 consecutive), thinking_off PASS, tool_call PASS (`get_weather`), hermes_two_turn PASS, needle PASS (8192 hit; 20480 c=2 both hit).

Bench (`bench_compare.py`, ±5% on decode and aggregate). Two runs, as PARITY.md allows one re-run. Run 1 is `bench-run1.*` / `parity-run1.txt`; run 2 is `bench.*` / `parity.txt` and is the verdict.

| Row | ref decode | run 1 | run 2 | ref aggregate | run 1 | run 2 | verdict (run 2) |
|---|---:|---:|---:|---:|---:|---:|---|
| prose c=1 | 21.2 | 18.9 (−11.0%) | 19.2 (−9.2%) | 21.2 | 18.8 (−11.0%) | 19.2 (−9.2%) | FAIL |
| prose c=2 | 16.6 | 17.7 (+6.5%) | 15.1 (−9.1%) | 33.2 | 35.0 (+5.3%) | 28.4 (−14.6%) | FAIL |
| structured c=1 | 67.6 | 66.9 (−1.1%) | 66.6 (−1.6%) | 67.6 | 66.9 (−1.1%) | 66.5 (−1.6%) | PASS |
| structured c=2 | 60.4 | 59.9 (−0.9%) | 55.5 (−8.1%) | 120.8 | 119.7 (−0.9%) | 111.0 (−8.1%) | FAIL |

Reading: the kit's `bench_decode.py` sends the same request body as the pre-kit `bench_decode.py` on `main` (same `PHASES` prompts, `temperature 0`, `max_tokens 200`, `chat_template_kwargs {"enable_thinking": false}`), so this is not a kit regression in the ruler. The prose rows are noisy at 3 runs: the reference's own three prose c=1 runs were 23.40 / 18.75 / 21.18 tok/s (a 22% spread, median 21.18), and this run's were 21.18 / 18.85 / 16.86 then 20.64 / 19.23 / 18.70. Run 1's prose c=1 first sample equalled the reference median exactly. Structured c=2 passed in run 1 and missed in run 2 (one 49.95/51.98 wave). Not tuned, not re-published; `recipe.yaml measured:` is unchanged.
