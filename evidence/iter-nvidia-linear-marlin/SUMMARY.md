# nvidia/GLM-5.3-Flash-NVFP4 @09b04e5 — first successful boot (2026-09-27)

**Outcome: ready on the first boot.** One exclusive TP=2 boot, no fallback rungs. The only change from the failing 2026-09-16 boots is `EXTRA_ARGS='--linear-backend marlin'`; run.sh, the pack and the image (`glm53-sm121-v11`) are the same.

- run.sh 12:27:06Z → `/v1/models` 12:46:59Z (19.9 min). Head load 769.7 s, worker 262.3 s, 90.36 GiB/rank.
- KV pool 372,877 tokens (1.14× at 327,680), same as the LibertAI pin. Block size actually used: 4608.
- Watcher (`kit/uma_watch.py`) never fired. Its kill latency, measured on dummy containers on both nodes, was 0.144 s for `docker kill` to return and 0.174 s until the containers were confirmed stopped.

## Root cause: FlashInfer FP4 GEMM JIT — supported, not proven

- Both ranks logged `Using MarlinNvFp4LinearKernel for NVFP4 GEMM` at 12:28:36, before `Filesystem type` (12:28:40). The MoE backend is MARLIN.
- The compiler census has 2,641 rows. None contain `fp4_gemm`.
- After TileLang `mhc_post` compiled, memory stepped from about 11.3/12.0 GiB to 9.1/9.4 GiB, then held near 10 GiB. On 09-16 the same point fell from 19 to 0.45–0.59 GiB within 7 s.
- Not proven: the 09-16 logs did not record which NVFP4 kernel was auto-selected, and the isolated build test (PLAN D12) was not run.
- Other JIT builds still run on every boot, because each boot uses a fresh container with no persistent JIT cache:
  - FlashInfer topk during PROFILE. One ptxas peaked at about 1.9 GiB; all compilers together peaked at 3.5–3.9 GiB.
  - FlashInfer batch_mla, batch_prefill, xqa and sampling during KV_READY.
  - deep_gemm and TileLang.

## MemAvailable minimum per phase (GiB)

| Phase | Floor | spark1 | spark2 |
|---|---:|---:|---:|
| LOAD | 10 | 14.62 | 15.47 |
| PROFILE | 8 | 8.90 | 8.89 |
| KV_READY | 3 | 4.16 | 6.02 |
| SERVING (to 12:55) | 2 | 4.82 | 8.73 |

- PROFILE sits only 0.9 GiB above its floor.
- Swap grew during LOAD (spark1 3.2 → 6.7 GiB, spark2 1.6 → 4.25 GiB) and was not reclaimed.
- `free -h` available after the benches: 5.5 / 9.5 GiB.

## Probes (all pass)

| Probe | Result |
|---|---|
| smoke | OK |
| thinking-off | `PING`, no leak |
| count | 200 consecutive (`--need 200 --max-tokens 1024`) |
| vision (legacy 1×1 smoke) | HTTP 200, answered "Black" (the check only requires not-400) |
| needle 8192 | hit. First run 17,081 tokens at 664 tok/s (JIT during inference); rerun 16,400 tokens at 1,349 tok/s |
| tool call | `get_weather {"city":"Wellington"}` |

The `w1_weight_scale_2 must match w3_weight_scale_2` warning is gone on both ranks. It fired on every LibertAI boot.

## Legacy bench (ruler v1; detects only ~31% changes)

| Cell | tok/s per stream | Acceptance | ms/step |
|---|---:|---:|---:|
| prose c1, wave 1 | 19.67 | 2.309 | 117.4 |
| prose c1, wave 2 | 18.99 | 2.188 | 115.2 |
| prose c2, wave 1 (agg 28.85) | 15.21 | 2.311 | 152.0 |
| prose c2, wave 2 (agg 28.66) | 14.91 | 2.242 | 150.4 |
| structured c1 | 68.29 | 8.000 | 117.2 |
| structured c2 (agg 119.4) | 59.71 | 8.000 | 134.0 |

ms/step = 1000 × acceptance / per-stream tok/s.

- LibertAI rebench (2026-09-02) for comparison:

  | Cell | tok/s | Acceptance | ms/step |
  |---|---:|---:|---:|
  | prose c1 | 21.18 | 2.426 | 114.6 |
  | prose c2 | 16.64 | – | 150.0 |
  | structured c1 | 67.64 | – | 116.0 |
  | structured c2 | 60.41 | – | 131.1 |

- Step time is within 1–2% of LibertAI. Prose acceptance is lower, but inside ruler-v1 noise. Neither is a verdict on the pack.
- Prose acceptance per draft position (both waves): .632 .324 .155 .088 .057 .011 .000. Structured is 1.0 at every position.
- The first c=2 wave had a 6.2 s TTFT (the known first-new-shape stall).

## State left behind

- The serve `glm53-flash-nvfp4` is running on both nodes.
- The serving watcher on spark1 runs in SERVING mode with a 2 GiB floor. It logs to `/home/sfxnz/projects/data/glm53-serving-watch/20260927-nvidia-linear-marlin/`.
- DFlash2 drafter revisions bf582e4 and dc77ff1 are downloaded and sha256-verified on both nodes. On spark2 they were fetched inside the image container (no hf CLI; the cache folder is root-owned). `refs/main` was not touched.

Files: `memory-by-phase.txt`, `spec-acceptance.txt`, `backend.txt`, `uma.tsv`, `compilers.tsv`, engine logs for both ranks, probe outputs, and bench outputs.
