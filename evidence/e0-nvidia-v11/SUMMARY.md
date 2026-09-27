# E0 — nvidia pack baseline on the new recipe defaults (image v11, 2026-09-27)

This is the reference for every later stage: nvidia/GLM-5.3-Flash-NVFP4 @09b04e5, `glm53-sm121-v11`, recipe at `ac45f1e`, DFlash2-7 (7d74cdd), `--linear-backend marlin`, `EXTRA_ENV='MAX_JOBS=2 FLASHINFER_JIT_DEBUG=0 FLASHINFER_JIT_VERBOSE=1'`.

## Boots

**Boot 1 was killed by the watcher** during PROFILE (spark1 at 7.75 GiB, below the 8 GiB floor). Cause: `FLASHINFER_JIT_VERBOSE=1` in this image's FlashInfer also turns on debug JIT builds (`-O0 --device-debug`, `flashinfer/jit/core.py:525-528`) unless `FLASHINFER_JIT_DEBUG` is set. The topk ptxas grew to 3.28 GiB. See `boot1-killed/cause.txt`. The real kill latency was 3.7 s, against 0.144 s in the dummy test.

**Boot 2 served.**
- Timing: run.sh 13:41:43Z → `/v1/models` 14:01:55Z (20.2 min). Head load 741 s, worker 277 s.
- Kernels: Marlin NVFP4 GEMM and MARLIN MoE on both ranks. No `weight_scale_2` warning. KV pool 372,877.
- MemAvailable minima, spark1/spark2 (GiB):

  | Phase | spark1 | spark2 |
  |---|---:|---:|
  | LOAD | 14.48 | 15.56 |
  | PROFILE | 9.06 | 9.99 |
  | KV_READY | 4.91 | 7.70 |
  | SERVING | 4.06 | 8.75 |

  The SERVING low is from Tier-1 vision, while the MM cache fills.

## Fast checks

- count-200 lossless.
- QUAL-2 kwarg matrix 8/8 (`{thinking:true}`, `{enable_thinking:true}`, `{thinking:false}`, `{}`, each streaming and non-streaming).
- The head logs the overridden defaults `{temperature 1.0, top_p 0.95, max_tokens 65536}`.
- Vision smoke 13/13.

## Ruler v2 (bench-1 full panel; bench-2 fast-gate repeat in brackets)

| Cell | tok/s | Acceptance | step ms | TTFT p50 s |
|---|---:|---:|---:|---:|
| A prose (published) | 19.79 [19.66] | 2.285 [2.271] | 115.3 [115.3] | 0.32 |
| B code | 34.90 [34.39] | 4.207 [4.168] | 120.0 [120.1] | 0.32 |
| J structured c1 | 68.89 | 8.000 | 115.5 | 0.32 |
| J structured c2 (agg 109.6) | 55.50 | 7.922 | 141.9 | 0.35 |
| H distinct prose c2 (agg 26.1) | 13.69 | 2.207 | 161.1 | 0.42 |
| I distinct prose c4 | 13.43 | 2.177 | 162.8 | 18.0 (queued) |
| E 32k context | 20.70 | 2.327 | 111.5 | 24.7 |
| E 128k context | 25.04 | 2.822 | 112.7 | 98.4 |
| F image | 26.17 | 3.045 | 116.0 | 0.42 |
| G sampled | 18.23 | 2.127 | 116.5 | 0.26 |
| K legacy c1 | 18.67 | 2.198 | 116.6 | 0.32 |
| K legacy c2 | 15.73 | 2.246 | 142.1 | 0.36 |

- Prefill: 1,331 tok/s at 32k and 1,329 tok/s at 128k.
- Per-position acceptance, A: .640 .345 .169 .065 .029 .012 .003.
- Per-position acceptance, B: .817 .657 .515 .406 .322 .238 .182.
- Within-boot repeat (tok/s), all within ±4.5%: A −0.6%, B −1.5%, J@c1 −0.1%, J@c2 +2.5%, H +2.6%, K@c1 +2.8%, K@c2 −4.5%. The paired acceptance bootstrap for A is −0.6% [−3.8, +3.3].
- Sanity checks pass: identity error ≤ 1.5%, draft counts consistent, no INVALID cells, swap flat on both nodes.

## Tier 0 (reference `nvidia-v11-k7`, `~/projects/data/glm53-evals/ref/`)

- **A/A (reference rerun):** |dNLL| 0.000627, top-1 98.45%, KL 5.14e-3, greedy hazard 0.0096. The serve is **not run-to-run deterministic**: identical prefills disagree on top-1 1.55% of the time, and greedy output diverged in 14 of 20 prompts. So stage gates have to be judged relative to this A/A, not against fixed floors.
- **Same-boot compare:** dNLL −0.000146, top-1 98.45%, KL 5.09e-3, hazard 0.0099. Result 9/10. Passing: count, kwargs 12/12, utf8, needle 3/3 at 8k and 32k, tools JSON 1.0. Vision failed on video only: the answer flips between 3816 and 3186 depending on frame size, a model decision boundary.

## Tier 1 (1,010 items, 0 errors)

**Pooled: 857/1010 = 84.9%.**

| Task | Score |
|---|---|
| IFEval | 106/150 (70.7%) |
| GSM8K | 190/200 (95.0%) |
| MMLU-Pro | 234/280 (83.6%) |
| BFCL | 114/120 (95.0%) |
| ChartQA | 92/100 |
| OCRBench | 82/100 |
| MMMU | 39/60 (12 items hit the 64-token cap; kept for pairing) |

Vision group total: 213/260 (81.9%).

## Tool fixes made during E0

- `256dbc4`: tier0 kwargs. `effort_low` may now return empty reasoning.
- `00e8492`: ChartQA scorer now strips markdown bold. It was scoring 62/100 on correct answers; the true score is 92/100.

## Anomalies

- **utf8 table:** the model writes wrong squares (2^2 as 2) and degenerate rows in 3 of 4 runs. These are the target's own top-1 choices, not drafter errors.
- **Serve config:** this serve ran at `ac45f1e`; later lane merges (JIT cache, shard warmer) were not active.
- **Unknown poller:** a localhost process polls `/health`, `/version`, `/metrics` and `/v1/models` at about 0.5 Hz. Left alone.
