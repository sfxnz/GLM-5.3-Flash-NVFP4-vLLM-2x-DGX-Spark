# GLM-5.3-Flash NVFP4 TP=2 architecture (hillclimb grounding)

Sources: repo `run.sh` / README, live `docker inspect` + boot logs, target `config.json`, three how-explorers (launch, spec, bench). KV explorer still running at write time.

## What this is

A two-node vLLM serve of LibertAIDAI NVFP4 GLM-5.3-Flash on GB10. Native context is 1,048,576. This recipe serves 327,680 with DFlash2-5 and CUDA graphs. Decode is limited by hybrid KDA+DSA attention, Marlin weight-only FP4, cross-node RoCE TP, and speculative acceptance. Context is limited by a 4.14 GiB fp8 KV pin that yields 400,497 tokens (1.22× at 327,680).

## Rulers (frozen)

Decode: `python3 bench_decode.py` from repo root. Do not change flags. Baseline 2026-08-29 live serve:

| phase | c | median_decode_tok_s | median_agg | acceptance_len | draft_acc |
|---|---:|---:|---:|---:|---:|
| prose | 1 | 28.30 | 28.30 | 2.875 | 0.375 |
| prose | 2 | 22.61 | 44.17 | 3.114 | 0.423 |
| prose | 4 | 16.99 | 65.70 | 3.004 | 0.401 |
| structured | 1 | 50.74 | 50.73 | 5.405 | 0.881 |
| structured | 2 | 41.78 | 83.54 | 5.423 | 0.885 |
| structured | 4 | 34.87 | 136.36 | 5.445 | 0.889 |

Primary decode metric: prose c=1 `median_decode_tok_s` (low-acceptance regime). Higher is better. Noise band ~1 tok/s vs README.

Context: boot `--max-model-len 1048576` and retrieve a needle at ≥95% of that window. Pass/fail. Prefill time is recorded, not the hill.

## Why 327,680 not 1M

`text_config.max_position_embeddings` is 1,048,576. `--max-model-len` is a recipe choice. `--kv-cache-memory 4445787956` skips profiling and ignores `--gpu-memory-utilization`. Pool is 400,497 tokens. The pool is 0.38× a 1M request (a 1M request is 2.62× the pool). Raising the pin to 5.0 GiB slowed decode ~20%. 5.14 GiB crashed under concurrent load (Tony ladder: every try ≥5.5 GiB NVRM OOM on GB10 UMA). Hybrid glm5 charging needs ~8.24 GiB of fp8 KV for one 1M request with DFlash. Host UMA is already ~115/121 GiB with swap. fp8 KV cannot buy 1M on 2× GB10.

Public 2× 1M result (drowzeys) uses `nvfp4_ds_mla` packed KV (~368 B/token vs ~656 B fp8), `block-size 7168`, `max-num-seqs 2`, and still decodes ~22 tok/s. That is a context win that likely loses decode vs this recipe's 28.

## Why decode sits here

Already spent: DFlash2-5 vs MTP-4, FULL_AND_PIECEWISE graphs (+2–5% at c=1/2, flat at c=4), FlashInfer 0.6.18 FA2 SM90 NoPE, Marlin NvFp4 MoE, fp8 KV pin, PDL off, NCCL 2.30.7 RoCE.

Leftover on this image: spec slots 6 (untested); `max_num_batched_tokens` stuck at 2048; spec=7 if max-num-seqs drops; inductor compile; draft FA4 if present. Spec=7 at max-num-seqs=4 starves the 4th request (KDA copies).

`flashinfer_cutlass` is closed. v12 (`cuda-nvrtc-dev-13-0`) only cleared the missing `nvrtc.h` error. The fused-MoE JIT then global-OOM'd spark2 (2026-08-31T11:19:38Z, `cudafe++`, `NV_ERR_NO_MEMORY`) after 90.67 GiB weights with ~18 GiB left. spark1 exited 255 after the TP partner died. `run.sh` refuses any `MOE_BACKEND` other than `marlin`.

## Joint predicate

Both numbers in one flag set fight each other. Occupancy lanes (decode vs 1M) are allowed if one pin cannot hold both. Do not claim a 1M fp8 boot that cannot fit a 1M request.

## Shared hardware

One TP=2 serve. Code may be drafted in parallel worktrees. Measurement is serial. Restart owns the lab containers.
