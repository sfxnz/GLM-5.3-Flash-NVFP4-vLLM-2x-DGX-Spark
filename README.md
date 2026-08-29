# GLM-5.3-Flash NVFP4 · vLLM · 2× DGX Spark

Serve [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

320B total / 18B active. Weight-only NVFP4 on routed experts (~181 GiB). Native context is 1,048,576. This recipe serves `--max-model-len` 327680 with the DFlash2 block-diffusion drafter (5 speculative tokens).

Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` loads on sm_121 and echoes the prompt. Build the local image chain through `glm53-sm121-v11` first.

## Measured on 2× DGX Spark (L.A.I.L lab)

Decode only. Streamed greedy, thinking off, 200 completion tokens, 3-run median. `max-num-seqs=4`, fp8 KV pinned at 4.14 GiB, context 327680, DFlash2-5, CUDA graphs. Prose is the low-acceptance regime (free text); structured (count 1→200, same protocol as the EXL3 recipe) is the high-acceptance regime.

| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 | Eager same-day |
|---|---|---:|---:|---:|---:|
| prose | 1 | 28.2 | 28.2 | 0.24 s | 27.6 |
| prose | 2 | 21.1 | 42.4 | 0.35 s | 20.0 |
| prose | 4 | 17.1 | 66.9 | 0.63 s | 17.1 |
| structured | 1 | 51.4 | 51.3 | 0.32 s | 49.6 |
| structured | 2 | 41.5 | 81.8 | 0.35 s | 41.8 |
| structured | 4 | 35.7 | 142.7 | 0.39 s | 35.0 |

CUDA graphs (previously avoided on sm_121 with `--enforce-eager`) resolved to FULL_AND_PIECEWISE with 4 uniform-decode graphs at the 6/12/18/24-token verify shapes, captured in 27 s at zero measurable memory cost, and held stable across two full benches: +2–5% at c=1/2, flat at c=4, acceptance unchanged. Greedy output stays lossless: a counting probe verified 100+ exact sequential tokens per stream at c=4 through graph replay, and the only text divergence vs eager was a near-tie token (0.125 nat margin) that flips run-to-run in either mode.

MTP-4 (eager, 262144 context) measured 24.7 / 20.9 / 16.6 per stream prose the same day the DFlash2-5 default landed. DFlash2-5 wins every concurrency and carries 25% more context.

KV pool at boot: 400,497 tokens (1.22× at 327680), identical eager or graphs. Acceptance length on 6 slots: 2.9–3.0 prose (draft accept ~0.40), 5.4–5.5 structured (~0.89), 3.7+ on thinking-on math. A 318,123-token prompt (97% of the window) prefilled in 4m05s and answered a needle question exactly. First wave after restart pays Triton JIT per batch shape; warm waves sit at 0.23–0.65 s TTFT. `python3 bench_decode.py` repeats both phases and prints acceptance per concurrency block.

## Requirements

- Two DGX Sparks on the QSFP RoCE link (stock `10.100.8.1` / `10.100.8.2`)
- Docker + NVIDIA Container Toolkit on both nodes
- About 200 GiB free disk per node for the weights
- SSH from the head node to the worker (`spark2` in this lab)

```bash
hf auth login
# or: export HF_TOKEN=hf_...
```

## Build the image

On both nodes, from this repo:

```bash
docker build -f docker/Dockerfile.sm121-v8 -t glm53-sm121-v8 docker
docker build -f docker/Dockerfile.sm121-v9 -t glm53-sm121-v9 docker
docker build -f docker/Dockerfile.sm121-v10 -t glm53-sm121-v10 docker
docker build -f docker/Dockerfile.sm121-v11 -t glm53-sm121-v11 docker
```

The v8 Dockerfile starts from `vllm/vllm-openai:glm53-flash-arm64-cu130` and applies the sm_121 patches (NoPE FA2 backend, FlashInfer 0.6.18, NCCL 2.30.7, PDL off, indexer init, fp8 tile cap). `run.sh` refuses the stock tag.

The next three layers are all required for `SPEC=dflash2` (MTP works on v8):

- v9 backports DFlash2 support, vLLM PR [#52816](https://github.com/vllm-project/vllm/pull/52816), missing from the image's vLLM snapshot.
- v10 teaches the fork-only Glm5Next model to capture aux hidden states for the drafter (the mHC stream contraction follows the reference integration, sglang [#36708](https://github.com/sgl-project/sglang/pull/36708)).
- v11 adds a dedicated draft KV group to the GLM5 bespoke KV layout so the draft's sliding-window layers share the pool.

## DFlash2 drafter

[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) is a 1B block-diffusion draft model that predicts a whole block per pass. Upstream reports it beating GLM's native MTP on acceptance length across every task they measured. Decoding is lossless; our greedy outputs matched MTP's byte for byte.

DFlash2 is the default drafter (`SPEC=dflash2`). Switch back with:

```bash
SPEC=mtp ./run.sh
```

`run.sh` downloads the draft weights (~2.2 GiB, snapshot pinned) and passes `{"method":"dflash","model":<draft>,"num_speculative_tokens":5}` to both ranks.

Why 5 and not the block's full 7: acceptance at draft positions 5–6 is under 15%, and each speculative slot costs a per-sequence KDA state copy (`num_spec+1` copies). At 7 slots the state of 4 sequences no longer fits the pool, so the 4th request queues for ~10 s at c=4; at 5 slots all four admit immediately and c=1 is faster too (26.8 vs 26.0). Going down to 4 slots truncates the block-diffusion draft too hard (acceptance 2.8, c=1 drops to 20.6).

The draft model's license is CC BY-NC-ND 4.0 (research and evaluation; commercial licensing via inco.ai). The base model and this recipe are unaffected when you stay on MTP.

## Quick start

On the head Spark (`spark1`):

```bash
chmod +x run.sh stop.sh
./run.sh
```

The head script copies itself to `spark2`, starts the worker, waits 25s, then starts rank 0. First boot is weight load plus warmup. About 15–20 minutes when the cache is warm.

If SSH is not set up, start the worker yourself, then the head:

```bash
# spark2
ROLE=worker ./run.sh

# spark1
ROLE=head ./run.sh
```

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "LibertAIDAI/GLM-5.3-Flash-NVFP4",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Stop both ranks from the head:

```bash
./stop.sh
```

## Defaults

| Setting | Value |
|---|---|
| Image | `glm53-sm121-v11` (local) |
| Model | `LibertAIDAI/GLM-5.3-Flash-NVFP4` |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 327680 |
| `--max-num-seqs` | 4 |
| `--kv-cache-dtype` | `fp8_e4m3` |
| `--kv-cache-memory` | `4445787956` (4.14 GiB) |
| `--moe-backend` | `marlin` |
| `--block-size` | 2304 |
| CUDA graphs | on, capture ladder 1/2/4 + 6/12/18/24 (`ENFORCE_EAGER=1` reverts to `--enforce-eager`) |
| Speculative | DFlash2-5 (`SPEC=mtp` for MTP-4) |
| Reasoning / tools | `glm45` / `glm47` |
| API | `http://<head>:8000/v1` |

`--kv-cache-memory 4445787956` stays the pin on TP=2. Dropping it OOMs GB10 (`NV_ERR_NO_MEMORY`). Raising it boots but backfires under UMA pressure: 5.0 GiB slowed decode ~20% at every concurrency, 5.14 GiB crashed under concurrent load. Do not turn on InstantTensor. That loader killed TP=2 ranks here.

## Environment

```bash
export HEAD_IP=10.100.8.1
export WORKER_HOST=spark2
export IFACE=enp1s0f1np1
export HCA=rocep1s0f1
export PORT=8000
export MAX_MODEL_LEN=327680
export MAX_NUM_SEQS=4
```

Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two of them are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`.

## Repeat the decode bench

```bash
python3 bench_decode.py                    # both phases, c=1,2,4, 3 runs
python3 bench_decode.py --phase structured # one phase only
```

## Logs

```bash
docker logs -f glm53-flash-nvfp4
ssh spark2 docker logs -f glm53-flash-nvfp4
```

## License

Recipe scripts are MIT. Model weights follow the base model license on Hugging Face (MIT).
