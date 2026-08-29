# GLM-5.3-Flash NVFP4 · vLLM · 2× DGX Spark

Serve [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

320B total / 18B active. Weight-only NVFP4 on routed experts (~181 GiB). Native context is 1,048,576. This recipe serves `--max-model-len` 327680 with the DFlash2 block-diffusion drafter (7 speculative tokens, two sequences).

Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` loads on sm_121 and echoes the prompt. Build the local image chain through `glm53-sm121-v11` first.

## Measured on 2× DGX Spark (L.A.I.L lab)

Decode only. Streamed greedy, thinking off, 200 completion tokens, 3-run median. `max-num-seqs=2`, fp8 KV pinned at 4.14 GiB, context 327680, DFlash2-7, CUDA graphs. Prose is the low-acceptance regime (free text); structured (count 1→200, same protocol as the EXL3 recipe) is the high-acceptance regime.

| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 27.3 | 27.3 | 0.34 s |
| prose | 2 | 20.4 | 40.4 | 0.58 s |
| structured | 1 | 61.9 | 61.9 | 0.33 s |
| structured | 2 | 53.9 | 107.7 | 0.37 s |

Default occupancy is the trained DFlash2 block (7 draft slots) at two sequences. That is the structured-decode win (50.7 → 61.9 at c=1 versus DFlash2-5 / four sequences). Prose c=1 did not rise. Four-way admission needs the rollback `NUM_SPECULATIVE_TOKENS=5 MAX_NUM_SEQS=4`. CUDA graphs capture 1/2/4 plus 8/16 (verify shapes for 1–2 sequences). Greedy count stays lossless: 200 consecutive integers with thinking off.

MTP-4 (eager, 262144 context) measured 24.7 / 20.9 / 16.6 per stream prose. A 318,123-token prompt (97% of the 327680 window) prefilled in 4m05s and answered a needle question exactly. First wave after restart pays Triton JIT per batch shape; warm waves sit at 0.23–0.65 s TTFT. `python3 bench_decode.py` repeats both phases at c=1,2.

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

`run.sh` downloads the draft weights (~2.2 GiB, snapshot pinned) and passes `{"method":"dflash","model":<draft>,"num_speculative_tokens":$NUM_SPECULATIVE_TOKENS}` to both ranks. Default is 7. CUDA graph sizes are derived as 1/2/4 plus `(num_spec+1)×{1..MAX_NUM_SEQS}`.

Seven slots is the trained block. At four sequences those extra KDA copies starve the 4th request (~10 s queue). The default therefore runs two sequences. Rollback to the old four-way occupancy:

```bash
NUM_SPECULATIVE_TOKENS=5 MAX_NUM_SEQS=4 ./run.sh
```

Positions 5–6 accept under 15% on prose, which is why that rollback does not lose much free-text speed. Structured decode is the one that pays for the full block.

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
| `--max-num-seqs` | 2 |
| `--kv-cache-dtype` | `fp8_e4m3` |
| `--kv-cache-memory` | `4445787956` (4.14 GiB) |
| `--moe-backend` | `marlin` |
| `--block-size` | 2304 |
| CUDA graphs | on, capture ladder 1/2/4 + 8/16 (`ENFORCE_EAGER=1` reverts to `--enforce-eager`) |
| Speculative | DFlash2-7 (`NUM_SPECULATIVE_TOKENS=5 MAX_NUM_SEQS=4` for four-way; `SPEC=mtp` for MTP-4) |
| Chat template | `chat_template.jinja` (honors `enable_thinking`) |
| Reasoning / tools | `glm45` / `glm47` |
| API | `http://<head>:8000/v1` |

`--kv-cache-memory 4445787956` stays the pin on TP=2. Dropping it OOMs GB10 (`NV_ERR_NO_MEMORY`). Raising it boots but backfires under UMA pressure: 5.0 GiB slowed decode ~20% at every concurrency, 5.14 GiB crashed under concurrent load. Do not turn on InstantTensor. That loader killed TP=2 ranks here.

Native `max_position_embeddings` is 1,048,576. This pin yields a ~400k-token fp8 hybrid pool (1.22× at 327,680). A 1M request needs ~8.2 GiB of this layout. That is above the UMA crash point, so `run.sh` refuses `--max-model-len` above 327,680 on `fp8_e4m3` unless `FORCE_UNSAFE_CTX=1`. Packed NVFP4 MLA KV is the published 2× GB10 path that actually needles 1M. It is a different attention backend and a different image, and it measured ~22 tok/s prose versus 28 here. This recipe does not pretend one flag set holds both numbers.

`chat_template.jinja` honors `enable_thinking`. The stock Hugging Face template always opens `<think>`, so `enable_thinking: false` used to leak chain-of-thought into `content`.

## Environment

```bash
export HEAD_IP=10.100.8.1
export WORKER_HOST=spark2
export IFACE=enp1s0f1np1
export HCA=rocep1s0f1
export PORT=8000
export MAX_MODEL_LEN=327680
export MAX_NUM_SEQS=2
```

Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two of them are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`.

## Repeat the decode bench

```bash
python3 bench_decode.py                    # both phases, c=1,2, 3 runs
python3 bench_decode.py --phase structured # one phase only
```

## Logs

```bash
docker logs -f glm53-flash-nvfp4
ssh spark2 docker logs -f glm53-flash-nvfp4
```

## License

Recipe scripts are MIT. Model weights follow the base model license on Hugging Face (MIT).
