# GLM-5.3-Flash NVFP4 · vLLM · 2× DGX Spark

Serve [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

320B total / 18B active. Weight-only NVFP4 on routed experts (~181 GiB). Native context is 1,048,576. This recipe pins `--max-model-len` at 262144 with MTP-4.

Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` loads on sm_121 and echoes the prompt. Build the local `glm53-sm121-v8` image first.

## Measured on 2× DGX Spark (L.A.I.L lab)

Decode only. Streamed greedy, thinking off, 200 completion tokens, 3-run median. `max-num-seqs=4`, MTP-4, fp8 KV pinned at 4.14 GiB.

| Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---:|---:|---:|
| 1 | 25.6 | 25.6 | 0.30 s |
| 2 | 20.2 | 39.3 | 0.70 s |
| 4 | 17.0 | 64.9 | 0.68 s |

KV pool at boot: 507,041 tokens (1.93× at 262144). First wave after restart paid JIT. Later waves sat at ~0.25–0.44 s TTFT. `python3 bench_decode.py` repeats this.

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
```

The v8 Dockerfile starts from `vllm/vllm-openai:glm53-flash-arm64-cu130` and applies the sm_121 patches (NoPE FA2 backend, FlashInfer 0.6.18, NCCL 2.30.7, PDL off, indexer init, fp8 tile cap). `run.sh` refuses the stock tag.

The v9 Dockerfile layers the DFlash2 backport (vLLM PR [#52816](https://github.com/vllm-project/vllm/pull/52816), missing from the image's vLLM snapshot) on top of v8. Only needed for `SPEC=dflash2`.

## DFlash2 drafter

[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) is a 1B block-diffusion draft model that predicts a whole block per pass. Upstream reports it beating GLM's native MTP on acceptance length across every task they measured. Decoding is lossless.

```bash
IMAGE=glm53-sm121-v9 SPEC=dflash2 ./run.sh
```

`run.sh` downloads the draft weights (~2.2 GiB, snapshot pinned) and passes `{"method":"dflash","model":<draft>,"num_speculative_tokens":7}` to both ranks. `bench_decode.py` reports the measured acceptance length per concurrency block.

DFlash2 numbers on this cluster are not yet published here; the table above is MTP-4. The draft model's license is CC BY-NC-ND 4.0 (research and evaluation; commercial licensing via inco.ai). The base model and this recipe are unaffected when you stay on MTP.

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
| Image | `glm53-sm121-v8` (local) |
| Model | `LibertAIDAI/GLM-5.3-Flash-NVFP4` |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 262144 |
| `--max-num-seqs` | 4 |
| `--kv-cache-dtype` | `fp8_e4m3` |
| `--kv-cache-memory` | `4445787956` (4.14 GiB) |
| `--moe-backend` | `marlin` |
| `--block-size` | 2304 |
| Speculative | MTP-4 (`SPEC=dflash2` for DFlash2-7) |
| Reasoning / tools | `glm45` / `glm47` |
| API | `http://<head>:8000/v1` |

`--kv-cache-memory 4445787956` is the pin that kept MTP-4 alive on TP=2. Raising it or dropping it OOMs GB10 (`NV_ERR_NO_MEMORY`). Do not turn on InstantTensor. That loader killed TP=2 ranks here.

## Environment

```bash
export HEAD_IP=10.100.8.1
export WORKER_HOST=spark2
export IFACE=enp1s0f1np1
export HCA=rocep1s0f1
export PORT=8000
export MAX_MODEL_LEN=262144
export MAX_NUM_SEQS=4
```

Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two of them are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`.

## Repeat the decode bench

```bash
python3 bench_decode.py
```

## Logs

```bash
docker logs -f glm53-flash-nvfp4
ssh spark2 docker logs -f glm53-flash-nvfp4
```

## License

Recipe scripts are MIT. Model weights follow the base model license on Hugging Face (MIT).
