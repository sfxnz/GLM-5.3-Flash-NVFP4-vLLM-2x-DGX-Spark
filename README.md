# GLM-5.3-Flash NVFP4 · vLLM · 2× DGX Spark

Serve [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

320B total / 18B active. Official NVIDIA ModelOpt 0.47.0 NVFP4 W4A4 on routed experts and dense MLP (`nvfp4_experts_dense_mlp-kv_fp8_cast`, ~190.5 GiB). Attention, shared experts, and vision stay BF16. Checkpoint KV scheme is `kv_fp8_cast`. Native context is 1,048,576. This recipe serves `--max-model-len` 327680 with the DFlash2 block-diffusion drafter (7 speculative tokens, two sequences). Vision is on (`LANGUAGE_MODEL_ONLY=0`). Do not copy the NVIDIA card's TP=4 / expert-parallel / 32-sequence / 8192 batched-token flags onto 2× Spark.

Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` loads on sm_121 and echoes the prompt. Build the local image chain through `glm53-sm121-v11` first.

## Measured on 2× DGX Spark (L.A.I.L lab)

Published decode is prose only. Do not score decode from structured, code, or other cells. The table is greedy, but a request that omits sampling params is served at `generation_config.json`'s T=1.0 / top_p 0.95 (vLLM `--generation-config auto`), so default-path speed and quality are unmeasured. Streamed greedy, thinking off, 200 completion tokens, 3-run median. These cells were measured on the LibertAIDAI pin on this image; the nvidia pack is unmeasured on Sparks. `max-num-seqs=2`, fp8 KV pinned at 4.14 GiB, context 327680, DFlash2-7, CUDA graphs. Prose is the low-acceptance regime (free text); structured (count 1→200) stays `--phase structured` for occupancy / acceptance only.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 21.2 | 21.2 | 0.33 s |
| prose | 2 | 16.6 | 33.2 | 0.37 s |
<!-- END generated measured -->

Default occupancy is the trained DFlash2 block (7 draft slots) at two sequences. Structured is the occupancy ruler (50.7 → 68.1 at c=1 versus DFlash2-5 / four sequences). Prose now stops near the requested eighty words (~105 tokens) once thinking-off seeds `<think></think>`, so it is no longer a 200-token pad. `MAX_NUM_SEQS=3` at DFlash2-7 does not starve the third stream, but structured c=2 fell 59.5 → 50.7. Four-way admission needs the rollback `NUM_SPECULATIVE_TOKENS=5 MAX_NUM_SEQS=4`. CUDA graphs capture 1/2/4 plus 8/16 (verify shapes for 1–2 sequences). `ENFORCE_EAGER=1` is the rollback; it slowed structured c=1 68.1 → 65.0. Adding capture size 3 to the ladder was inside noise. Capture size 24 at two sequences was unused and stayed inside noise. `VLLM_USE_BREAKABLE_CUDAGRAPH=0` slowed structured c=2 59.7 → 52.1. Leave the engine auto-on. Async scheduling is already on: this vLLM auto-enables it for DFlash with the `mp` executor, so passing `--async-scheduling` was a no-op and every cell stayed inside noise. Do not pass `--no-async-scheduling`. Greedy count stays lossless: 200 consecutive integers with thinking off. `MAX_NUM_BATCHED_TOKENS=4096` was measured at two sequences and reverted (structured c=2 55.5 → 51.6, KV pool 372877 → 363476). Tony's 3.0 GiB KV pin cannot boot `--max-model-len` 327680 (vLLM wants 3.62 GiB). The displayed 3.62 GiB pin (3886945403) still estimates max len 327168 and refuses. A 4.0 GiB pin boots but structured c=2 fell 59.5 → 52.4.

MTP-4 (eager, 262144 context) measured 24.7 / 20.9 / 16.6 per stream prose. A unique-salt 8k-word needle prefilled at 1425 tok/s (TTFT 7.2 s, 10271 prompt tokens). Repeating an 8k prompt hit prefix cache (1427 → 2600 tok/s, 4608 cached tokens = two 2304-token blocks). A 318,123-token prompt (97% of the 327680 window) prefilled in 4m05s and answered a needle question exactly. First wave after restart pays Triton JIT per batch shape; warm waves sit at 0.23–0.65 s TTFT. A first concurrent structured wave on this boot can median ~52 tok/s while `VLLM::Worker_TP0` has ~1.3 GiB in swap (host `vm.swappiness=60`, 6.1 GiB swap used). This table is a second frozen wave after those pages faulted in. `run.sh` already calls `maybe_drop_caches`; it no-ops without passwordless sudo. `python3 bench_decode.py` repeats the published prose phase at c=1,2. The fp8 hybrid pool on this pin is 372,877 tokens (1.14× at 327,680).

Receipts for these numbers are in [`evidence/`](evidence/): `trail.tsv` and `decision.tsv` (what was tried, kept, reverted), `hypotheses.md`, and per-iteration `bench.txt` / `run.log` / `doctor.txt`.

## Requirements

- Two DGX Sparks on the QSFP RoCE link (stock `10.100.8.1` / `10.100.8.2`)
- Docker + NVIDIA Container Toolkit on both nodes
- About 220 GiB free disk per node for the weights (~190.5 GiB nvidia pack plus the DFlash2 draft)
- SSH from the head node to the worker (`spark2` in this lab)

```bash
hf auth login
# or: export HF_TOKEN=hf_...
```

## Build the image

On the head node, from this repo (then copy the image to the worker, below):

```bash
docker build -f docker/Dockerfile.sm121-v8 -t glm53-sm121-v8 docker
docker build -f docker/Dockerfile.sm121-v9 -t glm53-sm121-v9 docker
docker build -f docker/Dockerfile.sm121-v10 -t glm53-sm121-v10 docker
docker build -f docker/Dockerfile.sm121-v11 -t glm53-sm121-v11 docker
```

Separate builds on each node give different image IDs, so nothing proves the two ranks run the same bits. Prefer building on the head and copying it with `docker save glm53-sm121-v11 | ssh spark2 docker load`. With `ORCHESTRATE=auto` (and in `VALIDATE_ONLY=1` when SSH works) `run.sh` compares the image IDs on both nodes and warns on a mismatch.

The v8 Dockerfile starts from `vllm/vllm-openai:glm53-flash-arm64-cu130` and applies the sm_121 patches (NoPE FA2 backend, FlashInfer 0.6.18, NCCL 2.30.7, PDL off, indexer init, fp8 tile cap). `run.sh` refuses the stock tag.

The next three layers are all required for `SPEC=dflash2` (MTP works on v8):

- v9 backports DFlash2 support, vLLM PR [#52816](https://github.com/vllm-project/vllm/pull/52816), missing from the image's vLLM snapshot.
- v10 teaches the fork-only Glm5Next model to capture aux hidden states for the drafter (the mHC stream contraction follows the reference integration, sglang [#36708](https://github.com/sgl-project/sglang/pull/36708)).
- v11 adds a dedicated draft KV group to the GLM5 bespoke KV layout so the draft's sliding-window layers share the pool.

## DFlash2 drafter

[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) is a 1B block-diffusion draft model that predicts a whole block per pass. Upstream reports it beating GLM's native MTP on acceptance length across every task they measured. Decoding is lossless; our greedy outputs matched MTP's byte for byte.

DFlash2 is the default drafter (`SPEC=dflash2`). The MTP-4 rollback needs the LibertAI pack as well:

```bash
MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 SNAPSHOT_REV=caca4e6a4ebbd66f159d3d2fc256683fd6e27177 SPEC=mtp ./run.sh
```

`run.sh` refuses `SPEC=mtp` on the nvidia pack unless `FORCE_UNSAFE_SPEC=1`. Its layer-45 MTP weights are 13.84 GiB of BF16 that are not in the quant ignore list, so they cannot load (NVFP4 params expected) or fit (~6.9 GiB per rank). LibertAI's MTP experts are NVFP4.

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
    "model": "nvidia/GLM-5.3-Flash-NVFP4",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Vision smoke (OpenAI `image_url`; must not return HTTP 400 `is not a multimodal model`):

```bash
python3 smoke_vision.py
```

Stop both ranks from the head:

```bash
./stop.sh
```

## Defaults

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Image | `glm53-sm121-v11` (local) |
| Model | `nvidia/GLM-5.3-Flash-NVFP4` (served under `SERVED_NAME`, which defaults to `$MODEL`) |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 327680 |
| `--max-num-seqs` | 2 |
| `--kv-cache-dtype` | `fp8_e4m3` |
| `--kv-cache-memory` | `4445787956` (4.14 GiB) |
| `--moe-backend` | `marlin`, routed experts only (`run.sh` refuses `flashinfer_cutlass`; it OOM'd spark2) |
| `--linear-backend` | `marlin`, dense quantized linears (the nvidia pack's layer 0-2 MLP is NVFP4; `run.sh` refuses other values, whose NVFP4 GEMMs JIT on sm_121) |
| Checkpoint | `09b04e5e74bca08ca8549fc736d4cdd8624bfde3` (official NVIDIA ModelOpt 0.47.0; `MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 SNAPSHOT_REV=caca4e6a4ebbd66f159d3d2fc256683fd6e27177` rolls back) |
| Vision | on (`LANGUAGE_MODEL_ONLY=0`; the pack is `Glm5NextForConditionalGeneration` with `vision_config`) |
| `--mm-processor-cache-gb` | 1 (vLLM default is 4; caps processed image/video tensors cached in the head's EngineCore on UMA) |
| Output ceiling | `--override-generation-config '{"max_new_tokens": 65536}'` (clamps every request; `generation_config.json` T=1.0 / top_p 0.95 still apply; `MAX_NEW_TOKENS=0` drops the flag) |
| `--block-size` | 2304 |
| CUDA graphs | on, capture ladder 1/2/4 + (7+1) x 1..2 (`ENFORCE_EAGER=1` reverts to `--enforce-eager`) |
| Speculative | DFlash2-7 (`NUM_SPECULATIVE_TOKENS=5 MAX_NUM_SEQS=4` for four-way; MTP-4 rollback is the LibertAI pack plus `SPEC=mtp`) |
| Chat template | `chat_template.jinja` (honors `enable_thinking` and its `thinking` alias, the glm45 parser's rule) |
| JIT / compile cache | on (`JIT_CACHE=1`), `$HOME/projects/data/glm53-jit-cache/<image id>/` per node, mounted at `/jit-cache`; `JIT_CACHE=0` disables |
| Reasoning / tools | `glm45` / `glm47` |
| API | `http://<head>:8000/v1` |
<!-- END generated defaults -->

The official NVIDIA pack is W4A4 on experts and dense MLP. `--moe-backend` covers only the routed experts; the layer 0-2 dense MLP goes through `--linear-backend`. With `--linear-backend auto` vLLM picks a FlashInfer FP4 GEMM on sm_121 that JIT-compiles during the first profile forward, so `run.sh` pins `LINEAR_BACKEND=marlin` and refuses other values unless `FORCE_UNSAFE_LINEAR=1`. Both ranks should log `Using MarlinNvFp4LinearKernel for NVFP4 GEMM`. Marlin dequantizes weights and never reads an activation `input_scale`, so this pin is a weight-layout + dense-MLP change until an SM121 W4A4 MoE path exists that does not JIT-OOM. `flashinfer_cutlass` is not that path on 2× GB10. v11 dies at JIT (`nvrtc.h` missing). v12 with `cuda-nvrtc-dev-13-0` got past that and then global-OOM'd spark2 during `cudafe++` after 90.67 GiB weights (`NV_ERR_NO_MEMORY`, ~18 GiB left). `run.sh` refuses any `MOE_BACKEND` other than `marlin` unless `FORCE_UNSAFE_MOE=1`. Do not set `VLLM_GLM53_MOE_INPUT_SCALE=1.0`. That constant underflows per 16-element block. LibertAI's GB10 recipe ([glm53-flash-vllm-gb10](https://github.com/Libertai/glm53-flash-vllm-gb10)) is MTP-3, eager, 64K, about 24 tok/s. It is a different stack from this DFlash2-7 / graphs / 327680 bar. Rollback to that pack: `MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 SNAPSHOT_REV=caca4e6a4ebbd66f159d3d2fc256683fd6e27177`.

`--kv-cache-memory 4445787956` stays the pin on TP=2. Dropping it OOMs GB10 (`NV_ERR_NO_MEMORY`). Raising it boots but backfires under UMA pressure: 5.0 GiB slowed decode ~20% at every concurrency, 5.14 GiB crashed under concurrent load. Two concurrent 20k-word needles (Tony's three-way 20k shape, at two sequences) and two concurrent 64k-word needles (~98k prompt tokens each) returned `hit=1` with docker `OOMKilled=false`. `MemAvailable` stayed about 8 GiB. Tony's anti-oom kill was three sequences at this pin. Occupancy gate: `python3 .cursor/skills/verify-glm53-flash/scripts/needle_probe.py --prompt-tokens 20480 --concurrency 2`. Do not turn on InstantTensor. That loader killed TP=2 ranks here.

Native `max_position_embeddings` is 1,048,576. This pin yields a 372,877-token fp8 hybrid pool at DFlash2-7 (1.14× at 327,680; it was 400,497 / 1.22× at DFlash2-5). A 1M request needs ~8.2 GiB of this `fp8_e4m3` hybrid layout. That is above the UMA crash point, so `run.sh` refuses `--max-model-len` above 327,680 on `fp8_e4m3` unless `FORCE_UNSAFE_CTX=1`. Official NVIDIA KV is still `kv_fp8_cast`. Packed `nvfp4_ds_mla` is the published 2× GB10 path that actually needles 1M. It is a different occupancy lane (different attention backend and image), and it measured ~22 tok/s prose versus 28 here. This recipe does not advertise a 1M window on the fp8 pin.

`chat_template.jinja` honors `enable_thinking`. The stock Hugging Face template and the official NVIDIA Hub template always open `<think>`, so `enable_thinking: false` used to leak chain-of-thought into `content`. Thinking off now seeds an empty `<think></think>` so a Hermes-style tool follow-up does not prefix `</think>` onto `content`. Do not swap in the Hub template. `thinking` is an alias with the same rule as vLLM's `glm45` reasoning parser: thinking is on when both kwargs are unset, otherwise when either is true. Before this, `{"thinking": true}` against the server default `enable_thinking: false` rendered `<think></think>` while the parser waited for `</think>`, so the whole answer landed in `reasoning` and `content` came back empty. `python3 -m unittest discover -s tests` checks the kwarg matrix and that the template differs from the Hub copy in `tests/data/` only in the generation prompt.

`reasoning_effort` follows the Hub template unchanged: `low` renders `Reasoning Effort: Low`, `high` renders `High`, and every other value (`medium`, `minimal`, `xhigh`, `max`, `none`, unset) renders `Max`. The model card documents no other levels, so the recipe does not remap them. vLLM also turns thinking on for any `reasoning_effort` other than `none` when the request does not set `enable_thinking`. With thinking off the prompt still carries `Reasoning Effort: Max` before an empty think block.

Vision is on by default. The pack is `Glm5NextForConditionalGeneration` with `vision_config` and `processor_config.json`. Vision was already on, implicitly and at vLLM's default limits, in every published LibertAI measurement: the boot logs `Encoder cache will be initialized with a budget of 32242 tokens, and profiled with 1 video items of the maximum feature size`. `run.sh` only makes it explicit: it passes `--limit-mm-per-prompt '{"image":4,"video":1}'` and `--mm-processor-cache-gb 1`, and refuses `LANGUAGE_MODEL_ONLY=1` unless `FORCE_UNSAFE_VISION=1`. `python3 smoke_vision.py` posts an OpenAI `image_url`. Do not skip MM profiling; it sizes the encoder cache, and the published boots ran with it.

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

`EXTRA_ENV` adds container env on both ranks as space-separated `NAME=VALUE` pairs, for example `EXTRA_ENV='MAX_JOBS=2 FLASHINFER_JIT_VERBOSE=1'`. Names must match `^(NCCL|VLLM|PYTORCH|TORCH|CUDA|OMP|FLASHINFER|TRITON|TILELANG|GLM53)_[A-Z0-9_]+$` or be `MAX_JOBS`; names containing `TOKEN`, `KEY` or `SECRET` are refused. `EXTRA_ARGS` adds `vllm serve` flags on both ranks. The head forwards both, and every other setting, to the worker from one `FORWARD_ENVS` list; `VALIDATE_ONLY=1 ./run.sh` prints the exact worker command.

## JIT and compile cache

With `JIT_CACHE=1` (the default), each node mounts `JIT_CACHE_DIR/<image id>/` at `/jit-cache`. `JIT_CACHE_DIR` defaults to `~/projects/data/glm53-jit-cache`, and `<image id>` is the first 12 hex digits of that node's own `docker image inspect -f '{{.Id}}' glm53-sm121-v11`. `run.sh` points each engine's cache variable into the mount:

- `FLASHINFER_WORKSPACE_BASE`
- `VLLM_CACHE_ROOT`, which covers torch.compile, the FlashInfer autotune file and DeepGEMM
- `DG_JIT_CACHE_DIR`
- `TRITON_CACHE_DIR`
- `TILELANG_CACHE_DIR`

The first boot on an image still compiles everything. That covers FlashInfer topk during PROFILE; FlashInfer batch_mla, batch_prefill, xqa and sampling during KV_READY; and DeepGEMM and TileLang mHC. Later boots should load those kernels from disk instead of running nvcc and ptxas; on 2026-09-27 the compilers together peaked at 3.5–3.9 GiB during PROFILE. The skipped compile time and memory are not measured yet. A new image gets a new directory, so it never reuses kernels built by an older one. `JIT_CACHE=0` gives every boot an empty cache, as before. The worker gets the same settings from the head and keys its directory by its own image ID.

The container runs as root, so the cached files are root-owned. To clear a cache, stop the serve, then run this on each node:

```bash
ls ~/projects/data/glm53-jit-cache/                                # one directory per image ID
sudo rm -rf ~/projects/data/glm53-jit-cache/<id>                   # one image
docker run --rm -v ~/projects/data/glm53-jit-cache:/c --entrypoint rm glm53-sm121-v11 -rf /c/<id>   # no sudo
```

## Repeat the decode bench

```bash
python3 bench_decode.py                       # ruler v2: cells A,B,J,H,K (~10 min)
python3 bench_decode.py --full --out DIR      # all cells; writes DIR/bench.txt + bench.json
python3 bench_decode.py --cells J             # structured count only (acceptance ceiling)
python3 kit/compare.py --a A1/bench.json A2/bench.json --b B1/bench.json B2/bench.json
```

The published score is cell A: 8 distinct prose prompts, 512 forced tokens, greedy, thinking off, c=1. Every wave reports `acceptance_len`, `step_ms` and tok/s from `/metrics` deltas. Cell K reruns the old ~98-token prose prompt for continuity. A cell is INVALID if swap use grows more than 64 MiB while it runs. The bench exits 1 on any failed or short request, INVALID cell, or unreadable meminfo, and rewrites `DIR/bench.json` after every cell. `kit/compare.py` treats each boot as one sample and prints KEEP, REVERT or INCONCLUSIVE per cell.

## Logs

```bash
docker logs -f glm53-flash-nvfp4
ssh spark2 docker logs -f glm53-flash-nvfp4
```

## License

Recipe scripts are MIT. Model weights follow the base model license on Hugging Face (MIT).
