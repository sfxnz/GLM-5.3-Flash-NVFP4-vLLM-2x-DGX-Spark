# tools/

Measurement tools for the kernel campaign (PLAN section 5: "Expert census boot" and "nsys boot"; findings PROTO-4, PROTO-5, XP-10, SD-1). Each tool is a diagnostic, not a perf arm: census and profiling boots do not produce publishable decode numbers.

| Tool | Runs on | Does |
|---|---|---|
| `census_report.py` | CPU, numpy | Reads `GLM53_EXPERT_CENSUS` files and reports distinct experts per verify step, duplicates, per-position overlap, routed-MoE bytes per rank per step, and the SD-1 tail-masking saving |
| `nsys_step.sh` | head node, against a running serve | Records about 30 steady-state verify steps on both ranks with the image's torch profiler |
| `step_buckets.py` | CPU, ijson | Turns one rank's trace into a kernel-time table per verify step |
| `bench_fp8_marlin.py` | GPU (FP8 lane); `--report-bytes` on CPU | BF16 vs Marlin FP8, NVFP4, INT8 and INT4 GEMM bench, and (`--dequant`) the `GLM53_WQ_DEQUANT_MIN_M` dequant + cuBLAS path at prefill M, documented in its docstring |

CPU tests for the first three live in `docker/test_v13_census.py`:

```bash
GLM53_V11_SRC=/path/holding/v11/vllm python3 docker/test_v13_census.py
```

## Expert census (`GLM53_EXPERT_CENSUS`)

`docker/patch_v13_census.py` is part of the v13 image and is off by default. With the switch unset, nothing is bound or allocated.

When the switch is set, each rank records real forward steps `SKIP .. SKIP+STEPS-1`. For each step it stores:

- The routed top-8 expert ids of every token row at every MoE layer (layers 3-44, 42 layers). These are logical ids, taken where `--enable-return-routed-experts` takes them.
- One segment per request:
  - its rows (a verify block is one anchor row plus k draft rows);
  - the drafts scheduled;
  - the tokens sampled (accepted drafts + 1);
  - the context length;
  - whether it is prefilling.

Unlike `--enable-return-routed-experts`, rejected verify rows are kept. Those rows are what SD-1 needs.

**Design choices:**

- **CUDA graphs.** The router's capture hook copies `topk_ids` into a preallocated device buffer. The hook is bound after load and before compile and capture, so the copy is part of every captured graph. The host reads the buffer after the step's sampler. `ENFORCE_EAGER` is not needed.
- **Trigger.** A step window (`GLM53_EXPERT_CENSUS_SKIP`, `GLM53_EXPERT_CENSUS_STEPS`). Both ranks see the same scheduler outputs, so the windows line up without a cross-node trigger. A marker file would have to be touched inside two containers on two hosts.
- **Runner.** Only the V2 model runner is hooked. That is the one DFlash2 forces, i.e. the `SPEC=dflash2` default. `SPEC=mtp` records nothing.
- **Cost.** There is one device-to-host sync per step, which is why census timings are not published. Files are about 10 KB per c=2 verify step.

## GPU validation recipe (exclusive TP=2 slot, one boot)

Follow AGENTS.md: exclusive GPUs, read memory with `free -h`, and keep receipts in `evidence/iter-census/` (`trail.tsv` and `decision.tsv` rows).

1. **Build.** Build v13 with the census layer (`docker build -f docker/Dockerfile.sm121-v13 -t glm53-sm121-v13 docker`) and sync the image to spark2.

2. **P3.0 gate.** Boot v13 with nothing set.
   - `docker logs glm53-flash-nvfp4 2>&1 | grep -c GLM53_EXPERT_CENSUS` must print 0 on both ranks.
   - Count-200 must be lossless.

3. **Census boot.**
   ```bash
   RUN=census-$(date +%Y%m%d-%H%M)
   IMAGE=glm53-sm121-v13 EXTRA_ENV="GLM53_EXPERT_CENSUS=/cache/huggingface/glm53-census/$RUN GLM53_EXPERT_CENSUS_STEPS=2000" ./run.sh
   ```
   Both ranks must log `GLM53_EXPERT_CENSUS: recording 42 MoE layers x top-8 for real steps 0..1999`.
   - If the line is missing, the V1 runner is in use.
   - If you get a `ValueError`, the MoE kernel is monolithic, which is not expected with Marlin.

   The boot must reach ready. Graph capture includes the copy.

4. **Traffic.**
   - Run `python3 bench_decode.py`: prose and structured, c=1 and c=2 distinct prompts.
   - Then keep sending prose until both ranks log `GLM53_EXPERT_CENSUS: rank R recorded steps 0..1999`.
   - Chunks are flushed every 100 steps. An unflushed tail is lost only on a hard kill.

5. **Losslessness.** Count-200 must still be lossless, since the census only reads routing. Do not use this boot's tok/s.

6. **Report**, after `./stop.sh`, on the head. The files are in the HF cache mount. Copy rank 1's `census-rank1*` from spark2's `~/.cache/huggingface/glm53-census/$RUN` to the same directory first.
   ```bash
   python3 tools/census_report.py ~/.cache/huggingface/glm53-census/$RUN --json evidence/iter-census/census.json
   ```
   Checks:
   - The line `rank0 vs rank1 routing identical: True` must appear (routing is replicated across TP ranks).
   - Prefix-curve `n=1` must be exactly 8.
   - Compare with the step model:
     - `distinct/layer` at 8 rows (c=1) against the 25-49 that PLAN section 1(b) infers;
     - the 58.1 of independent routing;
     - DSv4.1's duplicate fraction of 0.30.
   - The SD-1 oracle `saved_ms` feeds the P3.7 gate: at least 2 ms per tail token.

## Kernel-time profile (`nsys_step.sh` + `step_buckets.py`)

`nsys` is not in `glm53-sm121-v11`: `docker history` has no nsight layer. The host has it, at `/opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys` on both Sparks. However, `run.sh` has no hook to run the container's entrypoint under a bind-mounted `nsys`. The script therefore uses the image's torch profiler, which `run.sh` forwards to both ranks through `EXTRA_ARGS`. The torch profiler records the kernels inside CUDA graphs, each tagged with the correlation id of its `cudaGraphLaunch`.

1. **Boot a profiling serve.** `DELAY` and `STEPS` are fixed at boot:
   ```bash
   IMAGE=glm53-sm121-v13 EXTRA_ARGS="--profiler-config $(tools/nsys_step.sh --print-config)" ./run.sh
   ```
   The defaults are `DELAY=8` (skip prefill and the first decodes) and `STEPS=30`.

2. **Record.** Run `tools/nsys_step.sh` for c=1, then `CONCURRENCY=2 tools/nsys_step.sh` for c=2 with distinct prompts.
   - The script sends a warmup, then calls `/start_profile`. It streams one fixed-length request per stream, then calls `/stop_profile`.
   - It copies the traces from `$HF_CACHE/glm53-prof` on each node to `~/projects/data/glm53-prof/<stamp>/rank{0,1}`.

3. **Parse.** Run `./stop.sh`, then parse one rank at a time:
   ```bash
   python3 tools/step_buckets.py ~/projects/data/glm53-prof/<stamp>/rank0/<trace>.pt.trace.json.gz --json rank0.json
   ```

   The table covers each pure decode/verify step (`--only`). It gives wall, busy and idle time, and ms per step for each bucket:

   | Bucket | Contents |
   |---|---|
   | `routed_moe` | Marlin MoE + routing |
   | `linear_marlin` | Dense Marlin: layers 0-2 NVFP4 and the FP8 lane |
   | `bf16_gemm` | Plus a per-signature table, with the module guessed from launches per step: 34 KDA, 11 MLA, 42 shared-expert |
   | `kda` | |
   | `mla_indexer` | |
   | `mhc` | |
   | `nccl` | |
   | `other` | |
   | `lm_head_logits` | |
   | `sampler_rejection` | |
   | `drafter` | |

   The kernel-name regexes are first guesses. Check the printed top `other` kernels and extend `TARGET_BUCKETS` if something large lands there.

   **Acceptance (PROTO-4):**
   - The bucket sum plus idle equals wall within 5%.
   - Wall is within 5% of the ITL step time of a non-profiled boot.

**The nsys route, for when `run.sh` gets a hook to wrap the entrypoint:**

- Mount the host's `/opt/nvidia/nsight-systems/2025.3.2` read-only.
- Boot with `--profiler-config '{"profiler":"cuda"}'`.
- Run the container as:
  ```
  nsys profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop --trace-fork-before-exec=true -o /cache/huggingface/glm53-prof/rank<R> vllm serve ...
  ```
- Drive `/start_profile` and `/stop_profile` the same way.
- Export with `nsys stats -r cuda_gpu_kern_sum -f csv`, then run `step_buckets.py --nsys-kern-sum CSV --steps N`. This gives name buckets only: there are no step phases, so drafter kernels are not separated.

Do not run `ncu` inside the TP=2 serve: kernel replay stalls the partner rank's NCCL. `ncu` also needs SYS_ADMIN (`RmProfilingAdminOnly: 1`).
