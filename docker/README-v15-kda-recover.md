# GLM53_KDA_RECOVER (v15): KDA verify without per-token state stores

`GLM53_KDA_RECOVER=1` stops the DFlash2 verify from storing a full fp32 KDA recurrent state for every verified token. Off by default. With the switch off, v15 runs v14's KDA code: no buffer is allocated and the verify call keeps v14's arguments.

Files: `docker/patch_v15_kda_recover.py` (the patch), `docker/test_v15_kda_recover.py` (CPU tests), `docker/gpu_check_v15_kda_recover.py` (GPU check). The patch edits three vLLM files and adds one module:

- `models/glm5next/nvidia/kda.py` (the spec verify call)
- `v1/attention/backends/gdn_attn.py` (metadata fields and the builder hook)
- `v1/worker/gpu/model_states/mamba_hybrid.py` (keeps the step's metadata and commits after sampling)
- new `models/glm5next/nvidia/ops/glm53_kda_recover.py`

The Dockerfile needs this layer:

```dockerfile
COPY patch_v15_kda_recover.py /tmp/patch_v15_kda_recover.py
RUN touch /tmp/v15kr.stamp \
    && python3 /tmp/patch_v15_kda_recover.py "$VLLM_ROOT" \
    && rm /tmp/patch_v15_kda_recover.py \
    && find "$VLLM_ROOT" -name '*.py' -newer /tmp/v15kr.stamp -print0 \
        | xargs -0 -r python3 -m compileall -q \
    && rm /tmp/v15kr.stamp
```

`run.sh` has no knob for it yet. Set it per boot with `EXTRA_ENV='GLM53_KDA_RECOVER=1'`, as AGENTS.md says for GLM53_* switches that are not recipe knobs.

## Mechanism

The stock spec verify is one `fused_recurrent_kda` call per KDA layer (`kda.py:546-563` in v14). Its kernel reads each row's initial state from `ssm_state_indices[row, num_accepted - 1]` (`fused_recurrent.py:130-138`). It then stores the state after every token `t` into `ssm_state_indices[row, t]` whenever that id is above 0 (`fused_recurrent.py:189-199`). At TP=2 one state is 32 heads x 128 x 128 fp32, which is 2 MiB per layer. A DFlash2-7 row has 8 tokens, so it stores 16 MiB per layer per request per step.

Later steps read only two of those columns:

- **Column `n - 1`**, where `n` is `num_sampled`. The next verify reads it through `num_accepted = n` (`mamba_hybrid.py:304-310`, `fused_recurrent.py:130`), and so does the align pre-copy (`mamba_utils.py:418`, `:233`).
- **The boundary column**, align mode only. When the accepted tokens reach a 1152-token block boundary, the post-copy moves the state at column `bias` into the cached block (`mamba_utils.py:333-350`, `:233`).

With the switch on:

1. **Builder** (`gdn_attn.py`, outside any graph). The builder writes one index row per verify row as `[NULL x 8, src]`, where `src = spec_state_indices[row, num_accepted - 1]`. It also writes a `num_accepted` of 9 for every row. Both go into persistent buffers, so a FULL graph replays the addresses it captured.
2. **Verify** (`kda.py`, inside the FULL graph). This is the same `fused_recurrent_kda` call with those two index arguments. The kernel reads its initial state from column 8 (`src`, the same block as today) and finds NULL in columns 0-7, so it stores no state. One Triton copy per layer saves the token inputs into per-layer records: post-conv k and v, raw gate g, raw beta. The conv update keeps the real `num_accepted_tokens` and is unchanged.
3. **Commit** (`MambaHybridModelState.postprocess_state`). This runs after `post_update` and before the align post-copy. One small kernel per KV group builds the replay index rows `[S[n-1] at column n-1, S[bias] at column bias if a boundary was crossed, NULL elsewhere, src at column 8]`. The boundary rule is `postprocess_mamba_fused_kernel`'s, from the same inputs. Then one `fused_recurrent_kda` per layer runs the records from `src` with the verify's exact keyword set.

The replay runs every token of the row and stores only at those columns. Every state it stores comes from the same kernel, the same initial state, the same input values and the same token order as the stock verify's store into that column. Everything vLLM does with `num_accepted`, the pre-copy, the post-copy and the conv state is untouched. The patch changes which columns get written. It does not change which columns get read.

## Why it is bit-exact, and where that claim rests

- **Same kernel and specialisation.** The replay passes contiguous records. Under `GLM53_KDA_TRIM=1` they satisfy `_glm53_kda_trim_ok`, so `STRIDED_QKVB` is the same constexpr the stock call gets. Under `=0` both calls hit `.contiguous()` first. `scale`, `num_warps`, `num_stages`, the gate flags, the state tensor and its stride all match.
- **The one difference.** `stride_indices_seq` is a constexpr (`fused_recurrent.py:53`). It is 8 for the stock index tensor and 9 for the recover one, so recover compiles a second specialisation. That value only enters the scalar address of the index load. It does not enter the [8, 128] state-tile pointers or the fp32 math. That this changes no rounding on sm_121 is an inference. `gpu_check_v15_kda_recover.py` tests it bitwise. Kindling's port (`experimental/fixes/recoverssm.py` @ `c540c91`) also replays through the stock kernel, with a Q+1 index row, and reports bitwise equality. Its own notes say a copied recurrence was 1-2 ulp off, which is why this port copies no arithmetic.
- **Records are bitwise copies.** Same dtype in and out. `store_records` refuses on any dtype or layout mismatch.
- **Induction.** The next step's verify reads the block the replay wrote, which holds today's value. So outputs match at every step, and greedy text is byte-identical.

## Constraints, and how each is met

| constraint | how | evidence |
|---|---|---|
| bit-exact | stock kernel for verify and replay, bitwise records, same initial state | CPU tests (below); GPU check pending |
| FULL CUDA graphs at fixed 8-row shapes | the verify index rows and the `num_accepted` of 9 live in persistent builder buffers filled in `build()` (as `gdn_attn.py:420-464` does for the stock buffers); the records copy has fixed shapes, no host sync and writes persistent per-layer buffers allocated in the builder `__init__`, before capture | `_forward` is not a graph break with breakable CG off (`breakable_cudagraph.py:90`, `envs.py:758`); GPU check captures verify+records and the commit and compares replays with eager |
| adaptive verify (rows masked at fixed shape) | shapes unchanged; a masked row's later tokens only feed tokens after the last store, because `n - 1 <= m` | `glm53_adaptive_verify.py` header |
| no host sync, async scheduling | the commit reads `num_sampled` and `num_computed_tokens` on the device; the mixed-batch row map goes up via `async_tensor_h2d` (pinned) | `mamba_hybrid.py` hook; `model_runner.py:1489-1519` |
| replay before the next verify | the commit is enqueued on the main stream in `sample_tokens(t)` (`postprocess_sampled`, `model_runner.py:1516`), and `execute_model(t+1)` enqueues the pre-copy and graph after it; the worker pairs them (`execute_model_state` is single-slot, `model_runner.py:1820-1832`) | stream order |
| align prefix caching at 1152 | the replay stores the boundary column the post-copy reads, before the post-copy runs; the rule is `mamba_utils.py:333-350` on the same `num_computed` | CPU align chain runs both copy rules on both pools; GPU check at block 1152 |
| `KPOOL_TAIL_FIX` independent | the indexer ring is not touched; only the KDA recurrent call's index arguments change | patch diff |

Startup refusals (in `make_builder_state`): V1 model runner, pipeline parallel above 1, mamba cache mode other than `none`/`align`, non-GLM KDA layers in a GDN group, or a mamba block shorter than one verify window. Each would leave a verify uncommitted.

vLLM's own Kimi-K3 RecoverSSM is not in this tree. `grep -ri recover` over `models/kimi_k3/` and `v1/` of the v14 image finds no RecoverSSM, and the `487ecf187` base predates it. So the port follows kindling's replay design. It does not keep kindling's verify-kernel copy, its conv compaction, or its one-slot layout.

## Expected effect (inference, not measured)

These are per KDA layer, per request, per step, on one TP=2 rank:

| | stock | recover |
|---|---:|---:|
| state read | 2 MiB | 2 MiB (verify) + 2 MiB (replay) |
| state stored | 16 MiB | 2 MiB (+2 MiB at a block boundary, once per ~530 steps) |
| records | 0 | ~0.2 MiB written + read |
| total | ~18 MiB | ~6.4 MiB |

Over 34 KDA layers that saves ~390 MiB per step at c=1 and ~780 MiB at c=2. At the ~250 GB/s the E5 profile implies for the 2.33 ms `kda` bucket, the c=1 saving is ~1.6 ms of GPU time. The added launches (34 record copies in the graph, 34 eager replays, 4 plan kernels and 4 index kernels) cost roughly 0.1-0.3 ms. The net guess for cell A is -1.3 to -1.6 ms on a 69.3 ms step, about +2%. Cell H should gain more (5.53 ms bucket). Acceptance must not change.

**Memory.** Per-request state memory does not change. Each request still holds 1 + 7 state blocks per KDA group, because the replay writes into column `n - 1` of the existing layout. Cutting that to one block needs a different `num_accepted` contract for the align pre-copy, the post-copy and the conv window, plus `MambaSpec.num_speculative_blocks` on the scheduler side. That is a follow-up. New fixed memory is about 13 MiB per rank at `MAX_NUM_SEQS=2` (records: 34 layers x 16 tokens x 32 heads x 385 bf16), and about 26 MiB at 4.

## Risks

- **The constexpr stride.** The GPU check settles it. If it reports any mismatch, do not boot with the switch.
- **A spec row followed by a non-spec decode of the same request.** This happens when the scheduler trims a running decode to 0 drafts: token budget starvation behind a running chunked prefill, or `max_model_len`. The non-spec path reads column 0 and ignores `num_accepted` (`gdn_attn.py:218`). Today that column holds the state after token 0, which is already wrong for `n > 1`, and the conv window is also read at offset 0. Under recover it holds an older state. Both are wrong. Only the wrong value differs. Ruler v2 cells A and H do not reach it, because prompts fit one chunk and 16 decode tokens never meet the 4624 budget.
- **Eager commit launches.** 34 replays and 4 plan kernels launch from Python in `sample_tokens`, roughly 1 ms of CPU per step (guess). Async scheduling hides that while the CPU runs ahead of the GPU. If the profile shows new idle before the drafter graph, capture the commit in a graph keyed by batch size. The GPU check shows the commit is capturable.
- **First-boot JIT.** Four new Triton specialisations compile in warmup and capture: the stride-9 recurrent kernel and three small kernels. `JIT_CACHE=1` keeps them for later boots.
- **+13 MiB per rank after profiling.** Watch spark1's MemAvailable gate.
- **The commit hooks `MambaHybridModelState`.** A model-specific model state that overrides `prepare_attn` or `postprocess_state` would skip it. None exists in this image (`model_states/__init__.py`; GLM defines no `get_model_state_cls`).

## Tests run here (CPU)

```bash
GLM53_V14_SRC=/path/to/v14src python3 docker/test_v15_kda_recover.py -v
```

The suite needs torch and triton. It runs v14's own `fused_recurrent_gated_delta_rule_fwd_kernel` and v14's `fused_recurrent_kda` wrappers (lifted unchanged from `ops/kda.py`) under `TRITON_INTERPRET=1`, with `GLM53_KDA_TRIM` 0 and 1. The checks:

- **Single step.** Accepted counts 0..7, mamba modes none and align (block 12), a FULL-graph padded row, and a mixed batch whose spec rows are batch rows 1..3 with one filtered request. Verify outputs must be equal. The recover verify must store nothing. The states later steps read must be equal. No other block may change.
- **Chains.** None mode at c=1, 2 and 4. Align mode at c=1 and 2 with blocks 8, 12 and 13, running the V2 pre-copy and post-copy rules on both pools. Outputs, running states and every cached boundary block must be equal at every step.
- **Mutations.** Each was caught by its own assertion. Storing column `n` instead of `n - 1` gives "stray store". Dropping the boundary column gives "read state differs". Reading the source at `num_accepted` instead of `num_accepted - 1` gives "verify outputs differ".

The interpreter proves the index plumbing and the replay semantics. It does not prove GPU codegen. That is the GPU check's job.

## Live checks (spark1 and the pair)

1. Inside the image on spark1, with no serve up: `docker run --rm --gpus all --entrypoint python3 -v "$PWD":/work -w /work glm53-sm121-v15 docker/gpu_check_v15_kda_recover.py`. It must print `ALL BITWISE EQUAL`. It also prints the per-step timing of both paths over 34 layers at c=1 and c=2.
2. Do a plain v15 boot, then a v15 boot with `EXTRA_ENV='GLM53_KDA_RECOVER=1'`. On `bench_decode.py` cells A and H, the greedy outputs must be byte-identical prompt for prompt, with identical acceptance and a lower step time. Repeat for the multi-boot means the README table needs before changing `measured`.
3. Prefix cache: send one prompt longer than 1152 tokens twice (the second is a cache hit) with the switch on. The greedy output must match the switch-off boot.
4. Run the thinking-off smoke and the 200-integer greedy count. Read `free -h` MemAvailable after ready on both nodes and compare with the switch-off boot.
5. Profile one prose step. The `kda` bucket should fall and no new idle should appear before the drafter.
