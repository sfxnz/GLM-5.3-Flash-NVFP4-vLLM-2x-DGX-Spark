# Upstream changes the v14 image lacks (research, 2026-11-02)

Read-only desk research for r3. No serve, GPU or ssh command ran. The image source was read from CPU-only containers (`docker run --rm --entrypoint bash glm53-sm121-v14 ...`, never `--gpus`) and copied into a scratch tree. Upstream facts come from `gh` against vllm-project/vllm, flashinfer-ai/flashinfer, vllm-project/humming, vllm-project/FlashKDA and vllm-project/DeepGEMM. This note extends `research-fp4local/evidence/research-20261030/fp4-local-and-spark-moe.md` (sections 2 and 4) and does not repeat it.

"Verified" means read this session in a primary source: an installed file in the image, a PR diff or body, a release note, or a log under `evidence/`. "Inference" means derived by us and not measured. File paths without a prefix are in the image's installed `vllm/` tree.

## What the image is

- vLLM `0.1.dev20051+g487ecf187` is upstream commit `487ecf187` (2026-08-25). It is the tip of the GLM-5.3 day-0 branch: `main` at `b908a21` (2026-08-13) plus 142 branch commits. Verified (`gh api compare`).
- Every release tag diverges from the image at `b908a21`. v0.31.0 (2026-10-05) is 2211 commits ahead, v0.32.0 (2026-10-18) is 3024 ahead, and v0.33.0 (2026-11-02, tag `v0.33.0rc3`) is 3951 ahead. Verified.
- So a fix merged to `main` after 2026-08-13 is missing unless the day-0 branch had an earlier form of it or we backported it. GLM-5.3 reached `main` as #53906 on 2026-09-03, after the branch. Verified.
- Kernel pins in the image: FlashInfer 0.6.18.dev20260819, humming-kernels 0.1.12 (released 2026-07-31), DeepGEMM vendored under `third_party/deep_gemm`, and FlashKDA `b5d1101` (cmake at `487ecf187`). `_flashkda_C.abi3.so` carries sm_90a, sm_100 and sm_120 cubins. Verified (pip list, cuobjdump, cmake).
- The serve runs Model Runner V2, `FLASHINFER_MLA_SPARSE_SM90` (FA2 on sm_121 through our v8 patch), async scheduling, and mamba cache mode `align`. It logs "No MLA prefill backend supports this model", so every prefill token takes the sparse top-k path. Verified (`evidence/r2/r3/head.log` lines 16, 31, 39, 41).

## Summary

- Upstream has little for the decode step. The Marlin MoE kernel has not changed since August apart from a header move (#60721). The best GB10 alternative published, FlashInfer #5735, is at parity with Marlin at 8 to 16 tokens. The host-side items below add up to about 0.5 to 3 ms of our 69.3 ms step. Verified for the kernel facts; the total is inference.
- The largest upstream gains for us are outside cell A. They are prefix-cache reuse with DFlash2 (multi-turn TTFT, plus a latent crash), FlashKDA prefill, and about 1.2 GiB of UMA per rank from the indexer prefill workspace.
- Two corrections to the 2026-10-30 desk note. FlashInfer #5648 is a dense W4A16 GEMM, not a MoE kernel. The SM12x W4A16 routed-MoE kernel is FlashInfer #5735. Verified.
- One safety note. `KPOOL_TAIL_FIX=0`, part of the documented v11 rollback, brings back the tail-seed stride bug that upstream fixed in #57477. That bug writes raw tail rows into other requests' indexer blocks. Verified in the image (`models/glm5next/nvidia/ops/kpool_compress.py:529-535`).

## Ranked table

Ranked by expected value to this recipe: size of the effect on our config, times the chance it applies on sm_121, divided by backport cost. The step and prefill baselines are `recipe.yaml` (69.3 ms, 1593 tok/s at 32k).

| # | Item | Merged (release) | Kind | In v14? | Expected effect on us | Backport | Label |
|---|---|---|---|---|---|---|---|
| 1 | #54163 + #55601: prefix-cache reuse with DFlash2 | 10-23 (v0.33.0); 10-07 (v0.32.0) | correctness, TTFT | Absent. `config/speculative.py:1513-1517` counts dflash as EAGLE, so the EAGLE last-block drop applies. `v1/worker/gpu/model_states/mamba_hybrid.py:104` seeds the KDA state slot in `cache_config.block_size` units | A cached 32k prefix skips about 20 s of prefill per turn (32768 / 1593). Decode unchanged | Python. +26/-5 in 4 files and +22/-1 in 2 files, plus reconciling our off-by-default `GLM53_DFLASH_PREFIX_CACHE_FIX` | absence verified; hits and TTFT inference |
| 2 | #58684: SM90 sparse-MLA plan without a D2H sync, plan reused across steps | 09-27 (v0.31.0) | perf, decode host | Absent. `v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py:340-351` copies `positions` to host under async scheduling; `:368-380` replans on every build | 0 to -2 ms per verify step (0 to +3% cell A) | Python, 1 file plus tests, +381/-106. Reads FlashInfer `MLAPlanInfo`, which FA2 and FA3 share | absence verified; effect inference |
| 3 | FlashKDA chunked prefill: #55737, with FlashKDA at `17a037d` or later (#58846) | 09-14 (v0.30.0); 09-26 (v0.31.0) | perf, prefill | Absent. Only Kimi-K3 calls FlashKDA (`models/kimi_k3/nvidia/kda.py:163-236`). The image pin `b5d1101` predates FlashKDA #10, #12 and #13 | +2 to +5% prefill at 32k and 128k | Kernel rebuild of `_flashkda_C` for sm_120/121, plus a ~150-line port into `models/glm5next/nvidia/kda.py` | absence verified; effect inference |
| 4 | #55222 part 2: indexer prefill workspace sized in pool rows | 09-29 (v0.32.0) | memory | Absent. `v1/attention/backends/mla/indexer.py:680-700` sizes `max_model_len * 40` token entries. Our `GLM53_INDEXER_WS_FACTOR` is off | About 1.2 GiB UMA freed per rank (1.61 GiB to about 0.40 GiB). No speed change | Python, under 40 lines of the +84/-19 | absence verified; size inference |
| 5 | #57632: DFlash context K/V precompute inside the draft CUDA graph | 09-24 (v0.31.0) | perf, decode | Absent. `v1/worker/gpu/spec_decode/dflash/speculator.py:74` says "NOT captured by CUDA graphs" | -0.2 to -0.6 ms per step | Python, 2 files, +62/-30, re-diffed against our DFlash2 backport | absence verified; effect inference |
| 6 | #61253 + #64535: GLM tool arguments streamed in linear time | 10-12 (v0.33.0); 10-29 (main) | robustness, API server | Absent. No `arg_value_extender` in `parser/glm47_moe.py` | Long tool calls stop costing quadratic API-server CPU. Upstream saw a liveness kill at 275 s at 200 tok/s | Python, +226/-26 and +112/-8 | verified |
| 7 | humming-kernels 0.1.18 or later (humming #150, sm121 heuristics) with vLLM #57421 | 10-19; 09-20 (v0.31.0) | perf, decode MoE candidate | Absent. 0.1.12 maps sm_121 to `Sm120Heuristics` (`humming/tune/__init__.py:29-30`) | Bounded at 0 to +7% cell A (desk note 4.1). No GB10 number exists | pip upgrade plus vLLM-side changes. Desk-note X2 microbench first | verified; effect inference |
| 8 | FlashInfer #5735: SM12x NVFP4 W4A16 routed MoE in CuTe DSL | 10-02 (FlashInfer 0.7.1) | perf, decode MoE candidate | Absent (FlashInfer 0.6.18). No merged vLLM backend; #54617 and #63829 are open | On GB10 against Marlin: 1.36x at 1 token, 1.03x at 8, 1.00x at 16 (Qwen3.8 shape). 0 to -2 ms at our 8-row verify | FlashInfer upgrade plus a new vLLM MoE backend. Large | PR table verified; effect inference |
| 9 | #59528: kpool tail kept out of the null block | 10-08 (v0.32.0) | correctness hygiene | Absent. `indexer.py:577-584` maps every row, padding included, to `block_table[req, 0]` | Dummy and padding rows stop writing block 0. Upstream reports no output change | Python, about 30 lines | verified |
| 10 | #59126: GLM vision tower crashes | 10-02 (v0.32.0) | robustness, vision | Absent. Rope table `max_position=8192` (`models/glm5next/nvidia/multimodal.py:411-414`); no aspect cap in `transformers_utils/processors/glm5next.py:224-255`; pageable `non_blocking` copies at `multimodal.py:497,613` | An extreme-aspect image could index past the rope table and kill the engine | Python, +4/-3 | absence verified; trigger inference |
| 11 | #57534, #58450, #57458: sparse-MLA and kpool metadata host work | 09-20 to 09-25 (v0.31.0) | perf, decode host | Absent. The tail slot mapping is a torch chain (`indexer.py:573-584`) | -0.1 to -0.4 ms combined | Python. #58450 and #57458 edit `flashinfer_mla_sparse.py`, which our SM90 builder subclasses | verified; effect inference |
| 12 | #47616: trim token_ids and logprobs past a stop string under spec decode | 10-09 (v0.32.0) | correctness, API | Absent. No `_trim_surplus_tokens` in `v1/engine/detokenizer.py` | Correct usage and logprobs when a stop string ends inside an accepted batch | Python, +158/-13 | verified |
| 13 | #55738: dense-MHA prefill for the NoPE (256, 0, 256) layout | 09-14 (v0.30.0) | perf, short-prompt prefill | Absent (log line 41) | TTFT for prompts of 2048 tokens or fewer, and only if a FlashAttention MLA prefill backend runs on sm_121. Masked MHA is SM100-only (`model_executor/layers/attention/sparse_mla_attention.py:88`) | Python, +96 | verified; effect inference |

## Notes per item

### 1. Prefix-cache reuse with DFlash2 (#54163, #55601)

- **Mechanism.** #54163 adds `needs_eagle_block_drop()`, which is false for DFlash and DSpark. Their draft KV comes from the target hidden state alone, so the trailing prefix-cache block need not be dropped. #55601 seeds the running mamba state slot with `mamba_block_size`, not `cache_config.block_size`. Verified (diffs).
- **Why it matters here.** `v1/engine/core.py:329` resets `cache_config.block_size` to the smallest prefix-caching group. Our draft group block is a quarter of the attention block (`docker/patch_v11_dflash_kv_groups.py:78-79`), and `SlidingWindowSpec` takes part in prefix caching by default (`v1/kv_cache_interface.py:131`). On a prefix hit, `mamba_hybrid.py:104` would then index the KDA state in the wrong units. Upstream's report is the same model on a DGX Spark with DFlash2: HTTP 500 and `Xid 31 FAULT_PDE` on the first hit. Code path verified; the crash on our serve is inference.
- **Current state.** Tony's EXT-5 measured 0 hits on this design without a fix, and our `GLM53_DFLASH_PREFIX_CACHE_FIX` has never run on GPU (`docker/README-v13.md:106-117`). Our r2 logs show a 0.0% hit rate, but `bench_decode.py:300` salts long prompts so that no block can hit. Our own hit rate is unmeasured.
- **Order.** Land #55601 before turning on any fix that produces hits. Then add #54163. Then run the README-v13 9216-token probe, a needle at 32k with a cached prefix, and Tier 0.
- **Follow-ups to check after that.** #59146, #59175 and #60533 fix internal prefill checkpoints. The image has no `prefill_checkpoint` code in `v1/core/single_type_kv_cache_manager.py`, so they may not apply. #61589 (lookahead hashes) targets EAGLE and MTP. Inference.

### 2. SM90 sparse-MLA plan without a sync (#58684)

- **Mechanism.** The builder plans on the host from the upper bound plus 32 tokens of slack. A Triton kernel then clamps each work item's `kv_end` to the exact count on the GPU. The plan is reused while the bounds still fit. Verified (PR body).
- **In the image.** Under async scheduling `_kv_lens_host` calls `positions[:num_rows].cpu()`, and `build()` calls `wrapper.plan()` every step. Verified.
- **Effect.** Upstream measured c=1 TPOT 11.09 to 6.86 ms on H100 with MTP-5, where the sync ran on every draft step. DFlash2 builds this metadata once per step, so our gain is smaller. r2 B6 moved 2.1 ms of other host work ahead of the sync and saw no change (69.9 vs 69.7 ms). That suggests the sync is only partly on the critical path. E5 F3 shows 5.5 ms of GPU idle per profiled step. Hence 0 to -2 ms. Inference.
- **Risk.** The clamp writes into FlashInfer's planned int workspace. FA2 (`data/csrc/batch_mla_plan.cu:41`) and FA3 (`batch_mla_sm90_plan.cu:37`) both fill `MLAPlanInfo` from `MLAPlan`. Verified. Upstream validated FA3 on H100 only, so whether the FA2 kernel honours a clamped `kv_end` on sm_121 is inference. Our fixed-shape adaptive verify keeps the host query split equal to the device one, so the batch-max fallback in the PR does not trigger.

### 3. FlashKDA prefill (#55737, #58846, #61614)

- **Mechanism.** One FlashKDA kernel replaces the ~15-kernel Triton `chunk_kda` path for KDA chunked prefill. Upstream measured 1.7x to 3.8x per layer on GB300 and gated it to SM90, SM10x and SM12x. Verified.
- **In the image.** `_flashkda_C` exists, but only Kimi-K3 calls it. Its op has 14 arguments; #55737 passes 16. The pin `b5d1101` predates FlashKDA #10 (unstable inverse), #12 (bf16 checkpoint state) and #13, "Keep the recurrent state in fp32 between tiles", a GLM-5.3-Flash fix. Verified (schema strings in the `.so`, FlashKDA commit log). Using the current binary for GLM is not safe.
- **Effect.** In the E5 F3 prefill profile, `kda` is 75.9 ms of a 1001.6 ms chunk (7.6%). If most of that bucket is the chunk kernel and FlashKDA is 2x to 2.5x faster, prefill gains 2% to 5%. Inference. Cell E is the gate. Decode is untouched, because spec-decode tokens keep the Triton recurrent kernel.
- **Cost.** Build FlashKDA at `17a037d` (v0.33.0's pin) or `47fc9fe` (#61614, row-strided QKV) against CUTLASS for sm_120/121 on aarch64, then port the GLM selection and workspace code. The GLM file moved upstream to `models/glm5next/common/kda.py`.

### 4. Indexer prefill workspace (#55222 part 2)

- **Mechanism.** `get_max_prefill_buffer_size()` returns token entries (`max_model_len * 40` of 132 B), but the kpool indexer's K cache is pool-granular (`compress_ratio == index_kpool == 4`). Upstream sizes it in compressed rows. Verified (PR body).
- **Effect.** README-v13 puts the workspace at 1.61 GiB per rank at 327680. A quarter of that is about 0.40 GiB, so about 1.2 GiB per rank is freed. Inference. At our context the chunk planner never splits, so outputs should not change. Inference. That headroom matters on spark1. E5 skipped the four-sequence boot (+1.5 GiB KV) because spark1 missed its 9 GiB MemAvailable gate (AGENTS.md).
- **Part 1** (fp8 plan dtype) is already present (`flashinfer_mla_sparse_sm90.py:434`).

### 5. DFlash context K/V in the draft graph (#57632)

- **Mechanism.** The draft's FULL decode graph also captures `precompute_and_store_context_kv`, with one verify of context rows per request. Rows past the batch write to `PAD_SLOT_ID`. Verified.
- **Effect.** The drafter bucket was 5.24 ms per prose step in E5 F3. The eager precompute is a small, launch-bound part of it. -0.2 to -0.6 ms. Inference.
- **Cost.** Our DFlash2 is a backport of #52816 onto an August tree. The upstream speculator has moved since, so expect a re-diff, not a clean apply.

### 6. GLM tool-argument streaming (#61253, #64535)

- **Mechanism.** #61253 streams the open argument value incrementally, so cost per chunk no longer grows with the argument: 6.4 ms at 100 KB becomes 7 µs. #64535 parses each closed `<arg_key>/<arg_value>` pair once. Upstream's replay with the GLM-5.3 tokenizer hit a liveness kill at 275 s. Verified.
- **Why it matters here.** We serve `--tool-call-parser glm47` (`run.sh:586`). Agentic clients send file writes of tens of KB. With DFlash2 each delta holds about 2.2 tokens, so the per-chunk count drops, but the quadratic term remains. Inference.

### 7 and 8. W4A16 MoE kernels (Humming, FlashInfer #5735)

- **Humming.** Image 0.1.12 has no `tune/sm121.py`; 0.1.18 has it (humming #150 merged 2026-10-19, release 2026-10-21). Our vLLM also predates #57421, so a Humming MoE backend would hold its permutation scratch in every one of the 42 MoE layers. Verified. Run desk-note X2 first.
- **FlashInfer #5735.** It reads the cuTile NVFP4 layout with no second weight copy. On DGX Spark at the Qwen3.8-Flash-Next shape it is 1.36x Marlin at 1 token and 1.00x to 1.03x at 8 to 16 tokens. Verified (PR table). Our verify step runs 8 rows, and masked rows reuse their anchor's experts (`head.log` line 103). The distinct-expert load sits between the 1-token and 8-token cases. Inference. Nothing in vLLM calls this backend yet.
- **Marlin itself.** Since 2026-08-13 no commit to `csrc/moe/marlin_moe_wna16`, `fused_moe/experts/marlin_moe.py` or the Marlin utils changes kernel code. The only kernel-tree commit is #60721 (headers). Verified.

### 9 to 13. Smaller items

- **#59528.** Since vLLM #35431 the runners mark stateless rows with block 0, not `PAD_SLOT_ID`. The kpool tail slot mapping then writes dummy, capture and padding rows into the null block. Upstream reports no output change, because real requests never own block 0. Low priority. Verified.
- **#59126.** Part 1 only fires under `VLLM_GPU_SYNC_CHECK`. Part 2 sizes the vision rope table from `text_config.max_position_embeddings` (about 64 MiB bf16 per GPU). An overflow needs a grid side above 8192 patches. That takes an aspect ratio near 4000:1 after the 8000-token budget fit. Inference. The fix is 4 lines.
- **#57534, #58450, #57458.** #57534 cuts the kpool tail slot mapping from 12 launches to 1 (157 to 38 µs host at bs=1 on GB300). It must keep our `pos % RING` addressing and the `GLM53_TAIL_STASH_MASK` fill (`indexer.py:585-591`). #58450 and #57458 shave sparse-MLA metadata and query prep. Verified mechanisms; effect inference.
- **#47616.** A stop string that ends inside an accepted batch leaves later token ids and logprobs in the output, and in usage. Our bench does not use stop strings. Clients may.
- **#55738.** It only helps prompts of 2048 tokens or fewer, and needs `FlashAttnPrefillBackend` on sm_121, which is unverified.

## Already present

| Upstream | What it does | Where the image has it | Label |
|---|---|---|---|
| #58454 (09-25) | kpool tail ring of `kpool * cdiv(kpool + num_spec, kpool)` slots | `GLM53_KPOOL_TAIL_FIX=1`, `glm53_tail_ring_slots` (`ops/kpool_compress.py:25-45`) | verified |
| #57477 (09-20) | Tail-seed kernel addressed by the view's padded strides | Same knob, `ops/kpool_compress.py:529-535`. With `KPOOL_TAIL_FIX=0` the v11 dense addressing returns | verified |
| #58704 (09-25) | SM90 builder reads `index_kpool` from the model config | The image always did (`flashinfer_mla_sparse_sm90.py:306`). Upstream broke and refixed it | verified |
| #55222 part 1 (09-29) | fp8 KV plans as `float8_e4m3fn` | `flashinfer_mla_sparse_sm90.py:434` | verified |
| #58065 (09-22) | Async scheduling with DFlash | `DFlashModelTypes` is inside `EagleModelTypes` (`config/speculative.py:67-69`); no "Async scheduling not supported" warning in r2 logs | verified |
| #55736 part 3 (09-10) | No duplicate router GEMM | `GLM53_ROUTER_DEDUPE=1` through `DECODE_FIXES` | verified |
| #55736 part 1 | Strided KDA recurrent inputs | `GLM53_KDA_TRIM` exists but is off and has no GPU measurement (README-v13). Part 2, the token-major NoPE query, is absent (`mla_attention.py:916-919`) | verified |
| #56983, #60763 | DFlash2 on GLM-5.3 and aux hidden-state capture | v9 (`dflash2_backport.diff`) and v10 (`patch_v10_dflash_glm5.py`) | verified |
| #59973 (10-15) | Quantized draft `lm_head` shared with the target | INT8 g128 `lm_head` that DFlash2 shares (E4a) | verified |
| #60229 (10-29) | GLM-5.3 adaptive verification | Our fixed-shape `GLM53_ADAPTIVE_VERIFY`. Upstream's version needs varlen shapes that sm_121 rejects (desk note) | verified |
| #61228, #60140 | FP8 KV default | We already run `fp8_e4m3` | verified |
| #55277 | NoPE sparse MLA on SM12x | Our v8 patch routes sm_121 to the SM90 FlashInfer backend on FA2 | verified |

## Not relevant to this config

| Upstream | Why not | Label |
|---|---|---|
| #60618 fused MQA scoring and top-k for indexer prefill | SM100 family only (UMMA and TMEM) | verified |
| #57327, #62171 cooperative top-k | The dispatch excludes `is_device_capability_family(120)` | verified |
| #54951 TP-sharded indexer prefill | Needs at least 16384 x TP scheduled prefill rows; our `max-num-batched-tokens` is 4624 | verified |
| #59464 index conversion reused across layers | Edits only the FA3 backend (`flashattn_mla_sparse.py`). A port to our SM90 path would save about 0.1 to 0.2 ms | verified; size inference |
| #57443, #61873 fused multi-step draft decode, in-place draft metadata | MTP drafter path through sparse MLA; the DFlash2 drafter has no MLA | verified |
| #62102 `concat_and_cache_mla` scale skip | Kernel rebuild for microseconds per layer at 8 rows | inference |
| #60927 per-token-group quant | FP8 activation quant; our linears are W4A16 and W8A16 | verified |
| #59246 `fp8_ds_mla` on SM90 | SM90 FlashMLA path, not ours | verified |
| #57701 indexer decode workspace | 2 x 8 x 327680 x 4 B = 21 MB here (`model_executor/layers/sparse_attn_indexer_kpool.py:300-313`) | verified |
| #62784 skip drafting on non-final prefill chunks | The drafter was 0.9% of an E5 prefill chunk | verified; size inference |
| #59468 no draft slots in the token budget | `MAX_NUM_BATCHED_TOKENS=4624` already pads the 16 slots | verified |
| FlashInfer #5648 W4A16 tactics on SM121 | Dense GEMM, not MoE. At M=1 it moves 50 MB in 253 µs (about 198 GB/s), below Marlin's 220-230 GB/s in `evidence/e1-microbench` | verified; bandwidth inference |
| #58911 flush text after reasoning with no tool parser | We always configure `glm47` | verified |
| #60497 fallback detokenization | Slow-tokenizer path only | verified |
| #59147 honour `--mamba-ssm-cache-dtype` for KDA state | KDA state is fp32 by default. bf16 would halve the state page but risks quality | verified; risk inference |
| vllm-project/DeepGEMM #9 (via vLLM #57218) | Wrong FP8 MQA logits at D32 and D64 on the fork's SM120 port. Our indexer head dim is 128, and the image vendors a different pin | inference |
| U+FFFD in streaming with spec decode | No upstream change found. Since 2026-08-13, `v1/engine/detokenizer.py` and `tokenizers/detokenizer_utils.py` only got #47616, #59046 and #60497 | verified |

## Rebase or backport

Upstream moved GLM code into `models/glm5next/common/` and replaced the bespoke KV grouping with the generic packed layout (#57169: 83 files, +1281/-1987). Our v11 draft-group patch, the v13 kpool ring and the DFlash2 backport all extend the bespoke layout. A rebase onto v0.33.0 would mean rewriting them. Inference. Backporting rows 1 to 6 is cheaper, and each one can run as its own knob, A/B'd with `bench_decode.py` (cell A, H, E) and the Tier 0 gate.

## Open questions

- Our prefix-cache hit rate with and without `GLM53_DFLASH_PREFIX_CACHE_FIX`. The bench salts prompts, so it needs the README-v13 probe.
- Whether FlashInfer's FA2 MLA kernel honours a clamped `kv_end` the way FA3 does (#58684).
- Whether FlashKDA at `17a037d` builds for sm_121 on aarch64, and its prefill share at 4608-token chunks.
- Humming and FlashInfer #5735 timings at GLM shapes (H 4096, 288 experts, top-8, TP=2) on spark1.

## Sources (accessed 2026-11-02)

- vLLM: `gh api repos/vllm-project/vllm/compare/...` for the image base and the tags; release notes v0.29.0 to v0.33.0; `commits?path=vllm/models/glm5next` since 2026-08-01; PR bodies and diffs for every number cited above (`gh pr view` and `gh pr diff`).
- FlashInfer PRs #5648, #5735, #4732, #5896, #4661 and #6292. Humming tags and PR #150. FlashKDA commits `b59532f`, `17a037d` and `47fc9fe`. vLLM `cmake/external_projects/flashkda.cmake` at `487ecf187`, v0.30.0, v0.31.0 and v0.33.0. vllm-project/DeepGEMM PR #9.
- NVIDIA Model-Optimizer `docs/source/announcements/20261028-glm53-flash-iq2.rst` @ `584a80c`, lines 108-111 and 364. It says to serve with vLLM 0.31 or later, or with 0.30 plus `--no-enable-prefix-caching`. It does not name the PR. The v0.31.0 notes list three GLM-5.3 kpool corruption fixes (#57477, #58454, #58704), and our image covers all three (Already present). That the ModelOpt bug is one of them is inference. #57477's body describes the same symptom: a cached system prompt slowly corrupted by later requests.
- Image files as cited, read from `glm53-sm121-v14` (9599a9fc5fe5) in CPU-only containers.
- Local evidence: `evidence/r2/r3/head.log`, `evidence/e5-final/notes.txt` (F3 buckets), `evidence/e5-final/f3/buckets/prefill-rank0.txt`, `docker/README-v13.md`, `docker/README-v14.md`, `docker/patch_v11_dflash_kv_groups.py`, `run.sh`, `bench_decode.py`.
