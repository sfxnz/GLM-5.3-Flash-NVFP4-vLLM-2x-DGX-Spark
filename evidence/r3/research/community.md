# Community lanes per decode step, and which levers port to our image (research, 2026-11-02)

Read-only desk research. No GPU, docker, ssh or serve command ran. Cloned repos sat in `/tmp/claude-1000/r3-community/` and were read, never executed. One CPU script read BF16 tensors from our local `09b04e5` snapshot (section 4.4). This note extends `GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-research-fp4local/evidence/research-20261030/fp4-local-and-spark-moe.md` sections 5 and 6 (the "prior note"). It does not repeat FP4-Local, Humming or the X1 pack refresh.

"Verified" means read in a primary file (code, config, PR body, result file) or computed by us from one. "Inference" means our reasoning or arithmetic on top of that, not measured. `[n]` points into Sources (section 10).

Our reference is cell A of `bench_decode.py`, as recorded in `recipe.yaml` `measured`. It runs 8 distinct prose prompts, 512 forced tokens (`min_tokens = max_tokens`, `bench_decode.py:386`), greedy, thinking off. It reads 31.23 tok/s, step 69.3 ms, 2.17 tokens per step. The per-step profile below is E5 F3 (`evidence/e5-final/f3/buckets/prose-rank0.txt`), an older config with a 79.4 ms profiled step, rank 0, 29 verify steps of 8 rows:

| bucket | ms/step | launches/step |
|---|---:|---:|
| routed MoE (Marlin NVFP4 W4A16) | 33.57 | 210 |
| linear_marlin (INT8 W8A16 targets, NVFP4 drafter) | 18.58 | 180 |
| bf16_gemm | 6.30 | 335 |
| drafter | 5.24 | 343 |
| nccl | 3.81 | 91 |
| kda | 2.33 | 102 |
| other (elementwise) | 2.08 | 981 |
| mhc | 2.05 | 186 |
| lm_head_logits | 1.64 | 14 |
| mla_indexer | 0.74 | 109 |
| idle | 5.50 | |

v14's `DECODE_FIXES` has since removed the duplicate router GEMM (about 0.72 ms of bf16_gemm) and the fp32 head-gate gemv (about 0.4 ms) [`docker/README-v14.md`].

## 1. Summary

- **The cleanest comparison is knapcio's stack on two Sparks.** tonyd2wild ported knapcio's TP=4 stack to TP=2 and measured prose at 35.30 tok/s with 2.04 tokens per step. That is a 57.8 ms step against our 69.3 ms at 2.17 [3]. Verified numbers; the step is our arithmetic. knapcio's image is tonyd2wild's v11, built on vLLM `487ecf187`. Our image reports the same vLLM build (`v0.1.dev20051+g487ecf187` in `evidence/r2/b0-base/head-lines.txt`). Verified. Its overlays therefore target the source tree our v14 patches, which makes them the most portable code found.
- **kindling's 42.6 tok/s is not on our ruler.** Its "thinking off" maps to `Reasoning Effort: Low`, so every reply opens a short `<think>` block (about 67 characters on their probe) [1]. Its 16 "Explain X" prompts are not forced to 512 tokens, and its prose prompts accept about 2.7 to 3.7 tokens per step at k=7 (24.5% to 38.2% per draft token, TP=4), against our E0 prompts' 2.26. Verified inputs; the comparison is inference. No acceptance length is published for the TP=2 row.
- **kindling's "GB10 W4A16 MoE decode kernel" (megamoe) is source-available but not a knob for us.** It is Apache-2.0 CUDA in `experimental/megamoe/`, JIT-built at boot, and it reads the FlashInfer CUTLASS backend's processed NVFP4 tensors. It needs `MOE_BACKEND=flashinfer_cutlass`, which our `run.sh` refuses (JIT OOM on spark2, and vLLM #63473 reports silent corruption on SM120/121). No megamoe-vs-Marlin number is published anywhere. Verified.
- **knapcio's "per-step draft truncation" is the same idea as our adaptive verify, with three differences.** It changes the real verify shape (2 to 8 rows) instead of masking rows inside an 8-row graph. It scores each depth with a calibrated table and picks the length that maximises expected tokens minus a cost term. A device-side parent graph picks the child graph, so the host never waits for the draft. knapcio also built our exact mechanism (fixed shape, dead rows remapped to the anchor's experts) and rejected it at TP=4 with prose −5 to −9%; the README does not name the baseline [4]. Verified.
- **knapcio's "lossless8" claim holds on our checkpoint.** We re-encoded sample tensors of `09b04e5` on CPU. KDA projections sit on an MXFP8 grid (0.03-0.07% relative error). MLA and shared-expert weights sit on a block-128x128 FP8 grid (0.16-0.17%). Our INT8 g128 is 0.65-0.68% on the same rows. `lm_head` is on no 8-bit grid. Verified (section 4.4). Same bytes as INT8, so this is Tier 0 headroom, not speed.
- **No alternative drafter is a one-knob win.** canada-quant `DFlash2-G` (Apache-2.0) reports +1.2% acceptance over incoai at K=7 on B300, thinking on. Its 8 full-attention layers would break our 327680 KV guard and our v11 draft KV group. modal-labs publishes no acceptance. RedHat's DSpark preview is a different method. incoai has no commit after `bf582e4`. Verified.
- **No faster vLLM 2x Spark lane appeared after 2026-10-30.** New activity is EXL3 on custom engines, Flare guides for FP4-Local, and eugr's b12x pin with the GLM recipe unchanged. vLLM #62956 (SM121 TP=2 skinny GEMM) merged on 2026-10-31 but covers BF16 at M of 1, 2 and 4 only. Verified.
- **Portable levers, by expected gain on cell A (inference, section 7):**
  1. RDMA one-shot all-reduce (b12x RoCEnante as an image layer), +2 to +3.5%.
  2. KDA verify without per-token state stores (knapcio stash or kindling RecoverSSM), +2 to +3%.
  3. Real-shape, cost-aware draft truncation with device-side selection, +1.6 to +2.4% alone and +0.6 to +0.9% after item 2.
  4. Faster 8-bit dense decode kernels, after a switch to grid-matched MXFP8 and block FP8, +1.5 to +2%.
  5. 8-bit `fused_qkv_a` and indexer `wq_b`, the remaining replicated BF16 linears, about +1%.
  6. L2 prefetch of the next weights during all-reduce waits, about +0.7%.
  7. `GLM53_KDA_TRIM=1`, already in our image and never measured, about +0.6%. The cheapest first test.
  8. mHC `hc_*_fn` weights read as BF16, about +0.4%.
  9. shm reader spin cut from 1 s to 2 ms, 0 to +1%, mostly thermal.

## 2. Why 42.6 and 35.3 are not 31.23

| | ours, cell A | kindling TP=2 | knapcio stack, TP=2 port (tonyd2wild) |
|---|---|---|---|
| prompts | 8 distinct prose | 16 distinct "Explain {topic}, in flowing prose" plus a per-run tag | 1 prompt (CPU branch predictor) |
| length | 512, forced by `min_tokens` | `max_tokens` 512, not forced | `max_tokens` 700, reached |
| thinking | off via our template | "off" means `Reasoning Effort: Low`, short `<think>` always opened | `reasoning_effort: low`, 0 thinking characters on prose |
| sampling | greedy | temperature 0 | temperature 0 |
| runs | three boots, two panels each | one boot, 16 recorded runs, bootstrap CI 41.7-43.6 | median of 3 |
| result | 31.23 tok/s, 69.3 ms, 2.17 tokens/step | 42.6 tok/s, AL not published | 35.30 tok/s, 2.04 tokens/step, step 57.8 ms |
| rails | both | both | one |

Sources: `bench_decode.py`, `recipe.yaml` [ours]; `gate/bench/README.md`, `gate/bench/bench.py:42-83`, `NOTES.md` "Thinking off" [1]; `runs/2026-09-29-knapcio-tp2/results/laneA-thinking-low-kv6.txt` and `bench/bench_tp2_night.py:68-110` [3]. All verified.

- **Acceptance differs by prompt set (inference).** kindling's NOTES give prose acceptance per draft token at k=7 of 38.2% on v4 and 24.5% on v8 (TP=4, `dev/repro/decode.py` prompts). That is roughly 3.7 and 2.7 tokens per step. Our E0 cell A gave 2.26 at k=7 (`docker/README-v13.md`). kindling also reports that mapping "off" to low effort raised DFlash2 acceptance (v6 vs v4). Some of the 42.6 vs 31.23 gap is the ruler.
- **`min_tokens` may cost us a little on our own ruler.** Our bench sets it (verified). knapcio's reading of vLLM `487ecf187` is that a `min_tokens` request disables the greedy fast path, so every step all-gathers full-vocab logits [4, `overlay/glm_target_argmax.py`]; we did not check that code path ourselves. knapcio measured −0.59 ms per step at TP=4 by keeping the greedy path for `min_tokens`. Real traffic samples at T=1.0 (`generation_config.json`), so that saving would move our ruler, not users.
- **The only like-for-like step number is knapcio's TP=2 port.** Its 57.8 ms at 2.04 tokens per step is 11.5 ms (17%) below ours at a similar acceptance. Inference from the published file. It ran one rail, and knapcio's TP=4-keyed dense and Marlin tables fall back to stock at TP=2 [3, RUNBOOK].

## 3. kindling, `kindlingai/glm-5.3-flash-gx10` @ `c540c91`

`c540c911839058c725d54b21ff0ff3a116f1dd13` (2026-10-24) is still HEAD on 2026-11-02 (`pushedAt` 2026-10-24). Verified.

### 3.1 Identity and licence (verified)

- Base `vllm/vllm-openai:nightly-ddd6fbca148a867aad1fcab7ec72f582b9977db4` with anchored patches, FlashKDA `17a037d`, mentat instead of Ray. Not our vLLM.
- The repository has no top-level licence (GitHub reports none). `experimental/megamoe/` has an Apache-2.0 `LICENSE` (added in #42). Most experimental files carry `SPDX-License-Identifier: Apache-2.0`. `gb10_sparse_mla.py` and `draft-trunc/draft_trunc.py` have no header, so their terms are unstated.
- TP=2 defaults (`image/entrypoint.sh:60-125`): KV pin 8 GiB, `MAX_MODEL_LEN` 163840, `MAX_NUM_BATCHED_TOKENS` 8192, RecoverSSM on with `MAX_NUM_SEQS` 16, adaptive-k prior `32.1, 1.32, 0.427` (ms, ms per expert touched, ms per verified token).
- MoE backend `flashinfer_cutlass` (NVFP4 W4A4) everywhere, with megamoe for decode. The README says Marlin "ran on earlier images and is untested on this one". NOTES say the 2026-09-21 switch from Marlin is not recorded.

### 3.2 megamoe, the GB10 W4A16 MoE decode kernel (verified unless marked)

- **Source.** `experimental/megamoe/megamoe_rm.cu` (decode), `moe_prefill.cu` (W4A4 prefill), `megadense4.cu` (NVFP4 W4A16 dense), `megamoe_vllm.py` (wiring). Apache-2.0.
- **Build.** `torch.utils.cpp_extension.load` at boot, `-O3 -gencode=arch=compute_121a,code=sm_121a`, behind an flock so a killed build cannot hang later boots (#61). The weight-snapshot loader calls `install()` after load.
- **Wiring.** `install()` swaps each `ModelOptNvFp4FusedMoE` method's class for a subclass whose `apply()` dispatches by batch rows. Up to `VLLM_MEGAMOE_MAX_TOKENS` (8) rows go to megamoe. 1024 rows or more go to `moe_prefill.cu`. Rows in between stay on CUTLASS W4A4. A layer qualifies only if it is not monolithic, has a synchronous finalize and has no expert map (no EP).
- **What it reads.** CUTLASS's own processed tensors: per expert `w13 [2I][K/2]` bytes, `w2 [H][I/2]`, e4m3 block scales in FlashInfer's 128x4 swizzle, and `weight_scale_2 / input_scale` folded into alpha. Decode and prefill share one weight copy.
- **Math.** Activations are converted bf16 to fp16 in `route_prep` (`__float2half_rn`). It uses `cvt.rn.f16x2.e2m1x2` and `cvt.rn.f16x2.e4m3x2` dequant, `mma.sync m16n8k16` with fp32 accumulate, 8 tokens per expert entry (the MMA's N), and `cp.async` streaming of contiguous 16-row tiles. The header says direct fragment loads reach "only ~205 GB/s". Expert id -1 makes no entry and reads no weights (this is how draft-trunc saves bytes).
- **Shapes.** E=288, top-8, H=4096. Per rank I=512 at TP=4 and 1024 at TP=2. Variant 20 by default (tens digit FC1 K split, units digit FC2 warps).
- **Claimed speed.** None against Marlin. "All overrides" against stock at TP=4 (PR #4): prose 38.7 to 65.8, code 91.3 to 120.8, structured 121.9 to 170.3, not isolated per piece. `megadense4.cu` claims 230 GB/s.
- **Effective bandwidth (inference).** kindling's fitted TP=4 cost prior is 0.635-0.660 ms per expert touched. One expert across 42 layers is 3.54 MB x 42 = 149 MB per rank at TP=4, so 225-234 GB/s. Our census-matched Marlin is 7.17 GB in 33.6 ms, 213 GB/s (`evidence/e5-final/notes.txt`). A kernel at 230 GB/s on our bytes would save about 2.5 ms per step (+3.5%). The TP=2 prior (1.32 ms per expert) was set to twice TP=4's, not fitted, so it says nothing new.
- **Portability.** Not a knob. It needs the CUTLASS weight layout, and loading that layout runs the FlashInfer CUTLASS backend we refuse. A port means rewriting the kernel against Marlin's repacked layout, or re-laying experts at load with no room for a second copy. The effort is weeks. The fp16 activation cast is a numerics difference from Marlin's bf16 path that Tier 0 would have to clear.

### 3.3 FP8 and NVFP4 copies of the dense layers (verified)

`snapshot/dense_fp8.py` converts the BF16 dense linears after load. FP8 is W8A8 with per-channel weight and per-token activation scales on CUTLASS `scaled_mm`, which kindling measured at +14-20% decode and +0.5% prose NLL. Layers matching `VLLM_DENSE_W4` (KDA `in_proj`, MLA `o_proj`, shared experts, every drafter layer) go to NVFP4 W4A16 through `megadense4.cu`, with an FP8 copy kept for batches above 32 rows. That adds +6-10% decode for about the same NLL again; the lm_head is FP8. The README puts the quality cost at "about 1% NLL on prose", and AGENTS.md records that NVFP4 attention projections once took GSM8K to 95.6 and HumanEval to 154. Our E2a-E2d (FP8 per-channel) and E2b (NVFP4) failed Tier 0 on KL (`docker/README-v13.md`). Not portable under our gate; our drafter is already NVFP4.

### 3.4 Triton sparse MLA (verified)

`fixes/gb10_sparse_mla.py` is a Triton NoPE sparse MLA, 3.3x faster than FlashInfer for prefill. At decode sizes the first version was 3-6% slower than FlashInfer; a Flash-Decoding split below 96 query tokens "recovers most of that" and avoids FlashInfer's per-step host sync. Separately, `fixes/_fa_common.py:268-272` removes a 136 MB `kv_indices_buf` clone per decode step. That clone is FlashInfer 0.7.x code. FlashInfer `v0.6.18` and `v0.6.18.post1` `flashinfer/mla/_core.py` contain no `clone` at all, and our image ships 0.6.18. So neither piece is a decode lever for us. The Triton kernel is a prefill candidate.

### 3.5 adaptive-k and draft-trunc (verified)

- **adaptive-k** (`adaptive-k/adaptive_k.py`) is a scheduler that picks k from {2, 3, 4, 5, 7} before drafting. It uses per-request and global per-position acceptance and a step-cost model learned online by an extended Kalman filter, `step_ms = base + per_expert * 288(1-(1-8/288)^(d*t)) + per_token * t`. Graph levels come from `num_speculative_tokens_per_batch_size`. At one request the schedule drafts 7 and the scheduler still chooses the verified count. A "code mode" keeps separate rates inside code fences and tool calls.
- **draft-trunc** (`draft-trunc/draft_trunc.py` plus whole copies of five vLLM files, pinned to `ddd6fbca`) runs after drafting. It marks a request's drafts dead where the running product of vLLM's `OnlineAcceptanceEstimator` predictions falls below `VLLM_DRAFT_TRUNC_TAU` (0.3). That estimator is a logistic fit on each draft's top-token log-odds with per-position intercepts (AUC 0.91-0.93 per position). Dead drafts keep the shape, their MoE routes become -1 (skipped by megamoe and CUTLASS), and the sampler sees -1. Every 4th step skips the cut and is the only step that trains the estimator, which fixed a drift that took structured from 166 to 126 tok/s.
- **Measured (PR #50, TP=4, paired gate/bench after a mixed warm-up):** prose +3.2% (Holm p 0.025), code +3.3% (p 0.63), structured −1.1%, mixed 8 +3.5%. "The cut accepts 4-5% fewer tokens per step and makes each step 7-8% faster."
- **Against ours.** draft-trunc is our adaptive verify with a calibrated, online-trained predictor (tau 0.3 on calibrated survival) instead of the raw selector product (tau 0.4). The -1 route and our anchor remap save the same expert bytes. The calibrated predictor is the transferable part (section 4.3).

### 3.6 Other decode pieces (verified unless marked)

- **RecoverSSM** (`fixes/recoverssm.py`, vLLM Apache-2.0 header, ported from vLLM's Kimi-K3 RecoverSSM). KDA verify reads one checkpoint state per request, records each token's k, gate, v and beta, and after sampling replays only the accepted tokens through the stock `fused_recurrent_kda`. Bit-exact, 15,288 checks (PR #9). TP=2 RigMark (one run each): prose 35.0 to 36.4, structured 79.3 to 89.5. TP=4 one-stream prose flat. Its main purpose is concurrency, from 3 to 16 requests at TP=2.
- **mHC projection weights in BF16** (PR #47). The fused post plus pre-norm kernel read `hc_*_fn` as fp32 (1.5 MB a call). A cached BF16 copy, widened before the FMA, keeps every product. `mhc_fused` 11.9 to 8.2 µs, 1.06 to 0.74 ms per step, GPU time per step 36.79 to 36.17 ms (TP=4, one stream). Greedy logprobs identical.
- **arx** (`experimental/arx/`). RDMA all-reduce up to 256 KB, 13-27 µs against NCCL's 80-90 µs (TP=4, through a switch). The GPU publishes a sequence number and a CPU thread posts the writes over both ConnectX roots. `VLLM_GLM_ARX_PREFETCH` asks L2 for the next weights while a decode all-reduce waits, which saved 0.6 ms per 32 ms step.
- **Spin wait** (`image/patches/spin_wait.py`). `busy_loop_s` 1 to 0.002 in `shm_broadcast.py`. "Raises decode and drops the SoC ~20 °C", with no decode number. Always-block (0) "measured worse".
- **Drafter KV in its own pool** (`fixes/kv_cache_*.py`) gives about 40% more KV tokens with acceptance unchanged. A capacity lever, not cell A.

### 3.7 Acceptance numbers kindling reports (verified)

None for the published TP=2 or TP=3 rows. NOTES report acceptance per draft token at TP=4, k=7 for structured / code / prose. v4 (2026-09-23, thinking off as an empty `<think></think>`) read 88.5 / 65.0 / 38.2%. v8 (2026-09-26, low effort, its own prompt set) read 97.2 / 74.7 / 24.5%. On the 2026-09-23 image the checkpoint's MTP head at k=4 decoded 57.2 / 54.4 / 45.6 tok/s against DFlash2's 109.8 / 88.8 / 52.6 (v6).

## 4. knapcio's stack and tonyd2wild's TP=2 port

- knapcio `knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4` @ `770d1153062aa916b06591411c61d5c593ea0f03` (2026-09-29, still HEAD). MIT for repo-owned files; files with an Apache-2.0 SPDX line keep it (`NOTICE`). The vLLM-derived overlays are Apache-2.0. Verified.
- tonyd2wild `GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark` @ `d061f26ad3ec5c3c04f64aad7516dbe62a8aa6af` (2026-09-30), `runs/2026-09-29-knapcio-tp2/`. No top-level licence. The port changes `start.sh` to two ranks, turns off three TP=4-only prefill pieces, and pins 6 GiB of KV. "Everything on the decode side stays on and armed at TP2." Verified.

### 4.1 Same vLLM as ours (verified)

knapcio builds `Dockerfile.roce` on `ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef6…` and every overlay docstring names "Tony v11 image, vLLM 0.1.dev20051+g487ecf187". Our engine log reports `v0.1.dev20051+g487ecf187`. Our v8 to v14 chain differs from tonyd2wild's (our own DFlash2 backport, our v13/v14 patches), so each overlay must be re-anchored, but against the same upstream source.

### 4.2 What the TP=2 port measured (verified)

Lane A, 6 GiB KV, thinking low, one rail, median of 3: prose 35.30 tok/s, 2.04 tokens per step, 700 tokens, 0 thinking characters; narrative 35.31; code 68.91 at 5.12; JSON 71.00 at 4.97. Prefill 1,121 / 1,200 / 1,225 tok/s at 6k / 30k / 114k, below our 1,593 / 1,562 at 32k / 128k. The decode metric is `(completion_tokens - 1) / (t_last - t_first)`. The per-position counters read all zero in the JSON, so only the mean acceptance length is usable.

### 4.3 Draft truncation and device-side selection, against our adaptive verify

| | ours (`ADAPTIVE_VERIFY`, v13) | knapcio `GLM_DRAFT_TRUNC` + `GLM_DEVSELECT` | kindling draft-trunc |
|---|---|---|---|
| when it decides | after the draft, for the next verify | after the draft | after the draft (adaptive-k before it) |
| confidence | running product of DFlash2 selector max-softmax | per-depth log max-softmax binned into a fitted table `g[depth][bin]` (`glm_bav_table_seg.json`, 17 edges) | vLLM `OnlineAcceptanceEstimator`, logistic, trained online |
| rule | keep drafts while product >= tau 0.4 | `L* = argmax E(L) - lam * T(L+1)`, E from the table, T from a measured step-cost table, lam a running throughput EMA | keep drafts while calibrated survival >= 0.3; every 4th step uncut |
| shape | fixed 8 rows; masked rows take the anchor's experts at weight 0, draft id -1 | real M = L+1 rows, the captured FULL graph of that family | fixed; dead rows route to expert -1 |
| per-row work of cut rows | still paid (KDA state writes, attention, mHC, norms, logits rows) | not paid | still paid |
| host cost | none | late decision waits for the draft (about 1.7 ms per step) until `DEVSELECT`; a parent graph then selects the child graph on the device at one request | none |
| measured | E3b/E4b/r2 B5 on our ruler: +14.9%, +5.2%, +4.2% | TP=4: prose c1 +5.8% (trunc), +1.9% more (devselect, step −0.45 ms), over a host adaptive-k 3/7 policy | TP=4: prose +3.2% over adaptive-k |

Sources: `docker/README-v13.md` [ours]; `overlay/glm_draft_trunc.py:1-200`, `glm_devselect.py:1-60`, `glm_draft_trunc_cost.json`, README "Added on 2026-09-29" [4]; section 3.5 [1]. All verified.

- **knapcio's TP=4 cost table** (`glm_draft_trunc_cost.json`, step probe run 2146): 2 rows 30.1 ms, 3 33.5, 4 36.1, 5 39.8, 6 42.6, 7 45.7, 8 49.3. Verified.
- **knapcio rejected our mechanism.** `overlay/glm_verify_cut.py` is a fixed-shape confidence cut with the dead rows' MoE input gathered from the anchor row (`GLM_VERIFY_CUT=conf:<T>`). The README lists it as "tried and rejected: confidence-based verify cut (prose −5 to −9 %)". Verified. The README does not name the baseline. At TP=4 a rank reads half the expert bytes, so per-row costs weigh more there than on our TP=2, and the result may not transfer (inference).
- **What real-shape truncation would add for us (inference).** On prose c=1 our adaptive verify keeps 4.33 live rows of 8 (`evidence/e5-final/notes.txt`, census). The 3.67 masked rows still cost per-row work. The largest per-row item is KDA state stores (section 7, L2). The others are small at 8 rows. They are mHC per-token L2 re-reads, MLA and the indexer, logits rows and elementwise kernels. Estimate 0.3-0.45 ms per row, so 1.1-1.6 ms per step (+1.6 to +2.4%), less whatever L2 already recovers. A better rule than tau 0.4 may add or lose a few percent; no data on our ruler.
- **Port effort: large.** It needs FULL graphs for every verify width (vLLM's per-batch-size spec schedule, `SPEC_TABLE`), a device-side selector of about 1,500 lines, and rework of our `KPOOL_TAIL_FIX` ring and AV state, which assume a fixed width. The calibrated table alone (the policy, without shape change) is a small change to our `patch_v13_verify.py` rule. It would need its own fit on our drafter and checkpoint.

### 4.4 lossless8, checked on our checkpoint

- **knapcio's claim (verified in code).** `scripts/glm_quant_mix.py:346-349`: "The released BF16 weights already sit on 8-bit grids: KDA on an MXFP8 grid (re-quantization error 0.05 %), MLA projections and shared experts on a block-128 FP8 grid (0.16 %)." The `lossless8` arm encodes `kda_in`, `kda_o` as MXFP8 and `mla_qa`, `mla_qb`, `mla_o`, `sh_gu`, `sh_down` as block-128x128 FP8. It keeps `kv_b` BF16 (absorbed at load). Both formats run as Marlin W8A16 in `487ecf187`: MXFP8 through ModelOpt `MIXED_PRECISION`, block FP8 through `Fp8LinearMethod` with `VLLM_TEST_FORCE_FP8_MARLIN=1` (`overlay/qmix_patch.py`). The step saving over BF16 dense was −7.5 ms at TP=4.
- **Our check (verified, CPU only).** We read four 128-row tiles (512 rows) per tensor from `09b04e5` and computed relative Frobenius error for three encodings. MXFP8 used knapcio's rule, which keeps per 32 elements the better of `ceil(log2(amax/448))` and `floor(log2(amax)) - 8`. Block FP8 used scale = 128x128 tile amax / 448. INT8 g128 used our quantizer: BF16 scale amax/127 per 128 along K. The e4m3 rounding was nearest on the full e4m3fn grid.

| tensor | shape | MXFP8 | block-128x128 FP8 | INT8 g128 (ours) |
|---|---|---:|---:|---:|
| layer 0 KDA `q_proj` | 8192 x 4096 | **0.00031** | 0.02665 | 0.00667 |
| layer 0 KDA `o_proj` | 4096 x 8192 | **0.00050** | 0.02501 | 0.00678 |
| layer 20 KDA `k_proj` | 8192 x 4096 | **0.00068** | 0.02640 | 0.00661 |
| layer 11 MLA `q_a_proj` | 1536 x 4096 | 0.02619 | **0.00157** | 0.00652 |
| layer 11 MLA `q_b_proj` | 16384 x 1536 | 0.02532 | **0.00170** | 0.00671 |
| layer 11 MLA `o_proj` | 4096 x 16384 | 0.02345 | **0.00159** | 0.00665 |
| layer 3 shared `gate_proj` | 2048 x 4096 | 0.02398 | **0.00167** | 0.00656 |
| layer 3 shared `down_proj` | 4096 x 2048 | 0.02624 | **0.00162** | 0.00680 |
| layer 30 shared `up_proj` | 2048 x 4096 | 0.02444 | **0.00160** | 0.00652 |
| `lm_head` | 154880 x 4096 | 0.02658 | 0.02647 | **0.00674** |

- **Reading.** The right 8-bit grid has 10-20x less weight error than INT8 g128 on KDA and about 4x less on MLA and shared experts, at the same bytes (1 byte plus 1/32 byte of E8M0 scale, or 1/16384 of an fp32 tile scale, against INT8's 2/128). `lm_head` is on no 8-bit grid, so INT8 g128 stays best there. The small residual is BF16 rounding of the stored values. Our inference is that these weights were trained or released on FP8 grids (Z.ai's default release is FP8 block-128), and the BF16 copies inherited them.
- **What it buys.** No step time by itself. Our Tier 0 KL on the five INT8 groups is 6.63e-3 to 6.76e-3 against a 6.86e-3 limit (E4a, F1). Moving `kda_in`, `kda_o`, `mla` and `shared` to their native grids should recover most of the +0.7e-3 KL INT8 added (inference; Tier 0 has to show it). That headroom is what an NVFP4 `lm_head` (prior note X6) or an 8-bit `fused_qkv_a` would spend. It is also the prerequisite for knapcio's dense decode kernels (L4).

### 4.5 FP8 drafter (verified)

`scripts/drafter_fp8.py` re-encodes the incoai drafter's q, k, v, o and gate, up, down as block-128 FP8 on Marlin W8A16. It keeps `model.fc` BF16 (so it can be split over TP), and the conv, selector and norms. The draft graph went 4.24 to 3.10 ms at TP=4, "acceptance unchanged". The drafter is trained in BF16 and is on no grid, so this is lossy for acceptance only. Our NVFP4 drafter (E2e, step −4 ms) reads fewer bytes, so this is not a lever for us.

### 4.6 The other decode pieces in `profiles/current.env` (verified; TP=4 gains)

| switch | mechanism | knapcio's measurement | for us |
|---|---|---|---|
| `GLM_ROCE_ALLREDUCE=1` (`Dockerfile.roce`, `roce/`) | b12x RoCEnante one-shot all-reduce and all-gather over RDMA, both rails; CPU proxy in C over libibverbs; captured in graphs; fail-stop | not isolated at TP=4; tonyd2wild TP=2 (2026-09-18): single stream flat on a clock-clamped fleet, aggregate +5-18% | L1 |
| `GLM_KDA_STASH=1`, `GLM_KDA_STASH_NOCOPY=1` | verify writes slot 0's full state and small replay records (about 1.7 KiB per head per token) instead of a full fp32 state per token; ulp-level, not bit-exact | "−0.90 ms" for stash, no-copy and fused flags together | L2 |
| `GLM_KDA_NOCOPY` | the recurrent kernel reads q, k, v, g, beta by token stride; bit-exact | audit 0.19-0.25 ms per step | same as our `GLM53_KDA_TRIM` (L6) |
| `GLM_CERT_HEAD=1` | exact target argmax from an MXINT8 screen of the lm_head with a proven error bound; BF16 rows read only for candidates; two host syncs | with gate GEMV and early plan: −1.25 ms per c1 step | quality, not speed, for us: our INT8 head already reads 1 byte |
| `GLM_TARGET_VOCAB_ARGMAX=1` | vocab-parallel greedy argmax, 16 bytes per row gathered instead of full logits; also for `min_tokens` | −0.59 ms on `min_tokens` requests | moves our ruler only (section 2) |
| `GLM_DS_SPLIT=fc:128,qkv_a:64,wq_b:16` | replicated BF16 linears (`fused_qkv_a` 185 MB, indexer `wq_b` 138 MB, drafter `fc` 168 MB per rank per step) split by output columns over TP and all-gathered | with FP8 draft head and one-gather top-k: dense −0.48 ms | L5 |
| `GLM_DENSE_FAST=1` | custom W8A16 decode kernels (`glmk_dense.cu`) that read Marlin's MXFP8 and block-FP8 buffers in place, deterministic split-K | 1.03-1.13x per shape at M<=32; 0.78 ms per step at M=8; not bitwise (about 1 bf16 ulp on 0.1% of outputs); tables keyed to TP=4 shapes | L4 |
| `GLM_L2_PREFETCH*` | side-stream `cp.async.bulk.prefetch.L2` of the next weights while KDA recurrence or an all-reduce leaves DRAM idle; exact | −0.610 ms at c1, 95% CI [0.548, 0.671] | L7 |
| `GLM_MHC_BF16W=1` | TileLang source edit so the decode mHC kernel reads `hc_*_fn` as BF16 (90 matrices, 135 MiB of fp32 reads per step at c1); install self-check is bitwise | in the −1.90 ms "rest" bundle | L8 |
| `GLM_MARLIN_TUNE_ON=1` | per-shape Marlin launch configs; a persistent MoE lock workspace instead of a zero-filled one per call | tuned table has 3 MoE entries (gate_up, M=4/16/32), no dense entries | small; TP=2 shapes differ |
| `GLM_EARLY_PLAN=1`, `GLM_GATE_GEMV=1`, `GLM_ROUTER_DEDUP=1` | MLA plan during the draft; head-gate GEMV; router GEMM dedup | in the −1.25 ms bundle; router dedup −0.5 ms | we have the last two (v14); our late-plan A/B was flat (r2 B6) |
| `GLM_SHM_BUSY_LOOP_S` | shm reader spin shortened | implemented, not set in the profile | L9 |

Rejected by knapcio, with numbers: the fixed-shape confidence cut (−5 to −9%), Marlin tile M=32 (+1.2 ms), fused mHC kernels (not bit-exact), a CUDA graph for the DFlash context KV (+0.05 ms), W4A4 or MXFP4 experts for prefill (16-21% MoE output error) [4, README].

## 5. Alternative drafters, against incoai `7d74cdd`

Reference `incoai/GLM-5.3-Flash-DFlash2` @ `7d74cdd881ed7e32c31175984a67823127b66cfe`: `DFlash2DraftModel` (`model_type` qwen3), 5 sliding-window layers (2048), 32 heads / 8 KV heads, head_dim 128, `target_layer_ids` [5, 14, 24, 33, 42], block 8, selector rank 256 top-k 16, conv kernel 2 group 16, vocab 154880, `mask_token_id` 154856, CC BY-NC-ND 4.0. HEAD is still `bf582e4` (2026-08-31); its `config.json` is identical to `7d74cdd`'s (diffed). Verified.

| drafter @ sha | differences from the reference (verified) | licence | loads through our `SPEC=dflash2` path unchanged? | reported acceptance and protocol (verified) |
|---|---|---|---|---|
| `canada-quant/GLM-5.3-Flash-DFlash2-G` @ `b8ba68b022d612958d1a9c447f294f148a156ed1` (2026-10-04) | same class; **8 full-attention layers**, no sliding window; 9 taps [5, 9, 14, 19, 24, 28, 33, 38, 42]; learnable mask embedding in `mask_embedding.pt`; ships its own `embed_tokens` and `lm_head` (6.2 GB bf16); trained against canada-quant's W4A16 quant on self-generated data | Apache-2.0 | **No** (inference from our patch text). `patch_v11_dflash_kv_groups.py` carves only uniform sliding-window drafter layers into the draft group. Full-attention drafter KV costs about 3x context per GiB (card: 19-22 KB per token against 7.1), so the 327680 guard at 4.14 GiB fails. Whether our v9 backport loads `mask_embedding.pt` is unknown; the card says a silent miss must not be served | K=7, c16: 3.676 vs incoai 3.632; c1: 3.697 vs 3.667; K=4: 3.088 vs 3.136. 500 holdout prompts, thinking on, T=1.0, top_p 0.95, `max_tokens` 1024, TP=4 on B300, `FLASHINFER_MLA_SPARSE`. At `max_tokens` 4096: 3.478 vs 3.473 (parity). No Spark acceptance A/B |
| `modal-labs/GLM-5.3-Flash-DFlash` @ `dae6d319510889a7f1ac5242d5b7fb2d5d95eb05` (2026-09-20) | same class; 6 sliding-window layers of **4096**; 6 taps [23, 27, 31, 35, 39, 43]; `is_causal: true`; rope_theta 2e6; `output_multiplier` 1.0; no eos/pad ids | MIT | **Plausibly**, untested (inference). The v10 capture reads `target_layer_ids` from the config, our backport honours `is_causal`, and the layers are sliding-window. The 4096 window doubles the draft KV group. The card says "keep the draft model unquantized", which conflicts with `DRAFT_WEIGHTS=nvfp4` | none published; the card documents SGLang only |
| `RedHatAI/GLM-5.3-Flash-speculator.dspark-preview` @ `1972f1f0ebb5c1219e0a1f3344d642c857f5dfb4` (2026-09-25) | **`DSparkDraftModel`** (Speculators 0.7.0.dev174); 5 sliding layers, 64 heads, head_dim 64; aux layers [20, 28, 32, 36, 40, 44] plus implicit 45; Markov head (rank 256) and confidence head; block 8, 8 drafts | MIT | **No.** Needs `method: dspark` and "a vLLM build with compatible DSpark support and the Speculators plugin". Our adaptive verify reads DFlash2 selector scores and `run.sh` refuses it for other `SPEC` values. Its confidence head would be the natural AV signal after a port | weighted 3.77 tokens per step; writing 3.13, qa 3.11, HumanEval 4.74, math 6.12. T=0, top_p 1, 8 drafts, TP=4, 16k context, up to 200 requests per dataset, verifier `zai-org/GLM-5.3-Flash` |
| `RelaxingSnorlax/...dspark-preview-{BF16,FP8-block,FP8-dynamic}[-10L,-15L,-30L]` (2026-10-29) | FP8 requants and layer-cycled dummies of the RedHat preview; the 10L/15L/30L cards say "Not a trained checkpoint" | MIT | No (DSpark) | none; for latency experiments |
| `Enntity/glm-flash-drafter` (GitHub, `8d9bf2392134`, 2026-11-01) | from-scratch DFlash2-style trainer | Apache-2.0 | no weights released | none yet |

`chimcis/GLM-5.3-DSpark-fast-2026-10-27` targets `zai-org/GLM-5.3` (hidden 6144), not Flash, and is excluded. vLLM #64773 (open) adds a RedHatAI GLM recipe with the DSpark preview. Verified.

Reading (inference). canada-quant's +1.2% acceptance is under thinking on and T=1.0, and their 8 layers cost about 1.6x the reference's drafter bytes. On cell A the drafter bucket (5.24 ms) would grow by more than the acceptance gain returns. Its value is the licence. modal-labs is the only one that might load as is; it needs an acceptance A/B on our ruler before it is worth a boot.

## 6. Lanes published or updated after 2026-10-30 (verified unless marked)

- **eugr/spark-vllm-docker.** Commits 2026-10-29 to 2026-11-01 (sync, memory-leak fix, "Pin b12x", docs). `recipes/glm-5.3-flash.yaml` is unchanged since `b62168e` (2026-10-26).
- **Flare (spark-local-inference) guides.** `gpdev-Pilcothink/flare-spark` `c56bf1583204` (2026-10-31), forum t/386738 and t/387486. FP4-Local with MTP 5 on 2x Spark: llama-benchy tg128 38.21 tok/s and tg1024 37.46, pp2048 1,493. A different stack and ruler.
- **EXL3 on other engines.** `AmesianX/dspark-serve` `dce5e89e26a0` (2026-11-02, vLLM TP=2, EXL3 4bpw), `gilesco/GLM-5.3-Flash-EXL3-2x-GB10-TensorFold-ZIG-testing` `3ec36c450376` (2026-11-02), `jayleaton/glm53-tensorfold-spark` `1b4fb60929b1` (2026-10-30), `fly9i/spark-engine` (2026-11-02; its GLM rows are from 2026-10-08: prose greedy thinking off c1 52.9 tok/s, two prompts, 400 tokens). Different weights and engines.
- **Other.** `kindlingai/gr8` is the full GLM-5.3, not Flash. `tenhkspark/glm53-flash-nvfp4-2node` last changed 2026-10-01 (35.09 tok/s on 64 Japanese prose prompts, MTP K=2, W4A16 requant).
- **vLLM.** #62956 merged on 2026-10-31 (`e5196e2959a4`). It adds a CuTe DSL skinny GEMM for GLM-5.3-Flash BF16 linears at SM121 TP=2, M of 1, 2 and 4 only; "M=8/16 are within 2% of cuBLAS and are not listed". It does not touch our 8-row verify, and our targets are INT8. #64163 (open) changes Triton fused-MoE on integrated GPUs, not Marlin. #64916 (open, WIP) "DFlash2 draft sampling optimizations" has no description yet.
- **Not found (inference from GitHub, Hugging Face and forum searches on 2026-11-02):** any new vLLM 2x Spark lane on the NVFP4 family that reports a faster prose number than kindling or knapcio.

## 7. Portable levers

Expected gain is on cell A (69.3 ms step, 2.17 tokens per step). A step saving of d ms gives about d / (69.3 - d). All expected gains are inference. One knob means a `recipe.yaml` knob that `run.sh` turns into `GLM53_*` on both ranks, possibly on a new image layer.

| id | lever | mechanism | evidence | self-reported gain and protocol | our expected gain | one knob? | effort |
|---|---|---|---|---|---|---|---|
| L1 | RDMA one-shot all-reduce | replace NCCL for small TP all-reduces with b12x RoCEnante (GPU flags, CPU proxy posting RDMA writes on both rails) | knapcio `Dockerfile.roce`, `roce/` @ `770d115` (b12x `b58f34ea`, Apache-2.0, "without editing any vLLM file"); kindling `arx/` @ `c540c91` | arx 13-27 µs vs NCCL 80-90 µs per call (TP=4); tonyd2wild TP=2: single stream flat (clock-clamped fleet), aggregate +5-18% | our 91 collectives per step average 41.9 µs (3.81 ms); at 15-27 µs that is −1.4 to −2.4 ms: **+2 to +3.5%** | yes, as a v15 layer plus `ROCE_ALLREDUCE=1` | medium: build the layer on v14, check no b12x in the base, set `B12X_ROCE_HCA=rocep1s0f1,roceP2p1s0f1`, spin limit, memory of pinned slots |
| L2 | KDA verify without per-token state stores | write one full state plus replay records (knapcio, ulp-level) or keep one checkpoint and replay accepted tokens through the stock kernel (kindling RecoverSSM, bit-exact) | knapcio `overlay/kda_stash.py`, `glm_kda_stash.py`; kindling `fixes/recoverssm.py`, PR #9 | knapcio: −0.90 ms (with no-copy, TP=4); kindling TP=2 RigMark prose 35.0 to 36.4 (one run each) | at TP=2 each verify token stores 32 heads x 64 KiB = 2 MiB per KDA layer, 544 MiB per 8-row step, about 2.3 ms, which matches our 2.33 ms kda bucket; saving −1.5 to −1.9 ms: **+2 to +3%**; more at c=2 (5.53 ms bucket) | yes | medium: Triton kernel plus a commit hook in our runner; must compose with `KPOOL_TAIL_FIX`, prefix-cache block boundaries and our AV state |
| L3 | real-shape, cost-aware truncation | verify `L*+1` rows from captured graphs; L* from a calibrated table and a step-cost model; device-side graph choice | knapcio `glm_draft_trunc.py`, `glm_devselect.py` | TP=4 prose c1 +5.8% and +1.9% over host adaptive-k; their fixed-shape cut measured −5 to −9%, baseline unstated | per-row work of 3.67 masked rows, −1.1 to −1.6 ms: **+1.6 to +2.4%** alone; after L2 the KDA stores are gone and the rest is −0.4 to −0.6 ms (**+0.6 to +0.9%**); the rule change can add or lose a few percent | yes once built | large |
| L4 | faster 8-bit dense decode kernels | deterministic split-K W8A16 kernels reading Marlin's MXFP8 and block-FP8 buffers; needs the grid-matched formats of section 4.4 | knapcio `glm_dense_fast.py`, `glm_dense_fast_csrc/` | 1.03-1.13x per shape, 0.78 ms per step at M=8 (TP=4); not bitwise | our `linear_marlin` is 18.58 ms at about 238 GB/s; 8% faster is about −1.4 ms: **+1.5 to +2%** | yes | large: format switch first, then TP=2 tables and a JIT extension |
| L5 | 8-bit (or TP-split) remaining replicated BF16 linears | `fused_qkv_a` (16.8 MB x 11) and indexer `wq_b` (12.6 MB x 11), about 323 MB per rank per step | knapcio `glm_ds_split.py`, lossless8 `mla_qa` as block FP8; our grid check (`q_a_proj` 0.157% in block FP8) | dense −0.48 ms with draft head FP8 and one-gather top-k (TP=4) | halving 323 MB is about −0.7 ms: **about +1%**; a TP split adds a gather per call, which only pays with L1 | yes, as new groups in `patch_v13_fp8.py` | small to medium; Tier 0 |
| L6 | `GLM53_KDA_TRIM=1` | KDA recurrent kernel reads q, k, v, beta by token stride; removes 4 copies per KDA layer | our `docker/README-v13.md` (bit-exact under the interpreter test, never measured); knapcio `glm_kda_nocopy.py` | knapcio audit 0.19-0.25 ms per step at 16 heads per rank | 32 heads per rank: about −0.4 ms: **+0.6%** | already in the image (`EXTRA_ENV`); promote to a knob if it pays | trivial |
| L7 | L2 prefetch during all-reduce and recurrence | side stream prefetches the next weights into L2 while DRAM is idle | knapcio `glm_l2_prefetch.py`; kindling `VLLM_GLM_ARX_PREFETCH` | −0.610 ms at c1, CI [0.548, 0.671] (TP=4); kindling −0.6 ms per 32 ms step | about −0.5 ms: **+0.7%**; works best with L1 | yes | medium |
| L8 | mHC `hc_*_fn` as BF16 | widen BF16 to fp32 inside the TileLang kernel instead of reading an fp32 copy | kindling PR #47; knapcio `glm_mhc_bf16w.py` | `mhc_fused` 1.06 to 0.74 ms per step, prose +1.2% (TP=4) | replicated weights, so about −0.3 ms: **+0.4%** | yes | small; bitwise self-check |
| L9 | shm reader spin 1 s to 2 ms | stop vLLM's `MessageQueue` readers spinning `sched_yield()` for 1 s after each message on a shared CPU/GPU power budget | `vllm/distributed/device_communicators/shm_broadcast.py:134` @ `487ecf187` (default 1); kindling `spin_wait.py`; nacyot | nacyot (DSv4 FP8, TP=2 GX10): c=1 decode "within restart-to-restart noise", SoC 77 to 66 °C, vLLM CPU 333.6% to 88.7%; kindling: "raises decode", no number; knapcio did not enable it | **0 to +1%**; lower SoC temperature on spark1 | yes | trivial |

Not portable as a knob, or not a speed lever for us:

- **megamoe** (section 3.2). Needs the CUTLASS layout. If rewritten for Marlin's layout at about 230 GB/s, about −2.5 ms (+3.5%) (inference). Weeks.
- **Grid-matched 8-bit** (section 4.4). Speed neutral; Tier 0 headroom and the prerequisite for L4.
- **Certified LM head.** Exact output at about our INT8 head's bytes. A quality lever; a 4-bit screen could save about 1.2 ms per step but adds two host syncs (inference).
- **Vocab-parallel argmax for `min_tokens`.** Moves our ruler by about 0.5% and does nothing for T>0 traffic.
- **Calibrated confidence for our existing AV** (the policy part of L3 without shape change). Small, unmeasured either way; worth a tau-style sweep only after L2.
- **Alternative drafters** (section 5). None.

Do not stack. One change per boot against `bench_decode.py`, as AGENTS.md requires.

## 8. Explaining the 11.5 ms step gap to knapcio's TP=2 port (inference)

The port runs 57.8 ms at 2.04 tokens per step; we run 69.3 ms at 2.17. Summing the levers above that the port had armed at TP=2 gives:

- L1, 1.4-2.4 ms.
- L2 with no-copy, 1.9-2.3 ms.
- L3's per-row part net of the KDA stores L2 already removed, 0.4-0.6 ms.
- L5, about 1 ms (split plus FP8 draft head).
- L7, about 0.5 ms. L8, about 0.3 ms.
- The vocab argmax and certified head, about 0.5 ms (their run had no `min_tokens`, ours does).

That is about 6.0 to 7.6 ms. L4 and the Marlin table fell back to stock at TP=2 shapes, so they explain nothing here. The remaining 4 to 5.5 ms is unexplained. Candidates are host-side steps (early plan, GDN metadata fast path, devselect removing the host wait) and our 5.5 ms of idle in the E5 profile. Only a profile of their lane on our ruler would settle it, which is the prior note's X2b.

## 9. Open questions

- No megamoe-vs-Marlin measurement exists at any TP. kindling has never run Marlin on its current image.
- The acceptance length behind kindling's 42.6 is not published; its per-run `tok_per_step` is in result JSONs that are not in the repo.
- The knapcio TP=2 port's per-position acceptance counters read zero, so its truncation histogram at TP=2 is unknown.
- Whether `mask_embedding.pt` (canada-quant) and modal-labs' `is_causal: true` behave correctly in our v9 DFlash2 backport needs a boot.
- Whether RoCEnante's single-stream gain survives on an unclamped fleet at TP=2 has never been measured.
- `gb10_sparse_mla.py` and `draft_trunc.py` in kindling carry no licence header and the repo has no top-level licence.

## 10. Sources (all read 2026-11-02)

1. `kindlingai/glm-5.3-flash-gx10` @ `c540c911839058c725d54b21ff0ff3a116f1dd13`: `README.md`, `NOTES.md`, `KNOBS.md`, `AGENTS.md`, `kindling.json`, `experimental/README.md`, `experimental/megamoe/{megamoe_rm.cu,megamoe_vllm.py,megadense4.cu,LICENSE}`, `experimental/adaptive-k/adaptive_k.py`, `experimental/draft-trunc/{draft_trunc.py,grouped_topk_router.py:359-365}`, `experimental/fixes/{_fa_common.py:268-272,recoverssm.py}`, `experimental/compose/{megamoe,adaptive-k,fp8,fixes}.yaml`, `image/patches/spin_wait.py`, `image/entrypoint.sh:60-125`, `gate/bench/{README.md,bench.py}`, `gate/decode_stream.py`. PRs #4 (`95a93cd`), #9, #11, #13, #34, #47, #50, #51: https://github.com/kindlingai/glm-5.3-flash-gx10/pull/<n>
2. vLLM `OnlineAcceptanceEstimator`, `vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py` (local upstream copy of v0.30.0 under `~/projects/data/dsv41-1017/perf2/src/`), added by vLLM #52228.
3. `tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark` @ `d061f26ad3ec5c3c04f64aad7516dbe62a8aa6af`: `README.md`, `CURRENT.md:111`, `speed-night-2026-09-18/LOG.md:24-28`, `runs/2026-09-29-knapcio-tp2/{RUNBOOK.md,env.tp2-low,env.tp2-high,results/laneA-thinking-low-kv6.{txt,json},bench/bench_tp2_night.py}`.
4. `knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4` @ `770d1153062aa916b06591411c61d5c593ea0f03`: `README.md`, `CHANGELOG.md`, `LICENSE`, `NOTICE`, `docs/{runtime.md,weights.md,history.md,results/2026-09-29-release.md}`, `profiles/current.env`, `Dockerfile.roce`, `roce/glm_roce/adapter.py`, `roce/b12x/PROVENANCE.json`, `scripts/{glm_quant_mix.py,build_lossless8.sh,drafter_fp8.py}`, `overlay/{glm_draft_trunc.py,glm_devselect.py,glm_verify_cut.py,glm_bav_table_seg.json,glm_draft_trunc_cost.json,kda_stash.py,glm_kda_stash.py,glm_kda_nocopy.py,glm_cert_head.py,glm_target_argmax.py,glm_ds_split.py,glm_dense_fast.py,glm_dense_fast_table.json,glm_l2_prefetch.py,glm_mhc_bf16w.py,glm_marlin_tune.py,glm_marlin_tune.tuned.json,glm_shm_spin.py,qmix_patch.py}`.
5. Hugging Face, `config.json` and `README.md` at the pinned revisions, plus `/api/models/<repo>`: `incoai/GLM-5.3-Flash-DFlash2` @ `7d74cdd881ed7e32c31175984a67823127b66cfe` and `bf582e4eacc1810f76656d1811693ff6c6737d2a`; `canada-quant/GLM-5.3-Flash-DFlash2-G` @ `b8ba68b022d612958d1a9c447f294f148a156ed1` (with `PROVENANCE.txt`); `modal-labs/GLM-5.3-Flash-DFlash` @ `dae6d319510889a7f1ac5242d5b7fb2d5d95eb05`; `RedHatAI/GLM-5.3-Flash-speculator.dspark-preview` @ `1972f1f0ebb5c1219e0a1f3344d642c857f5dfb4` (commit list); `RelaxingSnorlax/GLM-5.3-Flash-speculator.dspark-preview-FP8-block-15L` @ `5eac1399b25ef5472cc6ef7dd6e9d6fe5d3986fe`; `chimcis/GLM-5.3-DSpark-fast-2026-10-27` @ `96fd459fe17bfdadad1c2a7a8db2a71d0c39665a`; listing `/api/models?search=GLM-5.3-Flash&sort=lastModified`.
6. vLLM `487ecf187d3dfe74d2cf6119a92881dba403c219`, `vllm/distributed/device_communicators/shm_broadcast.py:129-204`. PRs #62956 (merged 2026-10-31, `e5196e2959a4`), #64163, #64773, #64916, #63473: https://github.com/vllm-project/vllm/pull/<n>
7. FlashInfer `v0.6.18` and `v0.6.18.post1` `flashinfer/mla/_core.py` (no `clone`); `v0.7.0` tree has `flashinfer/mla/_batch_mla/_backends/_fa_common.py`.
8. nacyot, https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/ (no publication date; correction dated 2026-08-17).
9. NVIDIA Developer Forums: search `GLM-5.3 after:2026-10-29`; topics https://forums.developer.nvidia.com/t/386738, https://forums.developer.nvidia.com/t/387486, https://forums.developer.nvidia.com/t/382140.
10. Other repos (GitHub API, 2026-11-02): `eugr/spark-vllm-docker` commits since 2026-10-29 and `recipes/glm-5.3-flash.yaml` history; `gpdev-Pilcothink/flare-spark` @ `c56bf1583204` `Guide/GLM53-Flash.md`; `AmesianX/dspark-serve` @ `dce5e89e26a0`; `gilesco/GLM-5.3-Flash-EXL3-2x-GB10-TensorFold-ZIG-testing` @ `3ec36c450376`; `jayleaton/glm53-tensorfold-spark` @ `1b4fb60929b1`; `fly9i/spark-engine` README; `Enntity/glm-flash-drafter` @ `8d9bf2392134`; `tenhkspark/glm53-flash-nvfp4-2node` @ `014135f0d447`; `canada-quant/vllm-glm53-flash-sm121` @ `1c142c954436`; `Tutanka01/glm5.3-flash-2x-dgx-spark-nvfp4` @ `df1512671e98`.

Local evidence: `recipe.yaml`, `bench_decode.py:386`, `docker/README-v13.md` (adaptive verify, INT8 groups and errors, E2 table), `docker/README-v14.md`, `docker/patch_v11_dflash_kv_groups.py`, `docker/dflash2_backport.diff:158-160`, `evidence/e5-final/notes.txt` (census, masked rows), `evidence/e5-final/f3/buckets/prose-rank0.txt`, `evidence/r2/b0-base/head-lines.txt` (vLLM version), and the prior note. The grid check in section 4.4 read `~/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/09b04e5e74bca08ca8549fc736d4cdd8624bfde3` by safetensors header offsets with numpy; the script lived in the session scratchpad and is described fully in section 4.4.
