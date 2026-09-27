# Dimension: spec-decode

## Reviewer summary

I reviewed the speculative-decoding path end to end, reading source and evidence only, with no GPU use. Three measured data sets fit a bandwidth-bound step model: step_ms ≈ 73.4 + 0.75·D(k+1), where D(n)=288·(1−(1−8/288)^n) is the expected number of distinct routed experts per MoE layer for n verify tokens. The model reproduces DFlash2-7 (~117 ms, 19.2–21.2 tok/s), DFlash2-5 (~107 ms, 21.8–22.0 tok/s), structured k=5 (55.4 predicted vs 55.7 measured) and c=2 prose (~150 ms/step) to within a few percent. Prose acceptance collapses after position 3 (measured per-position 0.646/0.428/0.223/0.074/0.035/0.013/0.000). So k=7 pays for about 4 verify tokens whose MoE expert reads buy almost nothing on prose; the modeled prose optimum is k=3–4. Structured output wants k=7. The biggest lever is making the verify cost track acceptance, and I propose two ways. The first is a lossless kernel-level change that keeps shapes fixed: 'dead' tail verify tokens are not routed to new experts (modeled +20–25% prose, structured unchanged). The second extends vLLM's existing AdaptiveVerificationManager, which is DSpark-only today, to DFlash2. The cheap immediate step is k=4–5. PR #9 measured k=5 at +14% prose c=1 and +16–30% c=2 per stream, and that run was reverted for a render/lint failure, not a regression. k also sets KV-pool cost: each extra slot costs 4 KDA spec-state block ids per request. At k=7 that is 28 of ~35 ids, which matches the measured 26.7% pool per short request, so the 3.62 GiB context guard only holds for k=7. The v11 draft KV group puts a ~17.6% tax on every pool block. The v10 aux-hidden capture matches sglang #36708 (layers +1, hc_post then mean over 4 streams), and 98% structured acceptance backs this up. Low prose acceptance is therefore a distribution/quantization gap, not a capture bug. The recipe's default greedy drafting is lossless under both greedy and sampled targets. The probabilistic DFlash2 path, however, reuses the same Gumbel noise in the draft walk and in the residual resample, which biases it; do not enable it. SPEC=mtp is probably broken on the nvidia pack: layer 45 is BF16 (14.9 GB) and missing from hf_quant_config exclude_modules. It would add ~6.9 GiB per rank on a box already near its UMA limit, and MTP-4 measured no faster than DFlash2 anyway. The draft itself costs ~1.9 GB/rank/step (~8 ms, 7%), which is a secondary target.

## SD-1: MoE-aware verification: skip routed experts for low-survival tail verify tokens (lossless, keeps static CUDA-graph shapes)

- kind=perf component=glm5next MoE layer + DFlash2 speculator (kernel/model patch, new image layer v12) impact=5 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** At k=7 every step verifies 8 tokens per request, and each one pulls its own top-8 experts from LPDDR5X. On prose, positions 4-7 are almost never accepted (measured 0.074/0.035/0.013/0.000). About half of the MoE weight bytes per step are spent on tokens whose logits are never used. Tokens after the first live draft that is expected to fail can have their routed-expert contribution dropped without changing any accepted output. Causal attention/KDA means earlier (live) tokens never read later (dead) tokens. The rejection sampler treats a -1 placeholder draft as rejected, and the resample for the first dead position uses logits from the last live token.

**Mechanism:** Decode on GB10 is bandwidth-bound. The MoE term of a verify step scales with D(n)=288*(1-(1-8/288)^n) distinct experts per layer: D(8)=58.1 vs D(3.4)=26.3. Attention, KDA, shared-expert and lm_head bytes are per-weight, not per-token, so they do not depend on n. Removing the routed experts of dead tokens removes their expert reads without changing any shape.

**Evidence:**
- evidence/rebench-20260902T204243Z/engine.log.tail:256,260,270 (prose c=1 windows; weighted per-position acceptance 0.646,0.428,0.223,0.074,0.035,0.013,0.000 over 229 drafts)
- v11src/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:556-566,588 (-1 placeholder draft => rejected; greedy stores target argmax)
- nvidia ckpt safetensors headers: routed experts 171.229 GB = 14.16 MB/expert (U8 [2048,2048]x2 + [4096,1024] + F8 scales); TP=2 => 7.08 MB/expert/rank x 42 MoE layers = 297 MB per distinct-expert slot
- Fitted slope 0.75 ms per distinct expert per layer: k=5 steps 105.1 ms (21.97/2.308, PR#9 iter-H1 bench.json @e898a80) and 108.2 ms (21.76/2.354, rebench-dflash5); k=7 steps 119.2 ms (19.23/2.292 parity), 114.5 ms (21.18/2.426, rebench-20260902), 118 ms (66.55/7.844 structured)
- Bytes-only prediction at 237 GB/s would be 1.25 ms/slot; fitted 0.75 => ~60% of independent-routing D, consistent with intra-block routing locality

**Proposed action:** Add a persistent GPU buffer spec_live_mask[max_num_tokens] and anchor_idx[max_num_tokens]. The speculator fills them before target graph replay. After the draft, compute per-position survival = cumprod(conf_i), with conf_i = softmax(realized selector scores)[chosen] from DFlash2Speculator._selector_scores (walk kernel already stores it). Alternatively use a per-request acceptance EMA. Mark live = positions whose survival >= tau, plus one. For dead positions: (a) write draft token -1 into the verify input so the rejection sampler rejects them; (b) in Glm5NextMoE after select_experts, replace dead tokens' topk_ids with the anchor token's topk_ids and set topk_weights=0. This adds no new distinct experts, needs no -1 handling in Marlin, and stays graph-safe with torch.where. Ship it as a v12 patch layer that follows the v10/v11 style.

**Est. impact:** Prose c=1 at k=7 with ~3.4 live tokens avg: step 73.4+0.75*26.3 = 93.1 ms vs 117 ms; with acc ~2.35-2.42 gives 25.2-26.0 tok/s vs 20.7 modeled / 21.2 measured (+20-25%). Structured is unchanged (all positions have confidence ~0.98, so all stay live): ~67 tok/s. c=2 prose: D(16)=102 -> D(~7)=47 experts, step ~150 -> ~109 ms, per-stream ~16.6 -> ~22 (+30%).

**Validation:** Change one knob: SPEC_TAIL_MASK=1 at k=7. Run bench_decode.py prose c=1,2 plus structured c=1,2. Greedy count probe must stay 200/200, and the thinking-off smoke must pass. Log vllm:spec_decode_num_accepted_tokens_per_pos before and after; acceptance must not fall by more than the tau-induced truncation. Sweep tau in {0.3,0.2,0.1} only after the first pass works.

**Risks:** Selector softmax calibration is unknown, so a tau that is too high truncates accepted tokens (use an acceptance EMA as fallback). The fork's fused MoE path (Marlin, maybe SP) must accept post-routing edits of topk_ids/weights; the router may be fused into the kernel. Shared expert and attention still run on dead tokens (that cost is unchanged). EPLB counters are skewed by dead tokens.

**Verifier reasoning:** The lossless part holds up. rejection_sampler_utils.py:556-590 treats draft id -1 as rejected: greedy stores target_argmax at that index, and _resample_kernel uses raw target logits for a -1 placeholder (lines ~767-770). Attention and KDA are causal, so live positions never read dead ones. Dead-position KDA states and aux hidden states are discarded on rejection. Expert bytes check out: 14.16 MB/expert gives 7.08 MB/rank, times 42 layers = 297 MB per distinct-expert slot. D(8)=58.1 and D(6)=44.8 are correct.

Three problems. (1) The step model has two parameters fitted to k=5 and k=7. A linear-in-tokens model (step≈79.5+4.75·n ms) fits the same receipts about equally well, including c=2 prose k=7 (155 vs 150 ms measured; the D-model gives 152). Nothing yet shows that the k-slope comes from distinct-expert reads rather than per-token work. The fitted slope (0.75 ms/expert) is 40% below the bytes prediction (1.25), and that gap is patched with an untested 'routing locality' assumption. SD-1's saving exists only to the extent the D mechanism is real. (2) Routing is not in Glm5NextMoE. model.py:183-248 passes gate, use_grouped_topk and e_score_correction_bias into FusedMoEFactory, so topk is computed inside the fused-MoE router. The patch has to hook the fused_moe router/runner. -1 can go only into the rejection sampler's draft buffer: in the target input_ids it would index the embedding out of bounds. (3) On prose, SD-2 at k=3 already captures nearly all of this gain with zero code (the model gives 23.8 tok/s for k=3 vs SD-1's 25-26). SD-1's real incremental value is keeping structured at k=7. Its +20-25% prose figure is an upper bound that depends on the model.

**Verifier corrected claim:** Masking the routed-expert contribution of low-survival tail verify tokens (reroute them to the anchor's experts with weight 0, and send -1 only to the rejection sampler's draft ids) is lossless. It cuts step time only as much as the k-dependence of step time really comes from distinct-expert reads, which the current receipts cannot tell apart from per-token cost. It needs a patch in the fused-MoE router, not in Glm5NextMoE.

**Verifier corrected impact:** Prose c=1 upper bound ~+15-25% over k=7 if the D mechanism holds (0% if the k-slope is per-token compute). Its incremental gain over SD-2 at k=3-4 on prose is small (~+5-10%); the main value is keeping structured at ~67 tok/s.

## SD-2: Default k=7 is past the prose optimum; k=4-5 wins the published (prose) ruler

- kind=perf component=recipe.yaml serve.env.NUM_SPECULATIVE_TOKENS / run.sh:65-75 impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The throughput model tok/s = (1+sum_{i<=k} p_i)/step(k), using measured per-position prose acceptance and the fitted step cost, peaks at k=3 (23.8 tok/s) and k=4 (23.3). Modeled k=5 is 22.5 against 21.8-22.0 measured, and k=7 is 20.7 against 19.2-21.2 measured. PR #9 already measured k=5 at +14% prose c=1 and +16% c=2 per stream in the same session. That run was reverted only because 'render or lint failed after keep', not for a performance regression. Structured c=1 drops from 66.6 to 55.7 (-16%), which the model predicts (55.4).

**Mechanism:** Each extra draft slot adds a verify token and about 0.75 ms × ΔD of MoE reads, plus ~80 MB/rank of KDA per-position state writes (34 layers × ~2.34 MB). On prose, position 5+ adds ≤0.035 expected tokens. A prior decision (2026-09-02, adopting H3 DFlash2-7 at seqs=2) was driven by structured decode. H3 itself recorded prose 28.30→27.32, and the later k=5 reversion was a tooling failure.

**Evidence:**
- PR #9 @e898a80 evidence/iter-H1-20260903-20260903T010419Z/verdict.json: reason 'render or lint failed after keep', before_c1 19.23 after_c1 21.97, c2 15.13->17.52
- PR #9 evidence/rebench-dflash5-20260903T045815Z/summary.json: prose c1 21.76 (acc 2.354), c2 19.62/stream, structured c1 55.73 (acc 6.0)
- PR #9 evidence/rebench-parity-20260903T005500Z/bench.json: k=7 prose c1 19.23 (acc 2.292), structured c1 66.55
- evidence/rebench-20260902T204243Z/bench.txt SUMMARY: k=7 prose c1 21.18 acc 2.426; c2 16.64/stream acc 2.50
- Model table prose c=1: k1 19.3, k2 22.8, k3 23.8, k4 23.3, k5 22.5, k7 20.7; structured (p~0.985/pos): k3 41.0, k4 48.5, k5 55.4, k7 67.5
- AGENTS.md: 'Published decode score is prose only'

**Proposed action:** Set NUM_SPECULATIVE_TOKENS=4 as the next single-knob test, then k=3, against the k=5 receipt. Adopt whichever wins prose c=1 without regressing prose c=2. Record structured as a documented occupancy lane (k=7) rather than the default. Leave the capture ladder logic unchanged: it derives sizes (k+1)×{1..seqs}.

**Est. impact:** Prose c=1: +8-15% (21.2 -> 22.5-23.8 modeled; +14% measured for k=5). Prose c=2 per-stream: +16-30% (k=5 measured 17.5-19.6 vs 15.1-16.6). Structured c=1: -16% (k=5) to -39% (k=3).

**Validation:** python3 bench_decode.py (prose, c=1,2), once each for NUM_SPECULATIVE_TOKENS=4 and 3. Run the count probe (200) and thinking-off smoke. Record in evidence/trail.tsv + decision.tsv with acceptance and per-position metrics.

**Risks:** DFlash2 was trained with block 8. A shorter block (1+k) changes the non-causal mask context and the conv block_size (dflash2_backport.diff:339-341); k=5 showed no acceptance loss (2.31-2.35 vs 2.29), but k=3/4 are unmeasured. Structured/code users lose throughput; SD-1/SD-3 remove the tradeoff.

**Verifier reasoning:** The model-table arithmetic reproduces from p=[.646,.428,.223,.074,.035,.013,0]: k1 19.3, k3 23.8 (1+1.297=2.297 over 96.4 ms), k4 23.3, k5 22.5, k7 20.7. Old-era receipts back the step-time vs k trend independently: evidence/baseline-bench.txt, iter-h1 and iter-h3 give prose steps of 101.6/109.8/116.2 ms for k=5/6/7, and structured 106.5/113.7/117.8.

The measured gain is overstated. PR#9 H1 at k=5 gave 21.97 at c=1, and rebench-dflash5 gave 21.76. The +14% is against the low parity k=7 run (19.23). Against the same-config rebench k=7 (21.18) the gain is +2.7-3.7%, inside the ±11% run spread (18.75-23.40). c=2 prose went from 15.13/16.64 to 17.52/19.62 (+5% to +30%). The 'tooling failure' was not incidental: H1 lint.log shows 'FAIL VALIDATE_ONLY=1 did not print "cudagraph_capture_sizes":[1,2,4,8,16]'. The recipe lint hard-codes the k=7 ladder, so any k change must update recipe.yaml and the lint in the same change or the kit reverts it again. The reviewer also left out that structured c=2 regressed at k=5: 55.50 (parity) / 60.41 (rebench) → 41.56 (rebench-dflash5 summary.json), -25% to -31%. Old era: k=5 prose 28.30 vs k=7 27.32 (only -3.5%, and k=7 had higher acceptance there, 3.17 vs 2.875). Every receipt is on the LibertAI pack; the PR#11 nvidia default is unmeasured.

**Verifier corrected claim:** The step model and one measured k=5 point suggest k=3-5 beats k=7 on prose. The measured c=1 gain at k=5 is +3-14% depending on which k=7 receipt is the baseline, which is within run-to-run noise. c=2 prose improves 5-30%, structured c=1 drops 16%, and structured c=2 drops 25-31%. Changing k also requires updating the lint's hard-coded capture ladder.

**Verifier corrected impact:** Prose c=1 +3-12% (modeled k=3-4; measured k=5 +3-14%); prose c=2 +5-30%; structured c=1 -16% (k=5) to -39% (k=3); structured c=2 -25-31% at k=5. Must be re-measured on the nvidia pack.

## SD-3: Adaptive verification already exists in this vLLM but is gated to DSpark; DFlash2 has the needed confidence signal

- kind=perf component=vllm/config/speculative.py, v1/worker/gpu/spec_decode/adaptive_verification.py, dflash2/speculator.py impact=4 confidence=2 effort=L needs_gpu=True
- **verdict: refuted** (corrected confidence 4)

**Claim:** The v11 image ships AdaptiveVerificationManager. It profiles draft and verify cost curves, then admits a global top-k of (request, step) slots by survival probability (cumprod of per-position confidence), which gives variable-length verification. A validator rejects it for any method other than dspark. DFlash2's selector walk already produces per-step candidate scores (_selector_scores), so a confidence vector costs one softmax-gather. The smallest patch touches about 30 lines.

**Mechanism:** Variable per-request verify length makes cost track expected acceptance, the same effect as SD-1, but through vLLM's varlen path rather than expert masking.

**Evidence:**
- v11src/vllm/config/speculative.py:242-244 (enable_adaptive_verification: 'Currently only supported for method=dspark'), :1177-1178 (raise ValueError if method != dspark)
- v11src/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:35-60 (_assign_draft_token_budget survival=cumprod(confidence)), :63-107 (cost tables from profiled curves)
- v11src/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:82-87,209-226 (draft_token_confidence_probs + enable flag interface)
- docker/dflash2_backport.diff:712-716 (walk kernel stores realized scores per candidate), :766-772 (_selector_scores buffer)
- v11src/vllm/config/speculative.py:182-187 (num_speculative_tokens_per_batch_size: batch-size-only schedule, does not help c=1)

**Proposed action:** Patch v12 in three parts. (1) Relax the validator to allow method=='dflash' with a DFlash2DraftModel architecture. (2) In DFlash2Speculator, add self.enable_adaptive_verification and draft_token_confidence_probs [max_num_reqs,k]. After _sample_path, fill them with softmax(_selector_scores[r,i,:])[chosen_idx] (add one tl.store of max-prob in _selector_walk_kernel). (3) Check that the target's attention groups (FLASHINFER_MLA_SPARSE_SM90 NoPE, KDA mamba) pass get_query_lens_mismatch_unsupported_backend. Override the cost curves from measured step times: the profiler's dummy batches route every token to the same experts and will badly underestimate the MoE slope (use table derived from step_ms = 73.4+0.75*D(n)).

**Est. impact:** Same envelope as SD-1: prose c=1 ~+15-25% over k=7 while structured stays ~67 tok/s. It also shrinks the verify batch, so attention/KDA compute and KDA state writes drop too (~0.34 ms per dropped token).

**Validation:** Boot with speculative-config enable_adaptive_verification=true at k=7. Confirm no backend-mismatch refusal. Run bench_decode.py prose and structured c=1,2, plus the count probe, and compare with SD-1 and SD-2 receipts.

**Risks:** Varlen decode across the sparse-MLA indexer (kpool tail) and KDA spec states is untested in this fork. CUDA-graph FULL capture may fall back to PIECEWISE for varlen. Profiling with dummy tokens misstates MoE cost. It is a bigger patch surface than SD-1.

**Verifier reasoning:** The validator gate is real (speculative.py:242-244, 1177-1178), and the model runner wires the manager generically through getattr(speculator,'enable_adaptive_verification') (model_runner.py:532-534). But AdaptiveVerificationManager needs every attention backend to support a device/CPU query-length mismatch (attn_utils.py:184-197, backend.py:390-400), and three backends in this serve do not. (1) The DSA indexer: indexer.py:136-139 returns _supports_varlen_paged_mqa_logits(), which needs device capability family 100 plus DeepGEMM (indexer.py:649-654). The engine log confirms 'DSA indexer decode path: use_flattening=True supports_varlen=False' (rebench engine.log.tail:70). (2) KDA, the 34 linear-attention layers: backend.py:295 'return not cls.is_ssm()', so SSM backends opt out because their recurrent-state planning uses CPU per-request boundaries. (3) The FlashInfer draft non-causal attention: flashinfer.py:432-435 returns False (engine.log.tail:80 shows FlashInfer is used for the draft). This is not a ~30-line patch. It needs a varlen indexer on sm_121, varlen KDA spec-state planning, and a different draft attention backend.

**Verifier corrected claim:** Adaptive verification exists in v11 but cannot be enabled for this model on sm_121. The DSA indexer (no varlen paged MQA logits off sm_100), the KDA/SSM backend and the FlashInfer draft attention all opt out of device/CPU query-length mismatch. The fixed-shape SD-1 or a smaller k (SD-2) are the practical routes.

**Verifier corrected impact:** None achievable without a large multi-backend rewrite (effort XL). Drop from the plan.

## SD-4: k also sets KV-pool and UMA cost: 4 KDA spec-state block ids per slot per request; context guard constant is k=7-specific

- kind=ops component=KV cache layout (kv_cache_utils.py glm5next grouping, mamba spec blocks) + run.sh refuse-guard impact=3 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 5)

**Claim:** Each request holds (1+k) mamba state block ids in each of cdiv(34 KDA, 11 MLA)=4 mamba groups. Every id costs the full per-block byte sum, about 33.5 MB/rank. At k=7 that is 32 of ~35 ids for a short request, which matches the measured 26.7% of the pool per request. Going from k=7 to k=3 frees 16 ids ≈ 536 MB/rank per request. The run.sh guard (KV pin <= 3886945403 cannot hold 327680) and the AGENTS.md 4.14 GiB pin rationale were derived at k=7.

**Mechanism:** vLLM keeps one KDA recurrent state per speculative position so it can roll back rejected tokens. In the GLM-5-Next layout, mamba states share block ids with MLA pages, so every spec slot burns a full-layout block id in each of the 4 groups.

**Evidence:**
- v11src/vllm/model_executor/layers/mamba/abstract.py:74-78 (num_speculative_blocks = num_speculative_tokens)
- v11src/vllm/v1/kv_cache_interface.py:812-821 (mamba usage = page*(1+num_speculative_blocks) in mode none)
- v11src/vllm/v1/core/kv_cache_utils.py:1381 (num_groups = cdiv(num_mamba, num_mla) = 4), :1700-1705 (per_block = 11*mla_page + 11*idx_page + 5*draft_page)
- evidence/rebench-20260902T204243Z/engine.log.tail:54-55 (block size forced to 4608; mamba page padded 0.70% => MLA page 4608*512 B = 2.36 MB), :69 (1.14x concurrency at 327680), :226 (1 req = 26.7% KV usage), :265 (2 req = 53.4%)
- Computed: per_block = 11*2.359 + 11*0.152 + 5*5.898/5... = 25.95+1.67+5.90 = 33.5 MB -> 4445787956/33.5e6 = 132 ids; short request 32+1+2 = 35 ids = 26.5% (measured 26.7%)
- run.sh:108-111 (3886945403 guard)
- evidence/trail.tsv H1: 'extra slot taxes KDA copies and admission' pool 400k->386k

**Proposed action:** When changing k (SD-2), recompute the minimum pin for 327680 as ids = ceil(327680/4608) + 4*(1+k) + 3, and pin_bytes = ids*per_block/1.0x. Make the run.sh guard a function of NUM_SPECULATIVE_TOKENS instead of a constant (k=3: 91 ids vs 107 at k=7). Use the freed pool either to lower KV_CACHE_MEMORY by ~0.5 GiB/rank to relieve UMA swap (a KV pin of 5.0 GiB slowed decode ~20%, and swap was observed), or to allow MAX_NUM_SEQS=3-4 without the k=5 compromise. Longer term (L), consider checkpoint-and-replay for KDA states (store only the pre-step state and recompute accepted tokens) to decouple pool cost from k.

**Est. impact:** k 7->3: per-request short-context footprint 35 -> 19 ids (26.5% -> 14.4% of pool). Max concurrency at 327680 improves from ~1.14x to ~1.34x at the same 4.14 GiB. Alternatively the pin could drop by ~0.6 GiB/rank for equal capacity. k 7->5: 8 ids ≈ 268 MB/rank per request.

**Validation:** Boot with k=3 and read the log lines 'GPU KV cache size' and 'Maximum concurrency for 327,680'. Send a single short request and read the GPU KV cache usage %. Expect ~14-15%. Run free -h under load to check swap.

**Risks:** Lowering the pin changes the UMA balance that the OOM history was tuned on, so change one knob at a time. The guard math depends on the draft block (1152) and MLA page (512 B/token) staying the same.

**Verifier reasoning:** per_block = 11×2,359,296 + 11×152,064 + 5×1,179,648 = 33.52 MB. 4445787956/33.52e6 = 132 ids. A short request uses 35 ids; 35/131 usable = 26.7%, matching engine.log.tail:226. The reviewer's 327680 formula is wrong, though. 1.14x (engine.log.tail:69) = 132/116.0, so a max-length request costs 116 ids at k=7, not 107. About 9 ids come from the kpool tail and draft groups and are missing from their formula. Old H1-era pools confirm 4 ids per spec slot exactly: 400497/327680 = 1.2222 = 132/108 at k=5, and 386194/327680 = 1.1786 = 132/112 at k=6. The run.sh:110 guard 3886945403/33.52e6 = 115.97 ids falls just below 116, so the constant is exactly the k=7 threshold. For k<7 it errs conservatively.

**Verifier corrected claim:** Each speculative slot costs 4 block ids (33.5 MB/rank each) per request. A 327680 request needs 116 ids at k=7, 108 at k=5 and 100 at k=3. The 3886945403 guard equals 115.97 ids, the k=7 threshold, and is conservative for smaller k.

**Verifier corrected impact:** k 7→3: short-request footprint drops from 35 to 19 ids (26.7% to 14.5%), and max concurrency at 327680 rises from 1.14x to 1.32x (not 1.34x). For equal capacity the pin could drop by 16×33.5 MB ≈ 0.50 GiB/rank. k 7→5 frees 8 ids ≈ 268 MB/rank.

## SD-5: v11 draft KV group puts ~17.6% tax on every pool block id; decouple it

- kind=perf component=docker/patch_v11_dflash_kv_groups.py (kv_cache_utils glm5next layout) impact=3 confidence=3 effort=L needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** patch_v11 charges five draft sliding-window pages (1152 tokens × 5 layers × 4 KV heads/rank × 128 × 2 × 1 B fp8 ≈ 5.9 MB) on every block id: MLA ids, mamba-state ids and draft ids alike. A request needs only ~3 draft ids for its 2048-token window. Of ~132 ids × 5.9 MB ≈ 0.78 GB/rank of draft tensors, about 2 × 3 × 5.9 = 35 MB are ever live.

**Mechanism:** vLLM's GLM-5-Next slot-sharing layout sizes every tensor by a single num_blocks. Giving the draft its own tensors indexed by the shared pool ids makes the draft cost scale with the whole pool instead of with window × seqs.

**Evidence:**
- docker/patch_v11_dflash_kv_groups.py:69-86 (draft_block = attn_block//4, own group), :192-203 and :235-244 (per_block adds len(draft)*draft_page for every block id), :262-265 (draft KVCacheTensor size = draft_page*num_blocks)
- engine.log.tail:54 (attn block 4608 => draft block 1152)
- draft config: num_key_value_heads 8, head_dim 128, num_hidden_layers 5, sliding_window 2048 (DFlash2 config.json)
- Computed tax: 5.898 / 33.52 MB = 17.6% of per-block bytes

**Proposed action:** Allocate the draft KV outside the shared pool, as a fixed ring of max_num_seqs × (ceil(2048/draft_block)+1) pages per layer with its own small BlockPool or a static per-request slot map. Remove the draft term from per_block. A cheaper interim step is draft_block = attn_block//16 (288): the per-id tax falls to 4.4%, at the price of 9 draft ids per request. That helps long-context capacity (a 327680 request costs ~3.66 vs 3.94 GB/rank at k=7) but hurts short requests slightly.

**Est. impact:** Reclaims ~0.73 GiB/rank of the 4.14 GiB pin. That means either +21% block ids (132 -> ~160) or a pin cut of ~0.7 GiB for the same capacity (UMA headroom). No decode tok/s change, apart from indirect gains from less swap.

**Validation:** Check the boot log for GPU KV cache size and maximum concurrency at 327680. Run the needle probe at 20480 c=2 and bench_decode.py prose. Acceptance must be unchanged.

**Risks:** vLLM's coordinator assumes one BlockPool across groups, so the change touches the scheduler/KV manager. Prefix-cache interplay (already coarse with KpoolTailManager).

**Verifier reasoning:** patch_v11_dflash_kv_groups.py sets draft_block = attn_block//4 and adds len(draft_names)*draft_page to per_block in all three consumers. It also emits KVCacheTensor(size=draft_page*num_blocks). Draft page: 1152 × 4 KV heads/rank × 128 × 2 × 1 B = 1.18 MB per layer, ×5 layers = 5.90 MB. That is 17.6% of 33.52 MB. 132 × 5.9 MB = 0.72 GiB of draft tensors against a few tens of MB live. The patch docstring shows this was a deliberate trade-off (block-id burn vs tensor tax). The effort is correctly rated L.

**Verifier corrected impact:** ~0.69 GiB/rank of pool reclaimable (132→~160 ids, or a lower pin for UMA headroom). No direct tok/s change. Impact is more like 2 at max_num_seqs=2 with short prompts.

## SD-6: Losslessness: default greedy draft is exact; the DFlash2 probabilistic path reuses target Gumbel noise and biases the residual resample

- kind=correctness component=docker/dflash2_backport.diff (_selector_walk_kernel), v1/worker/gpu/spec_decode/rejection_sampler_utils.py (_resample_kernel) impact=3 confidence=3 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The recipe uses the default draft_sample_method='greedy'. With a one-hot proposal, Leviathan's rule (accept with prob p(x) using an independent uniform, then resample from p with x removed) is exactly lossless at any temperature, and the count probe confirms greedy. If someone enables draft_sample_method='probabilistic', things change. The DFlash2 walk samples candidates by Gumbel-max keyed on (request seed, position Q-1, token id), and the residual resample after a rejection uses the identical key (seed, pos, token). The resample noise is therefore conditioned on the event that selected the rejected draft, so the output is not distributed exactly as p. Counter-example: p=(.2,.4,.4), q=(.7,.25,.05); on rejecting 'a', token b's noise is constrained more tightly than c's, so P(b) < 0.4.

**Mechanism:** For the Leviathan rejection-sampling proof to hold, the residual sample must be independent of the draft sample. Sharing Gumbel noise is only correct for Gumbel-coupled acceptance (accept iff the argmaxes match), not for a ratio test followed by a residual resample.

**Evidence:**
- v11src/vllm/config/speculative.py:291-297 (draft_sample_method default 'greedy', one-hot in rejection)
- docker/dflash2_backport.diff:700-710 (walk: gumbel_noised_argmax(scores, candidates, ..., seed, position = sample_pos-1, temperature)), :537-557 (gumbel_noised_argmax: tl.randint(seed,pos) then tl_rand32(gumbel_seed, keys=token ids))
- v11src/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:568-569 (accept test uses independent u = tl_rand32(seed,pos)), :771-810 (probabilistic residual max(p-q,0)), :826-841 (resample gumbel_block_argmax with same seed/pos and token-id keys)
- dflash/speculator.py:261-262 comment 'verification keys Gumbel by the predecessor (Q-1)' (intentional coupling)
- evidence/rebench-20260902T204243Z/progress.md (count probe exit=0, greedy only)

**Proposed action:** Keep draft_sample_method greedy (the current default). Add a guard comment in run.sh/AGENTS.md. If probabilistic drafting is ever wanted, salt the draft walk's key (e.g. tl.randint(seed ^ DRAFT_SALT, pos)) so its noise is independent of the verifier's, and file this upstream against vLLM PR #52816 (the DSpark path has the same keying). Add a sampled losslessness probe: at temperature 1.0 with a fixed seed set on a low-entropy prompt, compare first-token/next-token histograms with and without spec decode (chi-square).

**Est. impact:** Correctness protection. No throughput change at the current config. Avoids a silent distribution shift for temperature>0 users if probabilistic drafting is turned on to raise sampled acceptance.

**Validation:** Sampled-distribution probe (N≈2000 single-token completions at T=1) with SPEC off vs SPEC=dflash2 greedy draft vs probabilistic draft. Expect the first two to match within chi-square noise and the third to deviate.

**Risks:** The bias size may be small in practice. The analysis is from source and needs empirical confirmation. The upstream authors may argue a different proof.

**Verifier reasoning:** The DFlash2 walk calls gumbel_noised_argmax(scores, candidates(token ids), seed, position=sample_pos-1) (dflash2_backport.diff:700-710). The residual resample calls gumbel_block_argmax, which in turn calls gumbel_noised_argmax with the same seed, pos_ptr[token_idx] and token-id keys (gumbel.py:125-170; rejection_sampler_utils.py ~826-841). dflash/speculator.py:261-262 documents the intended equality of keys ('verification keys Gumbel by the predecessor (Q-1)'). The accept test uses an independent tl_rand32(seed,pos), so the coupling only affects the residual. Given the draft argmax x, the other tokens' Gumbels are truncated at M - log q_y, which biases the residual argmax toward low-q tokens. The counter-example is valid. The recipe uses the default greedy draft (speculative.py:291-297), so the path is inactive today. This is upstream design, not a local backport bug.

**Verifier corrected impact:** None at the current config. It is a latent distribution bias if draft_sample_method='probabilistic' is ever enabled.

## SD-7: Sampled traffic (GLM's recommended T=1.0, top_p 0.95) gets one-hot acceptance p(x); block verification is a free lossless gain

- kind=perf component=--speculative-config rejection_sample_method impact=2 confidence=3 effort=S needs_gpu=True
- **verdict: refuted** (corrected confidence 5)

**Claim:** With greedy drafting and a sampled target, per-position acceptance equals p_target(draft token). That is lower than both the greedy bench cells and the DFlash2 card's T=1.0 numbers, which used a coupled sglang sampler. The v11 rejection sampler already implements block verification (Sun et al. 2024), including the one-hot residual. It is lossless and raises expected accepted length for multi-token drafts under sampling.

**Mechanism:** Block verification accepts the longest prefix under a joint-ratio criterion instead of stopping at the first per-token ratio failure. It has the same output distribution and a weakly higher accepted length, and it costs a few extra vocab reductions per step.

**Evidence:**
- v11src/vllm/config/speculative.py:220-226 (rejection_sample_method 'standard'|'synthetic'|'block')
- v11src/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:595-627 (block verification), :812-823 (one-hot residual under block verification)
- DFlash2 README (snapshot 7d74cdd README.md): eval at temperature 1.0 top-p 0.95, MT-Bench acceptance 4.03, GSM8K 5.78
- bench_decode.py stream_one: temperature 0 only (published cells never exercise sampling)

**Proposed action:** Add '"rejection_sample_method":"block"' to SPEC_CONFIG as a single-knob test. Add a sampled bench cell (T=1.0, top_p=0.95) that reports acceptance only; per AGENTS.md it is not published as decode.

**Est. impact:** Greedy cells: 0. Sampled prose: typically +3-10% accepted length (literature range; unmeasured here), so about +3-8% tok/s for T>0 clients.

**Validation:** bench with a temperature=1.0 variant: acceptance_len with rejection_sample_method standard vs block. Run the SD-6 distribution probe to confirm losslessness.

**Risks:** Extra vocab-wide kernels (residual mass) add ~0.1-0.5 ms per step at vocab 154880. Block verification with DFlash2 is untested in this fork.

**Verifier reasoning:** With greedy (one-hot) drafting, the recipe default, block verification cannot raise the expected accepted length. Any valid verifier must satisfy P(accepted_len ≥ i) ≤ P_target(output starts with d1..di) = Π_{j≤i} p(dj). Token-wise Leviathan with q=δ accepts each position with probability min(1, p/1) = p(dj), so it already attains that bound for every i. Block verification (Sun et al.) helps only when q is a non-degenerate distribution. The code's 'one-hot residual under block verification' branch (rejection_sampler_utils.py ~812-823) just makes it valid, not better. At T=0 (all published cells) the greedy branch runs anyway (line ~570). The warmup log also shows block_verify=False (engine.log.tail:73).

**Verifier corrected claim:** With the recipe's greedy draft, block verification gives exactly zero expected-acceptance gain at any temperature, because token-wise verification is already optimal for one-hot proposals. It could help only with probabilistic drafting, which SD-6 advises against.

**Verifier corrected impact:** 0 tok/s in every cell, plus a small extra-kernel cost. Do not spend a boot on it.

## SD-8: SPEC=mtp rollback is likely broken on the nvidia pack and would add ~6.9 GiB/rank; MTP-4 has no performance case

- kind=ops component=run.sh SPEC=mtp / nvidia checkpoint layer 45 / models/glm5next/nvidia/mtp.py impact=3 confidence=3 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** In nvidia/GLM-5.3-Flash-NVFP4@09b04e5 the MTP layer (layers.45) is fully BF16 (14.865 GB, experts BF16 [2048,4096]), yet hf_quant_config.json exclude_modules does not list layers.45. mtp.py builds the MTP decoder with vllm_config.quant_config (ModelOpt NVFP4), so vLLM will create NVFP4 expert and linear params for BF16 tensors that have no weight_scale. That likely fails to load. Even if it loaded as BF16, the MTP layer adds ~7.4 GB/rank against ~1.3 GB/rank for DFlash2, on nodes measured at ~115/121 GiB. The LibertAI pack's MTP layer is NVFP4 at 4.447 GB. Measured MTP-4 prose c=1 was 19.22 vs DFlash2-7 19.23 in the same session, with lower acceptance (2.175 vs 2.292).

**Mechanism:** ModelOpt only calibrated and exported the layers the HF forward runs; the MTP head was left BF16 and left out of the exclude list. The per-step MTP-4 draft cost is also high: 4 sequential passes, each reading the lm_head shard (634 MB), BF16 experts (~201 MB/rank), attention (~125 MB) and a replicated eh_proj (67 MB), ≈1.05 GB × 4 ≈ 18 ms per step versus ~8 ms for DFlash2.

**Evidence:**
- nvidia ckpt headers: model.language_model.layers.45.mlp.experts.0.gate_proj.weight BF16 [2048,4096], eh_proj BF16 [4096,8192]; mtp_layer total 14.865 GB
- nvidia hf_quant_config.json exclude_modules: 132 entries, none containing 'layers.45'
- LibertAI caca4e6 headers: layers.45.mlp.experts.0.gate_proj.weight U8 [2048,2048] + weight_scale F8_E4M3; mtp_layer total 4.447 GB
- v11src/vllm/models/glm5next/nvidia/mtp.py:44,69-83 (quant_config = vllm_config.quant_config passed to MTP decoder layer)
- PR #9 @e898a80 evidence/iter-H2-20260903-20260903T012725Z/bench.json: SPEC=mtp prose c1 19.22 acc 2.175 (LibertAI pack)
- engine.log.tail:52 (Model loading took 90.67 GiB per rank with DFlash2; MTP not loaded)
- AGENTS.md: 'SPEC=mtp rolls back to MTP-4'

**Proposed action:** In run.sh, refuse SPEC=mtp when MODEL is nvidia/GLM-5.3-Flash-NVFP4 unless FORCE_UNSAFE_SPEC=1, and say that MTP rollback requires the LibertAI pin (MODEL=LibertAIDAI/... SNAPSHOT_REV=caca4e6). Update the AGENTS.md rollback line. If MTP on the nvidia pack is ever needed, add layers.45* to the quant exclusion via a local hf_quant_config override, or quantize the MTP experts offline to NVFP4.

**Est. impact:** Prevents a boot failure or UMA OOM on the documented rollback path. No decode change (MTP-4 ≈ DFlash2-7 measured; DFlash2 at tuned k beats it by ~15%).

**Validation:** CPU-only: VALIDATE_ONLY run of run.sh with SPEC=mtp on the nvidia pin should now refuse. Later, in an exclusive GPU slot, a single boot attempt of SPEC=mtp FORCE_UNSAFE_SPEC=1 would confirm the load error.

**Risks:** The fork may special-case BF16 MTP weights somewhere not inspected, in which case load succeeds but the memory risk remains.

**Verifier reasoning:** A CPU header read of nvidia 09b04e5 shows layers.45 at 14.865 GB, all BF16 (experts.0.gate_proj BF16 [2048,4096], eh_proj BF16 [4096,8192]). hf_quant_config.json exclude_modules has 132 entries; entries with 'layers.4' prefixes stop at layers.44. config.json quantization_config.ignore has none for 45. num_nextn_predict_layers=1. The LibertAI caca4e6 MTP is 4.447 GB with NVFP4 experts (U8 [2048,2048] plus F8 scales). mtp.py:42-83 passes vllm_config.quant_config into SharedHead and Glm5NextDecoderLayer. The loader (mtp.py:301-440) has FP8 attention/indexer dequant helpers but no BF16-into-NVFP4 path, so a load failure is likely (not proven). Correction: PR#9 H2 measured MTP-4 at c=2 prose 17.70 vs DFlash2-7's 15.13 (+17%), so MTP has some c=2 case, though it is no better than DFlash2-5 (17.52-19.62). At c=1: 19.22 vs 19.23.

**Verifier corrected impact:** Prevents a probable boot failure or a +~6.9 GiB/rank UMA hit on the documented rollback. MTP-4 is not better than DFlash2 at tuned k, but it did beat DFlash2-7 at c=2 prose.

## SD-9: v10 aux-hidden capture matches the sglang reference; the low prose acceptance is a distribution/quant gap, and acceptance doubles as a quant-fidelity metric

- kind=quality component=docker/patch_v10_dflash_glm5.py, eagle3_utils.py, mhc.py impact=3 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The capture semantics are correct. vLLM maps target_layer_ids [5,14,24,33,42] to capture points [6,15,25,34,43] (+1), which is the same as sglang's set_dflash_layers_to_capture. At each capture point, v10 materializes the deferred hc_post of the previous layer and averages the 4 streams (hc_contract = mean over dim 1). That equals sglang's _prepare_aux_hidden_state (hidden_states+residual, then hc_contract mean over hc_mult). Structured acceptance of 0.978-0.989 is only possible if the target features reaching the draft K/V are right, because DFlash drafts see context only through those features. The prose gap (acceptance length 2.42 at T=0, thinking off) against the card's MT-Bench 4.03 (T=1, Max reasoning, BF16 target) is better explained by: thinking-off answers vs reasoning-trace training data; an NVFP4 target whose argmax drifts from the BF16 target the drafter was trained on; fp8 draft KV; and a single 80-word prompt.

**Mechanism:** Acceptance length measures how often the quantized target's argmax agrees with a drafter trained on the BF16 target's features and outputs. Given a fixed drafter and prompt set, higher acceptance means the target is closer to BF16.

**Evidence:**
- docker/patch_v10_dflash_glm5.py:46-66,81-105 (capture at top of iteration layer_idx, hc_post + hc_contract)
- v11src/vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py get_eagle3_aux_layers_from_config ('Add 1 to convert DFlash's aux layer id semantics')
- v11src/vllm/model_executor/layers/mhc.py:561-563 (hc_contract = x.mean(dim=1))
- gh pr diff 36708 --repo sgl-project/sglang: _prepare_aux_hidden_state + set_dflash_layers_to_capture: layers_to_capture = [val + 1 ...]; test asserts [6,15,25,34,43] and mean over hc_mult
- sglang glm5_next.py:1084-1094,1158-1167 (reduce_output then contract)
- evidence/rebench-20260902T204243Z/engine.log.tail:297,304 (structured per-position 0.983-0.989)
- DFlash2 README: evaluated with zai-org/GLM-5.3-Flash (BF16) target, temperature 1.0, Max reasoning
- engine.log.tail:51 (draft gets text-only inputs; image tokens reach it only through target hidden states)

**Proposed action:** Nothing to fix in v10. Add acceptance-only diagnostic cells that are not published: (a) the prose prompt with enable_thinking=true; (b) 8-16 diverse prose prompts; (c) a GSM8K-style math prompt. Use them to A/B the nvidia pack against the LibertAI caca4e6 pack at identical settings, as a GPU-cheap quality proxy. Only the per-position acceptance vector is needed, and it is already in /metrics as vllm:spec_decode_num_accepted_tokens_per_pos.

**Est. impact:** No direct tok/s. It gives a quantitative quality signal for the nvidia-vs-LibertAI decision (PR #11 is unmeasured) and rules out a silent capture bug (the ~0.2 acceptance rate was the main worry).

**Validation:** Two boots (nvidia pin, LibertAI pin), same bench prompt set, compare per-position acceptance. Expect the thinking-on cell to be clearly higher (>3) if the thinking-distribution explanation holds.

**Risks:** Acceptance is a proxy. Agreement with the BF16 argmax is not the same as task quality, so pair it with the existing count/thinking-off/tool probes.

**Verifier reasoning:** eagle3_utils.py:45-46 adds 1 to DFlash target_layer_ids ([5,14,24,33,42] from the draft config.json gives capture points [6,15,25,34,43]). patch_v10 captures at the top of iteration layer_idx, before that layer runs, which equals the output of layer_idx-1. It materializes prev_layer.hc_post and contracts with hc_contract, the same hc_post then hc_contract sequence the model itself uses for its final layer (model.py:508-511), so the shapes and semantics are consistent. Structured per-position acceptance of 0.983-0.989 (engine.log.tail:297,304) could not happen with broken features. I did not re-verify the sglang PR diff. The prose-gap explanation is a hypothesis. Also note the card's eval used an FP8 target KV cache on GB300 TP4 with T=1.0 and Max reasoning (draft README:52-54).

## SD-10: Draft forward reads ~1.9 GB/rank per step (~8 ms, ~7% of step); FP8 draft and sharding replicated pieces trim ~3-4 ms

- kind=perf component=qwen3_dflash.py / qwen3_dflash2.py / DFlash2 checkpoint impact=2 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** DFlash2 is BF16, 2.18 GiB, with no own embed or lm_head (it shares the target's). Its 5 layers are TP-sharded (QKVParallel, RowParallel, MergedColumn MLP 12288): 965 MB/rank. fc [4096,20480] (168 MB) and 10 conv kernel_projections (84 MB) are ReplicatedLinear and so read in full on both ranks. The precompute builds a concatenated copy of the KV weights (42 MB/rank). compute_candidates runs the target lm_head shard (154880/2 × 4096 BF16 = 634 MB) over 7 rows each step. Total ≈ 1.89 GB/rank ≈ 8.0 ms at the fitted 237 GB/s. The draft forward is captured as a FULL CUDA graph; the context K/V precompute runs eagerly outside it.

**Mechanism:** On bandwidth-bound decode the draft pays per weight byte. FP8 halves the matmul weight bytes, and sharding the replicated layers halves their per-rank reads.

**Evidence:**
- DFlash2 model.safetensors header: 2,342,160,896 B BF16; fc.weight [4096,20480]; layers.N.mlp.{gate,up,down} [12288,4096]/[4096,12288]; attention_conv/mlp_conv.kernel_projection [1024,4096]; candidate_selector codebooks [154880,256] x2; no lm_head/embed
- v11src/vllm/model_executor/models/qwen3_dflash.py:195-210,314-320 (TP-sharded attn/MLP), :417-427 (fc ReplicatedLinear), :453-454 (torch.cat copy of KV weights)
- docker/dflash2_backport.diff:278-286 (conv kernel_projection ReplicatedLinear), :491-496 (candidates via self.lm_head)
- v11src/vllm/v1/worker/gpu/spec_decode/dflash/utils.py (draft lm_head/embed replaced by target's)
- engine.log.tail:96-98 (dflash2 FULL CUDA graphs captured, 2 sizes)
- v11src/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:403-420 (precompute eager, outside graph)

**Proposed action:** Test these one at a time. (1) '"quantization":"fp8"' in SPEC_CONFIG (online FP8 of the BF16 drafter; check that an sm_121-capable FP8 GEMM path exists, since _C lacks 12.1a) saves ~0.6 GB, about 2.5 ms. (2) Make fc and the conv kernel_projections column-parallel plus all-gather, or keep them replicated; this saves ~0.13 GB (~0.5 ms) but adds 1-2 collectives over RoCE, so measure. (3) Do not add a separate FP8 lm_head copy (+317 MB UMA for ~1.3 ms).

**Est. impact:** Best case ~3 ms per step out of ~107-117 ms: +2.5-3% tok/s in every cell. The draft is a secondary cost; the target MoE and non-expert BF16 weights dominate.

**Validation:** Boot with the draft quantization knob only. Run bench_decode.py prose and structured; acceptance must stay within noise (per-position vector) and the step time should drop ~2-3 ms.

**Risks:** FP8 on the drafter may lower acceptance, which would erase the gain. FP8 scaled_mm availability on sm_121 in this image is uncertain (TORCH_CUDA_ARCH_LIST lacks 12.1a).

**Verifier reasoning:** The arithmetic reproduces. 5 layers × (q,o 4096² + k,v 4096×1024 + MLP 3×4096×12288) = 193M params × 2 B × 5 = 1.93 GB, giving 965 MB/rank. fc [4096,20480] BF16 = 168 MB replicated, kernel_projection ReplicatedLinear (diff:278-286), plus the target lm_head shard 634 MB via compute_candidates (diff:491-496). Total ≈ 1.89 GB/rank ≈ 8 ms. The candidate codebooks ([154880,256]×2) appear to be gathered by candidate id, not streamed in full. FP8 would save ~0.5 GB (~2.2 ms, ~2%). That is below the ±11% single-prompt noise, so it cannot be validated with the current ruler. Online FP8 support on sm_121 in this image is unverified.

**Verifier corrected impact:** ≤ +2-3% tok/s, not resolvable with the current bench noise.

## SD-11: Draft context K/V is stored in fp8_e4m3 (inherited) although the drafter was trained BF16; test only after SD-5

- kind=quality component=load_dflash_model cache_config / SpeculativeConfig.kv_cache_dtype impact=2 confidence=2 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** load_dflash_model uses speculative_config.kv_cache_dtype, falling back to the target's fp8_e4m3, for the draft's sliding-window cache. The draft's context K/V (projected from 5 target features) are quantized per-tensor with no calibrated scales, which may cost some prose acceptance. Switching to '"kv_cache_dtype":"auto"' (BF16) doubles the draft page (5.9 → 11.8 MB per id), raising the SD-5 pool tax from 17.6% to ~30%.

**Mechanism:** fp8 e4m3 has 3 mantissa bits (~6% relative error) on V and on RoPE'd K without per-head scales. This perturbs the draft's attention over context, which directly drives the prediction quality at later positions.

**Evidence:**
- v11src/vllm/v1/worker/gpu/spec_decode/dflash/utils.py (cache_config replaced only if speculative_config.kv_cache_dtype is not None)
- v11src/vllm/config/speculative.py:127-129 (draft kv_cache_dtype default inherits target)
- engine.log.tail:71 (FlashInfer kv_cache_dtype=torch.float8_e4m3fn), :80 (FlashInfer used for draft non-causal attention)
- patch_v11 per_block formula (draft pages charged on every id)

**Proposed action:** After SD-5 decouples the draft KV pool, run a one-knob A/B of kv_cache_dtype auto vs fp8 for the draft and compare per-position prose acceptance. Before decoupling, only test it at a reduced MAX_MODEL_LEN.

**Est. impact:** Unknown and probably small: 0 to +5% prose acceptance length, i.e. 0 to +5% tok/s. It costs ~0.8 GB/rank of pool if done before SD-5.

**Validation:** bench_decode.py prose c=1 with per-position acceptance, fp8 vs auto draft KV. Everything else is held fixed.

**Risks:** Pool shrinkage and UMA pressure. The effect may sit inside the ±11% run-to-run noise of the single-prompt bench (see SD-12).

**Verifier reasoning:** dflash/utils.py:33-35 replaces cache_config only when speculative_config.kv_cache_dtype is set, and speculative.py:127-129 says the draft inherits the target dtype. So the draft KV is fp8_e4m3 (engine.log.tail:71). But the reference evaluation itself ran with an 'FP8 target KV cache' (draft README:52), so the premise 'trained/evaluated BF16' is weaker than stated. BF16 draft KV doubles the draft page (5.9 → 11.8 MB/id) under the v11 layout. The expected effect is below measurement noise.

**Verifier corrected claim:** Draft KV inherits fp8_e4m3. The card's own eval also ran with an FP8 target KV cache, so any acceptance loss from fp8 draft KV is speculative and probably below bench noise.

**Verifier corrected impact:** 0 to low single-digit %, unmeasurable with the current ruler. Costs ~0.8 GB/rank of pool before SD-5.

## SD-12: Choose k (and judge spec changes) from per-position acceptance plus the step model, not single-prompt tok/s

- kind=methodology component=bench_decode.py / evidence process impact=3 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The published ruler is one 80-word prose prompt: ~98 tokens, 3 runs, ~40 verify steps per run. Within one session, runs spread 18.75-23.40 tok/s (±11%) and per-window acceptance spreads 2.31-2.78. That exceeds the 8% noise gate that hillclimb-night uses, so k decisions made on it are fragile. The per-position acceptance vector is exported by vLLM and, with the validated step model, predicts tok/s for every k offline. The model was within 1-5% on four independent cells.

**Mechanism:** tok/s = (1 + Σ_{i≤k} p_i) / (a + b·D(n_reqs·(k+1))). The p_i come from one k=7 run and truncate cleanly to smaller k, while a and b come from two measured k values. Every future k/occupancy decision then needs one boot rather than a sweep.

**Evidence:**
- evidence/rebench-20260902T204243Z/bench.txt (prose c=1 runs 23.40/18.75/21.18; median_completion_tokens 98)
- engine.log.tail:256,260,270 (acceptance 2.78/2.31/2.42 across windows)
- PR #9 body: 'Noise: 8%'
- v11src/vllm/v1/spec_decode/metrics.py:30-49,193-195 (num_accepted_tokens_per_pos; vllm:spec_decode_num_accepted_tokens_per_pos)
- Model checks: structured k=5 predicted 55.4 vs 55.7 measured; c=2 prose k=7 predicted 150 ms/step vs 16.64/2.50 -> 150 ms; c=2 k=5 predicted 18.5 vs 17.5-19.6

**Proposed action:** Extend bench_decode.py (or the kit bench in PR #8) to scrape vllm:spec_decode_num_accepted_tokens_per_pos and num_drafts before and after each cell and print the p_i vector. Add a small fixed prose prompt set (8 prompts, 200 tokens each) as a diagnostic, not published. Store a kit/spec_model.py that fits a and b from receipts and prints predicted tok/s for k=1..7 at c=1 and 2.

**Est. impact:** Removes 3-5 serve boots (18 min each) per k decision and reduces false keeps and reverts. Enables SD-1/SD-3 tau tuning offline.

**Validation:** CPU-only: fit on the existing receipts (rebench-20260902, PR #9 H1/parity/dflash5) and check leave-one-out error below 5%. Then confirm with one GPU run at k=4.

**Risks:** The model ignores UMA swap episodes, which change the effective bandwidth (a KV pin of 5.0 GiB slowed decode ~20%). Receipts taken under swap must be flagged.

**Verifier reasoning:** Run spread is confirmed: prose c=1 23.40/18.75/21.18 (bench.txt), and PR#9 'Noise: 8%'. The metrics exist (metrics.py:30-47, 194). Corrections: (a) The 'four independent cells' do not test the D functional form. The model has two parameters fit to k=5 and k=7. Structured k=5 only tests that step time does not depend on content. c=2 prose is the one real out-of-sample test, and a linear-in-tokens model passes it too (155 vs 150 ms). (b) c=2 structured cells are non-monotone in k: 144 ms at k=5 (41.56/5.98) vs 131-142 ms at k=7. Old-era k=5/6/7 gave 130/141/135 ms. So c=2 is not well predicted. (c) The 'prose c=1' per-position vector includes engine.log.tail:270, a window dominated by c=2 traffic (Running: 2 reqs from 21:07:25; ~125 drafts/10 s). Using only lines 256+260 gives ~[.683,.404,.173,.096,.048,.010,0], so the conclusions barely change. The recommendation (scrape the p_i vector, a multi-prompt diagnostic set, an offline fit) is sound and needs no GPU.

## SD-13: Spec-path runtime JIT and eager per-step work outside the CUDA graph

- kind=perf component=dflash/speculator.py prepare_dflash_inputs / rejection kernels impact=1 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** _prepare_dflash_inputs_kernel takes BLOCK_SIZE = next_pow2(max_target_query_len + 1 + k) as a constexpr, so each new prefill or query shape can trigger a Triton JIT during serving. The log shows it and _compute_local_logits_stats_kernel compiling on live requests. Each step also runs the context K/V precompute (rms_norm, GEMM, permute+contiguous, rms_norm, rope, 5 cache writes) and rebuilds the FlashInfer non-causal metadata eagerly, outside the FULL draft graph.

**Mechanism:** Triton on the Grace CPU compiles slowly (seconds), which shows up as first-hit TTFT spikes. The eager per-step ops add CPU launch latency on every decode step.

**Evidence:**
- engine.log.tail:189 (JIT during inference: _compute_local_logits_stats_kernel), :225 (JIT during inference: _prepare_dflash_inputs_kernel)
- v11src/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:672-676 (BLOCK_SIZE depends on max_target_query_len), :403-420 (eager precompute), :436-445 (metadata rebuilt even for FULL replay)
- evidence/rebench-20260902T204243Z/bench.txt (prose c=2 run 1 TTFT 6.286 s vs 0.37 s later)

**Proposed action:** Fix BLOCK_SIZE at 256 (the kernel already loops over blocks) or warm up every power of 2 up to 256 at boot. Extend spec_decode_rejection_warmup to the stats kernel shapes used at c=1,2. At decode (num_target_tokens == reqs×(k+1)), capture the precompute into the draft graph by padding its context to the fixed decode shape.

**Est. impact:** Removes multi-second first-request TTFT spikes (one observed at 6.3 s; attribution not proven). The graph change could save ~0.3-1 ms per step (≈0.5-1%).

**Validation:** After a fresh boot, the first c=1 and c=2 requests should log no jit_monitor warnings. Compare TTFT of run 1 against runs 2-3.

**Risks:** The 6.3 s spike may have another cause (e.g. TileLang mHC compiles for a new shape). Capturing the precompute requires stable context slot buffers.

**Verifier reasoning:** BLOCK_SIZE = min(256, next_pow2(max_target_query_len + num_query_per_req)) (dflash/speculator.py:672-675), so at most 9 compile variants. The JIT warnings (engine.log.tail:189-191 at 21:03:54, 225 at 21:06:13, 230 at 21:06:21) all fired during smoke/probes, before bench_decode started at 21:07:02 (progress.md). The first-run c=2 TTFT spike repeats in three sessions (6.286 s rebench, 6.472 s H1, 6.422 s dflash5), but the rank0 log shows no jit_monitor warning in that window (engine.log.tail:253-268). The spike is therefore not explained by these Triton JITs. The eager-precompute savings claim (~0.3-1 ms/step) is unquantified.

**Verifier corrected claim:** Spec-path Triton JITs exist, but probes absorb them before the bench. The reproducible ~6.3-6.5 s first c=2 TTFT has another, unidentified cause.

**Verifier corrected impact:** Negligible for published cells. Maybe ≤1% per step from graph-capturing the precompute.

## Open questions
- Is routing locality inside a verify block really ~60% of independent (fitted slope 0.75 vs 1.25 ms per expert-slot)? A one-off CPU-side dump of topk_ids during a GPU slot would pin down D(n) and sharpen SD-1/SD-2 estimates.
- Does the fork's Glm5NextMoE / Marlin path expose topk_ids/topk_weights after routing so dead-token masking (SD-1) can be done in-graph, or is routing fused into the kernel?
- Do FLASHINFER_MLA_SPARSE_SM90 (NoPE, kpool tail) and the KDA spec-state path support varlen verify queries? This is a prerequisite for SD-3.
- What is DFlash2 prose acceptance with enable_thinking=true and on the nvidia pack vs LibertAI caca4e6? It separates distribution mismatch from quantization drift (SD-9); PR #11 is unmeasured.
- Does SPEC=mtp on the nvidia pack actually fail at load, or does some path special-case BF16 layer-45 weights? This is a single boot check (SD-8).
- Is the Gumbel-coupled probabilistic DFlash2/DSpark path intentionally designed with a proof we have missed? It should be confirmed with the sampled-distribution probe before filing upstream (SD-6).
- Hybrid MTP+DFlash2 drafting is not supported by this vLLM (one method per SpeculativeConfig) and would add a second draft cost per step. I did not pursue it; is there any appetite for tree or multi-draft verification? On bandwidth-bound MoE it grows D(n) and is expected to lose.
- After other dimensions cut the fixed cost (BF16 attention and shared experts ≈ 8 GB/rank per step), the optimal k shifts lower. Should k be re-derived with the SD-12 model after each such change?

## Verifier: missed issues
- SD-3 is blocked in hardware and backends, and the reviewer did not see it. The DSA indexer reports supports_varlen=False on sm_121 (engine.log.tail:70; indexer.py:649-654 requires capability family 100 plus DeepGEMM). KDA/SSM backends opt out of query-length mismatch (backend.py:295 'return not cls.is_ssm()'). FlashInfer draft attention returns False (flashinfer.py:432-435). Adaptive verification is not an option here, which makes fixed-shape SD-1 or a smaller k (SD-2) the only routes to acceptance-proportional verify cost.
- Structured c=2 regressed 25-31% at k=5 (rebench-dflash5 summary.json: 41.56 per stream vs 55.50 parity and 60.41 rebench at k=7). Its step time (41.56/5.98 → 144 ms) is longer than at k=7 (131-142 ms), which contradicts the step model. SD-2 says 'without regressing c=2' but only looked at prose.
- The PR#9 H1 'render/lint failure' is a hard-coded check. H1 lint.log shows 'FAIL VALIDATE_ONLY=1 {} did not print "cudagraph_capture_sizes":[1,2,4,8,16]'. Any NUM_SPECULATIVE_TOKENS change must update recipe.yaml and the lint expectation (the capture ladder (k+1)×{1,2} changes, run.sh:85-97), or the kit auto-reverts it again.
- Block verification gives zero gain with one-hot (greedy) drafts. Token-wise verification already achieves the upper bound P(accept≥i)=Π p(d_j). SD-7 should be dropped, not tested.
- Every spec-decode receipt (acceptance vectors, step fits, k=5 results) is from LibertAI caca4e6. The PR#11 default nvidia pack is unmeasured. Headers show identical routed-expert bytes (171.229 GB in both), but nvidia non-expert bytes are 17.198 GB vs 17.843 GB (≈0.32 GB/rank less, ≈1.4 ms/step lower intercept), and its quant differs, so acceptance, and therefore the k optimum, must be re-measured on the nvidia pin before the default k is changed.
- Old-era receipts give an independent check of step time vs k that the reviewer did not use: evidence/baseline-bench.txt (k=5), iter-h1 (k=6) and iter-h3 (k=7) give prose 101.6/109.8/116.2 ms and structured 106.5/113.7/117.8 ms per step. In that era k=7 had higher prose acceptance (3.17 vs 2.875 at k=5), so tail acceptance depends on the content/template. The truncate-p_i method should be checked on more than one prompt.
- A first-run c=2 TTFT of ~6.3-6.5 s reproduces after every boot (rebench 6.286, H1 6.472, dflash5 6.422) with no jit_monitor warning in the window (engine.log.tail:253-268). It is an unexplained warmup gap for real 2-user traffic, not caused by the spec-path JITs SD-13 cites.
- The reviewer's prose per-position vector labels engine.log.tail:270 as c=1, but that window is c=2-dominated (Running: 2 reqs from 21:07:25). The c=1-only vector from lines 256+260 is ~[.683,.404,.173,.096,.048,.010,0]. The conclusions hold, but the evidence citation is wrong.
