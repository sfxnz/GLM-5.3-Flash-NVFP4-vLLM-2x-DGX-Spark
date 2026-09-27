# Dimension: byte-roofline

## Reviewer summary

I computed exact per-rank bytes for one verify step. Inputs were the 33 safetensors headers of the nvidia pack (204,419,110,596 B, 190.38 GiB), LibertAI caca4e6 (194,644,952,184 B, 181.28 GiB) and the DFlash2 draft (2,342,160,896 B), plus the TP=2 layouts read from the v11 fork source. As a check, the model predicts 90.11 GiB of resident weights per rank; the engine log reports 'Model loading took 90.67 GiB'. At the default DFlash2-7, c=1, one step reads 22.2-28.2 GB per rank, depending on how concentrated expert routing is. The measured step is 114.5-119 ms. That implies 193-245 GB/s, while the best measured GB10 DRAM read rate is about 231-234 GB/s. So decode is already roughly 85-95% bandwidth-bound, and larger gains must come from reading fewer bytes per accepted token, not from lower latency. Two-point calibration (205 GB/s achieved, rho=0.58 routing concentration, O=10 ms fixed overhead) reproduces k=5 105.0 ms and k=7 116.9 ms. In that model, per-step bytes break down as follows. Routed MoE: 11.0 GB (54 ms). BF16 non-MoE weights: 10.2 GB (50 ms), which does not depend on k. That 10.2 GB is KDA projections 4.72 GB (23 ms), MLA+indexer 1.55 GB, shared experts 1.06 GB, lm_head read twice 1.27 GB, draft 1.26 GB, plus about 0.35 GB of dense-MLP, router and mHC. KDA spec-state traffic is 0.66 GB (3.2 ms), and fixed overhead is about 10-18 ms. KV reads barely depend on context: top-k 2048 plus the kpool-4 indexer cap them at 0.3 ms at 128k. The main levers: (1) Move BF16 non-MoE weights to FP8 W8A16. Model: prose c=1 +26% (20.1→25.3), structured +26% (66.8→84.1), about 4.3 GiB freed per rank. With NVFP4 W4A16 instead, about +42%. (2) Lower k for prose. k=5 was measured at +14% (19.23→21.97) and then reverted only because render/lint failed. The model puts the optimum at k≈3 (22.8, flat from k=2 to 5). Choosing k by workload keeps structured at k=7. (3) Cut fixed and per-sequence overhead. (4) Store only the KDA states that get accepted. All levers together model at about 34 tok/s prose c=1 (+70%). That estimate depends on an assumed per-position prose acceptance curve (not in evidence) and needs GPU validation one knob at a time. One method caveat: bench_decode.py sends the same prompt to both c=2 streams, so the published c=2 numbers understate the real cost of two independent streams (about 159-170 ms per step predicted, against 150 measured).

## RF-1: Verify step is ~85-95% DRAM-bandwidth-bound; exact per-rank byte ledger

- kind=perf component=whole decode step (target verify + DFlash2 draft), TP=2 per rank impact=5 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** At DFlash2-7, c=1, each rank reads 22.2-28.2 GB per step (rho 0.6-1.0), with ctx 1k. The measured step is 114.5-119.1 ms, which needs 193-245 GB/s with zero overhead. That is 83-106% of the GB10 read ceiling of about 232 GB/s. Almost none of the step is compute or latency, so real gains must come from reading fewer bytes per accepted token.

**Mechanism:** Each step streams every BF16 non-MoE weight once, streams every routed expert touched by the T=(k+1)*c verify tokens once (Marlin reads the whole expert tile per expert), and reads/writes per-sequence KDA state. On UMA LPDDR5X that is all DRAM traffic. GEMMs at M<=16 are far below the FP4/BF16 compute roofline.

**Evidence:**
- Header sums (scratch agents/roof/nv.tsv, agg.py). Routed experts: 171,228,556,800 B = 159.47 GiB. One expert (gate+up+down U8 weights, F8 block scales, F32 globals) = 14,155,800 B = 13.50 MiB; per rank 6.75 MiB, because the intermediate dim is split 2048→1024 (FusedMoE TP, no EP: model.py:227-250).
- KDA layer = 262.9 MiB full, 132.45 MiB per rank. q/k/v/o are each BF16 8192x4096 = 64 MiB and TP-halved. f_a and g_a are replicated (kda.py:204-219), while f_b, g_b and o_proj are column/row-parallel (kda.py:221-285).
- MLA/DSA layer = 238.3 MiB full, 134.75 MiB per rank. fused_qkv_a (16 MiB) is replicated, while q_b, kv_b and o_proj are TP-split (attention.py:457-503). The indexer's wq_b, wk_weights_proj and compress_gate are replicated, plus a fp32 copy _wp_fp32 (attention.py:245-266, 329-336).
- Shared expert: 24 MiB per rank (model.py:118-133). Dense layers 0-2: 40.5 MiB per rank. Router: 2.25 MiB, replicated GateLinear (model.py:183-188). mHC: 3 MiB per layer, replicated and held in fp32 (model.py:391-402). lm_head: 605 MiB per rank (ParallelLMHead, model.py:945).
- Resident weights per rank from this ledger: 90.11 GiB, against 'Model loading took 90.67 GiB' in evidence/rebench-20260902T204243Z/engine.log.tail (LibertAI, whose dense MLP is BF16: +0.30 GiB). The ledger agrees within 0.3%.
- Measured ms/step = 1000*acceptance_len/tok_s: 114.5 (rebench prose), 116.0 (structured), 119.1 (PR #9 parity), 105.0 (k=5, PR #9 H1).
- GB10 saturating-read kernel reaches 231-234 GB/s, and decode reaches 85-90% of that: https://github.com/antirez/ds4/issues/773 ; 273 GB/s nominal: https://petronellatech.com/blog/dgx-spark-cluster-bandwidth-what-400g-really-means/
- Scratch model: /tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-GLM-5-3-Flash-NVFP4-vLLM-2x-DGX-Spark/29f0f0df-0b0e-4224-a895-0ea5d735b7de/scratchpad/agents/roof/model.py and levers.py

**Proposed action:** Treat bytes per accepted token as the objective. Rank levers by GB saved per step divided by BW (see RF-2..RF-7), and stop chasing latency micro-tuning in isolation.

**Est. impact:** Calibrated split at k=7, c=1 (BW 205 GB/s, rho 0.58, O 10 ms): MoE 11.02 GB = 53.7 ms; KDA proj 4.72 GB = 23.0 ms; MLA+indexer 1.55 GB = 7.6 ms; shared 1.06 GB = 5.2 ms; lm_head x2 1.27 GB = 6.2 ms; draft excluding lm 1.26 GB = 6.1 ms; KDA/conv state 0.66 GB = 3.2 ms; dense+router+mHC 0.37 GB = 1.8 ms; KV 0.01 GB; fixed 10 ms. Total 116.9 ms, against 114.5-119.1 measured.

**Validation:** On an idle Spark pair, run nsys on 20 steady-state prose steps. Sum DRAM bytes per kernel (ncu dram__bytes_read.sum on the Marlin MoE, the BF16 GEMMs and fused_recurrent_kda) and compare with the ledger per component. Also run a standalone read-bandwidth microbenchmark to fix BW_eff.

**Risks:** rho (routing concentration) and BW_eff cannot be separated from two measured points. The split between MoE and overhead could shift by about ±10 ms.

**Verifier reasoning:** I re-ran the reviewer's scratch model (agents/roof/model.py, levers.py) against nv.tsv and df.tsv. The ledger reproduces: one expert is 14,155,800 B (6.75 MiB per rank); KDA is 132.45 MiB per rank; MLA is 134.75; shared 24; lm_head 605; draft 1807 MiB. The k=7, c=1 total is 22.22 GB at rho=0.6 and 28.18 GB at rho=1.0. TP layouts match source: kda.py:204-219 merges q/k/v/b/f_a/g_a with f_a and g_a replicated; kda.py:168-174 strips quant_config; model.py:331 builds MLA with quant_config=None. Resident check: the engine log says 'Model loading took 90.67 GiB' (rebench engine.log.tail:52, target plus draft). That is LibertAI, so the ledger comparison is 90.11+0.30=90.41 GiB, which is within 0.3% but still includes guesses (vision TP-sharded, W_UK/W_UV copy). The ds4 #773 citation is real (231-234 GB/s read, 85-90% in decode), but it describes ds4's own CUDA engine, not vLLM/Marlin. The headline '85-95% bandwidth-bound, almost none of the step is compute or latency' does not hold up. (1) The calibration fits 2 free parameters (rho, O) at a fixed BW to exactly 2 points (rss 0.02), so matching 105.0 and 116.9 ms is by construction, not validation. (2) The inputs are noisy. Nine k=7 prose c=1 runs over three boots range 16.86-23.40 tok/s (rebench bench.txt; PR #9 parity bench.txt and bench-run1.txt). The same config gave 114.5 ms/step in one session and 119.1 in another. (3) There is a direct non-byte counter-signal. In PR #9's rebench-dflash5 (k=5), the synchronized identical structured c=2 run (run 3, 48.8 tok/s, acc 6.0) takes 123 ms, against 107.7 ms at c=1. Adding 6 identical tokens with no new expert bytes costs about 15 ms, but the model allows about 2.5 ms of state bytes for it. Levers.py shows the same thing at k=7: 131.1 ms measured vs 120.2 modeled. So per-token non-byte cost is about 1.5-2 ms/token. That competes with 'distinct experts' as the explanation for the k-slope, which pushes the fit toward lower rho and higher O. (4) Prose and structured have nearly identical step times at the same k (114.5 vs 116.0 at k=7; 108.2 vs 107.7 at k=5) despite very different token content. That is weak evidence against a strong dependence on routing diversity. Bytes are probably the largest single component, but the 85-95% share is not established.

**Verifier corrected claim:** The per-rank byte ledger is correct: 22.2-28.2 GB per verify step at k=7, c=1, of which about 10.2 GB is k-independent BF16 non-MoE weights. Whether decode is 85-95% bandwidth-bound is not established. The 2-point calibration is exactly determined, the per-step inputs carry about ±10% noise, and the c=2 identical-token controls show about 11-15 ms of non-byte cost per extra 6-8 tokens. Memory traffic is likely the largest component, somewhere between roughly 60% and 90% of the step. The split between MoE bytes and per-token overhead is unknown until nsys/ncu is run.

**Verifier corrected impact:** Framing and prioritization only. Byte levers remain the best-grounded, but expect about 60-90% of the modeled gains, not 100%.

## RF-2: BF16 non-MoE weights are ~43% of step time; FP8 (or NVFP4) weight-only for KDA/MLA/shared/lm_head/draft is the largest lever

- kind=perf component=KDA q/k/v/o projections, MLA projections, shared experts, lm_head, DFlash2 draft (all BF16) impact=5 confidence=3 effort=L needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Per rank, each step reads 10.23 GB of BF16 non-MoE weights no matter the k or acceptance: 50 ms of about 117 ms. KDA projections alone are 4.72 GB (23 ms). Converting them to FP8 W8A16 halves 9.86 GB of it. NVIDIA excluded all self_attn*, shared_experts* and lm_head from NVFP4 (config.json ignore list), and this recipe inherited that choice.

**Mechanism:** At M = 8-16 tokens, a W8A16 GEMM (fp8 Marlin, sm_80 code that runs on sm_121) reads half the bytes of BF16 at the same memory-bound speed. W4A16 (NVFP4 Marlin) reads 0.5625 B per param.

**Evidence:**
- config.json quantization_config.ignore: every layers.N.self_attn*, mlp.shared_experts*, mlp.gate, lm_head and model.visual* entry.
- Per-rank non-MoE bytes (model.py output): KDA 4503 MiB, MLA 1482, shared 1008, lm_head 605 (target) + 605 (draft re-read), draft excluding lm 1202 MiB. Sum 9405 MiB = 9.86 GB.
- The KDA and MLA constructors strip quant_config, so these layers run as UnquantizedLinear BF16: kda.py:168-174; model.py:331 'quant_config=None  # MLA projections are BF16 in checkpoint'.
- The step-time model is levers.py 'Lever table'.

**Proposed action:** Add an image-layer patch (v12-style) that online-quantizes selected BF16 linears at process_weights_after_loading to FP8 per-output-channel (or per-128-block) W8A16 via the fp8 Marlin path. Scope: KDA in_proj_qkvbfg_a/o_proj/f_b/g_b, MLA q_b/kv_b/o_proj/fused_qkv_a, indexer wq_b, shared_experts, lm_head, and draft qkv/o/mlp/fc. Leave the router gate, mHC, norms and the indexer fp32 weights_proj alone. Put it behind an env flag. NVFP4 W4A16 for the same set is the stretch option.

**Est. impact:** FP8 saves 4.93 GB per step, which is 24.1 ms at 205 GB/s. Step 116.9 → 92.8 ms. Prose c=1 2.34/0.0928 = 25.2 tok/s (+26%). Structured 7.81/0.0928 = 84.1 tok/s (+26%). NVFP4 W4A16 saves 7.09 GB = 34.6 ms, giving prose about 28.5 (+42%). Side benefit: about 4.3 GiB per rank of UMA freed (KDA 4.40 + MLA 1.62 + shared 0.98 + lm 0.59 + draft ~1.2 GiB, halved). That addresses the 116/121 GiB used + 6 GiB swap seen in evidence/rebench-20260902T204243Z/free-after-bench.txt.

**Validation:** Change one knob at a time. (a) lm_head+draft only. (b) + shared experts. (c) + MLA. (d) + KDA. Run bench_decode.py prose c=1/2 for each. Quality gate: greedy count 200, thinking-off smoke, the needle-8192/20480 probes already in evidence, and a small MMLU/GSM8K or HLE slice against BF16. KDA goes last because recurrent error can compound over long context.

**Risks:** Quality: NVIDIA deliberately kept attention BF16, and KDA's recurrent state may amplify weight error. FP8 is usually near-lossless, NVFP4 on attention is riskier. Kernel risk: fp8 Marlin at N=8192/K=4096 on sm_121 is untested here. CUTLASS block-FP8 needs sm_120a/121a cubins, and _C was built without 12.1a (image TORCH_CUDA_ARCH_LIST).

**Verifier reasoning:** The facts check out. config.json quantization_config.ignore (132 entries) covers layers.N.self_attn*, layers.N.mlp.shared_experts*, mlp.gate, embed_tokens, lm_head and model.visual*. kda.py:168-174 and model.py:331 build KDA/MLA as unquantized BF16. The arithmetic reproduces: the quantizable set is KDA 4503 + MLA 1482 + shared 1008 + lm 605x2 + draft 1202 = 9405 MiB = 9.86 GB, and halving it saves 4.93 GB, which is 24.1 ms at 205 GB/s. The magnitude, though, depends on RF-1's unvalidated calibration. Under the reviewer's own calB (232 GB/s, O=18.5 ms) the same saving is 21.3 ms, giving +22%. If these small-M BF16 GEMMs are not already near roofline, or if per-token overhead is larger than modeled, the gain shrinks. Kernel path: FP8 Marlin (W8A16) would need to be forced (VLLM_TEST_FORCE_FP8_MARLIN-style), because sm_121 has native FP8 and auto-select prefers the CUTLASS/Triton scaled-mm paths. Online quantization of a ModelOpt checkpoint's excluded layers needs a fork patch, so effort L is right. Quality: NVIDIA deliberately left all attention and shared experts in BF16. lm_head is shared by target and draft (v1/worker/gpu/spec_decode/dflash/utils.py load_dflash_model), so FP8 there changes target logits and needs the count-200 and quality gates. The UMA-freed estimate (about 4.4 GiB per rank) is arithmetically right and relevant: free-after-bench.txt shows 116/121 GiB used and 6 GiB of swap.

**Verifier corrected claim:** About 9.86 GB per rank per step (about 48% of the modeled bytes) is BF16 non-MoE weight that does not scale with k. FP8 W8A16 on that set halves it. The saving is +20-26% prose c=1 only if the step is as bandwidth-bound as modeled. Realistically it is +12-26%, and it needs an image-layer patch plus a forced FP8-Marlin path. Quality risk is highest for KDA and lm_head (lm_head is shared with the target).

**Verifier corrected impact:** +12-26% decode (model-dependent) and about 4.4 GiB of UMA freed per rank. The NVFP4 W4A16 stretch figure (+42%) carries substantially more quality risk and should not be planned on.

## RF-3: k=7 is past the prose optimum; k=5 was a measured +14% win reverted for a tooling failure; model optimum k≈3, workload-aware k keeps structured at 7

- kind=perf component=speculative config (NUM_SPECULATIVE_TOKENS), DFlash2 impact=5 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 4)

**Claim:** Each extra verify token costs about 6-7 ms per step, measured, which matches the marginal distinct-expert bytes. Prose positions 6-7 add under 0.05 accepted tokens. PR #9 measured k=5 at prose c=1 21.97 vs 19.23 (+14%, beyond the 8% noise) and c=2 17.52 vs 15.13. That run was reverted because 'render or lint failed after keep', not for performance. The same reasoning overturns h3/H3-2026-08-29 as the default: that comparison ran on the old CoT-leaking template, where prose acceptance at k=7 was 3.17 rather than today's 2.29-2.43.

**Mechanism:** Every added draft slot adds a verify token. That token adds new routed experts across 42 layers (about 1.2-1.9 GB per rank), 68 MiB of KDA state writes, and graph-size growth, while the prose acceptance probability at position ≥5 is under 7%.

**Evidence:**
- PR #9 verdict.json H1-20260903: {'reason': 'render or lint failed after keep: render: wrote README.md result=fail', 'before_c1': 19.23, 'after_c1': 21.97, 'before_c2': 15.13, 'after_c2': 17.52}. Acceptance_len: 2.31 at k=5 vs 2.29 at k=7 (parity baseline).
- Step slope from same-day runs (evidence/baseline-bench.txt, iter-h1, iter-h3): prose 101.6 → 109.8 → 116.2 ms and structured 106.5 → 113.7 → 117.8 ms for k = 5, 6, 7. That is about 6-7 ms per verify token.
- Uniform-routing marginal experts: dE/dT at T=7 is 6.6 experts per layer. x 42 layers x 6.75 MiB = 1.95 GB = 7.2 ms at 273 GB/s. With the calibrated rho=0.58 it is about 5.5-6 ms.
- levers.py calA: modeled prose tok/s by k=0..7 is 15.9, 20.2, 22.5, 22.8, 22.5, 21.8, 20.9, 20.1. Structured: 15.9, 25.2, 34.7, 42.8, 49.9, 56.1, 61.7, 66.8.
- run.sh:68-74 comment already notes 'positions 5-6 accept <15% on prose'.

**Proposed action:** (1) Re-run PR #9 H1 (NUM_SPECULATIVE_TOKENS=5, seqs=2) and keep it if it repeats, because the published score is prose. (2) Then try k=4 and k=3 one at a time. The capture ladder derives automatically (run.sh:88-97). (3) Longer term: set k per request or per step from the DFlash2 candidate confidences. v11 already ships adaptive_verification.py (a survival-probability draft budget) for DSpark only (vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:34-60); port it to dflash2 so structured keeps k=7.

**Est. impact:** Static k=5: +8.7% modeled (+14% measured). k=3: +13.8% modeled (22.8 vs 20.1). The curve is flat from k=2 to 5, so k=3-5 are within noise of each other. Structured loses at static k=5 (66.8 → 56.1, −16%). Workload-aware k keeps prose at about 22.8 and structured at about 66.8. Per-step adaptive k could plausibly add another 5-10% on prose (unquantified).

**Validation:** bench_decode.py prose c=1/2 with NUM_SPECULATIVE_TOKENS in {5,4,3}, one per boot. Scrape vllm:spec_decode_num_accepted_tokens_per_pos to get the real per-position acceptance curve. Record keeps and reverts in trail.tsv/decision.tsv.

**Risks:** The prose per-position acceptance curve used here (p = 0.60, 0.33, 0.18, 0.11, 0.07, 0.035, 0.02) is fit to acceptance_len values only. Structured and code workloads lose throughput at low static k. Four-way occupancy at k=5 needs MAX_NUM_SEQS=4 (AGENTS.md).

**Verifier reasoning:** The core k=5 win holds and is stronger than the reviewer showed, but the framing is incomplete. PR #9 verdict.json H1 does say 'render or lint failed after keep', and lint.log shows the failure was a stale expect_case pinning '[1,2,4,8,16]', a tooling problem. The reviewer missed that PR #9 then hand-applied k=5 as the default (commit a76474e; decision.tsv row 'H1-20260903-apply ... kept'; e898a80:recipe.yaml line 49 NUM_SPECULATIVE_TOKENS: 5) and ran a second boot, rebench-dflash5-20260903T045815Z. That run measured prose c=1 21.76 (acc 2.354), structured c=1 55.73 (acc 6.0) and prose c=2 19.62. It then FAILED the gate on needle-8192 (hit=0, model refused the planted-code prompt; commit f2538e5), so proposal (1), 'Re-run PR #9 H1', has already been done. Pooled per-run numbers: k=7 has 9 prose c=1 runs, mean 19.87 (range 16.86-23.40); k=5 has 6 runs, mean 22.08. That is +11% (t≈2.5), not +14%, since +14% compares against the lowest k=7 session. Structured at k=5 measured −17.6% (55.7 vs 67.6), which matches the model's −16%. Acceptance data support 'positions 6-7 add ~0': k=5 acc 2.31/2.35 vs k=7 acc 2.43/2.29/2.20. The k≈3 optimum rests entirely on an assumed per-position acceptance curve (p1=0.60, ...) fit only to aggregate acceptance_len, which cannot separate p1 from tail mass. It is not evidence. Per-step slope is 5-6 ms per token on the recent data (105.0/108.2 at k=5 vs 114.5/119.1 at k=7). The 6-7 ms figure comes from the 08-29 runs, which mixed seqs=4 and seqs=2. adaptive_verification.py exists and is DSpark-only, requiring a confidence_head (dspark/speculator.py:111), so a DFlash2 port is not a drop-in.

**Verifier corrected claim:** k=5 beats k=7 on prose c=1 by about +11% (pooled 22.08 vs 19.87 tok/s across 2 vs 3 boots) and loses about 18% on structured. It has already been hand-applied on the PR #9 branch, and its only rebench failed the needle-8192 quality gate (model refusal, probably unrelated to k but not yet shown). k=3-4 is unmeasured, and the k≈3 optimum is an assumption. Per-request or adaptive k needs new work for DFlash2 because the existing adaptive verifier requires a DSpark confidence head.

**Verifier corrected impact:** +11% prose c=1 measured at k=5 (−18% structured). k=3/4 is unknown, somewhere between 0 and +14% modeled.

## RF-4: bench c=2 runs identical prompts, so both streams share experts; published c=2 understates real concurrent cost

- kind=methodology component=bench_decode.py impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** wave() sends the same prompt with temperature 0 to every stream. Two greedy streams produce the same (or mostly the same) tokens, so they touch the same experts. Structured c=2 at 131.1 ms per step is effectively a shared-expert control. Prose c=2 at 150 ms per step is below the 156 ms that two independent streams would need at the 273 GB/s peak (42.6 GB per rank, uniform routing), which is physically impossible unless experts are shared.

**Mechanism:** Routed-expert bytes scale with the distinct experts across all verify tokens in the batch. Identical token streams add almost no new experts, while independent streams add E(16)-E(8) ≈ 26-46 experts per layer (rho 0.58-1.0), which is 7.4-13 GB per rank.

**Evidence:**
- bench_decode.py:97 `futs = [pool.submit(stream_one, url, model, prompt, max_tokens) for _ in range(concurrency)]`; bench_decode.py:36 `temperature: 0`
- evidence/rebench-20260902T204243Z/bench.txt: prose c=2 16.64 tok/s, acc 2.496 → 150.0 ms; structured c=2 60.41 tok/s, acc 7.921 → 131.1 ms; c=1 is 114.5 / 116.0 ms
- model.py: at k=7, c=2 with independent streams, rho=1 gives 42.63 GB per rank = 156.2 ms at 273 GB/s and 183.8 ms at 232 GB/s
- levers.py calA: independent c=2 at k=7 is 159.2 ms before about 11 ms of per-sequence overhead (structured c=2 control: 131.1 measured vs 120.2 modeled), i.e. about 170 ms → 2.34/0.170 = 13.8 tok/s per stream

**Proposed action:** Add a c=2 prose phase with two different prompts (and different seeds), and publish that as the concurrency row. Keep the identical-prompt run only as a shared-expert control. Do not change the c=1 published ruler.

**Est. impact:** Real independent c=2 predicted at about 14-15 tok/s per stream (159-170 ms per step), against the published 16.6. That is 10-17% optimistic today. Every later k and seqs decision made at c=2 is biased in favor of more verify tokens.

**Validation:** Run bench with two different 80-word prose prompts at c=2 and compare ms/step with the identical-prompt run under the same boot.

**Risks:** Prose streams can diverge after a near-tie, so identical-prompt prose c=2 is partly independent. The size of the bias is bounded, not exact.

**Verifier reasoning:** bench_decode.py:97 submits the identical prompt to every stream, and bench_decode.py:36 sets temperature 0. The 'physically impossible' argument is weak because it assumes rho=1, which the reviewer's own fit rejects (independent c=2 at rho 0.58 is 159 ms vs 150 measured, which is not impossible). Stronger direct evidence exists that the reviewer missed. In PR #9 rebench-dflash5 (k=5), structured c=2 run 3, with the streams synchronized (TTFT 0.352/0.352), ran at 48.8 tok/s/stream (123 ms/step). Runs 1-2, with the streams desynchronized (TTFT 0.225 vs 0.573 and 0.258 vs 0.635, so the same text at offset positions and different tokens per step), ran at 39.3-42.3 tok/s/stream (about 142-153 ms/step). Decorrelating the two streams' tokens costs about 20% per stream. Prose c=2 streams partly diverge already (per_stream 17.28/16.66, 16.59/14.85), so the bias on prose is smaller than on structured.

**Verifier corrected claim:** The c=2 rows use identical greedy prompts, so synchronized streams share routed experts. Desynchronized identical-prompt structured streams at k=5 were about 20% slower per stream than synchronized ones (rebench-dflash5). The published c=2 rows, structured especially, overstate independent-stream throughput. The prose c=2 bias is smaller and bounded, roughly 5-15%.

**Verifier corrected impact:** The structured c=2 row is optimistic by about 15-20%. Prose c=2 is likely optimistic by about 5-15%. Any c=2-based k or seqs decisions are biased toward more verify tokens.

## RF-5: DFlash2 draft costs 1.9 GB/rank/step (≈9 ms, 8%): re-reads the full lm_head, a replicated 160 MiB fc, BF16 12288-wide MLPs

- kind=perf component=DFlash2 drafter (qwen3_dflash*.py) and lm_head sharing impact=3 confidence=4 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Per step, per rank, the draft reads 5 layers x 200 MiB (qkv+o+MLP TP-halved: 184 MiB, plus replicated attention_conv/mlp_conv kernel_projection: 16 MiB), fc 160 MiB replicated, a fused context-KV copy of 40 MiB, hidden_projection 2 MiB, and the shared target lm_head 605 MiB again for get_top_k_tokens. Total 1807 MiB = 1.895 GB, which is 9.2 ms at 205 GB/s. By comparison, MTP-4 reads about 4 x 0.93 GB serially.

**Mechanism:** The draft is BF16 and partly replicated, and it runs a full vocab-parallel lm_head GEMM over its query tokens to get the top-16 candidates.

**Evidence:**
- df.tsv: fc.weight 4096x20480 BF16 = 167,772,160 B; each layer mlp gate/up/down = 3 x 100,663,296 B; q/o 33,554,432 B; k/v 8,388,608 B; kernel_projection 8,388,608 B x 2; the two codebooks 79,298,560 B each
- qwen3_dflash.py:417-427 fc = ReplicatedLinear; qwen3_dflash2.py:69-77 kernel_projection = ReplicatedLinear; qwen3_dflash.py:195-210 QKVParallel/RowParallel; qwen3_dflash.py:453-454 _fused_kv_weight = torch.cat(...) (an extra copy that is read every step)
- qwen3_dflash2.py:282-287 compute_candidates → get_top_k_tokens(self.lm_head, ...); llm_base_proposer.py:1524-1541 shares the target lm_head when the draft has none (the draft safetensors has no lm_head)
- levers.py: MTP-4 modeled at 107.6 ms vs 113.2 measured (PR #9 H2)

**Proposed action:** (a) FP8 W8A16 for the draft linears and lm_head (covered by RF-2). (b) Shard fc column-parallel (−80 MiB per rank per step) and drop the duplicated _fused_kv_weight copy by slicing views, or fuse the context-KV GEMM (−40 MiB of memory; same bytes). (c) Leave MTP as rollback only; it reads twice the draft bytes.

**Est. impact:** FP8 draft+lm_head: −0.95 GB = −4.6 ms (+4% prose c=1). Sharding fc: −0.08 GB = −0.4 ms (<0.5%).

**Validation:** nsys: time the draft forward and get_top_k_tokens per step before and after, then bench_decode.py prose c=1.

**Risks:** FP8 on the draft only changes acceptance, not target output quality, so the risk is low. Sharding fc changes a replicated layer's semantics and needs an all-gather.

**Verifier reasoning:** The byte facts are verified. The draft config has 5 sliding_attention layers, 32 q / 8 kv heads, intermediate 12288, and target_layer_ids of 5 aux layers (so fc is 20480→4096, 160 MiB). qwen3_dflash.py:417-427 makes fc a ReplicatedLinear. qwen3_dflash2.py:69-77 makes kernel_projection a ReplicatedLinear. qwen3_dflash.py:453 builds _fused_kv_weight via torch.cat. qwen3_dflash2.py:282-287 computes candidates with get_top_k_tokens(self.lm_head, ...). The draft runs one forward per step (dflash2/speculator.py _generate_draft → _run_model once). Two corrections. (a) The serve path is model-runner v2, which shares the target lm_head via v1/worker/gpu/spec_decode/dflash/utils.py (load_dflash_model), not llm_base_proposer.py:1524-1541 (the v1 path). The effect is the same. (b) The risk statement 'FP8 on the draft only changes acceptance, not target output quality' is wrong for the lm_head: it is the same tensor the target uses, so quantizing it changes target logits unless a separate FP8 copy is kept for the draft (+302 MiB per rank). The MTP comparison (4 x 0.93 GB) depends on the calibrated model (107.6 modeled vs 113.2 measured).

**Verifier corrected claim:** The DFlash2 draft adds about 1.9 GB per rank per step (1202 MiB of draft weights plus a second 605 MiB read of the shared target lm_head), about 7-9 ms. FP8 on the draft-only linears is low-risk (acceptance only). FP8 on lm_head affects the target unless a separate draft-only copy is kept. Sharding fc saves under 0.5%.

**Verifier corrected impact:** About +2-4% prose c=1 from FP8 on the draft (+lm_head). fc sharding is negligible.

## RF-6: KDA spec-decode writes (k+1) fp32 recurrent states per layer per sequence: 628 MiB/seq/step at k=7

- kind=perf component=fused_recurrent_kda (spec path) + mamba state layout impact=2 confidence=4 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** On the spec path the recurrent kernel loads the state at num_accepted-1 and stores the full [32 heads x 128 x 128] fp32 state (2 MiB per rank) after every verify token. The traffic is 34 x 2 MiB x (1+T) per sequence per step: 141 / 210 / 280 / 350 / 419 / 489 / 628 MiB for k = 0/1/2/3/4/5/7. At k=7 that is 3.2 ms at c=1 and 6.4 ms at c=2, and it also costs 8 state slots per request of UMA.

**Mechanism:** Intermediate states exist only for rollback after rejection sampling. Only one of the T states survives.

**Evidence:**
- fused_recurrent.py:112-130 (initial-state load at num_accepted_tokens-1) and :176-185 (INPLACE_FINAL_STATE store for every i_t whose index is > 0)
- mamba_utils.py:131-137 kda_state_dtype returns (state_dtype, torch.float32); :293 recurrent shape (num_heads/tp, head_dim, head_dim)
- kda.py:546-563 spec path passes ssm_state_indices=spec_state_indices_tensor (num_spec+1 columns)

**Proposed action:** Kernel change: store only the initial and final states, and after acceptance re-advance the recurrence for the ≤T accepted tokens from the cached k/v/g/beta (tiny re-run). Alternatively store intermediate states in bf16. Guard behind an env flag and check that greedy output matches bit-for-bit.

**Est. impact:** Recompute variant: writes fall from 16 MiB to about 2 MiB per layer. −0.49 GB per sequence per step = −2.4 ms at c=1 (+2.2% prose), −4.8 ms at c=2. It also frees about 7 of the 8 state slots per request (memory headroom for a larger KV pin or seqs).

**Validation:** ncu dram bytes on fused_recurrent_kda before and after, count-200 greedy parity, and prose c=1/2 bench.

**Risks:** Correctness of rollback. vLLM's GDN metadata assumes per-slot states, so prefix caching and mamba 'align' mode interactions need testing.

**Verifier reasoning:** The mechanism is verified. third_party/flash_linear_attention/ops/fused_recurrent.py loads the initial state at num_accepted_tokens-1 (IS_SPEC_DECODING branch) and, under INPLACE_FINAL_STATE, stores b_h after every token i_t whose ssm_state_indices entry is >0. mamba_utils.kda_state_dtype returns float32 for the recurrent state. kda_state_shape gives (heads/tp, 128, 128), which is 2 MiB per layer per slot at TP=2. kda.py:546-563 passes spec_state_indices_tensor. The arithmetic 34 x 2 MiB x (1+8) + conv ≈ 628 MiB per seq per step holds. The impact is small (about 2%) and conditional on the byte model. The claim that the recompute variant 'frees about 7 of the 8 state slots per request' is overstated: slot allocation is done by vLLM's mamba/KV manager for num_spec+1 columns, so a kernel change alone frees nothing. Also, the kernel launches 128 one-warp programs per sequence that iterate serially over T. It is more latency-bound than byte-bound, so the byte-derived ms estimate is uncertain in both directions.

**Verifier corrected claim:** KDA spec verify writes T fp32 recurrent states (2 MiB per layer per rank each) per sequence per step: about 0.63 GB/seq/step at k=7. Writing only the needed states would save roughly 1-3 ms per sequence. Freeing state slots additionally requires changes to vLLM's mamba state allocation.

**Verifier corrected impact:** About +1-2% prose c=1 and +2-4% at c=2. Memory savings only with allocator changes.

## RF-7: 10-18.5 ms fixed + ~11 ms per-extra-sequence overhead not explained by bytes

- kind=perf component=engine loop: NCCL over RoCE, CPU scheduling (async off), breakable CUDA-graph partitions, per-seq small kernels impact=3 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Fitting O + bytes/BW to the k=5 and k=7 points leaves O = 10.0 ms at BW 205 GB/s (rho 0.58) or 18.5 ms at 232 GB/s (rho 0.64). Separately, the shared-expert control (structured c=2) measures 131.1 ms against a modeled 120.2 ms, which is about 11 ms per extra sequence beyond its 0.66 GB of state. Together that is roughly 9-25% of the step.

**Mechanism:** Each RoCE all-reduce has a latency floor of about 15-30 µs, so 105 of them cost about 2-3 ms. With async scheduling off, CPU work between steps is serialized: scheduling, rejection sampling sync, DFlash2 metadata, detokenize and stream. Graph breaks add launch gaps. Per-sequence indexer top-k, sampler and state-index kernels add latency that grows with the batch.

**Evidence:**
- model.py fit output: 'fit BW 232.0 rho 0.64 O ms 18.5', 'fit BW 205.0 rho 0.58 O ms 10.0'
- Collective count from code: 2 all-reduces per target layer (KDA/MLA o_proj RowParallel + MoE output) x 45 = 90, plus embed/logits, plus the draft's 5x2 + embed + top-k gather, for about 105 small (≈64-128 KiB) all-reduces per step over rocep1s0f1
- run.sh env: NCCL_NET=IB, NCCL_NVLS_ENABLE=0; AGENTS.md: 'Leave --async-scheduling off', 'VLLM_USE_BREAKABLE_CUDAGRAPH on auto'; kda.py:20 imports eager_break_during_capture (graph partitions)
- levers.py: 'structured c2 shared-expert control k7: 120.2 ms (meas 131.1)'

**Proposed action:** First profile one steady-state step with nsys on both ranks and attribute the gap between GPU busy time and wall time. Then evaluate, one at a time: NCCL_PROTO=LL/LL128 and NCCL_ALGO=Ring for small messages; fusing the MoE and attention all-reduces where mHC allows it; and async scheduling. Async scheduling contradicts an AGENTS.md rule, so it needs explicit justification with the count-200 and thinking-off gates before any keep.

**Est. impact:** Halving O (10 → 5 ms) gives +4.5% prose c=1 (levers.py). Cutting the per-extra-sequence 11 ms in half gives about +4% per stream at c=2.

**Validation:** nsys timeline: GPU idle gaps per step, NCCL kernel durations, CPU time between the verify graph and the draft graph. Each env change is its own bench run.

**Risks:** Async scheduling interacts with spec decode and the KDA state indices (it was disabled for a reason). NCCL protocol changes can destabilize the pinned HCA path.

**Verifier reasoning:** The O values (10.0 and 18.5 ms) are artifacts of the exactly-determined fit, not measurements. Supporting evidence the reviewer did not cite: rebench-dflash5 synchronized structured c=2 at k=5 is 123 ms vs 107.7 ms at c=1 (+15 ms with identical tokens), consistent with the 131.1 vs 120.2 ms gap at k=7. Structural sources are confirmed in code. kda.py:20/373 decorates KDA _forward with @eager_break_during_capture, so 34 eager Python breaks run per target step. The engine log shows 'Breakable CUDA graph enabled' plus PIECEWISE capture. run.sh:255-264 sets NCCL_NET=IB and NVLS off. The all-reduce count of about 2 per layer (o_proj RowParallel + fused MoE/shared output) is consistent with model.py. The per-all-reduce latency (15-30 µs) is an estimate; over cross-node RoCE, 30-60 µs is also plausible, which would make NCCL 3-6 ms. Async scheduling contradicts AGENTS.md and needs explicit justification, as the reviewer notes.

**Verifier corrected claim:** The data imply a non-byte overhead of roughly 10-20 ms fixed plus about 1.5-2 ms per extra verify token (from the identical-token c=2 controls at k=5 and k=7). Likely sources are 34 eager KDA breaks per step, about 100 cross-node all-reduces, and serialized host work. The split is unknown until an nsys profile is taken.

**Verifier corrected impact:** +4-10% if the overhead is halved. This may be larger than the reviewer's estimate, because the per-token non-byte cost also inflates the k-slope attributed to MoE.

## RF-8: Amdahl lever table (prose c=1, calibrated model) — combined path to ~+70%

- kind=perf component=recipe + image-layer kernels impact=5 confidence=2 effort=XL needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** Starting from the calibrated model (k=7: 116.9 ms, 20.1 tok/s modeled; 21.2 measured best, 19.2 parity), stacking the independent levers multiplies. Model results in tok/s: k=5 21.8 (+8.7%); best static k=3 22.8 (+13.8%); lm_head FP8 20.3 (+1.3%); all BF16 non-MoE FP8 25.3 (+25.9%); NVFP4 W4A16 28.5 (+42%); KDA-state recompute 20.5 (+2.2%); O 10→5 ms 21.0 (+4.5%); achieved BW 205→225 GB/s 21.8 (+8.8%); combo (k=2-3 + FP8 non-MoE + O=5 + KDA-state) 34.6 (+72.7%); stretch (NVFP4 non-MoE + 225 GB/s) 46.4 (+131%).

**Mechanism:** With non-MoE bytes halved, the per-step fixed cost drops, so the best k shifts lower: each step becomes cheaper relative to the per-token MoE cost.

**Evidence:**
- levers.py output (scratchpad/agents/roof/levers.py)
- Combo arithmetic at k=2: MoE E_eff(3) = 8 + 0.58 x 15.3 = 16.9 experts x 42 x 7.08 MB = 5.02 GB. Non-MoE FP8 about 5.30 GB, state 0.30 GB. (5.02 + 5.30 + 0.30 − KDA-state savings) / 205 GB/s + 5 ms ≈ 56 ms. Acceptance 1.93 → 34.6 tok/s.
- The structured regime with FP8 non-MoE at k=7 is 84.1 tok/s (vs 66.8).

**Proposed action:** Order of execution: RF-3 (k, S effort, recipe only) → RF-7 profile → RF-2 in stages (lm_head/draft → shared → MLA → KDA) → RF-6. Re-derive the best k after each byte-reduction lever, because the optimum moves.

**Est. impact:** Prose c=1 about 20-21 → about 34 tok/s if every lever lands with the assumed acceptance curve. Structured about 67 → 84+ with FP8 alone at k=7.

**Validation:** Change one knob at a time against bench_decode.py, 3-run medians, 8% noise floor, receipts in evidence/.

**Risks:** The combined figure multiplies model uncertainties: the prose acceptance curve (assumed), rho, BW_eff, and whether quality survives FP8/NVFP4 attention. Treat 34 tok/s as an upper-plausible target, not a forecast.

**Verifier reasoning:** levers.py reproduces the table exactly (combo k=2 at 34.6 tok/s, +72.7%). Every entry, however, comes from a model with 2 free parameters fit to 2 noisy points. The combo also depends on an assumed prose per-position acceptance curve that is not in the evidence, since vllm:spec_decode_num_accepted_tokens_per_pos was never scraped. And it assumes lever independence, even though RF-1's c=2 controls show per-token non-byte costs the model omits. The FP8/NVFP4 quality assumptions are untested. The model does match one out-of-sample point: structured k=5 was modeled at 56.1 and measured at 55.7 (rebench-dflash5), which gives some credibility to the byte model at the aggregate level.

**Verifier corrected claim:** Only the k=5 lever is measured (+11%). FP8 on non-MoE weights is the next best-grounded lever, at about +12-26% if quality holds. The +70% combo is an optimistic upper bound built on an assumed acceptance curve and an exactly-fit model. Plan for +25-45% prose c=1 if k tuning, FP8 non-MoE and overhead work all land.

**Verifier corrected impact:** +25-45% plausible range, with +70% as an upper bound.

## RF-9: nvidia vs LibertAI pack: identical expert bytes, nvidia saves 310 MiB/rank/step on dense MLP; nvidia MTP layer is BF16 and not excluded from quant (SPEC=mtp rollback risk)

- kind=correctness component=checkpoint 09b04e5 vs caca4e6; SPEC=mtp path impact=3 confidence=3 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** Routed experts are byte-identical in size across the two packs (171,228,556,800 B each). Dense layers 0-2 are NVFP4 in the nvidia pack (81 MiB/layer) but BF16 in LibertAI (288 MiB/layer), saving (288−81) x 3 / 2 = 310.5 MiB per rank per step (1.6 ms, about +1.3%). The nvidia MTP layer 45 ships BF16 experts: 14,495,514,624 B = 13.5 GiB, against LibertAI's NVFP4 3.8 GiB. hf_quant_config.json excludes only lm_head and model.visual*, and config.json's ignore list has no layers.45 entry. Under SPEC=mtp the modelopt NVFP4 method would therefore be built for BF16 [2048x4096] tensors: a probable load failure, or +4.85 GiB per rank if it were ever dequantized.

**Mechanism:** ModelOpt quantizes a module unless its name is in the exclude list. The checkpoint author left MTP unquantized but did not list it.

**Evidence:**
- agg.py on nv.tsv: 'mtp.routed_experts 14495514624 B {BF16}'; on lb.tsv: 'mtp.routed_experts 4076870400 B {U8, F8, F32}'
- nv dense_mlp 254,804,040 B (U8+F8) vs LibertAI 905,969,664 B BF16
- hf_quant_config.json exclude_modules (132 entries): the only matches for 45/visual/lm_head are ['lm_head', 'model.visual*']
- model.py:818-820 skips layers.45 when SPEC=dflash2 (no memory cost in the default); mtp.py:44 and :76-83 build the MTP decoder layer with vllm_config.quant_config

**Proposed action:** Before advertising SPEC=mtp as the rollback on the nvidia pin, validate it boots (VALIDATE_ONLY cannot catch this). Otherwise make run.sh refuse SPEC=mtp when MODEL=nvidia/..., or add layers.45.mlp.experts* to the ignore handling via a small loader patch.

**Est. impact:** Decode: nvidia pack about 1.6 ms per step faster than LibertAI (+1.3%). Rollback correctness: SPEC=mtp on the nvidia pin is likely broken (unverified).

**Validation:** Boot SPEC=mtp with the nvidia pin in an exclusive slot and watch for a weight_loader shape assertion. Separately, bench the nvidia pack against LibertAI at k=7 prose c=1.

**Risks:** vLLM's modelopt path might silently skip unmatched BF16 tensors instead of failing. Either outcome needs a run to confirm.

**Verifier reasoning:** agg.py on both header TSVs: routed experts are 171,228,556,800 B in both packs. The nvidia dense_mlp is 254,804,040 B (U8+F8+F32, NVFP4), against LibertAI's 905,969,664 B BF16. nvidia layer 45 has BF16 routed experts (14,495,514,624 B) plus BF16 shared experts, while LibertAI's MTP experts are NVFP4 (4,076,870,400 B). hf_quant_config.json exclude_modules covers only layer indices 0-44, plus lm_head, embed_tokens and model.visual*. modelopt.py:177-214 (is_layer_excluded/get_quant_method) would therefore give layers.45.mlp.experts (RoutedExperts) and layers.45.mlp.shared_experts (LinearBase) NVFP4 methods for BF16 tensors, which is a near-certain load failure under SPEC=mtp. mtp.py:44 passes vllm_config.quant_config. One minor correction: model.py:818-820 skips spec-layer weights in the main model regardless of SPEC, not only for dflash2. The shared experts of layer 45 are also affected, not just the routed experts. The +1.3% dense-MLP saving is right on bytes, but the reviewer missed that the NVFP4 dense layers do not go through Marlin (see the missed issues).

**Verifier corrected claim:** The expert bytes are identical across packs. The nvidia dense MLP is NVFP4, saving about 310 MiB per rank per step vs LibertAI BF16. The nvidia MTP layer 45 (routed and shared experts) is BF16 but not excluded from NVFP4, so SPEC=mtp on the nvidia pin will almost certainly fail at weight load. The README still advertises SPEC=mtp as the rollback.

**Verifier corrected impact:** The SPEC=mtp rollback is broken on the PR #11 pin (correctness). The dense-MLP decode delta is about +1% at most.

## RF-10: Decode bytes are nearly context-independent up to 128k (DSA top-k 2048 + kpool-4 indexer); long context is not a decode-speed tax

- kind=perf component=MLA/DSA sparse attention + kpool indexer KV reads impact=2 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Per sequence per step, KV and indexer reads are 11 MB at 1k, 32 MB at 32k and 66 MB at 128k, assuming the T query tokens reuse the top-k set in L2. The worst case with no reuse is 52 / 189 / 461 MB. That costs at most 0.3 ms (2.0 ms worst case) against a 117 ms step.

**Mechanism:** Sparse attention bounds each query's KV gather at 2048 tokens. The indexer scans compressed pools at 1/4 density and 132 B each.

**Evidence:**
- model.py kv_bytes: 11 x min(ctx, 2051) x 512 B latent fp8 + 11 x ctx/4 x 132 B (fp8 128 + 4 B scale per pool, attention.py:280-286) + draft SWA 5 x 2048 x 4 heads x 128 x 2 x 1 B
- config.json: index_topk 2048, index_kpool 4, kv_lora_rank 512, qk_rope_head_dim 0; model.py:604-629 topk buffer width 2048+3 rounded to 2176
- levers.py: 'ctx 131072 k7 c1 step 117.2 (KV no-reuse would add 2.0 ms)'

**Proposed action:** No change is needed for decode. Use this to decouple the context-window decision (a KV-pool/UMA question) from decode-speed concerns. Confirm with a 32k/128k-context decode bench.

**Est. impact:** Informational. Predicted decode drop from 1k to 128k is under 2%, excluding indexer top-k compute over 32k pools.

**Validation:** Run bench_decode.py prose with a 32k and a 100k filler prefix at c=1 and compare ms/step.

**Risks:** Indexer top-k over ctx/4 candidates, and paged-MQA kernels at next_n=8, could be latency-bound rather than byte-bound at 128k.

**Verifier reasoning:** config.json confirms index_topk 2048, index_kpool 4, index_n_heads 32 x 128, kv_lora_rank 512 and qk_rope_head_dim 0. The draft config uses sliding_window 2048 on all 5 layers with 8 KV heads (4 per rank). The KV byte arithmetic in model.py kv_bytes reproduces (10.9 / 32.4 / 66.4 MiB with reuse, 51.8 / 188.9 / 461.1 MiB without). Bytes are indeed small. At long context, though, indexer scoring and top-k over ctx/4 candidates for T=8 queries across 11 layers (DeepGEMM fp8 paged MQA, run.sh:42) is latency and compute work that the byte model does not capture. The reviewer flags this as a risk. No long-context decode measurement exists in evidence.

**Verifier corrected claim:** KV and indexer byte traffic stays under about 0.5 GB per step up to 128k. Decode slowdown at long context will come from indexer top-k latency, not bytes, and is unmeasured.

**Verifier corrected impact:** Informational. Long-context decode drop is likely under 10%, but unmeasured.

## RF-11: Achieved bandwidth of Marlin NVFP4 MoE at M≈1 token/expert is the biggest unknown; +10% BW ≈ +9% tok/s

- kind=perf component=Marlin MoE (W4A16 NVFP4) on sm_121 impact=3 confidence=2 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** MoE is about 54 ms of the calibrated 117 ms. Each touched expert is a skinny GEMM of 6.75 MiB per rank with 1-2 tokens. Whether Marlin reaches about 205 or about 230 GB/s there changes the step by 5-6 ms on MoE alone. Across all bytes, 205 → 225 GB/s gives +8.8% (levers.py).

**Mechanism:** For small-M MoE, the tile scheduler (moe_align_block_size padding, per-expert blocks), split-K and the number of SMs streaming concurrently (GB10 has 48 SMs) decide what fraction of DRAM bandwidth the kernel reaches.

**Evidence:**
- levers.py: 'achieved BW 205->225 GB/s 21.8 tok/s (+8.8%)'
- ds4 #773: GB10 decode at 85-90% of 231-234 GB/s measured read, https://github.com/antirez/ds4/issues/773
- run.sh:114-117 and AGENTS.md: Marlin is the only non-OOM MoE path on this UMA

**Proposed action:** Microbenchmark Marlin fused_moe on an idle Spark with a synthetic 288x(2048x4096) NVFP4 layer at M = 8/16 and 58/100 active experts. Measure GB/s. If it is under 200 GB/s, tune block size, thread_k/n and split for the M≤16 path in an image layer. Do not touch the flashinfer_cutlass/b12x paths (OOM / clamp blockers, per AGENTS.md).

**Est. impact:** +5-10% prose and structured if Marlin is currently below about 205 GB/s effective. Zero if it is already near 225 GB/s.

**Validation:** Standalone Marlin microbenchmark (no serve), then an ncu dram throughput counter on the in-serve kernel.

**Risks:** Needs an exclusive GPU slot. Kernel retuning risks numerical differences (still bf16 accumulate) and must pass count-200.

**Verifier reasoning:** This is correctly framed as an unknown. The ds4 #773 figures (231-234 GB/s read, 85-90% in decode) are verified from the issue, but they describe ds4's custom CUDA decode, not vLLM's Marlin MoE, so they only bound the ceiling. The +8.8% for 205→225 GB/s reproduces in levers.py under the model's assumptions. AGENTS.md and run.sh correctly keep Marlin as the only MoE path.

**Verifier corrected claim:** Marlin MoE's achieved bandwidth at 1-2 tokens per expert on sm_121 is unmeasured. It is worth a standalone microbenchmark before any kernel tuning.

**Verifier corrected impact:** 0 to about +8%, depending on the measured efficiency.

## RF-12: mHC hyper-connection weights are upcast to fp32 (2x checkpoint size) and read every step

- kind=perf component=Glm5NextDecoderLayer mHC params impact=1 confidence=3 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** hc_attn_fn and hc_ffn_fn are declared torch.float32 [24, 16384] although the checkpoint stores BF16. That is 3 MiB per layer x 45 = 135 MiB replicated per rank read per step (0.7 ms), where the BF16 values would be 67.5 MiB. The upcast is lossless, so keeping them BF16 reproduces the checkpoint values exactly.

**Mechanism:** The fp32 parameter doubles bytes read by the mHC pre/post fused kernels.

**Evidence:**
- model.py:391-402 nn.Parameter(torch.empty(mix_hc, d_model, dtype=torch.float32))
- nv.tsv: layers.N.hc_attn_fn BF16 24x16384 = 786,432 B

**Proposed action:** Only if the MHC kernels accept bf16 fn with fp32 accumulate: store it as bf16. Otherwise skip. Low priority.

**Est. impact:** −0.07 GB per step = −0.35 ms (+0.3%) and −67 MiB per rank.

**Validation:** Greedy count-200 parity and a microbenchmark of MHCFusedPostPreOp.

**Risks:** The kernel may require fp32 fn, and fp32 accumulate over 16384 inputs must be kept.

**Verifier reasoning:** model.py:391-402 declares hc_attn_fn and hc_ffn_fn as torch.float32 [24, 16384] (mix_hc=(2+4)*4=24, d_model=4*4096). The checkpoint stores them in BF16 (agg: mhc 1.50 MiB per layer in BF16 for both tensors). 45 x 3 MiB = 135 MiB fp32 vs 67.5 MiB. The impact is negligible, as the reviewer says.

**Verifier corrected impact:** About 0.3%. Not worth an image layer on its own.

## Open questions
- Per-position prose acceptance (vllm:spec_decode_num_accepted_tokens_per_pos) is in none of the evidence files. The prose curve here (p = 0.60, 0.33, 0.18, 0.11, 0.07, 0.035, 0.02) is fit only to acceptance_len 2.31 at k=5 and 2.29-2.43 at k=7. The best-k and combo numbers depend on it.
- Real routing concentration: how many distinct experts per layer a T=8 prose verify block touches. Two-point calibration gives rho≈0.58-0.64 (about 37-40 experts vs 58 under uniform routing). Logging router top-k ids for one prose request would settle it and pin BW_eff.
- Achieved DRAM GB/s of the Marlin NVFP4 MoE kernel and of the BF16 GEMMs at M=8-16 on sm_121. The step budget is highly sensitive to this, and it has only been inferred.
- Where the 10-18.5 ms fixed and about 11 ms per-sequence overhead goes (NCCL over RoCE vs CPU scheduling with async off vs graph-break gaps). Only an nsys profile on an exclusive slot can answer this.
- Does SPEC=mtp boot on the nvidia pin, given layer-45 BF16 experts that are not in the ModelOpt exclude list?
- Quality impact of FP8 or NVFP4 weight-only on the KDA/MLA projections (NVIDIA kept them BF16). A long-context and recurrent-sensitive eval is needed before RF-2 lands.
- Is fp8 Marlin (W8A16) usable for N=8192/K=4096 BF16 linears inside the v11 image on sm_121 (the _C arch list excludes 12.1a)? Or must RF-2 use NVFP4 Marlin W4A16 instead?
- Should PR #9 H1 (k=5, +14% prose, reverted for a render/lint failure) be re-run and kept before any deeper work?

## Verifier: missed issues
- The reviewer missed that PR #9 already hand-applied NUM_SPECULATIVE_TOKENS=5 as the default (commit a76474e; decision.tsv 'H1-20260903-apply ... kept'; e898a80:recipe.yaml:49) and re-benched it in evidence/rebench-dflash5-20260903T045815Z: prose c=1 21.76, structured c=1 55.73 (−18%), prose c=2 19.62. That rebench FAILED the gate on needle-8192 (hit=0, refusal; commit f2538e5), and no retry was run. Any k=5 proposal has to deal with that failed quality gate. Re-running H1 does not address it.
- Bench noise is large and it undermines the calibration. Nine prose c=1 runs at k=7 over three boots (rebench bench.txt, parity bench.txt, parity bench-run1.txt) range 16.86-23.40 tok/s. The first run of each wave is consistently the fastest. bench_decode.py pairs a per-phase aggregate acceptance_len with the median tok/s, which is a mismatched estimator for ms/step. With the 8% noise floor, the model's 2-point exact fit cannot separate rho, O and BW.
- The identical-token c=2 controls show a large non-byte per-token cost. rebench-dflash5 structured c=2 run 3 (synchronized, 48.8 tok/s, acc 6.0) takes 123 ms vs 107.7 ms at c=1, which is +15 ms for 6 extra identical tokens with no new expert bytes. The k=7 equivalent is 131.1 vs 116.0 ms. The same runs, desynchronized (runs 1-2, TTFT 0.225 vs 0.573), fall to about 40 tok/s/stream (about 150 ms). This both supports RF-4 and weakens RF-1's attribution of the k-slope to distinct-expert bytes.
- On the PR #11 nvidia pin, the NVFP4 dense MLP (layers 0-2) does NOT run through Marlin. MOE_BACKEND=marlin only covers MoE. init_nvfp4_linear_kernel (v11src vllm/model_executor/kernels/linear/__init__.py:980-1075, candidate list at :500-512) auto-selects FlashInferCutlassNvFp4LinearKernel, which is supported for any capability >=100 (model_executor/kernels/linear/nvfp4/flashinfer.py:106-119), before MarlinNvFp4LinearKernel. That is a W4A4 path that uses input_scale and may JIT a FlashInfer CUTLASS module on sm_121, the same class of JIT that global-OOM'd spark2 (evidence/oom-20260831). The LibertAI pack never exercised it because its dense MLP is BF16. This is an unmeasured boot/UMA and numerics risk for PR #11. The README line 'Marlin still dequantizes weights and never reads an activation input_scale' is only true for MoE.
- lm_head is one tensor shared by the target and the DFlash2 draft (v1/worker/gpu/spec_decode/dflash/utils.py, load_dflash_model). Any FP8 lm_head lever (RF-2/RF-5) changes target output, not only draft acceptance, and must pass the count-200 and thinking-off gates. Alternatively, keep a separate FP8 copy for the draft only (+302 MiB per rank).
- The SPEC=mtp rollback on the nvidia pin also breaks for layer-45 shared_experts (BF16, and hf_quant_config exclude_modules only lists shared_experts for layers 0-44), not just the routed experts. README.md:64 and recipe.yaml:81 still advertise SPEC=mtp as the rollback on PR #11.
- UMA swap is a probable noise and slowdown source that the reviewer did not model. evidence/rebench-20260902T204243Z/free-after-bench.txt shows 116/121 GiB used, 2 GiB free and 6 GiB of swap used. README.md:22 records the worker with about 1.3 GiB swapped and slow first waves. decision.tsv h-snap shows wave2 prose falling to 16.92 under swap. Cutting resident bytes (FP8 non-MoE frees about 4.4 GiB per rank) may reduce variance as well as bytes.
- Each target step contains 34 eager KDA breaks (kda.py:373 @eager_break_during_capture) under breakable CUDA graphs. Each one runs Python metadata handling plus Triton launches for causal_conv1d_update and fused_recurrent_kda. fused_recurrent_kda itself launches 128 one-warp programs per sequence that iterate serially over T (fused_recurrent.py grid=(NK,NV,N*HV), num_warps=1). These latency-bound costs are a likely large share of the unexplained fixed and per-token overhead and should be the first nsys target.
