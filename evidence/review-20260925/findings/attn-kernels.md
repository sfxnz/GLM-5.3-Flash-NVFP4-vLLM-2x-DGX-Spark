# Dimension: attn-kernels

## Reviewer summary

Scope: non-MoE kernels on the decode/verify critical path. This was a read-only review with no GPU use. Paths shortened as v11src = /tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-GLM-5-3-Flash-NVFP4-vLLM-2x-DGX-Spark/29f0f0df-0b0e-4224-a895-0ea5d735b7de/scratchpad/v11src and repo = /home/sfxnz/projects/ai-lab/recipes/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark-opt-nvidia.

(1) Measured baseline. Steady-state verify steps already replay FULL CUDA graphs. engine.log.tail shows "Capturing CUDA graphs (FULL) 2/2" for the uniform 8- and 16-token verify batches. The 56 eager breaks (34 KDA, 11 MLA, 11 indexer) only apply to mixed prefill steps. Graph mode is therefore not the lever.

(2) A pure byte model from the nvidia safetensors headers says the BF16 non-MoE weights are read every step. They are 7.94 GiB per rank: 34 KDA layers at 132 MiB each, 11 MLA+indexer layers at 135 MiB, shared experts at 24 MiB x 42, and lm_head at 605 MiB. At 220-273 GB/s that is about 30-38 ms of the ~116 ms verify step, or 26-33%. Routed MoE accounts for most of the rest, about 78 ms modelled. The model reproduces the measured ~116 ms step at c=1 (21.2/2.43 prose and 67.6/7.84 structured) and supports the weight-bandwidth hypothesis.

(3) The largest non-MoE lever is FP8 weight-only (W8A16 Marlin) on the KDA, MLA and shared-expert projections. It saves about 3.3 GiB per rank per step, estimated at 13-16 ms per step (+12-16% tok/s). It also frees about 3.3 GiB of UMA per rank. The quality risk is manageable at 8 bits. The producer's own FP8 path already block-quantizes the MLA projections. KDA was kept BF16 even there, so it needs a KL gate.

(4) Smaller verified opportunities:
- KDA speculative state checkpoints write 8 fp32 128x128x32 states per layer per sequence: 612 MiB per step per sequence (~2.9 ms) and 608 MiB of pool per sequence. A replay/commit design cuts both about 4x and would allow 4 sequences at DFlash2-7.
- The sm_12x indexer flattening re-reads the kpool K cache 8x per step. This costs up to ~4 ms per step at 327k context.
- 4 .contiguous() copies per KDA layer, plus 2 tiny GEMMs that can be merged, can be removed bit-exactly.

(5) Two ops/quality risks found:
- PR #11's nvidia pack is the first to run an NVFP4 dense linear (layers 0-2) on the Sparks. Kernel auto-selection can pick FlashInfer CUTLASS (JIT, no jit-cache, nvrtc.h missing on v11) or W4A4. Pinning --linear-backend marlin avoids that.
- mHC TileLang kernels JIT-compile per n_splits variant during serving (~5 s each). The DSv4-only mHC warmup skips Glm5Next. This matches the recorded 6-7 s TTFT spikes at first c=2/c=4.

mHC itself (4 fused TileLang kernels per layer, Sinkhorn in-kernel, inside the graph) and MLA sparse attention (FA2 page_size=1, top-2048) are each under 2 ms per step and are not worth major work. Fixed overhead of about 1,600 kernels per step plus ~90 inter-node PyNCCL allreduces, with PDL off, is estimated at 5-8 ms and should be measured before acting. FlashInfer's KDA decode kernels are SM100-only with bf16 state, so they are not usable here.

## NMK-1: Quantize BF16 KDA/MLA/shared-expert projections to FP8 weight-only (Marlin W8A16)

- kind=perf component=glm5next/nvidia kda.py + attention.py/model.py MLA + Glm5NextMLP shared experts; kernels/linear Marlin FP8 impact=5 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The largest non-MoE per-step cost is streaming BF16 projection weights through cuBLAS at M=8-16. KDA in_proj (q|k|v|b|f_a|g_a) + o_proj, MLA fused_qkv_a/q_b/o_proj + indexer wq_b, and the shared experts are 6.9 of the 7.94 GiB/rank of non-MoE bytes read on every verify step. Storing them as FP8 with per-output-channel scales and running them through the Marlin W8A16 kernel (BF16 activations, no activation quant) halves those bytes.

**Mechanism:** Decode/verify at M=8 (c=1) or 16 (c=2) is weight-bandwidth bound on the 273 GB/s LPDDR5X. Each byte of BF16 projection weight is read once per step regardless of acceptance. FP8 weight-only halves the bytes. Marlin dequantizes in-register and keeps BF16 activations, so no activation-quant error is added and no extra quant kernels are needed. Marlin is already proven on sm_121 in this image (the MARLIN NvFp4 MoE backend runs).

**Evidence:**
- nvidia ckpt headers (header-only parse, scratchpad/agents/nonmoe/hdr.py): q_proj/k_proj/v_proj BF16 [8192,4096] x34 = 2176 MiB each; o_proj total 3712 MiB (34x64 + 12x128 MiB); q_b_proj [16384,1536] x12 = 576 MiB; shared_experts gate/up/down 688 MiB each
- scratchpad/agents/nonmoe/bytes.py: per-rank per-step non-MoE = KDA 4496.5 + MLA/indexer 1482.2 + shared 1008 + router 94.5 + mHC fn 135 + dense NVFP4 121.5 + lm_head 605 = 7942.8 MiB (8.33 GB) -> 30.5 ms @273 GB/s, 37.9 ms @220 GB/s
- nvidia config.json quantization_config.ignore: model.language_model.layers.*.self_attn*, *.mlp.shared_experts*, *.mlp.gate, lm_head all BF16
- v11src/vllm/models/glm5next/nvidia/kda.py:168-174 strips quant_config for the KDA layer; model.py:331 constructs MLA with quant_config=None
- v11src/vllm/model_executor/layers/utils.py:86-92,391-397: unquantized GEMM = torch.nn.functional.linear (cuBLAS) on CUDA
- v11src/vllm/models/glm5next/nvidia/model.py:1191-1200 and 837-848: the producer's FP8 checkpoint stores q_a/kv_a/q_b/o_proj as 128-block FP8 (and loads them by dequantizing to BF16); kda.py:168-170 says KDA stays BF16 even in FP8 checkpoints
- v11src/vllm/model_executor/kernels/linear/__init__.py:375-383 (MarlinFP8 first in the CUDA FP8 list); kernels/linear/scaled_mm/marlin.py:45-54 (needs VLLM_TEST_FORCE_FP8_MARLIN=1 on cap>=89) and 62-76 (block/channel support)
- v11src/vllm/model_executor/layers/quantization/online/fp8.py:339-398: an online per-channel quant helper (_fp8_quant_per_channel) exists; PTPC refuses Marlin (W8A8 only), so a small W8A16 per-channel method is needed
- Measured step ~116 ms: rebench-20260902T204243Z/bench.txt prose c=1 21.2 tok/s / acc 2.43 = 8.7 steps/s; structured 67.6/7.84 = 8.6 steps/s

**Proposed action:** Add a new image layer (v12-fp8proj) patch with per-channel symmetric FP8 e4m3 weights, quantized at process_weights_after_loading, per layer so peak memory stays flat, and applied with the Marlin FP8 kernel. Set VLLM_TEST_FORCE_FP8_MARLIN=1 in run.sh env. Scope, in rollout order (one knob per bench): (a) shared_experts gate_up/down; (b) MLA fused_qkv_a_proj, q_b_proj, o_proj, indexer.wq_b (producer-validated class), keeping kv_b (W_UK/W_UV BMM) BF16; (c) KDA o_proj; (d) KDA in_proj_qkvbfg_a. Keep f_b/g_b, router gate, mHC fn, embed and lm_head BF16 (see NMK-8). Use per-channel rather than 128x128 blocks because the merged in_proj has 32- and 128-row shards that are not 128-block aligned. Expose it as a recipe.yaml knob (e.g. PROJ_WEIGHT_DTYPE=bf16|fp8, default bf16 until measured). NVFP4/W4A16 for these layers would save another ~8 ms but is rejected: NVIDIA and LibertAI both excluded self_attn/shared at 4 bits.

**Est. impact:** Bytes saved per rank per step: KDA 2248 MiB + MLA 638 MiB + shared 504 MiB = 3390 MiB (3.55 GB). At 220-273 GB/s that is 13.0-16.2 ms/step. c=1 prose step 116 -> ~100-103 ms: 21.2 -> ~24.0-24.6 tok/s (+13-16%). c=2 prose step 150 ms (2.50/16.64) -> ~134-137 ms: 16.6 -> ~18.2-18.6 tok/s/stream. Breakdown: KDA alone 10.7 ms @220 GB/s, MLA 3.0 ms, shared 2.4 ms. Side effect: ~3.3 GiB/rank less UMA resident (b12x-A postboot shows 115 Gi used and 7.3 Gi swap on spark1).

**Validation:** Stage by stage, one knob each, on an exclusive TP=2 slot. (1) Boot log must show MarlinFP8ScaledMMLinearKernel for the targeted layers. (2) nsys --cuda-graph-trace=node on 20 verify steps: per-GEMM duration before/after (expect ~2x shorter on those GEMMs), plus achieved GB/s for cuBLAS BF16 vs Marlin FP8 at M=8/16. (3) python3 bench_decode.py prose c=1/c=2 3-run median, noting acceptance_len (target hidden states feed DFlash2 via aux layers). (4) Quality gates: greedy count 200 lossless; thinking-off smoke; needle 8192/20480 (existing evidence harness); KL(bf16 || fp8) on ~200 prompts via prompt_logprobs, with an expected mean KL < 1e-3 as a hard gate before the KDA stages. Record each stage in evidence/trail.tsv and decision.tsv.

**Risks:** KDA q/k/v feed an L2-normalised delta rule, and errors could accumulate in the recurrent state over long contexts. That is why KDA goes last, behind KL and needle gates. The producer kept KDA BF16 even in its FP8 checkpoint. Marlin dense-FP8 kernel efficiency on sm_121 is unmeasured, though Marlin cubins exist since the MoE Marlin runs. VLLM_TEST_FORCE_FP8_MARLIN is a test env var and could change semantics upstream. Load-time quantization adds boot time (~1-2 min). This is a fork-code patch, so it adds a new image in the v8->v11 chain.

**Verifier reasoning:** Checked the facts it relies on. (a) KDA strips quant_config (v11src/vllm/models/glm5next/nvidia/kda.py:168-174). MLA is built with quant_config=None (model.py:331). Unquantized GEMM is F.linear (layers/utils.py:86-92, 391-397). (b) A header-only parse of the nvidia shards reproduces the numbers: q/k/v_proj 2176 MiB x34 each; o_proj 3712 MiB over 46 = 34x64 + 12x128; shared gate/up/down 688 MiB each; q_b 576 MiB. Re-running nonmoe/bytes.py gives 7942.8 MiB/rank non-MoE and 3390 MiB FP8 savings, i.e. 13.0 ms @273 or 16.2 ms @220 GB/s. (c) The producer's FP8 checkpoint does store q_a/kv_a/q_b/o_proj as block-FP8 and dequantizes on load (model.py:1191-1200). (d) MarlinFP8 needs VLLM_TEST_FORCE_FP8_MARLIN on cap>=89 (scaled_mm/marlin.py:45-54). The merged in_proj's 12576 rows/rank are not a multiple of 64, but prepare_fp8_layer_for_marlin pads (marlin_padded_nk), so that is not a blocker. What stays unproven is the premise that this time is recovered one-for-one. The ms figure assumes cuBLAS BF16 and Marlin FP8 both stream at 220-273 GB/s at M=8. The byte model that 'supports' this does not actually validate (see NMK-2): it omits ~1.77 GiB/rank of drafter bytes and needs near-peak bandwidth with zero overhead. The percentage uplift is therefore an upper bound. The rollout also misses the lowest-risk FP8 target, the BF16 DFlash2 drafter, where errors only move acceptance and not output (see missed).

**Verifier corrected claim:** Non-MoE BF16 projections are ~7.9 GiB/rank per step by header arithmetic. FP8 weight-only (Marlin W8A16, padded shapes OK) would halve 3.31 GiB of that. Whether that becomes 13-16 ms depends on unmeasured achieved bandwidth of cuBLAS BF16 vs Marlin FP8 at M=8/16 on sm_121. The byte model does not establish that these bytes are on the critical path at the claimed share.

**Verifier corrected impact:** Upper bound 13-16 ms/step (+12-16% prose c=1) if both kernels run at the same fraction of peak bandwidth. Realistic range is unknown until nsys shows per-GEMM time. UMA saving of ~3.3 GiB/rank is solid arithmetic.

## NMK-2: Byte model explains the ~116 ms step; the ~15 ms per extra sequence is unexplained, so profile before kernel work

- kind=methodology component=whole verify step (target + DFlash2 draft) on 2x GB10 impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 4)

**Claim:** Step time is flat across acceptance: prose 21.2/2.43 = 8.72 steps/s and structured 67.6/7.84 = 8.62 steps/s at c=1, both ~116 ms. That is consistent with weight-bandwidth bound. The modelled non-MoE bytes (30-38 ms) plus routed MoE for ~58 distinct experts/layer (78.5 ms @220 GB/s) roughly reproduce 116 ms. Structured c=2 uses identical prompts, so identical experts are touched and MoE bytes are about the same as c=1, yet it takes 7.92/60.4 = 131 ms. That leaves ~15 ms of per-extra-sequence cost that weight streaming does not explain. KDA state traffic (NMK-5) accounts for only ~3 ms of it.

**Mechanism:** When experts are shared (structured c=2), the extra per-sequence cost scales with rows, not weights. Candidates: KDA state checkpoint writes (~2.9 ms/seq), MLA/indexer per-row gathers, DFlash2 drafter per-sequence work, rejection sampling over a 154,880-token vocabulary, host-side metadata (async scheduling off, FlashInfer MLA plan() on host every step: flashinfer_mla_sparse_sm90.py:224-276, 368-380), and PyNCCL allreduce payload doubling.

**Evidence:**
- repo/evidence/rebench-20260902T204243Z/bench.txt: prose c=1 21.2 acc 2.43; prose c=2 16.64 acc 2.50 (150 ms/step); structured c=1 67.6 acc 7.84; structured c=2 60.4 acc 7.92 (131 ms/step)
- scratchpad/agents/nonmoe/bytes.py: MoE distinct experts = 288*(1-(1-8/288)^8) = 58.1/layer at 8 tokens, 104.5 at 16; 6.75 MiB/expert/rank NVFP4 incl. scales -> 16475 MiB (8 tok) and 29625 MiB (16 tok) per rank per step
- b12x-A evidence (opt-b12x/evidence/iter-b12x-A-marlin/bench.txt): wave1 prose c=1 18.98 at acc 2.28 = 120 ms/step and wave2 20.85 = 109 ms/step. The 10% wave-to-wave spread happened with 7.3 GiB swap in use (postboot.txt:4)

**Proposed action:** Before any kernel project, take one nsys capture on the Sparks (both ranks) with --cuda-graph-trace=node for 30 steady-state verify steps at c=1 and structured c=2. Attribute time to: cuBLAS BF16 GEMMs, Marlin MoE, KDA recurrent/conv/copies, mHC TileLang, MLA/indexer, NCCL, drafter, sampler, and GPU-idle gaps between graph replays (host overhead). Save the per-bucket table to evidence/ as the ruler for NMK-1/4/5/6. Read-only instrumentation; no recipe change.

**Est. impact:** Enabling finding: determines whether NMK-1's 13-16 ms, NMK-5's ~3 ms/seq and the unexplained ~15 ms/seq are real. Misallocating effort on graph/mHC work (<2 ms each) is the main risk it prevents.

**Validation:** Target: the sum of per-bucket GPU time plus idle gaps equals the wall-clock step (116 ms c=1, 131 ms structured c=2) within 5%. If the GPU is idle for more than 5% of the step, host overhead is a first-class target, which may justify revisiting the async-scheduling rule with evidence.

**Risks:** Profiling perturbs timing a little. It needs an exclusive TP=2 slot, and the Sparks are currently busy.

**Verifier reasoning:** The arithmetic on bench.txt holds: 21.18/2.426 = 8.73 steps/s (114.5 ms); 67.64/7.84 = 8.62 (116 ms); structured c=2 60.41/7.92 = 7.63 (131 ms); prose c=2 16.64/2.50 = 6.66 (150 ms). The 'profile before kernel work' recommendation is sound. The supporting claims have three problems. (1) The byte model omits the DFlash2 drafter. The drafter snapshot header is 2233 MiB BF16 with no own embed/lm_head. Per rank it streams ~1162 MiB (sharded qkv/o/mlp 920 + replicated fc 160 + conv kernel_proj 80), plus a second full lm_head pass: qwen3_dflash2.py:283-287 get_top_k_tokens -> logits_processor.py:264 _apply_head, with lm_head shared per llm_base_proposer.py:1570-1573. That adds ~1767 MiB/rank. c=1 total becomes 7943+1767+16475 MiB = 27.5 GB, which needs >=237 GB/s with zero launch/NCCL/host overhead to hit 116 ms. The '116 ms reproduced' fit is a tuned bandwidth, not validation, and the uniform-routing distinct-expert count likely overcounts. (2) bench_decode.py submits the SAME prompt at temperature 0 to every stream in both phases (bench_decode.py:36, :97). Prose c=2 is also identical-prompt, so the per-extra-sequence cost is +15 ms (structured) to +34 ms (prose; streams diverge slightly: per_stream 17.28/16.66, 16.59/14.85), not just 15 ms. The 104.5-expert 16-token model does not describe either bench cell. (3) The b12x-A 'wave-to-wave 10% spread' is misread. wave1 18.98/2.281 = 8.32 steps/s and wave2 20.85/2.509 = 8.31 steps/s are both ~120 ms/step. The tok/s spread is entirely acceptance, not swap-induced step time.

**Verifier corrected claim:** Step time at c=1 is ~115-120 ms across acceptance and runs. A second identical-prompt stream adds 15-34 ms/step. A header byte model plus the omitted drafter (~1.77 GiB/rank) totals ~27.5 GB/step at c=1 under uniform routing. That would need >=237 GB/s with zero overhead, so the model neither confirms nor refutes weight-bandwidth dominance. Profiling is required before sizing any kernel work.

**Verifier corrected impact:** Enabling measurement. Its value is higher than stated because the MoE/non-MoE/overhead split is currently unknown, not 'reproduced'.

## NMK-3: PR #11 nvidia pack: pin the NVFP4 dense-MLP linear kernel to Marlin (--linear-backend marlin)

- kind=correctness component=layers 0-2 dense MLP (NVFP4 W4A4 in nvidia pack) -> kernels/linear NVFP4 selection impact=4 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The nvidia pack quantizes the dense MLP of layers 0-2 to NVFP4. The LibertAI pack left them BF16, so PR #11 is the first time an NVFP4 *linear* (not MoE) kernel is auto-selected on the Sparks. The selection order tries FlashInferCuteDsl (sm_10x only), then FlashInferCutlassNvFp4 (chosen whenever cutlass_fp4_supported() and cap >= 100), then b12x, then vLLM CUTLASS, then Marlin. FlashInfer CUTLASS FP4 GEMM on sm_121 must JIT-compile because flashinfer-jit-cache is uninstalled in v8, and PR #11 itself records that v11 dies at FlashInfer CUTLASS JIT (nvrtc.h missing) and v12 JIT global-OOM'd spark2. W4A4 also adds activation-FP4 error that Marlin (W4A16, ignores input_scale) avoids.

**Mechanism:** Auto-selection can land on a JIT-only FlashInfer CUTLASS FP4 GEMM, depending on whether the image's _C reports cutlass FP4 support for cap 121. That gives either a worker-init crash (no nvrtc.h) or a JIT memory spike on a UMA box already at ~115/121 GiB. Marlin avoids JIT and gives W4A16 numerics.

**Evidence:**
- nvidia config.json: quantization_config.ignore lists layers.0/1/2.self_attn* but not layers.0-2.mlp; headers show layers.N.mlp.gate_proj U8 [12288,2048] x3 with weight_scale/input_scale
- LibertAI caca4e6 config.json ignore includes '*.mlp.gate_up_proj', '*.mlp.down_proj', '*.mlp.gate_proj', '*.mlp.up_proj', so dense MLP was BF16 there
- v11src/vllm/model_executor/kernels/linear/__init__.py:500-512 (NVFP4 kernel priority); kernels/linear/nvfp4/flashinfer.py:106-119 (FlashInferCutlass supported if cutlass_fp4_supported && has_device_capability(100)), :35-39 (CuteDsl sm_10x only), :312-321 (b12x SM120+)
- repo/docker/Dockerfile.sm121-v8:67 'pip uninstall -q -y flashinfer-jit-cache'
- gh pr diff 11 README text: 'v11 dies at JIT (nvrtc.h missing). v12 ... global-OOM'd spark2 during cudafe++'
- v11src/vllm/engine/arg_utils.py:1596 (--linear-backend); kernels/linear/__init__.py:266-272 (marlin set includes MarlinNvFp4LinearKernel) and 331-359 (falls back per layer type, so BF16 layers are unaffected)

**Proposed action:** Add LINEAR_BACKEND=marlin to recipe.yaml serve.env and pass --linear-backend "$LINEAR_BACKEND" in run.sh, alongside the existing MOE_BACKEND=marlin refuse-guard. Optionally add a run.sh guard mirroring FORCE_UNSAFE_MOE for non-marlin linear backends. Re-render with kit/render.py.

**Est. impact:** Removes a plausible first-boot failure or JIT-OOM mode for the nvidia pin; the probability is unknown without the boot log. Perf: Marlin NVFP4 dense reads 40.5 MiB/rank/layer vs 144 MiB for LibertAI's BF16 dense MLP, so 3 x 103.5 = 310.5 MiB/step less, about 1.1-1.4 ms/step (~+1%) over the LibertAI baseline. Quality: W4A16 instead of W4A4 on the 3 dense layers.

**Validation:** VALIDATE_ONLY=1 ./run.sh shows the flag. On first nvidia boot, grep the worker log for the selected NVFP4 linear kernel (expect Marlin) and confirm there is no JIT/nvcc activity (free -h during init). Then run count 200, thinking-off smoke, and bench_decode prose c=1/c=2 vs the LibertAI numbers.

**Risks:** If _C lacks Marlin FP4 dense for sm_12x, the flag falls back with a warning (the MoE Marlin evidence suggests it exists). Future quantized layers would also be forced to Marlin, which is fine for W-only decode.

**Verifier reasoning:** Confirmed parts. The nvidia pack stores layers 0-2 dense MLP as NVFP4: headers show U8 [12288,2048] gate/up and [4096,6144] down with F8 weight_scale and F32 input_scale/weight_scale_2, x3. LibertAI caca4e6's ignore list includes '*.mlp.gate_proj', '*.mlp.up_proj', '*.mlp.down_proj', '*.mlp.gate_up_proj' and its input_activations is null (W4A16). NVFP4 linear auto-selection order is CuteDsl (sm_10x only), then FlashInferCutlass (cutlass_fp4_supported() && has_device_capability(100), which is >= and so true on 121), then b12x, then CUTLASS, then Marlin (kernels/linear/__init__.py:500-512, 980-1075; nvfp4/flashinfer.py:35-39, 106-119). --linear-backend exists (arg_utils.py:1596) and falls back per layer type (__init__.py:331-359). jit-cache is uninstalled (Dockerfile.sm121-v8:67). Unverified parts. Whether _C's cutlass_scaled_mm_supports_fp4(121) returns True cannot be read without the .so, so the selection outcome is unknown. The cited README 'v11 dies at JIT (nvrtc.h missing)' refers to the flashinfer_cutlass MoE backend (README.md:149), not the dense FP4 GEMM, so the failure mode is inferred rather than observed. All Spark evidence so far is on the LibertAI pack. The perf arithmetic is right: 144 vs 40.5 MiB/rank/layer, x3 = 310.5 MiB.

**Verifier corrected claim:** PR #11 is the first run of an NVFP4 dense linear on the Sparks. Auto-selection prefers FlashInfer CUTLASS FP4 whenever _C reports FP4 support for cap 121. That kernel would need FlashInfer JIT (no jit-cache), which is the same class of failure seen for the MoE backend. Pinning --linear-backend marlin removes that uncertainty and gives W4A16 numerics.

**Verifier corrected impact:** Cheap insurance against an unknown-probability first-boot failure or JIT memory spike. About -310 MiB/step vs LibertAI's BF16 dense (~1.1-1.4 ms).

## NMK-4: mHC TileLang kernels JIT-compile during serving (~5 s each); add a Glm5Next mHC warmup and persist JIT caches

- kind=ops component=vllm/model_executor/kernels/mhc/tilelang.py + tilelang_kernels.py; warmup/deepseek_v4_mhc_warmup.py; run.sh volumes impact=3 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** mhc_pre_big_fuse_with_norm_tilelang takes n_splits as a compile-time argument. For more than 16 tokens, n_splits = compute_num_split(64, 16384, cdiv(T,64)) because DeepGEMM is 'supported' on family 120, giving 12 distinct values on 48 SMs. Only 1, 4 and 8 are compiled at startup. Every prefill size in an unseen n_splits class triggers a ~5 s TileLang compile on both ranks while serving. The existing mHC warmup is gated to model_type=='deepseek_v4' and skips Glm5Next. TileLang, Triton and FlashInfer-autotune caches also live in the container's /root and are lost on every restart.

**Mechanism:** A new n_splits value means a new specialization, which means a synchronous TileLang/NVCC compile inside the forward on each TP rank, stalling the whole TP=2 step. A container restart discards ~/.tilelang, ~/.triton and ~/.cache/vllm, so the stalls recur every boot.

**Evidence:**
- v11src/vllm/model_executor/kernels/mhc/tilelang.py:517-532 (n_splits choice; small path <=16 tokens uses 8/4), 635-654 (big_fuse_with_norm called with n_splits)
- v11src/vllm/model_executor/kernels/mhc/tilelang_kernels.py:202-226 (@tilelang.jit, num_tokens dynamic, n_splits static int)
- v11src/vllm/platforms/cuda.py:716-722 support_deep_gemm includes family 120; utils/deep_gemm.py:110-115
- v11src/vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py:~175-181 (returns unless model_type=='deepseek_v4'; also matches class DeepseekV4DecoderLayer only)
- repo/evidence/rebench-20260902T204243Z/engine.log.tail: 'TileLang JIT compilation during inference: mhc_pre_big_fuse_with_norm_tilelang' at 21:06:21; startup compile of the same kernel took 20:59:22->20:59:27 (~5 s)
- repo/evidence/decision.tsv h1 'c=4 prose TTFT spikes 6s', h3 'c=2 prose TTFT spike 6.6s'; opt-b12x iter-b12x-A-marlin/bench.txt prose c=2 run1 ttft 7.149 s (runs 2-3: 0.37 s)

**Proposed action:** (a) Fork patch: a Glm5Next mHC warmup at kernel_warmup time that calls layer.hc_fused_post_pre(...) with norm_weight for token counts 64*g for every distinct grid g in 1..ceil(max_num_batched_tokens/64), deduplicated by resulting n_splits (12 compiles, ~60 s one-time, before the API goes ready). Or (a') clamp n_splits to {1,2,4,8,16,32} to cap variants at 6; this only re-associates an fp32 split-K sum. (b) In run.sh, mount a persistent host cache dir (under ~/projects/data/, per AGENTS.md) at /root/.tilelang, /root/.triton and /root/.cache/vllm so compiled kernels survive restarts.

**Est. impact:** Removes 5-7 s TTFT stalls at the first occurrence of each prefill-size class per container lifetime (observed at c=2/c=4 first runs), and removes their noise from bench medians. No steady-state tok/s change. Adds about 1 minute to cold boot, or ~0 with a warm persisted cache.

**Validation:** After boot, send prompts of 30, 100, 300, 700 and 1500 tokens. The engine log must show no 'JIT compilation during inference' lines and TTFT must stay under 1 s. Restart the container and confirm the startup mHC compile is skipped (cache hit).

**Risks:** Cache-dir mounts must be version-keyed. A stale cache across image rebuilds is normally invalidated by content hashes, but should be verified. Warmup adds UMA-transient allocations of (max_tokens, 4, 4096) bf16 = 64 MiB at 2048 tokens, which is negligible.

**Verifier reasoning:** n_splits is a static jit arg and num_tokens is dynamic (tilelang_kernels.py:202-226, 242). For more than 16 tokens with DeepGEMM supported, n_splits = compute_num_split(64, 16384, cdiv(T,64)) = min(48//g, 64) (tilelang.py:517-532; tilelang_kernels.py:39-48). support_deep_gemm includes family 120 (platforms/cuda.py:716-722). With max_num_batched_tokens 2048 (g<=32) that yields {48,24,16,12,9,8,6,5,4,3,2,1} = 12 variants. The warmup returns unless model_type == deepseek_v4 (deepseek_v4_mhc_warmup.py:175-176). run.sh mounts only HF cache and the template (run.sh:293-296), so /root caches are lost. The startup compile of the kernel took 20:59:22->20:59:27. Caveat on evidence: the logged in-inference TileLang JIT at 21:06:21 happened during smoke/tool-call probes (progress.md 21:06:20-21:06:27), not in the bench. jit_monitor uses logger.warning_once keyed on the kernel name (utils/jit_monitor.py _handle_jit_event), so later compiles of other n_splits variants are silent. The bench prose c=2 run1 TTFT 6.286 s spike is therefore consistent with, but not proven to be, this mechanism.

**Verifier corrected claim:** Mechanism verified in code: up to 12 n_splits specializations of mhc_pre_big_fuse_with_norm, a Glm5Next warmup gap, and no persisted caches. The attribution of the specific bench TTFT spikes is circumstantial because the JIT monitor logs each kernel name once. The >16-token path also calls DeepGEMM tf32_hc_prenorm_gemm, whose JIT (cache in ~/.cache/vllm/deep_gemm, deep_gemm.py:266-270) is not hooked by jit_monitor, so mounting /root/.cache/vllm matters for it too.

**Verifier corrected impact:** Removes ~5-7 s first-occurrence TTFT stalls per prefill-size class per container lifetime and cleans bench noise. No steady-state tok/s change.

## NMK-5: KDA speculative verify writes 8 full fp32 state checkpoints per layer per sequence (612 MiB/step/seq); replace with commit-by-replay

- kind=perf component=third_party/flash_linear_attention/ops/fused_recurrent.py + kda.py (fla), glm5next kda.py _forward, gdn_attn.py spec state slots impact=3 confidence=3 effort=L needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** In spec verify, fused_recurrent_kda loads the state from slot num_accepted-1 and stores the full [32 heads x 128 x 128] fp32 state (2 MiB/rank) after every one of the 8 tokens into per-token slots. That is 18 MiB of state traffic per KDA layer per sequence per step (612 MiB over 34 layers), and each sequence pins 8 state slots (608 MiB/rank) in the 4.14 GiB KV pin. This per-slot copy is the documented reason 7 speculative tokens only fit 2 sequences.

**Mechanism:** State stores dominate the kernel's memory traffic (16 of 18 MiB/layer/seq). Seven of the eight checkpoints are discarded after rejection sampling. On UMA the checkpoint slots also consume the scarce KV pin.

**Evidence:**
- v11src/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:113-130 (load initial state at num_accepted_tokens-1), 175-185 (store b_h into ssm_state_indices[n, t] for every token t)
- v11src/vllm/third_party/flash_linear_attention/ops/kda.py:55-59 (BK=128, BV=8, num_warps=1), 94 (grid (1,16,N*32))
- v11src/vllm/model_executor/layers/mamba/mamba_utils.py:131-137 (recurrent state dtype fixed float32), 271-294 (conv width 3+num_spec)
- v11src/vllm/v1/attention/backends/gdn_attn.py:266-268 (spec_state_indices = block_table[:, :num_spec+1])
- repo/run.sh:69-73 ('each extra slot is a KDA copy that starves the 4th request at 7')
- scratchpad/agents/nonmoe/bytes.py: 34 x 9 x 2 MiB = 612 MiB/seq/step (2.92 ms @220 GB/s); slots 34 x 8 x 2.24 MiB = 608 MiB/seq vs 152 MiB with 2 slots
- v11src/flashinfer/kda_decode.py:111-112 and kda_kernels/recurrent_kda.py:204,1609-1614,1782: FlashInfer's recurrent_kda is SM100-only with a bf16 state, so it is not a drop-in on sm_121 fp32

**Proposed action:** Kernel+metadata project (fork patch). Keep one committed fp32 state slot plus a small stash of per-token (k, v, raw g, raw beta) for the previous verify (~24.6 KB/token/layer/seq). At the next verify, the Triton kernel first replays the accepted prefix from the stash onto the committed state, writes the committed state once, then runs the 8 new tokens with outputs only and no checkpoint stores. Conv state already uses num_accepted_tokens. Allocate 1 state block (+stash) per request instead of num_spec+1. Cheaper interim experiment, not recommended as default: bf16 recurrent state (halves traffic and slots), with long-context needle and KL gates.

**Est. impact:** State traffic per layer per sequence drops from 18 MiB to about 4 MiB (read + commit). That saves 476 MiB/seq/step: 1.8-2.3 ms at c=1 (~2% of 116 ms) and 3.6-4.5 ms at c=2 (~3% of 150 ms). Replay adds at most 7 serial token updates (~0.2-0.3 ms). Memory: frees ~456 MiB/seq/rank, so ~0.9 GiB at 2 sequences, enough to revisit the run.sh:69-73 decision and admit MAX_NUM_SEQS=4 at DFlash2-7 on the same 4.14 GiB pin (aggregate-throughput gain via MoE amortisation, unquantified).

**Validation:** Unit check on CPU/GPU off-box first: the replay kernel is bit-identical to the current kernel's state at every num_accepted value (1..8) over random inputs. Then on the Sparks: greedy count 200 lossless, structured and prose bench c=1/c=2, KV pool token count in the boot log, and a MAX_NUM_SEQS=4 structured/prose c=4 run.

**Risks:** Touches vLLM mamba spec-decode allocation and prefix-cache semantics ('align' mode copy funcs). Chunked prefill interplay and CUDA-graph address stability of the stash buffers need care. This is the highest-effort item in this dimension.

**Verifier reasoning:** Code verified. The kernel loads the state at num_accepted-1 (fused_recurrent.py:113-130) and stores b_h for every token t into ssm_state_indices[n,t] (175-185). Grid is (1,16,N*32) with BV=8 and num_warps=1 (ops/kda.py:55-59, 94). Spec state indices are block_table[:, :num_spec+1] (gdn_attn.py:266-268). The state dtype is fp32 (mamba_utils.py:131-137). State is 32x128x128x4 B = 2 MiB/rank; 9 x 2 x 34 = 612 MiB/seq/step. The run.sh:69-73 comment does attribute the 4th-request starvation to KDA slots. Impact is an estimate. It assumes the stores are DRAM-bandwidth-bound on the critical path. The kernel is a serial 8-token loop in 1-warp CTAs and may be latency-bound instead. The MAX_NUM_SEQS=4 benefit is unquantified. The per-slot footprint arithmetic (608 vs 152 MiB) is right.

**Verifier corrected impact:** ~1.8-2.3 ms/step at c=1 (~2%) and ~3.6-4.5 ms at c=2 if store-bandwidth-bound. Plus ~0.9 GiB of pool at 2 sequences. Large effort for a small, uncertain step gain. The occupancy gain is the main motive.

## NMK-6: Indexer decode flattening on sm_12x re-reads the kpool K cache 8x per verify step (long-context cost)

- kind=perf component=v1/attention/backends/mla/indexer.py + model_executor/layers/sparse_attn_indexer_kpool.py (DeepGEMM fp8 paged MQA logits) impact=2 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Outside SM100, the DeepGEMM fp8 paged-MQA-logits kernel only supports next_n in {1,2}. With DFlash2-7 (next_n=8) the indexer 'flattens' each verify token into its own next_n=1 row, so all 8 rows of a request stream the full pool-compressed K cache separately. At long context this is 11 layers x 8 rows x (ctx/4 pools x 132 B) per sequence per step.

**Mechanism:** Flattened rows share no K tile loads, so K bytes scale with next_n. L2 may absorb part of it; GB10 L2 reuse across concurrently scheduled rows is unverified.

**Evidence:**
- repo/evidence/rebench-20260902T204243Z/engine.log.tail:70 'DSA indexer decode path: use_flattening=True supports_varlen=False (next_n=8 ...)'
- v11src/vllm/v1/attention/backends/mla/indexer.py:715-735 (use_flattening = not family 100 and next_n not in (1,2); comment cites deepgemm smxx_fp8_fp4_paged_mqa_logits next_n in (1,2))
- v11src/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:788-822 (fp8_fp4_paged_mqa_logits + persistent_topk per row)
- models/glm5next/nvidia/attention.py:280-286 (indexer cache head_dim 128 + 4 B scale = 132 B per pool entry)

**Proposed action:** Partial flattening: group the 8 verify tokens into 4 pseudo-requests of next_n=2, which the sm_12x DeepGEMM kernel supports. This needs the metadata builder to emit seq_lens (B*4, 2) and the kpool topk/expand path to map rows back. It is ~a builder + indexing patch. The longer-term option is a Triton MQA-logits kernel that loads each K tile once for all 8 query rows.

**Est. impact:** Excess K traffic = 11 x 7 x (C/4) x 132 B per sequence per step: 0.17 GB (0.6-0.76 ms) at C=64k and 0.83 GB (3.0-3.8 ms) at C=327,680. next_n=2 pairing removes 4/7 of the excess: ~0.35-0.45 ms at 64k and 1.7-2.2 ms/step/seq at 327k. There is ~0 effect on the short-prompt published prose ruler.

**Validation:** Long-context decode bench (reuse needle-20480 and add ~128k and ~300k prompts), prose continuation at c=1: decode tok/s before/after, and needle accuracy unchanged. nsys: paged_mqa_logits kernel time per layer.

**Risks:** Row-mapping bugs in the kpool expand/tail path would silently corrupt top-k (needle tests catch this). If L2 already absorbs the re-reads, the gain is smaller than estimated.

**Verifier reasoning:** use_flattening = not family 100 and next_n not in (1,2) (indexer.py:717-727). The boot log confirms use_flattening=True next_n=8 (engine.log.tail:70). The indexer entry is head_dim + head_dim/128*4 = 132 B (attention.py:280-286). The arithmetic holds: 11 x 7 x (327680/4) x 132 B = 0.83 GB. However, per-row K at 327k is 81,920 x 132 B = 10.8 MB per layer. The 8 flattened rows run in one launch over the same pages, so L2 reuse could absorb much of the 'excess'. The estimate is an upper bound, as the reviewer partly concedes. It has no effect on the published short-prompt ruler.

**Verifier corrected impact:** Upper bound ~3-3.8 ms/step/seq at 327k context, and less if L2 absorbs the re-reads. ~0 on the published prose cell.

## NMK-7: Remove 4 .contiguous() copies per KDA layer and merge f_b/g_b GEMMs (bit-identical kernel trimming)

- kind=perf component=glm5next/nvidia/kda.py forward/_forward; fla ops/kda.py fused_recurrent_kda impact=2 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** q/k/v come from split() views of the merged conv output, and beta is a slice of the merged in_proj output. fused_recurrent_kda then calls .contiguous() on q, k, v, g and beta, adding 4 copy kernels per KDA layer (g is already contiguous). f_b_proj and g_b_proj are two separate tiny GEMMs ([n,128]x[128,4096] per rank). Across 34 layers that is ~170 extra kernels per verify step inside the FULL graph.

**Mechanism:** Each tiny copy or GEMM in a graph costs a node launch gap plus ramp/tail (~2-5 us) with PDL off on SM12x, which is pure fixed latency.

**Evidence:**
- v11src/vllm/models/glm5next/nvidia/kda.py:327-336 (projected.split -> strided qkv/beta views), 343 (beta = beta_raw.unsqueeze(0)), 344-349 (separate f_b_proj and g_b_proj GEMMs), 492 (qkv_spec.split after conv)
- v11src/vllm/third_party/flash_linear_attention/ops/kda.py:167-173 (q/k/v/g/beta .contiguous())
- v11src/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py:89-95 (pointer math assumes contiguous [T,H,K] rows)

**Proposed action:** (a) Add token-stride arguments (stride_q_tok, stride_k_tok, stride_v_tok, stride_beta_tok) to fused_recurrent_gated_delta_rule_fwd_kernel and drop the .contiguous() calls when the inner dim is contiguous. (b) Stack f_b/g_b weights to [2,128,4096/tp] and run one torch.bmm on stacked [2,n,128] inputs, or add the two as a block-diagonal merged GEMM. Keep o_norm separate. Optional follow-up: fuse causal_conv1d_update into the recurrent kernel prologue (conv output -> registers).

**Est. impact:** Removes ~136 copy kernels + 34 GEMM launches = ~170 kernels/step. At ~2.5-5 us each that is ~0.4-0.9 ms/step (0.4-0.8%). Outputs are bit-identical.

**Validation:** Numerical: torch.equal on layer outputs over recorded inputs, which can be run as a Triton unit test on a single GPU when free. Perf: nsys kernel count per step and bench_decode prose c=1 (expect within noise-plus about +0.5%).

**Risks:** Low. Stride-aware pointer math must handle the cu_seqlens varlen path and the non-spec decode path.

**Verifier reasoning:** q/k/v come from qkv_spec.split (kda.py:492), which gives views with row stride 3*4096. beta is a slice of the merged projection (kda.py:328-343). fused_recurrent_kda calls .contiguous() on q, k, v, g and beta (third_party/fla/ops/kda.py:167-173). g1 comes from the f_b_proj output reshape and is already contiguous, so there are 4 copy kernels per layer. f_b_proj and g_b_proj are separate GEMMs (kda.py:344-349). Kernel pointer math assumes dense [T,H,K] (fused_recurrent.py:89-95). The trims are bit-exact by construction. The ~2.5-5 us/kernel savings figure is an unmeasured estimate.

**Verifier corrected impact:** ~170 fewer graph nodes per step; estimated 0.4-0.9 ms (unmeasured).

## NMK-8: lm_head FP8 per-channel (optional, after NMK-1): -302 MiB/step

- kind=perf component=ParallelLMHead (target) — BF16 [154880,4096], vocab-parallel 605 MiB/rank impact=2 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The target lm_head is read once per verify step: 77,440 x 4096 x 2 B = 605 MiB/rank. FP8 per-row weight-only halves it. Logit error, though, directly affects greedy ties, rejection sampling and the lossless count gate.

**Mechanism:** Same bandwidth argument as NMK-1. lm_head is ~7.6% of non-MoE bytes.

**Evidence:**
- nvidia ckpt header: lm_head.weight BF16 [154880,4096] = 1210 MiB; config.json ignore includes 'lm_head'
- v11src/vllm/models/glm5next/nvidia/model.py:945-950 (ParallelLMHead with the model quant_config, excluded by ignore)
- scratchpad/agents/nonmoe/bytes.py: 302 MiB saved -> 1.1 ms @273 GB/s, 1.44 ms @220 GB/s

**Proposed action:** Only after NMK-1 lands and its gates pass: apply the same W8A16 per-channel method to lm_head behind its own knob. Leave embed_tokens BF16 because it is a gather, not a stream.

**Est. impact:** -1.1 to -1.4 ms/step (~1-1.2% tok/s).

**Validation:** Greedy count 200 must stay lossless. Top-1 agreement vs BF16 over ~10k positions should be >99.9%. Also check acceptance_len and bench prose c=1.

**Risks:** Near-tie flips change greedy outputs, and the lab's greedy/lossless checks may fail for a ~1% gain. Default to not doing it unless the KL is negligible.

**Verifier reasoning:** lm_head is BF16 [154880,4096] (header 1210 MiB) and ignored in the quant config, i.e. 605 MiB/rank. The reviewer missed that the DFlash2 drafter has no lm_head of its own: the snapshot header lists none, and llm_base_proposer.py:1570-1573 shares the target lm_head. The drafter runs a full vocab-parallel head every step via get_top_k_tokens -> _apply_head (qwen3_dflash2.py:283-287; logits_processor.py:241-271). lm_head is therefore streamed twice per step, and FP8 would save ~604 MiB/step, not 302. The quality risk is still dominated by the target pass.

**Verifier corrected claim:** lm_head (605 MiB/rank) is read twice per step, once for target verify and once for the DFlash2 candidate top-k through the shared head. FP8 weight-only would save ~604 MiB/rank/step.

**Verifier corrected impact:** ~2.2-2.9 ms/step (~2-2.5%), about double the reviewer's estimate. The same greedy/tie-flip risks apply.

## NMK-9: CUDA graph status: verify runs FULL graphs; breakable mode is not the bottleneck and Inductor fusions would not help cross-node

- kind=methodology component=compilation/breakable_cudagraph.py, v1/worker/gpu/cudagraph_utils.py, config/vllm.py impact=2 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** Uniform verify batches (8 and 16 tokens) dispatch to FULL CUDA graphs. In FULL mode @eager_break_during_capture is a no-op, so KDA/MLA/indexer run inside one monolithic graph. The 56 eager breaks per forward (34 KDA + 11 unified_mla_attention_with_output + 11 sparse_attn_indexer_kpool) only apply to PIECEWISE (mixed prefill+decode) steps. Breakable mode forces CompilationMode.NONE, so there are no Inductor fusions. However, the only valuable fusion (allreduce+norm) is intra-node-only, and allreduce here is PyNCCL over RoCE.

**Mechanism:** Decode steady state is already graph-replayed. Eager segments only cost host Python time (~0.1-0.3 ms each, so ~5-15 ms) on steps that include prefill chunks, which affects TTFT and co-scheduled streams, not decode tok/s.

**Evidence:**
- repo/evidence/rebench-20260902T204243Z/engine.log.tail:94-95 'Capturing CUDA graphs (PIECEWISE) 0/5' and '(FULL) 2/2'
- v11src/vllm/compilation/breakable_cudagraph.py:100-103 (FULL mode -> run fn directly)
- v11src/vllm/v1/worker/gpu/cudagraph_utils.py:241-266 (uniform decode descriptors rounded to decode_query_len=8)
- v11src/vllm/config/vllm.py:1313-1347 (Glm5Next auto-enables breakable; mode forced NONE)
- eager-break decorators: glm5next kda.py:373, model_executor/layers/attention/mla_attention.py:1247, sparse_attn_indexer_kpool.py:242
- opt-b12x/evidence/iter-b12x-B-flashinfer/engine-spark1.log: "Using ['PYNCCL'] all-reduce backends"
- repo/run.sh:30-31 (measured: VLLM_USE_BREAKABLE_CUDAGRAPH=0 was slower)

**Proposed action:** No change: keep VLLM_USE_BREAKABLE_CUDAGRAPH on auto (AGENTS.md). Do not invest in torch.compile or Inductor fusion work for this model. If mixed-step latency ever matters (c>=2 with frequent new prompts), measure host time of the 56 eager segments with nsys NVTX first.

**Est. impact:** Negative finding that saves effort: graph-mode work has <1 ms/step expected upside on decode.

**Validation:** Already evidenced by the capture log. Optionally confirm in the NMK-2 nsys trace that verify steps show one cudaGraphLaunch per forward.

**Risks:** None for keeping the current setting.

**Verifier reasoning:** engine.log.tail:94-95 shows PIECEWISE 5/5 and FULL 2/2 captures (sizes 8/16 = uniform verify at 1-2 seqs, capture list 1,2,4,8,16 per run.sh:85-97). In FULL runtime mode, eager_break_during_capture runs fn directly (breakable_cudagraph.py:100-103). Glm5Next auto-enables breakable CG and forces CompilationMode.NONE (config/vllm.py:1313-1347). run.sh:30-31 records that the opt-out was slower. The negative conclusion holds.

## NMK-10: Fixed per-step overhead: ~1,600 kernels plus ~90 inter-node PyNCCL allreduces with PDL off (est. 5-8 ms); measure before acting

- kind=perf component=whole target forward (45 layers) + drafter + sampler; PDL gate in Dockerfile.sm121-v8 impact=2 confidence=2 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** Kernel count per layer (FULL graph):
- KDA layer: ~17 non-MoE kernels (2 mHC, in_proj, f_b, g_b, conv_update, 4 copies, recurrent, o_norm, o_proj, allreduce, 2 mHC).
- MLA layer: ~36 (2 mHC, qkv_a, fused qk-norm, q_b, ~19 indexer ops, kv-cache update, W_UK bmm, index convert/clamp/copy, FA2 MLA, W_UV bmm, o_proj, allreduce, 2 mHC).
- MoE block: ~12 per layer.
Total ~1,600 kernels/step. There are 2 PyNCCL allreduces per layer (90/step) over RoCE. PDL is disabled on SM12x, so every node boundary drains the GPU.

**Mechanism:** Per graph node there is ~1-2 us launch gap plus ramp/tail on tiny kernels. Inter-node allreduce of 8x4096 bf16 (64 KB) costs ~20-40 us of NCCL latency each.

**Evidence:**
- repo/docker/Dockerfile.sm121-v8:73-100 (is_arch_support_pdl -> major in (9,10); 'races KDA state kernels')
- v11src/vllm/models/glm5next/nvidia/kda.py:320-371 and 374-653 (per-layer op sequence); model/attention.py:315-411 (indexer ops); layers/mla.py:155-245
- v11src/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:618-883 (indexer decode ops incl. torch.full, persistent_topk, expand)
- opt-b12x/evidence/iter-b12x-B-flashinfer/engine-spark1.log: PYNCCL all-reduce backend for tp:0

**Proposed action:** Measure in the NMK-2 trace: sum of gaps between kernels and NCCL time per step. Only if gaps exceed ~5 ms, pursue the fusion items (NMK-7; merging indexer small ops such as positions.to(int32), torch.full, pool_ids.to(int64) and seq-len derivation into the kpool expand kernel). Keep PDL off; re-enabling it needs a KDA state race analysis, which is not worth ~1 ms.

**Est. impact:** Estimate: 1,600 x 2-3 us = 3.2-4.8 ms plus 90 x 20-40 us = 1.8-3.6 ms, so ~5-8 ms/step (4-7%). Realistically recoverable without comm changes: ~1 ms.

**Validation:** nsys: count kernels per replay, summed inter-kernel idle, and NCCL kernel time per step on both ranks.

**Risks:** Estimates are unmeasured, and the NCCL latency over RoCE could be larger. The comm dimension is owned elsewhere.

**Verifier reasoning:** PDL is gated off on SM12x (Dockerfile.sm121-v8:73-100). Kernel and allreduce counts are rough and unmeasured. The count also omits drafter work: 5 sharded layers (~10 allreduces) plus a second vocab-parallel head, and the verify logits gather over RoCE. The 5-8 ms estimate is a guess with no trace behind it.

**Verifier corrected impact:** Unknown; 5-8 ms is an unsupported estimate. Measure first.

## NMK-11: mHC is fused and graph-resident (<~2.5 ms/step); only a small bf16-fn saving remains

- kind=perf component=model_executor/layers/mhc.py -> kernels/mhc/tilelang.py (mhc_fused_tilelang + mhc_pre_big_fuse_with_norm_tilelang) impact=1 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Per layer, attn-side and ffn-side hc_fused_post_pre each run 2 TileLang kernels in the <=16-token path (post+prenorm-GEMM fused, then Sinkhorn(20 iters)+pre-mix+RMSNorm). That is 4 kernels/layer (180/step), captured in the FULL graph. The mix weights hc_{attn,ffn}_fn are held in fp32 ([24,16384], 1.5 MiB each) even though the checkpoint stores them in BF16, doubling their stream to 135 MiB/step. mhc_pre_big_fuse launches only num_tokens CTAs x 96 threads (8 CTAs at c=1 on 48 SMs).

**Mechanism:** Latency-bound small kernels plus 3 MiB/layer of fp32 weights. Sinkhorn on 4x4 matrices is negligible compute.

**Evidence:**
- v11src/vllm/models/glm5next/nvidia/model.py:391-406 (fn params float32), 443-513 (inter-layer fused post+pre)
- nvidia ckpt header: hc_attn_fn / hc_ffn_fn BF16 [24,16384] x45
- v11src/vllm/model_executor/kernels/mhc/tilelang.py:517-582 (small_fma path, n_splits 4-8), 615-654
- v11src/vllm/model_executor/kernels/mhc/tilelang_kernels.py:242 (T.Kernel(num_tokens, threads=96)), 292-299 (Sinkhorn loop in-kernel)

**Proposed action:** Low priority: (a) keep fn in BF16 and upconvert on load inside mhc_fused_tilelang. This is bit-identical because the values are BF16-exact. (b) Optionally split the layer_input write across several CTAs per token, recomputing the tiny Sinkhorn per CTA, to use more SMs at 8-16 tokens.

**Est. impact:** Model: 180 kernels x ~8-12 us ≈ 1.5-2.2 ms plus 135 MiB (0.5-0.6 ms), so ~2-2.5 ms/step total for mHC. bf16 fn saves 67.5 MiB, 0.25-0.3 ms. The CTA split saves maybe 0.2-0.4 ms. Total ≤0.7 ms (~0.6%).

**Validation:** torch.equal on layer outputs; nsys per-kernel time for mhc_* before/after.

**Risks:** TileLang kernel edits add another JIT variant (see NMK-4). Low value.

**Verifier reasoning:** The fn params are float32 (model.py:391-402) while the checkpoint stores BF16 [24,16384]. mhc_pre_big_fuse uses T.Kernel(num_tokens, threads=96) (tilelang_kernels.py:242) with Sinkhorn in-kernel (292-299). But the kernels assert fn.dtype == float32 in three places (tilelang.py:138, 328, 478), and the >16-token path feeds fn to DeepGEMM tf32_hc_prenorm_gemm. A BF16 fn therefore needs kernel edits, or two copies (fp32 kept for prefill), which forfeits the memory saving. The gain is small either way.

**Verifier corrected impact:** <=0.3 ms from bf16 fn; total <=0.7 ms. Not worth a new TileLang variant.

## NMK-12: MLA sparse attention (FA2, page_size=1) and absorbed BF16 BMMs are cheap and context-flat; deprioritize

- kind=perf component=v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py (v8 patch: FA2 on sm_121), model_executor/layers/attention/mla_attention.py impact=1 confidence=3 effort=L needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Decode MLA treats each verify token as a separate varlen row whose top-k slots form the page table (page_size=1). At contexts of 2048 or more, each row reads ≤2048+3 fp8 latent rows (512 B): 11 layers x 8 rows x 1 MiB ≈ 92 MB/step. The absorbed W_UK_T/W_UV BMMs are BF16, 16 MiB/rank/layer (176 MiB/step). plan() runs host-side every step. Combined with the O(1) KDA state, the decode step is nearly context-independent.

**Mechanism:** Bandwidth: 92 MB + 176 MiB ≈ 0.27 GB/step, about 1.0-1.3 ms at 220-273 GB/s. Compute: 32 heads x 8 rows x 2048 x 1024 x 2 ≈ 1.1 GFLOP/layer, which is small on tensor cores.

**Evidence:**
- v11src/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py:11-28 (design), 208-222 (wrapper fa2 on sm12x), 224-276 (per-step host plan), 464-500 (convert+clamp+copy then run)
- v11src/vllm/model_executor/layers/attention/mla_attention.py:905-919, 977, 1170-1192 (BF16 torch.bmm for W_UK_T/W_UV; fp8 BMM only via aiter on ROCm)
- repo/docker/patch_v8_fp8.py (EFF_CTA_TILE_KV capped at 32 for fp8 on sm12x)
- config: index_topk 2048, kv_lora_rank 512, 64 heads (32/rank)

**Proposed action:** No kernel work now. If NMK-2 shows the FA2 MLA kernel well above ~100 us/layer, consider a verify-aware kernel that shares KV tiles across a request's 8 rows (their top-k sets overlap heavily).

**Est. impact:** Upper bound of recoverable time ~0.5-1 ms/step.

**Validation:** nsys: BatchMLAPagedAttention kernel time per layer at 1k, 32k and 200k context; host time of FlashInferMLASparseSM90Builder.build per step.

**Risks:** None if left as is.

**Verifier reasoning:** The configuration facts match: topk 2048, kv_lora 512, kv_b_proj [32768,512] = 16 MiB/rank/layer from the header, and use of the FA2 sparse backend on sm12x. The byte and FLOP arithmetic is fine. Kernel time is not measured, and deprioritization is reasonable pending NMK-2.

## Open questions
- Actual achieved bandwidth on GB10 for cuBLAS BF16 skinny GEMMs (M=8/16) vs Marlin FP8 W8A16 vs Marlin NVFP4. The byte model assumes 220-273 GB/s; NMK-1's 13-16 ms range depends on it.
- Does the v11 image's vLLM _C report cutlass_scaled_mm_supports_fp4(121)/fp8(121) as true (built as 12.0f vs 12.0a)? This decides which NVFP4 dense-linear kernel auto-selection picks for the nvidia pack (NMK-3) and whether CUTLASS W8A8 FP8 is an alternative to Marlin.
- What is the ~15 ms extra per sequence at structured c=2 (131 ms vs 116 ms at c=1 with the same experts)? KDA state traffic explains ~3 ms. Drafter per-sequence work, the sampler over 154,880 vocab, host metadata (async scheduling off; FlashInfer MLA plan() per step) or MLA/indexer rows?
- How much GPU idle time is there between verify steps from host overhead with --async-scheduling off? If it exceeds ~5% of the step, the AGENTS.md 'leave async scheduling off' rule should be revisited with evidence.
- KL/needle sensitivity of FP8 KDA q/k/v projections. The producer kept KDA BF16 even in its FP8 checkpoint, and there is no public quality data.
- GB10 L2 size/behaviour: does the flattened indexer (8 rows over the same K pages) hit L2, shrinking NMK-6's long-context estimate?
- Actual distinct-expert count per verify step (router logging) to firm up the MoE share. Intra-sequence token correlation likely makes it lower than the independent-draw 58/layer.

## Verifier: missed issues
- The byte model leaves out the DFlash2 drafter. Header of incoai snapshot 7d74cdd: 2233 MiB BF16, no lm_head or embed. Per rank per step it streams ~1162 MiB: sharded q/k/v/o + MLP 920, replicated fc 160 (qwen3_dflash.py:417), conv kernel_projection 80. It also runs a second full pass over the shared target lm_head (605 MiB/rank; llm_base_proposer.py:1570-1573, qwen3_dflash2.py:283-287). That adds ~1.77 GiB/rank = 6.8-8.4 ms at 273-220 GB/s. With it, c=1 bytes are ~27.5 GB, which needs >=237 GB/s with zero overhead to fit 116 ms. So the '116 ms reproduced' claim is not evidence for the weight-bandwidth split.
- A lower-risk FP8 target than KDA/MLA: the drafter's ~1162 MiB/rank of BF16 weights. W8A16 saves ~581 MiB/rank/step (~2.1-2.7 ms). Target outputs stay lossless because only acceptance can move. This belongs as stage 0 of NMK-1's rollout (license is CC BY-NC-ND; runtime quantization is not redistribution, but note it).
- bench_decode.py sends the identical prompt at temperature 0 to every concurrent stream in BOTH phases (bench_decode.py:36, :97). The prose c=2 cell is identical-prompt too. The cost per extra stream is +34 ms/step for prose and +15 ms/step for structured over c=1. The 16-token uniform-routing MoE estimate (104.5 experts, 141 ms) does not apply to any published cell. The c=2 ruler also does not measure diverse-prompt concurrency.
- Evidence misread: b12x-A wave1 18.98 tok/s at acc 2.281 and wave2 20.85 at acc 2.509 are both ~120 ms/step (8.32 vs 8.31 steps/s). The spread is acceptance-driven, not a 10% swap-induced step-time spread.
- jit_monitor logs through logger.warning_once keyed on the kernel name (vllm/utils/jit_monitor.py _handle_jit_event). Only the first in-inference compile of mhc_pre_big_fuse_with_norm_tilelang is logged, and that one was at 21:06:21 during smoke probes (progress.md). Later n_splits compiles, such as a plausible one behind the bench prose c=2 run1 6.286 s TTFT, leave no log line. DeepGEMM tf32_hc_prenorm_gemm JIT on the >16-token mHC path is not hooked at all. Warmup and cache validation must use VLLM verbose JIT mode or timing, not grep. The persistent mount should cover /root/.cache/vllm, which holds the deep_gemm and flashinfer autotune caches.
