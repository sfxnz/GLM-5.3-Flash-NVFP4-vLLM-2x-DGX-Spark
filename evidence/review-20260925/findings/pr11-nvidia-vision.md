# Dimension: pr11-nvidia-vision

## Reviewer summary

I reviewed PR #11 line by line (gh pr diff 11) and checked it against the v11 image source, both checkpoints' safetensors headers (read header-only, 147,661 nvidia tensors / 150,226 LibertAI tensors), configs and the published boot log. The local chat_template.jinja matches the nvidia Hub template and the LibertAI template byte for byte, except the generation-prompt block. So image and video parts render the same way (<|begin_of_image|><|image|><|end_of_image|> and the video equivalent). vLLM already detects the content format as 'openai'. Thinking and tool rendering are unchanged, and the glm45 reasoning parser is an alias of glm47. The template is therefore not a defect. There are real defects elsewhere: (1) The nvidia pack quantizes the dense MLP (layers 0-2) to NVFP4 W4A4. `--moe-backend marlin` does not cover linear layers, so auto-selection picks a FlashInfer CUTLASS (or b12x CuTe-DSL) FP4 GEMM that has to JIT-compile at boot. The v8 image uninstalls flashinfer-jit-cache. This is the same failure class that killed flashinfer_cutlass on v11 (nvrtc.h) and OOM'd spark2 on v12 (cudafe++). (2) The pack ships the MTP layer 45 as unquantized BF16 (13.84 GiB) with no ignore entry, so the advertised `SPEC=mtp` rollback cannot load, and even as BF16 it would not fit UMA. (3) The rollback keeps SERVED_NAME=nvidia and the worker does not receive SNAPSHOT/LIMIT overrides. (4) 'Vision on' is not new: the published LibertAI serve already loaded the tower and profiled one 32,242-token video. `--limit-mm-per-prompt` does not change profiling, so the 'max-size dummy OOMs' rationale has no evidence behind it. The real new UMA risk is the default 4 GiB x2 MM processor cache on the head node. (5) smoke_vision.py sends a 1x1 grayscale (black) JPEG and checks only for non-empty text; it cannot detect broken vision. From the headers, per-rank resident weights change only from 88.57 to 88.27 GiB (-0.30 GiB), so the 4.14 GiB KV pin and the old OOM/swap analysis still hold. The dense-MLP FP4 saves about 1% of step bytes. No KV scales are present in either pack, so fp8 KV stays at scale 1.0 and nothing is silently dropped besides activation input_scales under Marlin. The PR has no Spark receipts at all. It needs a linear-backend pin, an MTP guard, rollback-name and worker-forwarding fixes, a cap on the MM cache, and a real vision correctness gate before merge.

## PR11-1: Dense-MLP NVFP4 W4A4 (layers 0-2) bypasses --moe-backend marlin and auto-selects a JIT FlashInfer FP4 GEMM on sm_121

- kind=correctness component=run.sh / vLLM ModelOptNvFp4LinearMethod kernel selection impact=5 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The nvidia pack stores layers 0-2 mlp.{gate,up,down}_proj as U8 NVFP4 with input_scale, while LibertAI stored them as BF16. The model builds Glm5NextMLP with quant_config (model.py:354-361), and the ignore list has no layers.[0-2].mlp entry, so the layers get ModelOptNvFp4LinearMethod -> init_nvfp4_linear_kernel(). `--moe-backend marlin` applies only to RoutedExperts. With --linear-backend=auto the CUDA priority list is CuteDsl(sm10x only), FlashInferCutlass(cutlass_fp4_supported && cap>=100 && flashinfer), FlashInferB12x(cap>=120 && b12x gemm present, which v11's flashinfer ships), then Cutlass, then Marlin. On GB10 (sm_121) the pick is therefore a FlashInfer CUTLASS or b12x FP4 GEMM, not Marlin. FlashInfer's SM120 CUTLASS FP4 GEMM is a runtime JIT module of 16 translation units (8 tiles x 2 dtypes), and Dockerfile.sm121-v8 uninstalls flashinfer-jit-cache. This is the same nvcc/cudafe++ JIT that failed on v11 (nvrtc.h missing) and global-OOM'd spark2 on v12 after 90.67 GiB of weights. It would now run during profile_run on both ranks. The README also wrongly says 'Marlin still dequantizes' for this pin.

**Mechanism:** Moving from the LibertAI pack (BF16 dense MLP) to the nvidia pack (NVFP4 dense MLP) adds quantized linear layers. These go through a separate kernel registry that the recipe never pinned, so the first forward pass (profile_run) compiles CUTLASS SM120 FP4 GEMMs with nvcc inside a UMA box that has about 18 GiB left.

**Evidence:**
- safetensors headers: nv model.language_model.layers.N.mlp.gate_proj.weight U8 (12288,2048) + weight_scale F8 (12288,256) + weight_scale_2 + input_scale x3 layers; LibertAI same key BF16 (12288,4096)
- /home/sfxnz/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/09b04e5.../config.json quantization_config.ignore: no layers.0-2 mlp entries (only self_attn*)
- v11src/vllm/models/glm5next/nvidia/model.py:353-361 (dense Glm5NextMLP gets quant_config)
- v11src/vllm/model_executor/layers/quantization/modelopt.py:1025-1026,1104-1107 (NVFP4 -> ModelOptNvFp4LinearMethod -> init_nvfp4_linear_kernel())
- v11src/vllm/model_executor/kernels/linear/__init__.py:500-512 (priority list), 980-1060 (auto selection)
- v11src/vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:106-119 (FlashInferCutlass is_supported), 312-321 (B12x is_supported), 154-170 (scaled_fp4_quant + mm_fp4 backend=cutlass)
- v11src/flashinfer/jit/gemm/core.py:256-290 (gen_gemm_sm120_module_cutlass_fp4: 8 tiles x 2 dtypes JIT .cu)
- docker/Dockerfile.sm121-v8:63-67 (pip uninstall flashinfer-jit-cache)
- README.md:149 (v11 JIT nvrtc.h failure, v12 cudafe++ OOM after 90.67 GiB)
- run.sh:340 (only --moe-backend is pinned)

**Proposed action:** Add `--linear-backend marlin` to run.sh through a recipe.yaml env LINEAR_BACKEND=marlin, plus a refuse-guard like MOE_BACKEND's. Forward it to the worker. Assert on first boot that the log contains 'Using MarlinNvFp4LinearKernel for NVFP4 GEMM' on both ranks. Fix README:149 to say that --moe-backend covers experts only and --linear-backend covers the dense MLP. Marlin W4A16 drops the three dense input_scale tensors, which is numerically closer to BF16 than W4A4.

**Est. impact:** Prevents a probable boot failure or UMA OOM on a Spark, which can also take down the co-tenant workload. Perf delta is tiny either way. Dense MLP is 3 x 3 x 12288 x 4096 = 453M params. At FP4 plus 1/16 scale that is about 0.24 GB total, or 0.12 GB per rank per step, which is about 0.45 ms at ~273 GB/s against a ~115 ms verify step (0.4%). Versus the LibertAI BF16 dense MLP (0.936 GiB, so 0.47 GiB per rank), per-rank bytes read per step drop by about 0.30 GiB, roughly 1.2 ms or ~1% faster decode, whichever FP4 kernel runs.

**Validation:** When a slot is exclusive, boot with only this flag added. grep the engine logs on both ranks for 'for NVFP4 GEMM' and 'flashinfer.jit' compile lines. Check free -h before and after profile_run. Then run count_probe (200 integers) and the thinking-off probe.

**Risks:** If _C's cutlass_scaled_mm_supports_fp4 returns False on this build, auto picks b12x (CuTe-DSL compile, flagged for Xid 31 on sm_121) instead of CUTLASS. It is still not Marlin, so the pin is needed either way. --linear-backend marlin only filters kernel lists and falls back with a warning for layer types that have no Marlin kernel; BF16 layers are unaffected.

**Verifier reasoning:** Header scan (my own, header bytes only): nvidia layers.0.mlp.{gate,up}_proj.weight U8 [12288,2048] + weight_scale F8_E4M3 [12288,256] + weight_scale_2 + input_scale; LibertAI caca4e6 layers.0.mlp.gate_proj.weight BF16 [12288,4096]. nvidia config.json quantization_config.ignore has 132 entries; the only layers.0-2 entries are self_attn* (no mlp). LibertAI's ignore had wildcards '*.mlp.gate_proj', '*.mlp.up_proj', '*.mlp.down_proj', which is why the dense MLP was BF16 before. model.py:354-361 builds the dense Glm5NextMLP with quant_config. quant_algo 'NVFP4' selects ModelOptNvFp4LinearMethod (modelopt.py:1024-1028, 1093-1107), which calls init_nvfp4_linear_kernel() with use_a16=False. With linear_backend 'auto', the list at kernels/linear/__init__.py:500-511 is walked in order: CuteDsl (is_device_capability_family(100), so False on sm_121), then FlashInferCutlass (cutlass_fp4_supported() && has_device_capability(100) && flashinfer), then B12x (cap>=120 && b12x gemm), then Cutlass, and only then Marlin. So Marlin is chosen only if every earlier kernel fails. run.sh:249-264 sets no --linear-backend, no VLLM_DISABLED_KERNELS and no MAX_JOBS. FlashInferCutlass.apply_weights calls mm_fp4(backend='cutlass'); gen_gemm_sm120_module_cutlass_fp4 (flashinfer/jit/gemm/core.py:256-300) renders 16 TUs plus 1 base .cu. Dockerfile.sm121-v8:67 uninstalls flashinfer-jit-cache. A new detail strengthens the OOM risk: flashinfer/jit/cpp_ext.py:346-365 passes -j only when MAX_JOBS is set, and the image env (docker image inspect) has no MAX_JOBS. Ninja's default parallelism (nproc+2 = 22 on 20 cores) therefore compiles all 17 CUTLASS TUs at once, next to about 17 GiB 'Available RAM' (engine.log.tail:45). --linear-backend exists in v11 (arg_utils.py:1596; _resolve_backend_kernels at __init__.py:331). MarlinNvFp4LinearKernel is in the non-a16 list, so the pin works. README:149 'Marlin still dequantizes weights ... this pin is a weight-layout + dense-MLP change' is wrong for the dense MLP. Arithmetic re-derived: dense MLP 3x3x12288x4096 = 453M params; nvidia bucket 0.237 GiB vs LibertAI 0.844 GiB. The reviewer's 0.330/0.936 are slightly off, but the delta of 0.607 GiB total (0.30 GiB/rank) is right, which is about 1.2 ms of a ~115 ms step. Whether the final pick is FlashInfer CUTLASS or b12x depends on _C's cutlass_scaled_mm_supports_fp4(121), which I cannot read (.so excluded); either way it is not Marlin.

**Verifier corrected claim:** Confirmed as stated, with one numeric correction: the dense-MLP bytes are 0.237 GiB (nvidia NVFP4) vs 0.844 GiB (LibertAI BF16), not 0.330/0.936. The delta is still 0.30 GiB per rank. Added evidence: the JIT runs with ninja default parallelism (no MAX_JOBS in the image or run.sh), so the 17 CUTLASS SM120 TUs compile concurrently on a node with about 17 GiB free.

**Verifier corrected impact:** Likely boot failure (nvrtc/JIT) or UMA OOM during profile_run on both ranks, the same failure class as the flashinfer_cutlass MoE OOM on spark2. Decode delta from the dense-MLP format is about 1% (0.30 GiB/rank of about 44 GiB/rank of bytes per step), far inside the measured prose noise (c=1 runs 18.75-23.40 tok/s).

## PR11-2: SPEC=mtp rollback is broken on the nvidia pack: MTP layer 45 is unquantized BF16 (13.84 GiB) and not in the ignore list

- kind=correctness component=run.sh SPEC=mtp / glm5next mtp.py / nvidia checkpoint impact=4 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** In the nvidia checkpoint, layers.45 (the MTP block) has 288 routed experts stored as BF16 (gate/up (2048,4096), down (4096,2048)), with no weight_scale, weight_scale_2 or input_scale. The MTP bucket is 13.84 GiB versus 4.14 GiB NVFP4 in LibertAI. Layer 45 does not appear in .quant_summary.txt, so ModelOpt never saw it. The config.json ignore list has no layers.45 entry. Glm5NextMTP builds its decoder layer with vllm_config.quant_config (mtp.py:44 and Glm5NextMoE -> FusedMoEFactory), so the experts and shared_experts expect NVFP4 U8 params with half the K dimension. Loading a BF16 (2048,4096) into a U8 (…,2048) param raises, or at best is skipped through the is_expert_weight/continue path, which leaves torch.empty garbage. Even an unquantized BF16 MTP would add 13.5 GiB / 2 = 6.9 GiB per rank against about 8 GiB MemAvailable. AGENTS.md:3 and README:64/143 still advertise SPEC=mtp as the MTP-4 rollback.

**Mechanism:** ModelOpt exported the HF module tree, which has no MTP module, and copied layer 45 through as BF16. vLLM applies one global NVFP4 config to the MTP draft, so layer 45's expected param layout does not match the checkpoint.

**Evidence:**
- header scan: nv model.language_model.layers.45.mlp.experts.E.{gate,up,down}_proj.weight BF16 x288 (4608 MiB each); LibertAI same keys U8 + weight_scale/_2/input_scale
- bucket totals: mtp(layer45) nv 13.844 GiB vs LibertAI 4.141 GiB; checkpoint total 190.38 vs 181.28 GiB
- nvidia .quant_summary.txt (3970 lines) has no 'layers.45'
- v11src/vllm/models/glm5next/nvidia/mtp.py:44,76-83 (quant_config passed to MTP decoder layer), 385-408 (expert load; unsuccessful expert weights 'continue')
- v11src/vllm/model_executor/layers/quantization/modelopt.py:1429-1523 (NVFP4 MoE param shapes hidden//2 U8)
- AGENTS.md:3, README.md:64,143, run.sh:76-77

**Proposed action:** In run.sh, refuse SPEC=mtp unless MODEL is LibertAIDAI/* (or FORCE_UNSAFE_SPEC=1), with a message that the nvidia MTP head is BF16 and 6.9 GiB per rank. Update AGENTS.md/README so the MTP rollback reads `MODEL=LibertAIDAI/... SNAPSHOT_REV=caca4e6 SPEC=mtp`. State in the README that the ~190.5 GiB includes 13.8 GiB of BF16 MTP that DFlash2 never uses. Add a CI VALIDATE_ONLY test for the refusal.

**Est. impact:** Prevents a boot crash, or a 6.9 GiB/rank UMA OOM if someone patches in BF16 loading, on the documented rollback path. The DFlash2 default is unaffected.

**Validation:** CPU only: header scan as above, plus VALIDATE_ONLY=1 SPEC=mtp ./run.sh must exit 1. Optional later: an exclusive boot with SPEC=mtp to confirm the loader error text.

**Risks:** A future nvidia revision could quantize the MTP head. Key the guard on the header dtype of layers.45.mlp.experts.0.gate_proj.weight rather than on the org name if you want it robust.

**Verifier reasoning:** Header scan: nvidia layers.45.mlp.experts.0.{gate,up}_proj.weight BF16 [2048,4096], down [4096,2048], with no scale tensors; shared_experts BF16 as well. The LibertAI layers.45 experts are U8 with weight_scale/_2/input_scale. Bucket totals are nvidia mtp 13.844 GiB vs LibertAI 4.141 GiB, and file totals 190.38 vs 181.28 GiB. .quant_summary.txt has 0 'layers.45' lines. The config.json ignore list has no 45 entry, and LibertAI's generic wildcards (*.mlp.shared_experts.*) covered its MTP shared experts. mtp.py:44 and :76-83 build Glm5NextDecoderLayer with vllm_config.quant_config. In model.py:216-250 Glm5NextMoE passes quant_config to shared_experts (a Glm5NextMLP with MergedColumnParallelLinear) and to FusedMoEFactory. The MTP prefix 'model.layers.45.mtp_block...' cannot match any ignore pattern, mapped or not, in is_layer_excluded (modelopt.py:139-175). The MLA self_attn is built with quant_config=None (model.py:331), so attention is not the problem, but the experts and shared experts are. The shared_experts gate_proj goes through the stacked mapping (mtp.py:365-382) into a U8 param of half the K width, so a shape assert or copy error is near-certain. Per-rank: 288x3x16 MiB = 13.5 GiB of experts, 6.75 GiB/rank, or about 6.9 GiB/rank with attention and shared parts. AGENTS.md:3 and README:64,143 still advertise SPEC=mtp.

**Verifier corrected impact:** Documented SPEC=mtp rollback cannot boot on the default nvidia pin. Most likely it crashes at load on a shape mismatch in the shared_experts/experts weight loader. Even a BF16-aware loader would add about 6.9 GiB/rank against about 8 GiB MemAvailable. DFlash2 default unaffected.

## PR11-3: Rollback serves LibertAI weights under the nvidia name; SNAPSHOT/LIMIT overrides are not forwarded to the worker

- kind=correctness component=run.sh (generated env block + ssh worker launch) impact=3 confidence=5 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** SERVED_NAME defaults to the literal nvidia/GLM-5.3-Flash-NVFP4 (run.sh:7) and does not follow MODEL. The documented rollback `MODEL=LibertAIDAI/... SNAPSHOT_REV=caca4e6` (AGENTS.md:5, README:149) therefore serves LibertAI weights as 'nvidia/GLM-5.3-Flash-NVFP4'. smoke.sh, bench_decode.py and the probes all default to and assert that name, so receipts from a rollback run are mislabeled and look like nvidia-pack results. Separately, the head computes SNAPSHOT/SNAPSHOT_IN_CONTAINER (run.sh:58-60) and LIMIT_MM_PER_PROMPT (62-64) from optional overrides, but the ssh line (run.sh:386) forwards only MODEL and SNAPSHOT_REV. A user override of SNAPSHOT (for example a local dir) or LIMIT_MM_PER_PROMPT therefore applies only to rank 0, and the two TP ranks can load different weight files or different MM budgets.

**Mechanism:** The generated defaults treat served name, model id and snapshot as independent knobs, and the worker bootstrap only forwards a hand-maintained subset.

**Evidence:**
- run.sh:6-7 (MODEL and SERVED_NAME independent defaults)
- run.sh:58-64 (SNAPSHOT/LIMIT derived locally), run.sh:386 (ssh env list lacks SNAPSHOT, SNAPSHOT_IN_CONTAINER, LIMIT_MM_PER_PROMPT)
- scripts/smoke.sh (asserts model == nvidia/GLM-5.3-Flash-NVFP4), bench_decode.py --model default nvidia
- AGENTS.md:5 rollback line has no SERVED_NAME

**Proposed action:** In recipe.yaml, make SERVED_NAME default to "$MODEL" (render as SERVED_NAME="${SERVED_NAME:-$MODEL}") or add SERVED_NAME to the documented rollback line. Add SNAPSHOT, SNAPSHOT_IN_CONTAINER, LIMIT_MM_PER_PROMPT and LINEAR_BACKEND to the ssh env list. Optionally have each rank log sha256 of config.json and hf_quant_config.json and have doctor.sh compare them.

**Est. impact:** Keeps evidence attribution honest (A/B between packs is the next measurement) and removes a silent cross-rank weight-mismatch mode, which would produce garbage logits without crashing.

**Validation:** VALIDATE_ONLY=1 MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 ./run.sh should print served=LibertAIDAI/...; shellcheck; add a recipe-lint check that every var in the head-side derived block appears in the ssh env string.

**Risks:** Changing the served-name default changes the API model id for the rollback; clients that hardcode the nvidia name must pass --model.

**Verifier reasoning:** run.sh:6-7 has independent literal defaults for MODEL and SERVED_NAME. The rollback lines (run.sh:46-47, AGENTS.md:5, README:149) set only MODEL and SNAPSHOT_REV. smoke.sh:20,42 hardcodes and asserts 'nvidia/GLM-5.3-Flash-NVFP4'. lib.sh:14, bench_decode.py:166 and smoke_vision.py:27 default to the nvidia name, so bench headers from a rollback run would print model=nvidia/... Compare bench.txt line 1 of the published run, which says model=LibertAIDAI/... The ssh env string (run.sh:386) forwards MODEL and SNAPSHOT_REV but not SNAPSHOT, SNAPSHOT_IN_CONTAINER or LIMIT_MM_PER_PROMPT. The worker recomputes them from defaults (run.sh:58-64), so only explicit overrides diverge. Those overrides are new in this PR; SNAPSHOT was hard-coded on main.

**Verifier corrected claim:** Served-name mislabel on rollback: confirmed. Non-forwarding: confirmed, but it matters only when a user explicitly overrides SNAPSHOT/SNAPSHOT_IN_CONTAINER/LIMIT_MM_PER_PROMPT. Defaults are recomputed identically on the worker from MODEL+SNAPSHOT_REV and LANGUAGE_MODEL_ONLY.

**Verifier corrected impact:** Evidence mislabeling on the rollback path (high relevance, since the next measurement is an A/B between packs). Cross-rank mismatch is an edge case that needs an explicit override.

## PR11-4: MM processor cache defaults to 4 GiB x (API + EngineCore) of head-node UMA; now the real vision memory risk

- kind=ops component=run.sh vision args / vLLM MultiModalConfig impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 4)

**Claim:** The PR's vision memory rationale is aimed at the wrong target. The real new UMA exposure once people actually send images is vLLM's mirrored MM processor cache. mm_processor_cache_gb defaults to 4 and is duplicated per API process and engine core, so up to 8 GiB of host RAM on spark1, which is the same LPDDR5X pool as the GPU. The README reports MemAvailable around 8 GiB at steady state. A max-size image (8000 merged tokens = 32,000 patches x 1176 values) is roughly 75-150 MB of pixel_values depending on dtype, so a few dozen distinct large images fill the cache. The KV pin, weights and encoder cache stay fixed while this cache grows until OOM or heavy swap (swappiness 60, 1.3 GiB already swapped on TP0).

**Mechanism:** Processed multimodal kwargs are cached in CPU memory on both the API server and the EngineCore. On GB10 UMA that memory competes directly with the pinned weights and KV pool.

**Evidence:**
- v11src/vllm/config/multimodal.py:152-160 (mm_processor_cache_gb default 4; 'duplicated for each API process and engine core process')
- README.md:151 (MemAvailable stayed about 8 GiB)
- nvidia processor_config.json image_processor.max_image_tokens 8000, patch 14, merge 2, temporal_patch 2
- run.sh:303-308 (only --limit-mm-per-prompt is set)

**Proposed action:** Add `--mm-processor-cache-gb 1` (or 0.5) to run.sh via recipe.yaml (MM_PROCESSOR_CACHE_GB). Optionally add `--mm-processor-cache-type shm` to avoid the second mirror. Document it next to the KV pin. Add a doctor check: free -h after a 20-image unique-hash soak.

**Est. impact:** Caps worst-case vision host-memory growth from about 8 GiB to about 2 GiB, which keeps roughly 6 GiB of the ~8 GiB MemAvailable margin under vision load. Cost: repeated identical images get re-preprocessed (CPU ms, not GPU).

**Validation:** With serve up, send 30 distinct 1792x1792 images one at a time and log free -h and /proc/meminfo MemAvailable and SwapUsed after each, once at the default and once with 1 GiB. Expect a plateau with the cap.

**Risks:** None beyond re-processing latency for repeated images.

**Verifier reasoning:** config/multimodal.py:152-163 does default mm_processor_cache_gb=4 with type 'lru', and the docstring mentions duplication. However, the v11 cache implementation (multimodal/cache.py:85-106, 409-450) shows that in mirrored-LRU mode MultiModalProcessorSenderCache (P0/API) stores only MultiModalProcessorCacheItemMetadata (item size plus prompt_updates). The tensor data lives only in the P1/EngineCore receiver cache. Worst-case tensor residency is therefore about 4 GiB, not 8 GiB. The risk also predates this PR: the published LibertAI serve already had vision active (engine.log.tail:57, 110-111), so the cache existed already. The PR only makes vision use more likely. The proposed '--mm-processor-cache-type shm to avoid the second mirror' is not right. shm replaces the LRU with a shared-memory ring buffer of the same GiB budget in /dev/shm (tmpfs, which is also UMA RAM), so it saves nothing. The per-image size estimate (32,000 patches x 1176 values, 75-150 MB) is plausible, but I did not verify the dtype; mm_device_do_normalize=True may keep pixels un-normalized on CPU.

**Verifier corrected claim:** Pre-existing (not introduced by PR #11): the mirrored LRU MM processor cache can hold up to about 4 GiB of processed pixel tensors in the EngineCore process on spark1 (the API process keeps only size metadata). On GB10 UMA that competes with weights and KV. Capping it with --mm-processor-cache-gb 1 is a cheap safety margin. shm mode is not a memory saving.

**Verifier corrected impact:** Caps worst-case vision host-memory growth from about 4 GiB to about 1 GiB on spark1 (about 3 GiB kept, not the 6 GiB claimed), and only under sustained unique-image traffic.

## PR11-5: 'Vision on by default' is not a behavior change; the mm cap does not alter profiling; README claims lack evidence

- kind=methodology component=README.md / AGENTS.md / run.sh vision block impact=2 confidence=5 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** The published LibertAI serve already ran Glm5NextForConditionalGeneration with the vision tower loaded and profiled. The log shows 'Encoder cache will be initialized with a budget of 32242 tokens, and profiled with 1 video items of the maximum feature size', 'Detected the chat template content format to be openai' and 'Multi-modal warmup completed'. Nothing in main passed --language-model-only. In vLLM, profiling uses max_items_per_batch = max(1, min(encoder_budget // max_toks_per_item, max_num_reqs*items_per_prompt)). encoder_budget is 32242, which is itself the fork-capped video item (_MAX_VIDEO_TOKENS = 30000 plus timestamp tokens), so profiling is 1 video with or without `{"image":4,"video":1}`. The statement 'a max-size image+video dummy OOMs UMA; this is not that dummy' (run.sh:61, README:157, AGENTS:15, vision-smoke.md) has no GLM evidence behind it; the max-size video dummy was already profiled on the published pin without OOM. The cap itself is fine: 4 images x 8000 = 32,000, which is at most 32,242 and fits one encoder batch. README:153 also still says '~400k pool (1.22x)' while the log and README:167 say 372,877 (1.14x).

**Mechanism:** The PR describes an existing capability as a new default and attaches an untested OOM rationale to a flag that does not affect profiling.

**Evidence:**
- evidence/rebench-20260902T204243Z/engine.log.tail:51,57,109-111
- v11src/vllm/multimodal/encoder_budget.py:147-187, 203-220
- v11src/vllm/transformers_utils/processors/glm5next.py:62-73 (_MAX_VIDEO_TOKENS=30000), glm4_1v.py:1044-1053 (video tokens include timestamps)
- git show main:run.sh has no --language-model-only / --limit-mm-per-prompt
- README.md:153 vs engine.log.tail:69 (372,877 tokens, 1.14x)

**Proposed action:** Reword the README and AGENTS text: vision was already on; the new flag only caps per-request items; the encoder budget is 32,242 tokens (about 0.25 GiB per rank at 4096 x BF16). Drop or evidence the 'max-size dummy OOMs' claim. Fix the 1.22x/400k line to 1.14x/372,877. Keep the refuse-guard.

**Est. impact:** Documentation accuracy only; confirms the old OOM/swap analysis stays valid for vision (tower bytes 1,127,254,016 are identical in both packs).

**Validation:** Grep the first nvidia boot log for the same 'Encoder cache ... 32242 tokens ... 1 video' line.

**Risks:** None.

**Verifier reasoning:** engine.log.tail on the LibertAI caca4e6 published boot: 'Encoder cache will be initialized with a budget of 32242 tokens, and profiled with 1 video items of the maximum feature size', 'Detected the chat template content format to be 'openai'', 'Multi-modal warmup completed' and 'Readonly multi-modal warmup completed'. git diff main...HEAD shows mm_args and --language-model-only are new in this PR, so main served with vision on. encoder_budget.py:147-220: max_items_per_batch = max(1, min(budget//max_toks, max_num_reqs*items_per_prompt)), and get_dummy_encoder_profile_inputs profiles the max-token modality (video). glm4_1v.py:986-1053: the supported limits are image None and video 1, and get_mm_max_tokens_per_item only checks mm_counts>0, so {image:4, video:1} leaves both modalities active with the same max tokens. Profiling is unchanged: 1 video item of 32242 tokens (_MAX_VIDEO_TOKENS 30000 + 320x7 + 2). README:153 says '~400k ... 1.22x' while engine.log.tail:69 says 372,877 / 1.14x. The reviewer cited README:167 for the correct figure; it is actually README:22 (and README:20). The 'max-size dummy OOMs' text is at run.sh:61, README:157 and AGENTS.md:15.

**Verifier corrected claim:** As stated. Citation fix: the correct 372,877/1.14x figure is at README.md:22, not README:167.

## PR11-6: smoke_vision.py cannot detect broken vision; replace it with a stdlib-only correctness suite

- kind=quality component=smoke_vision.py / verify skill vision-smoke feature impact=4 confidence=5 effort=M needs_gpu=True
- **verdict: confirmed** (corrected confidence 5)

**Claim:** The fixture named _RED_JPEG_B64 is a 1x1 single-component (grayscale) baseline JPEG (SOF0 ncomp=1) whose pixel decodes to 0, i.e. black. The processor upsamples it to the 16-token minimum, and the gate is only 'HTTP not 400 and content non-empty' (smoke_vision.py:72-80). A server that dropped the placeholder, mis-ordered patches (rot-pos/merge permutation), fed zeros from the ViT or returned NaN features would still pass, because the model always says something. The check also ignores usage.prompt_tokens, video, multi-image ordering and spec-decode behavior on image prompts (log line 51: the DFlash2 drafter gets text-only inputs).

**Mechanism:** A liveness check was presented as a vision gate.

**Evidence:**
- smoke_vision.py:16-20 (fixture), 72-80 (gate)
- CPU decode: SOF0 at byte 71 = ffc0000b08000100010101 -> 8-bit, 1x1, 1 component; PIL mode L, pixel 0
- engine.log.tail:51 (DFlash2 drafter uses text-only inputs for MM prompts)
- nvidia processor_config.json (patch 14, merge 2, min 16 / max 8000 tokens) -> 448x448 image = (448/28)^2 = 256 tokens

**Proposed action:** Keep it stdlib-only. Write PNGs with zlib+struct+crc32 (no PIL) or embed small pre-rendered base64 PNGs. Gates, all greedy, thinking off: (a) token accounting: prompt_tokens(with 448x448 image) - prompt_tokens(same text, no image) must equal 256 + 2 (begin/end tokens), which catches a missing or mangled placeholder expansion. (b) A four-quadrant image (TL red, TR green, BL blue, BR yellow) asking for each quadrant's color: 4/4 exact, which catches patch-order and merge bugs. (c) A pre-rendered PNG of the text '7F3A91' (OCR) with an exact match. (d) Two images (red, blue): 'which is red, first or second?' must answer first, which checks multi-image ordering. (e) Optional: an 8-frame mp4 fixture of digits 1..8 asking for the sequence. (f) Record the vllm:spec_decode acceptance delta for vision requests. Run the same suite on the LibertAI rollback as a parity baseline and commit results under evidence/.

**Est. impact:** Turns vision from 'returns 200' into a regression gate that fails on the realistic silent-breakage modes: placeholder loss, patch permutation, zero or NaN features and ordering. Cost is about 6 requests, under 1 minute.

**Validation:** Run it against the current serve once a slot is free; deliberately break it (for example send the image without a text anchor, or LANGUAGE_MODEL_ONLY=1 with FORCE) and confirm the suite fails.

**Risks:** Answers from small VLM-style prompts can vary in wording, so normalize (lowercase, strip punctuation) and ask for one word. A 256-token delta assumes no resize; compute the expected count from the same smart_resize formula.

**Verifier reasoning:** smoke_vision.py:16-21 fixture decoded with stdlib: 142 bytes, SOF0 at offset 71 = ffc0000b08000100010101, i.e. 8-bit, 1x1, 1 component (grayscale). The DC Huffman table has a single length-1 code for category 9; scan bits 0|010101010 give diff -341, x quant 3 = -1023, /8 + 128 which is about 0, i.e. black, despite the name _RED_JPEG_B64. The gate (lines 59-81) is HTTP error text / non-empty content only; color is explicitly not asserted (docstring lines 4-6). engine.log.tail:51 confirms the DFlash2 drafter uses text-only inputs for MM prompts. The 448x448 -> (448/14)^2/4 = 256 tokens arithmetic is right, and the template emits begin/end image tokens (chat_template.jinja:51), so +258 prompt tokens vs no-image is the correct expectation if smart_resize keeps 448 (divisible by 28).

## PR11-7: LibertAI experts load with a mismatched gate/up global scale under Marlin; the nvidia pack may fix it (verify on first boot)

- kind=quality component=ModelOptNvFp4FusedMoE.process_weights_after_loading (Marlin path) impact=3 confidence=3 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The published LibertAI boot logs 'w1_weight_scale_2 must match w3_weight_scale_2. Accuracy may be affected.' The NVFP4 MoE loader then keeps only w13_weight_scale_2[:,0], the gate scale, for the fused w13 (modelopt.py:1531-1538). So for any expert whose up_proj global scale differs, up_proj is dequantized with the wrong per-tensor scale, a multiplicative error of s2_gate/s2_up on every up_proj weight. That is a latent quality bug in the currently published recipe. ModelOpt 0.47's official export normally ties gate/up amax for fused experts; if the nvidia boot does not print the warning, switching packs fixes it. The dense linear path has the analogous check (modelopt.py:1186-1196, which takes the max of the fused scales).

**Mechanism:** Per-tensor FP4 global scales that differ between the gate and up shards collapse to one scale when the shards are fused.

**Evidence:**
- evidence/rebench-20260902T204243Z/engine.log.tail:39 (warning on LibertAI caca4e6)
- v11src/vllm/model_executor/layers/quantization/modelopt.py:1530-1538 (single gscale for w13), 1186-1205 (linear: max of fused scales)

**Proposed action:** On the first nvidia boot, grep for 'w1_weight_scale_2 must match' and 'global scale for input or weight are different'. If absent, record in evidence/ that the nvidia pack removes a known accuracy hazard. If present, compute offline which experts are affected (a scalar-only read when allowed) and consider rescaling the w3 block scales at load time in a v12 patch.

**Est. impact:** Unknown in magnitude until measured; it affects the up_proj of an unknown subset of 12,096 experts. It is a correctness-quality item rather than speed. The A/B should use greedy count, needle, and a small GPQA or IFBench slice.

**Validation:** First-boot log grep on the nvidia pack. Then a same-session A/B of the LibertAI rollback against nvidia on 50 GPQA-Diamond questions at T=1.0/top_p 0.95 and on the greedy count probe.

**Risks:** Reading the scalar tensors to quantify offline was out of scope under the header-only rule; the warning fires once per process (warning_once), so it cannot count affected experts.

**Verifier reasoning:** engine.log.tail:39 on LibertAI caca4e6: 'w1_weight_scale_2 must match w3_weight_scale_2. Accuracy may be affected.' modelopt.py:1530-1538 keeps only w13_weight_scale_2[:,0] for the fused w13, so the mechanism is confirmed. New CPU evidence that the nvidia pack avoids this for routed experts: the nvidia .quant_summary.txt quantizes experts through ONE fused quantizer per expert ('model.language_model.layers.3.mlp.experts.gate_up_proj_weight_quantizers.0 ... amax=1.56e-01', .1, .2, ...), so gate and up share one amax and their exported weight_scale_2 should be identical. Caveat for the dense MLP: in the same summary, layers.0-2 have SEPARATE gate/up weight_quantizers with different amax in layer 1 (0.156 vs 0.141) and layer 2 (0.219 vs 0.156). ModelOpt export normally ties fused-sibling amax before export, but that is unverified under the header-only rule. If they are not tied, the linear path takes max(weight_scale_2) (modelopt.py:1186-1204; Marlin uses the same weight_global_scale, nvfp4/marlin.py:52), which mis-scales up_proj by up to 1.40x in layer 2. The magnitude of the LibertAI hazard remains unmeasured.

**Verifier corrected claim:** Confirmed mechanism. Stronger than 'may fix': the nvidia .quant_summary.txt shows a single fused gate_up weight quantizer per routed expert, so the w1/w3 global-scale mismatch should disappear for experts on the nvidia pack. For the dense MLP (layers 0-2), gate/up had separate quantizers with different calibration amax, so the first-boot grep must include the linear warning 'global scale for input or weight are different'.

**Verifier corrected impact:** Quality: removes a known up_proj mis-scaling hazard on LibertAI experts (magnitude unknown). Watch item on the nvidia dense MLP (up to 1.40x mis-scale in layer 2 if the export did not tie amax).

## PR11-8: The chat template is multimodal-correct (identical to the nvidia Hub template); lock it with a render-parity CI test

- kind=quality component=chat_template.jinja impact=2 confidence=5 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** A diff of the local template against the nvidia 09b04e5 template shows a single difference: the generation prompt (local lines 255-261 against Hub 255-257). The LibertAI aa28e1f template is byte-identical to the nvidia one. The visible_text macro emits <|begin_of_image|><|image|><|end_of_image|> for image and image_url parts, <|begin_of_video|><|video|><|end_of_video|> for video parts, and passes text through. The token ids match config.json (image 154854, start 154830, end 154831, video 154855/154832/154833). vLLM auto-detects content format 'openai', so parts reach the template instead of being flattened. Tool rendering (<tool_call>name<arg_key>…) matches the glm47 tool parser. glm45 in v11 maps to the same Glm47MoeParserReasoningAdapter. Upstream quirks carried over unchanged: vLLM always passes reasoning_effort (None), so the template always prepends '<|system|>Reasoning Effort: Max', even with thinking off. History reasoning is read only from reasoning_content, not from vLLM's newer 'reasoning' field.

**Mechanism:** Not a defect. The risk is future drift: a Hub template update that changes MM or tool rendering would silently diverge from the vendored copy.

**Evidence:**
- diff local vs nvidia chat_template.jinja: only lines 256-261 differ
- diff LibertAI aa28e1f vs nvidia chat_template.jinja: SAME
- chat_template.jinja:51-53,63-66 (emit_image/emit_video)
- config.json image_token_id 154854 / image_start 154830 / image_end 154831 / video 154855/154832/154833
- engine.log.tail:109 ('openai' content format)
- v11src/vllm/reasoning/__init__.py:55-62 (glm45 == glm47 adapter)
- v11src/vllm/entrypoints/openai/chat_completion/protocol.py:569-583 (reasoning_effort always passed)

**Proposed action:** Add a GPU-free CI test that renders the local and Hub templates (vendor a copy of the Hub file with its sha) with jinja2 for fixtures: text, image+text, 2 images, video, tool call plus tool response, and a multi-turn with reasoning_content. Assert that the outputs are identical except for the enable_thinking=false suffix '<think></think>'.

**Est. impact:** Protects vision and tool correctness against template drift at zero runtime cost.

**Validation:** CI job runs pure jinja2 (transformers-style environment: trim_blocks and lstrip_blocks false, tojson filter).

**Risks:** Needs the same jinja2 extensions/filters HF uses (tojson with ensure_ascii); emulate them in the test.

**Verifier reasoning:** diff of the local template vs nvidia 09b04e5: only lines 256-261 differ (the generation prompt with enable_thinking). diff -q LibertAI aa28e1f vs nvidia: identical. Lines 51-52 and 63-66 emit <|begin_of_image|><|image|><|end_of_image|> for image/image_url and the video equivalent for video/video_url. engine.log.tail:110 shows content format 'openai'. reasoning/__init__.py: glm45 and glm47 both map to Glm47MoeParserReasoningAdapter. One correction: 'Reasoning Effort: Max' is prepended because chat_template.jinja:2 defaults effective_reasoning_effort to 'max' whenever reasoning_effort is undefined or not low/high. That comes from the template, not from vLLM passing None.

**Verifier corrected claim:** As stated, except that the always-on '<|system|>Reasoning Effort: Max' comes from the template default at chat_template.jinja:2-3 ('max' unless reasoning_effort is low/high), not from vLLM passing None.

## PR11-9: Weight bytes and memory: nvidia is -0.30 GiB per rank resident, but reads +9.1 GiB more at load (unused BF16 MTP)

- kind=perf component=checkpoint / loader / UMA budget impact=2 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** From headers: nvidia totals 190.38 GiB against LibertAI's 181.28 GiB. Resident without MTP (DFlash2 default): nvidia 176.54 GiB against 177.14 GiB, i.e. dense MLP 0.330 vs 0.936 GiB, with experts 159.469, self_attn about 11.29, shared 1.969, embed and lm_head 1.182 each, and visual 1.050 all equal. Per rank that is about 88.27 against 88.57 GiB, so the nvidia pack frees about 0.30 GiB per rank. The 4.14 GiB KV pin and the published '90.67 GiB loaded' figure remain valid to within about 0.3 GiB, and the old OOM/swap analysis stands. However, the loader reads and discards the 13.84 GiB BF16 MTP (model.py:818-820 skips after the iterator has materialized it). Each rank streams 190.4 GiB, which extends load from 629 s to about 629 x 190.4/181.3 = 661 s and adds page-cache churn on a box with swappiness 60 that already had 1.3 GiB of TP0 swapped.

**Mechanism:** The safetensors iterator reads every tensor before the model decides to skip the spec layer, and the unused MTP head is 3.3x bigger in the nvidia pack.

**Evidence:**
- header scan buckets (GiB) nv/lib: routed 159.469/159.469, mtp 13.844/4.141, self_attn 11.289/11.282, shared 1.969/1.969, dense 0.330/0.936, embed 1.182/1.182, lm_head 1.182/1.182, visual 1.050/1.050
- engine.log.tail:38 (629.20 s load), :52 (90.67 GiB), :68 (free 109.0 GiB, KV 4.14 GiB)
- v11src/vllm/models/glm5next/nvidia/model.py:818-820
- README.md:151,165 (swap, MemAvailable)

**Proposed action:** No flag change is needed for residency. Optionally: (a) run maybe_drop_caches before boot (already done where sudo exists) and set vm.swappiness=10 on both Sparks, a host change the user must approve; (b) in a later image layer, filter layer-45 names before get_tensor when speculative method != mtp, saving about 13.8 GiB of reads per rank.

**Est. impact:** Load time about +32 s per boot, from reading 190.4 instead of 181.3 GiB; resident memory -0.30 GiB per rank (a slight improvement). The optional filter would bring load to about 190.4-13.8 = 176.6 GiB read, roughly 610 s.

**Validation:** On the first nvidia boot, compare the 'Loading weights took' and 'Model loading took X GiB' lines with the LibertAI log, and check free -h after ready.

**Risks:** The exact per-rank split depends on which params are replicated (router gates, norms, hc params); the ±0.1 GiB estimate assumes a clean TP split.

**Verifier reasoning:** My bucket scan: nvidia 190.380 GiB total (mtp 13.844, routed 159.469, self_attn 11.289, shared 1.969, dense 0.237, embed 1.182, lm_head 1.182, visual 1.050, other 0.159); LibertAI 181.277 (mtp 4.141, dense 0.844, self_attn 11.282, others equal). Resident excluding MTP: 176.54 vs 177.14 GiB, delta 0.60 total, 0.30/rank. model.py:818-820 skips spec layers only after the iterator yields them. Layer-45 tensors sit in shards 1-3 (889 keys per the index), mixed with other layers, so the whole files are read. 629.20 s x 190.38/181.28 = 660.8 s, about +32 s, assuming linear bandwidth scaling. The reviewer's dense bucket (0.330/0.936) is off by about 0.09 GiB in each, but the conclusion is unchanged.

**Verifier corrected claim:** As stated; the dense-MLP buckets are 0.237/0.844 GiB rather than 0.330/0.936, with the same 0.30 GiB/rank delta.

## PR11-10: No Spark receipts; the PR changes several knobs at once and republishes LibertAI numbers under the nvidia default

- kind=methodology component=PR process / evidence impact=3 confidence=5 effort=M needs_gpu=True
- **verdict: confirmed** (corrected confidence 5)

**Claim:** The PR flips MODEL and SNAPSHOT, adds --limit-mm-per-prompt, changes ORCHESTRATE fallback behavior (a lone head now exits), changes the bench default phase, and edits CI/lint, all with zero boot evidence. The README decode table (21.2 / 16.6) is LibertAI data under an nvidia-default README. AGENTS.md requires changing one knob at a time against bench_decode.py with trail/decision rows. Findings PR11-1 (linear kernel JIT) and PR11-2 (MTP) show the swap is not layout-neutral.

**Mechanism:** Process gap: a clone-shape CI pass is being treated as a recipe change.

**Evidence:**
- gh pr view 11 body: 'Unmeasured on the nvidia pack', 'Do not merge without a later exclusive TP=2 Spark slot'
- AGENTS.md working rules ('Change one knob at a time', record in evidence/trail.tsv, decision.tsv)
- no evidence/iter-* for 09b04e5 in the worktree

**Proposed action:** Split the PR: (A) docs, lint, bench default and the smoke-suite code (mergeable now); (B) the MODEL pin together with --linear-backend marlin, the SPEC=mtp guard and the MM cache cap, merged only with evidence/iter-nvidia-pin/ containing run.log greps (NVFP4 GEMM kernel, 'MARLIN' MoE backend, loading GiB and s, encoder budget, absence of the w1/w3 warning), smoke, count 200, thinking-off, tool-call probe, the vision suite from PR11-6, and the prose bench c=1,2 against a same-session LibertAI ruler.

**Est. impact:** Avoids publishing an unbootable default and gives a clean A/B for the quality claim (NVIDIA reports GPQA 0.9211 NVFP4 vs 0.9217 BF16 on GB200 W4A4; on Marlin W4A16 it should be at least as close).

**Validation:** Two boots in one exclusive slot (LibertAI ruler, then nvidia with the fixes) using the unchanged bench command.

**Risks:** Needs an exclusive slot; the Sparks are currently shared with the other workload.

**Verifier reasoning:** git diff main...HEAD confirms several changes in one PR: the MODEL/SNAPSHOT pin, mm_args, the ORCHESTRATE lone-head exit, the bench default phase 'both' to 'prose', recipe-lint/CI edits and the README table rows. No evidence/iter-* exists for 09b04e5. Correction: the README does disclose the provenance. README.md:11 says 'These cells were measured on the LibertAIDAI pin on this image; the nvidia pack is unmeasured on Sparks', and the recipe.yaml conditions say the same, so 'republishes LibertAI numbers under the nvidia default' is overstated. Additional methodology point: the published prose cell is a 3-run median whose runs span 18.75-23.40 tok/s (bench.txt), about ±11%, so the ~1% dense-MLP effect and similar knobs cannot be resolved with this bench.

**Verifier corrected claim:** Process gap confirmed: multiple knobs, no Spark receipts for the nvidia pin. The README does label the decode table as LibertAI-pin data, so the numbers are not presented as nvidia results. The remaining issue is that an unbooted default is being published.

## PR11-11: Generation defaults: vLLM applies T=1.0/top_p=0.95 from generation_config; published tok/s are greedy only

- kind=quality component=generation_config.json / vLLM --generation-config auto impact=2 confidence=4 effort=S needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** Both packs ship temperature 1.0 and top_p 0.95; the nvidia pack also sets do_sample true, which vLLM ignores. vLLM logs that these override its defaults, so any client that omits temperature samples at T=1.0. That matches NVIDIA's eval settings, which is good for quality. However, spec-decode acceptance under rejection sampling at T=1.0 is lower than greedy, and every published decode cell is greedy (temperature 0). Real default-client decode is therefore below the table.

**Mechanism:** The published numbers measure only the deterministic regime, while the served default is stochastic.

**Evidence:**
- nvidia generation_config.json (do_sample true, temperature 1.0, top_p 0.95)
- engine.log.tail:112 ('Default vLLM sampling parameters have been overridden ... temperature 1.0, top_p 0.95')
- nvidia README evaluation note: benchmarked with temperature=1.0, top_p=0.95
- bench_decode.py (greedy)

**Proposed action:** Keep --generation-config auto for quality. Add a non-published bench cell `--temperature 1.0 --top-p 0.95` (prose, c=1) recording acceptance_len, and state in the README which regime the table uses.

**Est. impact:** Informational. It sets honest expectations: if acceptance drops from 2.43 to about 2.0 at T=1, decode falls to about 2.0/2.43 x 21.2 ≈ 17.5 tok/s, an estimate not yet measured.

**Validation:** One extra bench run in the same slot as PR11-10.

**Risks:** None.

**Verifier reasoning:** nvidia generation_config.json has do_sample true, temperature 1.0, top_p 0.95; LibertAI has the same without do_sample. engine.log.tail:112 shows the override warning. bench_decode.py:36 sends temperature 0. The nvidia README:205 says 'Benchmarked with temperature=1.0, top_p=0.95'. The 17.5 tok/s figure is an unmeasured estimate, as labeled.

## PR11-12: LANGUAGE_MODEL_ONLY accepts non-0/1 values inconsistently; CI does not run render --check

- kind=correctness component=run.sh vision guard / .github/workflows/ci.yml impact=1 confidence=5 effort=S needs_gpu=False
- **verdict: plausible** (corrected confidence 5)

**Claim:** The refuse-guard tests `!= 0` (run.sh:119), mm_args tests `== "1"` (304), and the LIMIT default requires `== "0"` (62). LANGUAGE_MODEL_ONLY=true FORCE_UNSAFE_VISION=1 therefore passes neither --language-model-only nor the cap, which means vision on with no per-prompt limit, the opposite of intent. Separately, recipe.yaml:2 says 'CI runs python3 kit/render.py --check', but ci.yml has no such step (pre-existing; the check passes today, rc=0 locally).

**Mechanism:** Tri-state string comparisons on a boolean env var.

**Evidence:**
- run.sh:62,119,304
- recipe.yaml:2 vs .github/workflows/ci.yml (no render --check)
- python3 kit/render.py --check -> rc=0

**Proposed action:** Normalize: `case $LANGUAGE_MODEL_ONLY in 0|1) ;; *) echo 'want 0 or 1'; exit 1;; esac`. Add a render --check step to the CI recipe-lint job. Add CI VALIDATE_ONLY cases for SPEC=mtp with the nvidia MODEL (PR11-2) and a LINEAR_BACKEND pin (PR11-1).

**Est. impact:** Minor robustness; prevents a misconfigured uncapped vision serve.

**Validation:** VALIDATE_ONLY=1 LANGUAGE_MODEL_ONLY=true FORCE_UNSAFE_VISION=1 ./run.sh must exit 1.

**Risks:** None.

**Verifier reasoning:** First half confirmed: run.sh:62 requires == "0", run.sh:119 tests != 0, and run.sh:304 tests == "1". So LANGUAGE_MODEL_ONLY=true FORCE_UNSAFE_VISION=1 passes neither --language-model-only nor --limit-mm-per-prompt, and vLLM runs with its default limits (image unlimited per glm4_1v.py:986-987, video 1). Second half refuted: .github/workflows/render-check.yml runs 'python3 kit/render.py --check' on pull_request and on push to main, so recipe.yaml:2's statement that CI runs it is true, just in a separate workflow from ci.yml.

**Verifier corrected claim:** The LANGUAGE_MODEL_ONLY tri-state inconsistency is real: a non-0/1 value with FORCE_UNSAFE_VISION=1 serves vision uncapped. The claim that CI does not run render --check is wrong: render-check.yml does.

**Verifier corrected impact:** Minor robustness only.

## PR11-13: Vision + DFlash2 + M-RoPE path has never been exercised; the drafter runs text-only on image prompts

- kind=perf component=DFlash2 speculator with Glm4v-style M-RoPE impact=2 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The drafter logs that it cannot take multimodal embeddings, so draft inputs for image positions are placeholder-token embeddings. The target receives M-RoPE positions from rope_state, while the speculator path uses the 1-D input_batch.positions (model_states/default.py:113 vs 181). No evidence directory contains any image request, so the combination is untested on this stack for both correctness (it should stay lossless) and speed, where acceptance after long image spans is unknown.

**Mechanism:** The DFlash2 backport (v9-v11) was validated only on text prompts.

**Evidence:**
- engine.log.tail:51
- v11src/vllm/v1/worker/gpu/model_states/default.py:104-122,181
- grep of evidence/: no image_url requests

**Proposed action:** Include the PR11-6 vision suite in the first boot, run it under DFlash2-7, and record per-request acceptance (the /metrics delta around each request) and decode tok/s for a 256-token description answer. If acceptance on vision prompts is under about 1.3, document that VL requests decode near target-only speed.

**Est. impact:** Unknown. At the measured ~115 ms verify step, acceptance of 1.3 against 2.43 prose means about 11 tok/s against 21 tok/s on image-conditioned answers (estimate).

**Validation:** Same slot as PR11-10; compare greedy vision answers between SPEC=dflash2 and a no-spec boot only if acceptance looks pathological.

**Risks:** An M-RoPE/1-D position mismatch could in principle trip an assertion in the drafter on long image prompts; that is exactly what the test will reveal.

**Verifier reasoning:** engine.log.tail:51 confirms text-only draft inputs for MM prompts. model_states/default.py:104-116 produces M-RoPE positions via rope_state for the target (Glm5NextForConditionalGeneration subclasses Glm4vForConditionalGeneration, model.py:1031), while attn metadata uses input_batch.positions (line ~181). The reviewer did not account for two things: (a) the target LLM is NoPE MLA (qk_rope_head_dim 0, mla_use_nope true) plus KDA with no RoPE, so M-RoPE positions affect only the DSA indexer's RoPE; (b) DFlash2 conditions on target aux hidden states from layers (6, 15, 25, 34, 43) (engine.log.tail:42), which do carry image information, so 'text-only' applies to draft input embeddings only. No evidence file contains an image request. The acceptance collapse to about 1.3 is speculation.

**Verifier corrected claim:** Untested combination: DFlash2 gets placeholder-token embeddings for image spans but still receives target aux hidden states that encode the image. Losslessness is guaranteed by verification; the acceptance and speed impact on image-conditioned answers is unknown and may be modest.

**Verifier corrected impact:** Unknown; the 11 tok/s figure is a worst-case guess, not an expectation.

## Open questions
- Does torch.ops._C.cutlass_scaled_mm_supports_fp4(121) return True on this build? It decides whether the dense-MLP auto-select lands on FlashInferCutlass (nvcc JIT) or FlashInferB12x (CuTe-DSL). Verify offline with cuobjdump on the image's _C.abi3.so or with the first boot log line 'Using ... for NVFP4 GEMM'. The --linear-backend marlin pin is recommended either way.
- Do the nvidia pack's per-expert gate/up weight_scale_2 scalars match, so the 'w1_weight_scale_2 must match w3_weight_scale_2' warning seen on LibertAI disappears? Reading the 12,096x2 four-byte scalars was outside the header-only rule; a first-boot log grep answers it.
- Were the nvidia pack's mHC hc_attn_base/hc_attn_scale/hc_ffn_* values rounded from F32 to BF16 by the ModelOpt export? LibertAI stores them as F32, and the model params are float32. Does that measurably change greedy outputs (count probe, needle) against the rollback?
- Is the GLM-5.3 video prompt layout that vLLM's inherited Glm4v _get_prompt_updates produces (per-frame image blocks plus 'X.X seconds' timestamps) the same one GLM-5.3-Flash was trained with? This needs the zai-org reference processor or an SGLang comparison; the recipe caps video at 1 per prompt but does not validate it.
- What dtype does the Glm5Next processor emit for pixel_values (fp32 vs bf16)? This sets the per-image MM-cache footprint (about 75 vs 150 MB for a max image) used to size --mm-processor-cache-gb.
- Can the loader skip reading layers.45 tensors when the speculative method is not mtp (a safetensors key filter before get_tensor), and is that worth a v12 image layer for about 13.8 GiB less I/O per rank per boot?

## Verifier: missed issues
- Ninja parallelism amplifies the PR11-1 JIT risk. flashinfer/jit/cpp_ext.py:346-365 passes -j only when MAX_JOBS is set. Neither the glm53-sm121-v11 image env (docker image inspect: no MAX_JOBS) nor run.sh:249-264 sets it. The 17-TU SM120 CUTLASS FP4 GEMM module would therefore compile with ninja's default (nproc+2 = 22) concurrent cudafe++/cicc jobs on a node with about 17 GiB 'Available RAM' (engine.log.tail:45). If someone deliberately tests a FlashInfer linear path, set MAX_JOBS=1-2 as well; the primary fix remains --linear-backend marlin.
- Dense-MLP gate/up global-scale risk on the nvidia pack. In .quant_summary.txt, layers.0-2 have SEPARATE gate_proj/up_proj weight_quantizers with different calibration amax: layer 1 gate 1.56e-01 vs up 1.41e-01, layer 2 gate 2.19e-01 vs up 1.56e-01. vLLM's linear path collapses fused shards to max(weight_scale_2) (modelopt.py:1186-1204), and the Marlin linear kernel uses the same weight_global_scale (nvfp4/marlin.py:52). If ModelOpt's export did not tie these (unverifiable under the header-only rule), up_proj in layer 2 would be mis-scaled by up to 1.40x. The first-boot gate must grep 'global scale for input or weight are different' in addition to the MoE w1/w3 warning. By contrast, routed experts use one fused gate_up_proj_weight_quantizers.E per expert (tied).
- The nvidia config.json quantization_config carries kv_cache_scheme {num_bits 8, type float, dynamic false}, but no k_scale/v_scale tensors exist. Glm5NextMLAAttention is built with quant_config=None (model.py:331), so the fp8 KV scale stays 1.0 exactly as on LibertAI. This confirms the reviewer's 'nothing silently dropped' claim from a second angle; no action is needed, but nobody should expect calibrated KV scales from the 'kv_fp8_cast' label.
- Published prose methodology is too noisy for the effects under discussion. rebench bench.txt prose c=1 runs are 23.40 / 18.75 / 21.18 tok/s (about ±11%) with median completion 98 tokens, yet README.md:11 and recipe.yaml conditions say '200 completion tokens'. The PR also changes bench_decode.py's default --phase from 'both' to 'prose', which drops the structured occupancy/acceptance ruler (acc 7.84) from default runs. Any A/B of the nvidia pin (a ~1% dense-MLP effect, W4A16 vs W4A4 quality) needs more runs or longer outputs, and should keep the structured phase as a secondary regression gate.
- The rollback guard for PR11-2 should key on the SPEC and checkpoint layout, not only on the org. The layer-45 BF16 tensors sit in shards 1-3 (889 keys per model.safetensors.index.json), so a VALIDATE_ONLY CPU check can read one safetensors header (e.g. layers.45.mlp.experts.0.gate_proj.weight dtype) without touching tensor data.
