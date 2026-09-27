# Dimension: system-uma-comm

## Reviewer summary

System-level review of the 2x GB10 TP=2 serve. The main finding: decode is already close to the LPDDR5X bandwidth roofline. I added up per-rank bytes read per verify step from the safetensors headers: about 16 GiB of NVFP4 routed experts (upper bound, assuming independent routing), 7.6 GiB of BF16 attention, shared-expert, lm_head and indexer weights, and 1.7 GiB for the DFlash2 draft, about 25 GiB in total. Dividing by the measured ~115 ms/step gives roughly 200-236 GB/s effective, which is 74-86% of the 273 GB/s peak. The comparison of DFlash2-5 with DFlash2-7 (about 108 vs 115 ms/step at the same acceptance-independent cost) is consistent with this. So NCCL, CPU and the scheduler are second-order. By estimate, NCCL is about 102 small all-reduces plus 2 logits gathers per step at the ~45-60 us small-message RoCE latency others have published for GB10, about 6-9 ms (5-8%). Async scheduling is already on by default and CUDA graphs are FULL for the verify and draft steps. The largest system levers are therefore: (1) stop UMA reclaim and swap from stealing bandwidth and stalling the serving processes; (2) remove bytes per step and free UMA at the same time, for example FP8 for the BF16 attention, shared-expert, lm_head and draft weights (about 4.3 GiB/rank resident and ~18% of per-step bytes); (3) a faster cross-node all-reduce (b12x one-shot RoCE all-reduce; LL protocol), worth about 3-5%. Several recipe beliefs are not backed by the evidence. The '5.0 GiB KV pin slows decode 20%' claim has no receipts, and KV bandwidth cannot explain it with ~100-token bench prompts. It is most likely swap/reclaim pressure: vm.swappiness=60, a 16 GiB swapfile, min_free_kbytes=45 MB, 6 GiB swapped on spark1 after the bench, and TiB-scale cumulative swap-out on both nodes. The 'leave --async-scheduling off' A/B was a no-op because vLLM turns it on automatically for DFlash with the mp executor. PR #12's 16 GiB MemAvailable wait-abort would kill healthy boots: the head was already at 17.16 GiB before the draft, KV and graphs were allocated. Orchestration gaps: run.sh cannot pass any NCCL_*/VLLM_* env into the containers (so the NCCL experiments cannot be run), LIMIT_MM_PER_PROMPT, HF_CACHE and SNAPSHOT are not forwarded to spark2, the v11 image was built separately on each node (patch-layer digests differ), and VALIDATE_ONLY checks nothing across nodes. Also, the SPEC=mtp rollback is probably broken on the nvidia pin: MTP layer 45 is 13.84 GiB of BF16 and is not in exclude_modules. PP=2 and EP are both worse than TP=2 for decode on this pair (PP about 2x slower step latency and not supported for Glm5Next; EP has expert load imbalance and all2all over RoCE). All GPU-dependent impacts here are estimates to validate one knob at a time; all numbers marked 'measured' come from the evidence logs.

## SYS-1: Decode is at ~74-86% of the LPDDR5X bandwidth roofline; system knobs are second-order

- kind=methodology component=whole step / UMA bandwidth impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** At c=1 each TP rank reads about 25 GiB of weights per verify step (k+1=8 tokens). At the measured ~115 ms/step that is ~200-236 GB/s, 74-86% of the ~273 GB/s peak. Step time is therefore set by bytes read, not by communication or CPU. Every system proposal should be ranked by how many bytes/step it removes or how much bandwidth contention it prevents.

**Mechanism:** Per rank, c=1: distinct experts per MoE layer E[D] = 288*(1-(1-8/288)^8) = 58.1 (independent-routing upper bound). 58.1 x 42 layers x 13.50 MiB/expert / 2 (TP) = 16.1 GiB. Non-expert BF16/2 (+ replicated indexer, mHC and router) = 7.56 GiB. Draft 1.09 GiB plus a second lm_head pass 0.59 GiB = 1.68 GiB. Total ~25.3 GiB = 27.2 GB in 115 ms = 236 GB/s. With correlated routing (D~45) it is ~23 GB = 203 GB/s. DFlash2-5 vs DFlash2-7 costs ~7 ms for 2 extra verify tokens (~3.7 ms/token), consistent with the expert-read term. Structured and prose have the same step time despite 3x different acceptance, so acceptance does not change per-step cost.

**Evidence:**
- safetensors headers of nvidia 09b04e5 (parsed header bytes only, scratchpad/agents/sys_st_bytes.py): routed_experts 159.469 GiB (U8 141.75 + F8 scales 17.72), kda_attn 8.729 BF16, mla_attn 2.406 BF16, shared_experts 1.969 BF16, lm_head 1.182, embed 1.182, visual 1.050, dense_mlp 0.237 NVFP4, indexer 0.153, router 0.092, mhc 0.066, MTP layer45 13.844 BF16; total 190.38 GiB
- draft config.json: 5 Qwen3 layers, hidden 4096, intermediate 12288, no own lm_head (model.safetensors 2.18 GiB per engine-rank1.log.tail:40)
- evidence/rebench-20260902T204243Z/bench.txt: prose c=1 21.18 tok/s acc 2.43 -> 8.73 steps/s = 114.6 ms; structured c=1 67.64 acc 7.84 -> 116 ms
- git show f2538e5:evidence/rebench-dflash5-20260903T045815Z/bench.txt: DFlash2-5 prose 21.76/2.354 = 108 ms/step; structured 55.73/6.0 = 108 ms/step
- models/glm5next/nvidia/model.py:819-820 (MTP layer skipped for main model when not used)

**Proposed action:** Put this roofline into evidence/ (bytes/step table) and use it to rank knobs. Anything that neither removes bytes/step nor protects bandwidth (NCCL, CPU, scheduler) is capped at single-digit percent. Confirm with one nsys or torch-profiler capture of ~20 decode steps on rank0: sum MoE+GEMM kernel time and compare to DRAM bytes (ncu dram__bytes on a single step if allowed).

**Est. impact:** No direct gain. It bounds the achievable: comm+CPU at most ~8-12% of a 115 ms step, while bytes/step levers can give 15-25% (see SYS-4).

**Validation:** Single nsys run at c=1 prose on the current default: sum per-kernel time by category (Marlin MoE, BF16 GEMM, KDA/MLA attention, NCCL, draft). Expect MoE+GEMM >= 75% of the step and NCCL <= 8%.

**Risks:** The 273 GB/s peak and the effective-bandwidth figure for GB10 are not measured here. The expert-distinct count is an upper bound. The LibertAI pin was measured, not the nvidia pin (same byte totals to within 0.3 GiB/rank).

**Verifier reasoning:** I re-ran scratchpad/agents/sys_st_bytes.py against the nvidia 09b04e5 headers and got the same totals: routed_experts 159.469 GiB (U8 141.75 + F8 17.72), kda 8.729, mla 2.406, shared 1.969, lm_head 1.182, MTP layer45 13.844 BF16, total 190.380 GiB. Against LibertAI caca4e6 (the pack actually benched) the non-MTP difference is only dense_mlp (0.844 BF16 vs 0.237), about 0.30 GiB/rank, so the 0.3 GiB/rank claim holds. The byte arithmetic also holds: 13.50 MiB/expert, E[D(8)] = 58.1, 16.08 GiB/rank of experts, about 7.57 GiB/rank of non-expert weights. Step times hold too: 21.18/2.426 = 114.6 ms and 67.64/7.844 = 115.9 ms (rebench bench.txt); DFlash2-5 21.76/2.354 = 108.2 ms and 55.73/6.0 = 107.7 ms (git show f2538e5). The inference fails in three places. (a) The DFlash2-5 vs -7 comparison contradicts the model rather than supporting it. Independent routing predicts D(6) = 44.8 to D(8) = 58.1, i.e. +13.3 experts x 42 layers x 13.5 MiB / 2 = +3.7 GiB/rank per step, which is +14.5 ms at 273 GB/s and +16.7 ms at 236 GB/s. The observed delta is +7 ms, across different boots on different days, inside ±11% noise. The implied marginal bandwidth is 0.297 GB per expert-slot / 0.526 ms = 565 GB/s, which is physically impossible. So either routing is strongly correlated (fewer bytes, lower real effective BW) or the step is not byte-dominated. (b) bench_decode.py:36 uses temperature 0 and the same prompt for every stream, so at c=2 both greedy streams route to the same experts. The step still grows from 115 ms to 131 ms (structured c=2: 60.41/7.92) and 150 ms (prose c=2: 16.64/2.50) with roughly zero extra distinct-expert bytes. That is 14-30% of step time scaling with token count, not with weight bytes. (c) 236 GB/s is 86% of theoretical peak for a mix of M=8 Marlin W4A16 MoE (log: 'Weight-only FP4 compression ... Marlin'), small BF16 GEMMs, KDA/MLA kernels and ~100 collectives. That is implausibly high and is really a byte upper bound divided by time. Bytes probably are the largest single term, but 'comm+CPU capped at 8-12%' and '74-86% of roofline' are not established.

**Verifier corrected claim:** Weight bytes per verify step are an upper bound of ~25 GiB/rank (verified from headers). The measured data (DFlash2-5 vs -7 delta about half the modeled expert term; identical-prompt c=2 adding 16-35 ms with shared experts) show that a sizeable token-proportional, non-byte component exists. The effective-bandwidth fraction and the comm/CPU cap are unproven until one profiler trace is taken.

**Verifier corrected impact:** Framing only. The comm/CPU/launch share may be well above 8-12%, so it should not be used to deprioritize comm or kernel-launch work before an nsys trace exists.

## SYS-2: Host reclaim/swap policy (swappiness 60, 16 GiB swapfile, 45 MB min_free) costs bandwidth and stalls serving processes

- kind=perf component=host kernel VM / UMA impact=4 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Both Sparks run with default desktop VM tuning on a box where GPU allocations are system RAM. Serving leaves only ~5 GiB MemAvailable. The kernel then swaps anon pages of the API server, EngineCore and workers, and direct-reclaims and compacts under load. This adds memory traffic and CPU stalls to a bandwidth-bound decode and makes run-to-run variance ±11%.

**Mechanism:** GPU weights, KV and graph pools are driver-pinned system pages and never swap. Everything else competes for the last ~5 GiB: CPU-side anon (API server tokenizer and MM processor, EngineCore, worker Python, torch/triton/tilelang heaps) and page cache. With swappiness=60 and a tiny low/high watermark gap, kswapd runs late, allocations take direct reclaim, and hot Python pages get swapped out and faulted back in on the serving path (streamed tokens and scheduler). Each swap or compaction also consumes LPDDR5X bandwidth that the GPU needs.

**Evidence:**
- host (read-only): vm.swappiness=60, vm.min_free_kbytes=45167, vm.watermark_scale_factor=10, /swap.img 16G on spark1 AND spark2
- evidence/rebench-20260902T204243Z/free-after-bench.txt: used 116 / available 5 / swap used 6 GiB (spark1, after bench)
- /proc/vmstat spark1 (6 days up, includes other workloads): pswpout 309,074,410 pages (~1.15 TiB), pswpin 58.5M, pgmajfault 115.8M, allocstall_movable 12.3M, compact_stall 2.1M; /proc/pressure/memory full total=3887 s
- spark2 /proc/vmstat (12 days): pswpout 695,975,628 (~2.6 TiB), allocstall_movable 6.6M, compact_stall 7.1M
- evidence/iter-h-snap/bench.txt vs bench-wave2.txt (same serve, wave2 noted '(swap)' in decision.tsv h-snap): prose c=1 18.08->16.92 (-6.4%), structured c=1 67.14->64.43 (-4%), wave2 prose run1 TTFT 3.635 s
- evidence/rebench-20260902T204243Z/bench.txt: prose c=1 runs 23.40/18.75/21.18 (±11% spread)
- tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark README (same model, 2x Spark): mandatory drop_caches + vm.swappiness=0, vm.min_free_kbytes=8388608, docker --memory 110g --memory-swap 110g; notes that swapoff entirely caused UVM livelock

**Proposed action:** This is outside ~/projects and needs explicit user authorization. On both nodes, only for the serving window: sysctl vm.swappiness=1 (keep swap enabled per tony's UVM-livelock note), vm.watermark_scale_factor=200 (kswapd wakes earlier; ~2.4 GiB gap), keep min_free_kbytes modest (<=1 GiB; 8 GiB would eat the headroom this pin needs). Drop caches before launch (see SYS-10 for a way to do this without sudo). In the recipe, add a doctor/preflight WARN when swappiness>10 or SwapFree<SwapTotal at launch, and record /proc/vmstat + /proc/pressure/memory deltas next to every bench.

**Est. impact:** Measured analogue: wave1 vs wave2 swap delta = 6.4% prose c=1, 4% structured. Expected +3-7% median decode, a much tighter spread (±11% -> ~±4%), and no multi-second TTFT spikes. Estimate, not measured on nvidia pin.

**Validation:** One knob: swappiness 60->1 only, same pin and image. Run 2 bench waves back to back and capture pswpin/pswpout/pgmajfault/allocstall deltas on both nodes. Pass if the vmstat deltas are ~0 and the prose c=1 median is >= wave1 with a smaller spread.

**Risks:** Host-wide change affects the other workload. Too-high min_free_kbytes shrinks usable UMA and can itself OOM the load (weights leave ~18-20 GiB during load). Swapoff risks the UVM livelock reported by tony.

**Verifier reasoning:** The host facts check out. On spark1 and spark2: vm.swappiness=60, min_free_kbytes=45167, watermark_scale_factor=10, /swap.img 16 GiB. Page size is 4096, so pswpout 309.09M ≈ 1.15 TiB (spark1) and 695.98M ≈ 2.59 TiB (spark2). free-after-bench shows available 5, swap used 6. The 'measured analogue' is weak. The iter-h-snap per-run spreads overlap: wave1 prose c=1 17.97/20.73/18.08 vs wave2 16.13/18.00/16.92, n=3, so -6.4% is inside noise. Nothing records the swap-in rate during wave2, and the '(swap)' tag in decision.tsv is an annotation, not a measurement. Wave2 did show multi-second TTFT stalls on a warm serve (prose c=1 run1 3.635 s, c=2 run1 6.630 s vs 0.403 in wave1). That supports stalls and variance more than a median-decode gain: once pages fault back in, a FULL-graph async decode loop does little CPU paging. Evidence corrections: tony's README says swap OFF made the worker die during Marlin repack, and the UVM livelock came from DEFAULT swappiness with swap on (mid-load paging), which is the reverse of the reviewer's wording. The scratchpad copy of tony's README does not contain min_free_kbytes=8388608 or --memory 110g; tony PR #22 says --memory 112g.

**Verifier corrected claim:** Default desktop VM tuning plus ~5 GiB MemAvailable at serve leads to swap-outs of serving-process pages and likely causes the multi-second TTFT stalls seen on warm serves (h-snap wave2). A steady-state decode gain has not been shown.

**Verifier corrected impact:** Mainly lower variance and no TTFT spikes. A median decode change of +0-3% is unproven. The 6.4% figure is noise-level. Tony's own reason for swappiness=0 is load-time UVM-livelock avoidance, not decode speed.

## SYS-3: The '5.0 GiB KV pin slows decode ~20%' rule has no receipts and cannot be a KV-bandwidth effect

- kind=methodology component=KV pin / UMA headroom impact=3 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** The rule that caps the KV pin at 4.14 GiB comes from a commit message only. Bench prompts are ~100 tokens, so attention reads scale with live tokens, not pool size; a larger pool cannot slow decode through KV reads. The slowdown, if real, is headroom-driven reclaim and swap (SYS-2), or noise. Each request also takes a fixed ~26.7% of the pool regardless of length, so raising the pin buys concurrency, not decode speed.

**Mechanism:** Going 4.14->5.0 GiB takes 0.86 GiB from the ~5 GiB MemAvailable left after the bench. That pushes the node from 'some swap' to 'continuous reclaim' at the default watermarks, and the serving processes' anon pages get faulted in on the critical path. The GPU kernels read the same bytes either way.

**Evidence:**
- git show a138ab7 (commit message only, no evidence dir): '5.0 GiB slowed decode ~20% at every concurrency'
- run.sh:35-41 comment encodes the same claim; AGENTS.md 'Raising it boots but backfires under UMA pressure'
- engine.log.tail:192 and :226: 'Running: 1 reqs ... GPU KV cache usage: 26.7%' for a ~100-token prompt; 2 reqs = 53.4%
- engine.log.tail:69: 'GPU KV cache size: 372,877 tokens' at 4.14 GiB
- tonyd2wild GLM 2x Spark PR #22: nvidia pack runs at a 6->8 GiB KV pin (pool 536,832->714,240 tokens) with swappiness=0 + --memory 112g
- rebench prose c=1 run-to-run spread 18.75-23.40 (±11%): a single 20% delta is close to noise

**Proposed action:** Do not treat 4.14 GiB as a hard physical limit. After SYS-2 host tuning (one knob), re-run 4.14 vs 5.0 vs 6.0 GiB with vmstat/PSI captured on both nodes and record the receipts in decision.tsv. Keep 4.14 as the default until then. The goal is headroom and concurrency, not decode speed.

**Est. impact:** 0% decode by itself. Expected to show the '20%' is reclaim, which would unblock a 6 GiB pin (~+45% pool, c=3 admission) if SYS-2 is applied.

**Validation:** A/B the pin with swappiness fixed. If decode at 5.0 GiB is within noise of 4.14 GiB and pswpin stays ~0, the rule is refuted. If it slows and pswpin rises, reclaim is confirmed as the mechanism.

**Risks:** The 5.14 GiB crash under concurrent load is a separate OOM risk. Long-prompt activation peaks (not profiled because of the pin) still need headroom.

**Verifier reasoning:** Confirmed that there are no receipts. The claim appears only in commit a138ab7's message (DFlash2-5/c=4 era), in run.sh:35-41 and in how-explanation.md:101. No evidence dir has a 5.0 GiB run, and git log -S finds nothing else. The KV-read argument is sound: engine.log.tail:192/226 show 26.7% per short request and 53.4% for two, so attention reads scale with live tokens. Needle requests do take more (30.5-36.6%), so 'regardless of length' only holds for short prompts. Attributing the slowdown to reclaim is a hypothesis with no data behind it. The tony evidence (PR #22: 6->8 GiB pool 536,832->714,240) is a different image, 262K ctx, eager, and --memory 112g, so it does not transfer directly.

**Verifier corrected claim:** The 20% slowdown at 5.0 GiB is unreceipted and cannot come from KV-read bandwidth on ~100-token prompts. Reclaim/swap is the leading hypothesis but has not been measured.

**Verifier corrected impact:** 0% decode. It could unblock a larger pin for concurrency only after host tuning, and at the 4.14 pin that leaves about 3 GiB MemAvailable. 5.14 GiB crashed.

## SYS-4: FP8 for the BF16 attention, shared-expert, lm_head and draft weights frees ~4.3 GiB/rank of UMA and removes ~18% of bytes per step

- kind=perf component=weights / UMA budget impact=5 confidence=3 effort=L needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** About 9.2 GiB/rank of the per-step bytes (~37-42%) are BF16: KDA 4.36, MLA 1.20, shared experts 0.98, lm_head 0.59 (read twice, target plus draft), indexer 0.15, draft 1.09. Halving them with FP8 cuts ~4.6 GiB of reads per step and frees ~4.3 GiB/rank resident. That solves the UMA headroom problem and speeds a bandwidth-bound step at the same time.

**Mechanism:** Bandwidth-bound step: 4.6 GiB fewer bytes out of ~25 GiB is ~18% at constant effective bandwidth. For the UMA budget, 8.65 GiB of resident BF16 non-expert weights per rank drop to ~4.3 GiB. That gives ~4 GiB of headroom, taking the node from ~5 to ~9 GiB MemAvailable, enough to stop reclaim (SYS-2) or to hold a larger KV pin.

**Evidence:**
- safetensors header totals (SYS-1): kda_attn 8.716 BF16, mla_attn 2.406 BF16, shared_experts 1.969 BF16, lm_head 1.182 BF16, mla_indexer 0.153 BF16 (replicated per rank: attention.py:250-265 ReplicatedLinear / disable_tp=True)
- hf_quant_config.json exclude_modules: self_attn*, shared_experts*, mlp.gate, lm_head, embed, visual excluded (BF16)
- model.py:331 'quant_config=None  # MLA projections are BF16 in checkpoint'
- engine logs: 'Model loading took 90.67 GiB'; head 'Initial free memory 109.0 GiB' (engine.log.tail:68) -> ~14.2 GiB left for KV+activations+graphs+NCCL+process RSS

**Proposed action:** Kernel/quant owners: build an FP8 (per-channel or 128-block) variant of KDA in/out projections, MLA q_b/kv_b/o_proj and shared experts via vLLM online fp8 quantization or a ModelOpt FP8 re-export. Check whether sm_121 cutlass FP8 GEMM exists in this image (_C was built for 12.0, not 12.1a). Keep lm_head and embed BF16 in the first pass. Change one group at a time (shared experts first: 0.98 GiB/rank, least quality risk).

**Est. impact:** Upper bound +18-22% decode at c=1 (prose 21.2 -> ~25-26 tok/s) if bandwidth-bound, plus ~4.3 GiB/rank of UMA headroom. Shared experts alone: ~0.49 GiB/step, about -2 ms (+2%).

**Validation:** Per group: bench_decode prose c=1/c=2 plus the greedy count-200 and thinking-off smokes, and a small quality eval (GSM8K/MMLU subset) against the BF16 baseline. Record free -h after the bench.

**Risks:** Quality loss in KDA gates and projections. Possible FP8 GEMM kernel gaps on sm_121 (JIT OOM risk as with cutlass MoE, see oom-20260831). This is outside the pure system dimension and should be coordinated with the quant/kernel reviewers.

**Verifier reasoning:** The header numbers are verified: KDA 8.716, MLA 2.406, shared 1.969, lm_head 1.182, indexer 0.153 BF16. The indexer is replicated. hf_quant_config/config.json ignore excludes self_attn*, shared_experts*, mlp.gate, lm_head, embed and visual*. The 18% figure includes lm_head (0.59 GiB x 2 reads), but the proposed action keeps lm_head BF16. Without it, the saving is (9.14-1.18)/2 = 3.98 GiB of about 25 GiB, i.e. 16%. The gain also assumes those GEMMs run at the same effective bandwidth, which SYS-1's evidence does not show. A stronger measured analogue was missed: tony's EXPERIMENTAL-NVFP4-ATTENTION (scratchpad tony_nvfp4_attn.md) measured x1.14-1.30 per category at TP4, predicts x1.21 at TP2 and frees 5.21 GiB/rank. It uses NVFP4 through a path that already works on sm_121 (Marlin W4A16), rather than an FP8 cutlass GEMM whose sm_121 availability in _C (built for 12.0) is unverified.

**Verifier corrected claim:** Quantizing the BF16 non-expert weights (FP8, or NVFP4/W4A16 as in tony's TP4 experiment) frees about 4-5 GiB/rank and removes about 16% (FP8, lm_head kept) to about 21% (NVFP4) of weight bytes per step.

**Verifier corrected impact:** About +10-20% c=1 decode (analogue: tony TP4 measured x1.14-1.30; untested at TP2), plus 4-5 GiB/rank of UMA headroom. Quality risk needs an eval.

## SYS-5: PR #12's 16 GiB MemAvailable wait-abort will kill a healthy boot

- kind=pr-review component=run.sh wait_ready (PR #12) impact=4 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** PR #12 aborts both ranks if MemAvailable < 16 GiB while /v1/models is not up. A healthy boot drops below 16 GiB before the API is ready: after weights, MemAvailable is ~17-20 GiB, and the draft (1.09), the 4.14 GiB KV pin, encoder profiling and graph capture still follow. The tripwire fires around KV allocation on every boot.

**Mechanism:** 17.16 - 1.09 (draft) - 4.14 (KV) - activations/workspaces ≈ 10-12 GiB < 16 GiB, reached ~2 min before /v1/models answers. abort_load then runs ./stop.sh on both ranks.

**Evidence:**
- engine-rank1.log.tail:31 'Available RAM: 20.55 GiB' at main-weight load start (psutil.virtual_memory().available == MemAvailable; weight_utils.py:687-691, no cgroup limit)
- engine-rank1.log.tail:40 'Available RAM: 18.25 GiB' before draft load (worker)
- engine.log.tail:45 'Available RAM: 17.16 GiB' before draft load (head, 20:55:36), then KV 4.14 GiB (engine.log.tail:68) + graphs (:97) before ready at 21:01
- PR #12 text itself: 'After the API is ready a healthy LibertAI boot can sit near 8 GiB'
- gh pr diff 12: wait_uma_or_abort uses UMA_ABORT_GIB=16 on both nodes every 5 s

**Proposed action:** Change PR #12 before merge. Arm the MemAvailable tripwire at ~3-4 GiB (not 16), or trip on swap-in rate or PSI 'full' avg10 > ~10% for 30 s instead of a static MemAvailable floor. Keep the 20 GiB start reserve (MemAvailable ~110 GiB at start). Add a unit test with a simulated 12 GiB MemAvailable during loading that must NOT abort.

**Est. impact:** Prevents a 100% false-abort rate on default boots (each costs ~15-18 min of boot).

**Validation:** CPU-only: replay the MemAvailable trajectory from the rebench logs through wait_uma_or_abort in a unit test. On the Sparks: one boot with the revised threshold and logging of min MemAvailable seen.

**Risks:** Too low a threshold loses the protection against real global OOM (oom-20260831: spark2 died with ~18 GiB left during cudafe++). A PSI/swap-rate trip is more robust than a static floor.

**Verifier reasoning:** pr12.diff: wait_uma_or_abort runs every 5 s in wait_ready on both nodes with UMA_ABORT_GIB=16 until /v1/models answers. Measured trajectory on the head: engine.log.tail:45 shows 17.16 GiB available before the draft load at 20:55:36. After that come the draft (2.18 GiB checkpoint, ~1.09/rank), the 4.14 GiB KV pin (:68 at 20:57:56), encoder profiling and graph capture. The API is ready at around 21:00-21:01 (progress.md: run.sh rc=0 at 21:01:18). The worker has 18.25 GiB before the draft (rank1:40). Both nodes therefore fall below 16 GiB about 2-4 min before readiness. The PR body itself says 'a healthy LibertAI boot can sit near 8 GiB' after ready, and the PR was never run on a Spark ('No Spark SSH or serve from this change'). The nvidia pack's resident footprint is within about 0.3 GiB/rank, so it behaves the same.

## SYS-6: Cross-node all-reduce costs ~6-9 ms/step; a one-shot RoCE all-reduce or LL protocol can recover ~3-5%

- kind=perf component=NCCL / RoCE / vllm.distributed impact=3 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** Each verify step does ~91 target all-reduces (embed + 45 x [attention o_proj + MLP/MoE]), ~11 draft all-reduces and 2 logits all-gathers, all on NCCL over one RoCE HCA. Custom AR and symm-mem are disabled and GDR is off on GB10. At the published GB10 small-message latency (~45-60 us), that is ~6-9 ms/step (5-8%).

**Mechanism:** Each tiny all-reduce pays two network hops plus proxy-thread staging through host bounce buffers (GDR 0). Latency, not bandwidth, dominates at 64-128 KiB. A one-shot RDMA-write all-reduce needs one hop. The LL protocol cuts per-hop overhead for small sizes.

**Evidence:**
- engine-rank1.log.tail:16-17: 'SymmMemCommunicator: Device capability 12.1 not supported' and 'Custom collectives are disabled because this multi-node group does not support MNNVL multicast' -> pynccl path (cuda_communicator.py:275-340)
- model.py:126-131 (dense down_proj RowParallel reduce), model.py:222 shared_experts reduce_results=False + moe_runner.py:451-487 single final all-reduce for routed+shared, attention.py:497 and kda.py:279 o_proj RowParallelLinear
- message size: 8 tok x 4096 x 2 B = 64 KiB (c=1), 128 KiB (c=2); logits gather 8 x 154880 x 2 B ≈ 2.4 MiB
- FujitsuPolycom/sparkring#193 (NCCL 2.30.7 on GB10): LL vs Simple 4 KB 42.7 vs 58.3 us (-27%), 32 KB -19%; mis-tuned 368 KB AR 923 -> 173 us after channel/protocol tuning
- tonyd2wild GLM-5.3-Flash 2x Spark PR #22: b12x RoCEnante one-shot RoCE all-reduce '2.5x faster than NCCL at decode-step size on two nodes'; single-stream decode flat to +8%, prefill +26-36%
- multimodalflow.net dual-node blog: NCCL reports GDR 0 on Spark, ~10.2 GB/s busbw single HCA
- run.sh:248-265 env list has no NCCL_PROTO/ALGO/NCHANNELS and no passthrough

**Proposed action:** (1) Add an env passthrough (see SYS-8) so NCCL knobs reach both containers. (2) Run nccl-tests all_reduce_perf between the Sparks at 32K-4M: default vs NCCL_PROTO=LL (<=128K) vs LL128, and NCCL_MIN_NCHANNELS/MAX_NCHANNELS=2/4. Set the winner via NCCL_TUNER or NCCL_PROTO. (3) Port tony's b12x RoCEnante all-reduce (valid on a 2-node pair) as an opt-in ROCE=1 lane. Check b12x issue #313 (a graph-replayed collective can wedge a rank) before making it default.

**Est. impact:** NCCL LL tuning: ~20% of ~7 ms ≈ 1.5 ms/step (+1-1.5%). One-shot RoCE AR: ~60% of the AR time ≈ 4-5 ms/step (+3-5% decode; tony measured flat to +8%) and +26-36% prefill.

**Validation:** First nccl-tests (needs exclusive slot, GPU use), then one knob per serve boot: bench prose c=1/c=2 and needle-20480 TTFT. Capture NCCL_DEBUG=INFO once to record the chosen proto/algo/channels and 'GDR 0/1'.

**Risks:** RoCE one-shot AR wedging under CUDA-graph replay (b12x #313). NCCL env changes can silently change algorithm selection for large prefill messages. Needs the exclusive-GPU window.

**Verifier reasoning:** The all-reduce count is right. kda.py:279 and attention.py:497 o_proj are RowParallel, model.py:126-131 is the dense down_proj, and model.py:222 gives a single MoE final reduce. use_sequence_parallel_moe is off, so the model.py:272 all_gather is not used. That is about 91 target all-reduces plus the draft. rank1:16-17 confirms symm-mem and custom AR are disabled. The latency figures do not fit this case. sparkring#193 (gh api) measured 4 KB 42.7 vs 58.3 µs on a 4-node TP4 DeepSeek cycle, and its LL gain shrinks with size (-19% at 32 KB; its threshold is 40 KB). Here the messages are 64-128 KiB, above that threshold, so the LL benefit is likely smaller. The same issue also says 'single-stream decode gain is masked'. Tony PR #22 (gh pr view) reports RoCEnante 'single stream flat' and aggregate +5-18% at C1-C6, measured in EAGER mode on a power-clamped fleet. The '2.5x faster' and 'flat to +8%' quotes are not in the PR body. The 6-9 ms total is a guess, since per-AR latency at 64 KiB over this RoCE link has not been measured.

**Verifier corrected claim:** About 100 cross-node NCCL all-reduces per step go through the pynccl path. Their cost at 64-128 KiB is unmeasured. The published analogues show LL gains mainly below 40 KB and flat single-stream decode for one-shot RoCE AR.

**Verifier corrected impact:** c=1 decode about 0-2% (analogue: flat). c=2 aggregate possibly +5% and prefill larger. Needs nccl-tests at 64K/128K first.

## SYS-7: The second active RoCE rail and MTU 1500 halve inter-node bandwidth for prefill

- kind=perf component=network / NCCL_IB_HCA impact=2 confidence=2 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** AGENTS.md is right that two of the four HCAs are DOWN, but both nodes have a second ACTIVE HCA (roceP2p1s0f1, 10.100.9.1/.2) that is never used. The netdevs run MTU 1500, so the RoCE active_mtu is 1024. Decode barely cares; long-prompt prefill (16 MiB all-reduces per 2048-token chunk) is bandwidth-bound on ~10 GB/s single-rail GDR-0 NCCL.

**Mechanism:** The Spark's CX-7 QSFP is exposed as two PCIe halves. NCCL on one HCA tops out around 10 GB/s busbw. Listing both HCAs lets NCCL stripe channels across rails. A 4096 RoCE MTU cuts per-packet overhead ~4% and the packet count per message 4x.

**Evidence:**
- ibv_devinfo spark1: rocep1s0f1 PORT_ACTIVE and roceP2p1s0f1 PORT_ACTIVE, both active_mtu 1024 (max 4096); rocep1s0f0 and roceP2p1s0f0 PORT_DOWN
- ip -br addr: enp1s0f1np1 10.100.8.1/24, enP2p1s0f1np1 10.100.9.1/24; spark2: 10.100.8.2 and 10.100.9.2; both mtu 1500, speed 200000
- run.sh:15 HCA=rocep1s0f1 only; run.sh:258 NCCL_IB_HCA=$HCA
- engine.log.tail:385 prefill ~4438 tok/s (needle) -> 2048-token chunk ≈ 461 ms; 93 all-reduces x 16 MiB at ~10 GB/s ≈ 150 ms ≈ 30% of the chunk (estimate)

**Proposed action:** Keep the default pin until measured. Add an opt-in HCA='rocep1s0f1,roceP2p1s0f1' (forwarded to the worker) and test it with nccl-tests plus a needle-20480 TTFT A/B. The MTU 9000 change is host-level and needs user authorization.

**Est. impact:** Prefill/TTFT on long prompts ~-10-15% (estimate: halves the ~30% comm share). Decode ≤1%.

**Validation:** all_reduce_perf 1M-64M busbw single vs dual HCA, then needle-20480 c=1 TTFT with dual HCA as the only change.

**Risks:** The second subnet needs routes and GID on both sides. NCCL_CROSS_NIC semantics with two rails. The AGENTS.md rule warns that unpinned HCA picks a dead port, so list both explicitly and never leave it unset.

**Verifier reasoning:** The topology checks out on spark1 (ibv_devinfo, ip): rocep1s0f1 and roceP2p1s0f1 are PORT_ACTIVE, both active_mtu 1024, netdev mtu 1500 (maxmtu 9978), with 10.100.8.1 and 10.100.9.1. run.sh:15/258 pins only rocep1s0f1. The impact arithmetic is wrong, though. engine.log.tail:385 '4438 tok/s' is a 10 s logger window average and not the prefill rate. needle-20480-c2.txt measured 44,386 tokens with TTFT 30.75 s = 1443 tok/s, and needle-8192 measured 884 tok/s. A 2048-token chunk therefore takes about 1.42 s, not 461 ms. 93 x 16 MiB at about 10 GB/s ≈ 156 ms is about 11% of a chunk, not 30%.

**Verifier corrected claim:** A second active RoCE rail is unused and the RoCE MTU is 1024. Communication is about 10% of long-prompt prefill time.

**Verifier corrected impact:** Dual rail would cut long-prompt TTFT by about 3-6% (half of an ~11% comm share at best). Decode ≤1%.

## SYS-8: Worker env/arg forwarding gaps and no NCCL/VLLM env passthrough make cross-node experiments unsafe

- kind=ops component=run.sh orchestration impact=3 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The head builds the worker command from a fixed variable list. LIMIT_MM_PER_PROMPT, HF_CACHE, SNAPSHOT/SNAPSHOT_IN_CONTAINER and HF_HUB_DISABLE_XET are not forwarded, so a head-only override silently gives the headless worker (which builds its own VllmConfig) different args. No NCCL_*/VLLM_*/OMP env can reach either container, because docker env is a hardcoded array. User values containing a single quote (EXTRA_ARGS, SPEC_CONFIG) break the ssh quoting. The 25 s sleep does nothing.

**Mechanism:** A mismatched --limit-mm-per-prompt changes the encoder cache budget and MM profiling on one rank only. HF_CACHE or SNAPSHOT overrides make the worker load a different path or re-download. NCCL tuning (SYS-6, SYS-7) cannot be A/B'd without editing the generated env block.

**Evidence:**
- run.sh:386 worker ssh variable list (no LIMIT_MM_PER_PROMPT, HF_CACHE, SNAPSHOT, SNAPSHOT_IN_CONTAINER, HF_HUB_DISABLE_XET)
- run.sh:62-64 and 303-308: LIMIT_MM_PER_PROMPT drives --limit-mm-per-prompt on each rank independently
- run.sh:248-265 env_args hardcoded; only VLLM_USE_BREAKABLE_CUDAGRAPH is conditionally added (266-268)
- engine-rank1.log.tail: serve.py:216 'Launching vLLM ... headless multiproc executor' (worker parses its own CLI)
- run.sh:387-388 'Waiting 25s for NCCL listen': worker initialized at 20:43:46 (engine-rank1.log.tail) and blocked on the head's TCPStore anyway (next line 20:44:28), so the order is self-synchronizing
- run.sh:238-239: maybe_drop_caches runs BEFORE stop_local removes the old container (drops cache while ~95 GiB are still held)

**Proposed action:** Forward LIMIT_MM_PER_PROMPT, HF_CACHE, SNAPSHOT, SNAPSHOT_IN_CONTAINER, HF_HUB_DISABLE_XET. Add EXTRA_ENV (space-separated KEY=VAL, validated against ^(NCCL|VLLM|TORCH|OMP|CUDA)_) forwarded to the worker and turned into -e flags on both ranks. Build the ssh command with printf %q instead of hand-written single quotes. Reorder to stop_local -> drop caches. Replace the sleep 25 with a check that the worker container is Running (or drop it).

**Est. impact:** Correctness/ops. Removes a class of silent rank-divergent configs, unblocks NCCL experiments, saves 25 s per boot.

**Validation:** CPU-only tests: a stubbed ssh records the worker command. Assert that every run.sh variable used in docker args is forwarded, that EXTRA_ENV round-trips, and that a single-quote in EXTRA_ARGS is preserved.

**Risks:** Forwarding arbitrary env could leak HF tokens into logs. Keep the allowlist and never echo tokens.

**Verifier reasoning:** run.sh:386 forwards a fixed list that omits LIMIT_MM_PER_PROMPT, HF_CACHE, SNAPSHOT, SNAPSHOT_IN_CONTAINER and HF_HUB_DISABLE_XET. The defaults re-derive the same values on the worker, so divergence happens only when a user overrides one (run.sh:44, 59-64). env_args at run.sh:248-265 is hardcoded, with only VLLM_USE_BREAKABLE_CUDAGRAPH conditional (266-268). Single-quote interpolation of EXTRA_ARGS/SPEC_CONFIG breaks on a value containing '. maybe_drop_caches runs before stop_local (238-239). rank1 log lines 15-16 show the worker waiting on the TCPStore from 20:43:46 to 20:44:28, so the 25 s sleep (387-388) is redundant. One nuance: env_args sits outside the '# BEGIN generated' block, so NCCL knobs can be added by editing run.sh; 'cannot be run' overstates it. It is still inconvenient and cannot be done per run.

## SYS-9: Image built separately on each node, and VALIDATE_ONLY checks nothing across nodes

- kind=ops component=image parity / preflight impact=2 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** glm53-sm121-v11 has different image IDs and different patch-layer digests on spark1 and spark2 (base identical). The code is probably the same but nothing proves it, and the next rebuild can diverge. VALIDATE_ONLY exits before role detection, so it never checks the worker's image, snapshots, HCA state, IFACE IP, SSH or MemAvailable.

**Mechanism:** TP ranks running different vLLM/FlashInfer/NCCL bits can deadlock or diverge numerically, and the logs only show up as NCCL hangs or shm_broadcast timeouts (see oom-20260831 spark1 symptom).

**Evidence:**
- docker image inspect spark1: sha256:88173725... created 11:39:23; spark2: sha256:bde907c4... created 11:39:24
- RootFS layers 34-44 differ between nodes (v8-v11 patch layers); base vllm/vllm-openai:glm53-flash-arm64-cu130 identical sha256:d4264906...
- recipe.yaml:17 image.digest: null
- run.sh:123-128 VALIDATE_ONLY prints and exits before ORCHESTRATE/ROLE logic (lines 378-403)
- docker/Dockerfile.sm121-v8 pip installs nightly flashinfer from an extra index (pinned version, but the wheel index can change)

**Proposed action:** Build once on spark1, then `docker save | ssh spark2 docker load` (same ID on both). Record the image ID in recipe.yaml (image.digest or local_id). In run.sh, refuse when the local and worker image IDs differ. Extend VALIDATE_ONLY with read-only cross-node checks: image ID match, both snapshot dirs present, ibv_devinfo PORT_ACTIVE for HCA, IFACE has an IPv4, MemAvailable, and no foreign GPU container (reuse PR #12's functions).

**Est. impact:** Ops robustness. Prevents a class of silent rank-divergence failures; a preflight costs about 2 s.

**Validation:** VALIDATE_ONLY=1 ./run.sh on the head prints both image IDs and all checks and exits non-zero on a mismatch (read-only, no GPU).

**Risks:** docker save/load of a ~20+ GB image takes minutes. Not a live risk.

**Verifier reasoning:** Locally, docker image inspect glm53-sm121-v11 gives sha256:88173725... created 2026-08-28T11:39:23. diff of scratchpad v11_layers_spark1/2.txt shows layers 34-44 differing. VALIDATE_ONLY exits at run.sh:123-128, before detect_role and orchestration (378-403). Different layer digests can come from tar mtimes alone, so this does not prove different code. It does prove that nothing guarantees parity. Tony's README separately records a 'silent image-version mismatch between ranks' incident.

## SYS-10: Boot takes 18 min: each rank streams the full checkpoint, the head loads 3.2x slower, and drop_caches is a no-op without sudo

- kind=ops component=weight loading / page cache impact=2 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Under TP, each node reads the whole 181-190 GiB checkpoint through ~20 GiB of free UMA, including the unused 13.84 GiB BF16 MTP layer on the nvidia pack. Head load took 629 s (0.29 GiB/s) vs worker 197 s (0.92 GiB/s). maybe_drop_caches silently no-ops without passwordless sudo, so another model's page cache (DeepSeek was stopped just before the rebench) has to be reclaimed during load.

**Mechanism:** Streaming 181 GiB through a ~20 GiB window forces continuous page-cache reclaim. On spark1 this competes with a cold cache full of another model and the interactive tooling, so throughput drops 3x and the head becomes the critical path (the worker waits on the TCPStore).

**Evidence:**
- engine.log.tail:38 'Loading weights took 629.20 seconds' (head); engine-rank1.log.tail:33 '197.46 seconds' (worker); both 'Checkpoint size: 181.30 GiB', worker 'Available RAM: 20.55 GiB'
- evidence/rebench-20260902T204243Z/progress.md: boot 18m17s; 'DeepSeek serve ... was stopped ... just before this'
- run.sh:177-182 maybe_drop_caches requires sudo -n
- nvidia pack MTP layer45 13.844 GiB BF16 (header parse) is read and then skipped at model.py:819-820 when SPEC=dflash2
- weight_utils.py:840-842: expert skip-before-read exists only under EP (local_expert_ids), not TP

**Proposed action:** (1) Evict without sudo: posix_fadvise(POSIX_FADV_DONTNEED) over the other models' snapshot blobs (user-readable files) in maybe_drop_caches, after stop_local. (2) Try --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' (default_loader.py:85-110) as one knob. (3) Persist Triton/TileLang/FlashInfer JIT caches on a host volume (TileLang mhc compile ~5 s each, repeated). (4) Long term: prefetch only this rank's shard slices, or a pre-sharded per-rank checkpoint.

**Est. impact:** Boot 18 min -> ~8-10 min if head load drops to the worker rate (629->~200 s) and JIT caches persist (estimate). No decode effect, but lower page-cache churn during load reduces later swap (SYS-2).

**Validation:** Boot with fadvise eviction as the only change and compare 'Loading weights took' on both ranks, then multithread load as the next single change.

**Risks:** The multithread loader raises peak page-cache pressure during load (the OOM class seen with InstantTensor). Watch MemAvailable. fadvise on files still mmapped by a running serve has no effect, which is harmless.

**Verifier reasoning:** The facts hold. engine.log.tail:38 shows 629.20 s (head) vs rank1:33 197.46 s. The checkpoint is 181.30 GiB with 20.55 GiB available on the worker. progress.md shows a boot of 18m17s, with DeepSeek stopped just before. model.py:818-820 skips spec layers only after the iterator has yielded (and read) them. weight_utils.py describes local_expert_ids skip-before-read as EP-only. default_loader.py:85-110 accepts enable_multithread_load/num_threads. Two corrections. The benched pack was LibertAI (MTP 4.14 GiB quantized), so the 13.84 GiB BF16 MTP read applies only to the new nvidia pin. The cause of the 3.2x head slowdown (another model's page cache vs desktop load) is inferred, not measured.

**Verifier corrected impact:** Boot time only. The 8-10 min target is speculative.

## SYS-11: Async scheduling is already ON; the 'leave --async-scheduling off' rule comes from a no-op A/B

- kind=methodology component=vLLM engine core / scheduler impact=2 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** In the v11 vLLM, async_scheduling defaults to None and resolves to True when the spec method is in EagleModelTypes (which includes DFlash) and the executor is the mp executor. Passing --async-scheduling therefore changed nothing, which explains 'every cell inside noise'. The recipe runs with async scheduling on, so per-step scheduler/CPU work is already overlapped. It is not an untried lever.

**Mechanism:** With async on (V2 runner, max_concurrent_batches = pp+1 = 2), the EngineCore schedules step N+1 while the GPU runs step N. Combined with FULL CUDA graphs for the verify (8/16) and draft (engine.log.tail: 'Capturing CUDA graphs (FULL) 2/2', 'dflash2 CUDA graphs (FULL)'), per-step Python overhead is hidden behind the ~115 ms bandwidth-bound step.

**Evidence:**
- config/scheduler.py:148 async_scheduling: bool | None = None; engine/arg_utils.py:729
- config/vllm.py:1184-1233 auto-enable unless the spec method is not Eagle/ngram/draft_model, padded drafter disabled, or the executor is unsupported
- config/speculative.py:67-69 EagleModelTypes includes DFlashModelTypes
- v1/executor/multiproc_executor.py:541-542 supports_async_scheduling -> True
- README.md (worktree) '--async-scheduling boots with DFlash2-7 ... every cell stayed inside noise ... Leave it off'; AGENTS.md 'Leave --async-scheduling off'

**Proposed action:** Fix README/AGENTS wording to 'async scheduling is auto-enabled; do not pass --no-async-scheduling'. If an A/B is wanted, test --no-async-scheduling (expect a regression). Do not spend hillclimb budget on scheduler/CPU knobs.

**Est. impact:** 0% (documentation/methodology). Avoids wasted iterations. Removing async would likely cost ~3-8% (estimate).

**Validation:** Grep a debug-level boot log for 'Batch queue is enabled with size 2' (core.py:211-213) or dump vllm_config.scheduler_config.async_scheduling at startup.

**Risks:** None, beyond the cost of a confirmatory boot.

**Verifier reasoning:** v11 source: config/scheduler.py:148 has async_scheduling=None. config/vllm.py:1184-1233 resolves None to True unless the spec method is outside EagleModelTypes etc. config/speculative.py:67-69 shows EagleModelTypes includes DFlashModelTypes=Literal['dflash'], and run.sh:74 passes method 'dflash'. The spec config does not set disable_padded_drafter_batch (default False, speculative.py:143). run.sh:328 passes --distributed-executor-backend mp, and multiproc_executor.py:541-542 returns True for supports_async_scheduling. So the explicit --async-scheduling A/B (commit a27e654) was a no-op relative to the default, which matches 'every cell inside noise'. AGENTS.md/README 'Leave --async-scheduling off' is misleading, since it is on.

## SYS-12: SPEC=mtp rollback is probably broken on the nvidia pin and would not fit in UMA

- kind=correctness component=run.sh SPEC=mtp / nvidia checkpoint impact=3 confidence=3 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The nvidia pack's MTP layer 45 is 13.84 GiB of BF16 (routed experts included) but is not listed in hf_quant_config exclude_modules. The MTP drafter builds its MoE with the NVFP4 quant_config, so loading BF16 expert tensors into NVFP4 (U8 packed) params should fail. Even if it loaded as BF16 it would add ~6.9 GiB/rank against ~5 GiB MemAvailable. AGENTS.md and README still advertise SPEC=mtp as the rollback.

**Mechanism:** ModelOpt NVFP4 applies to any module not excluded, so the layer-45 MoE gets NVFP4 params with no matching weight_scale tensors. If a BF16 fallback were forced, 13.84/2 = 6.92 GiB/rank exceeds the headroom (4.85 GiB more than the LibertAI MTP).

**Evidence:**
- header parse: nvidia mtp_layer 13.844 GiB all BF16 (layers.45.mlp.experts.N.{gate,up,down}_proj.weight); LibertAI caca4e6 mtp_layer 4.141 GiB (quantized)
- hf_quant_config.json: 132 exclude entries, none match layers.45
- models/glm5next/nvidia/mtp.py:44,70 uses vllm_config.quant_config for the MTP layer
- run.sh:76-78 SPEC=mtp path; AGENTS.md 'SPEC=mtp rolls back to MTP-4'
- free-after-bench available 5 GiB at the 4.14 pin

**Proposed action:** Make run.sh refuse SPEC=mtp when MODEL=nvidia/... (point to the LibertAI rollback pair MODEL=LibertAIDAI... SNAPSHOT_REV=caca4e6 for MTP). Or add a VALIDATE_ONLY header check: read the layer-45 dtypes from the index/shard header and refuse if BF16 and not excluded. Update the README/AGENTS rollback text.

**Est. impact:** Prevents a failed boot (~15 min) or a UMA OOM on the advertised rollback path.

**Validation:** CPU-only: header check in VALIDATE_ONLY. Optionally a single SPEC=mtp boot on the nvidia pin in an exclusive slot to confirm the load error.

**Risks:** The fork's loader might special-case MTP BF16 (not found in mtp.py). The low confidence reflects that no test boot was run.

**Verifier reasoning:** Header read: layers.45 has 889 tensors in shards 1-3, all BF16, e.g. experts.0.gate_proj [2048,4096], shared_experts.* BF16, mlp.gate [288,4096] BF16. config.json quantization_config.ignore (132 entries) and hf_quant_config exclude_modules contain nothing for layers.45 (only model.visual* and per-layer patterns 0-44). The modelopt.py:139-215 is_layer_excluded/get_quant_method path gives a NVFP4 FusedMoE and Linear method to any non-excluded prefix. mtp.py:44/70 and model.py:294/350/358 build the MTP layer's MoE and MLP with vllm_config.quant_config. Loading BF16 [1024,4096] TP shards into packed uint8 [.., 2048] params should fail on shape, and no BF16 special case exists in mtp.py load_weights. The load failure itself has not been seen (no boot). Per rank 13.84/2 ≈ 6.9 GiB vs LibertAI MTP 2.07 GiB.

## SYS-13: Head node (spark1) is a noisy straggler: API server, EngineCore and dev tooling share its UMA and bandwidth

- kind=perf component=node placement / TP straggler impact=2 confidence=2 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 2)

**Claim:** Rank0's node also hosts the API server, EngineCore, conduit and the lab's interactive tooling (agent CLIs, chrome, node, next-server; 97 logins, load ~7-8 now). At init it had 2.67 GiB less free memory than spark2, loaded weights 3.2x slower, and carries more swap history. Every TP step waits for the slower rank at each all-reduce, so head-side contention (bandwidth, reclaim, CPU) sets the pace.

**Mechanism:** In a bandwidth-bound step, CPU-side memory traffic on the same LPDDR5X (browser, node, Python agents, kswapd) lowers the GPU's effective bandwidth on rank0. Rank1 then idles at each of ~100 all-reduces per step.

**Evidence:**
- engine.log.tail:68 'Initial free memory 109.0 GiB' (head) vs engine-rank1.log.tail:56 '111.67 GiB' (worker)
- load 629 s vs 197 s (SYS-10)
- uptime spark1: 97 users, load avg 7.12/8.31/8.78; spark2: 3 users, 1.37
- ps spark1: python3 11.7 GiB RSS, several 2.1.280 agent CLIs, hermes, chrome, next-server, node
- oom-20260831/diagnosis.txt invokers include gsd-media-keys (desktop session on a Spark)

**Proposed action:** For benchmarking and serving windows, quiesce spark1 (stop the desktop session, browser and agent tooling; move conduit if feasible) or record them as a controlled variable. Add a doctor line with load average and top-RSS processes to each evidence bundle. Optionally measure per-rank NCCL wait time (NCCL_DEBUG or an nsys trace on both ranks) to confirm the straggler.

**Est. impact:** Estimated +2-5% decode and lower variance, unverified. Mostly overlaps with SYS-2.

**Validation:** One bench wave with spark1 quiesced vs the normal state, same serve, with vmstat and loadavg captured.

**Risks:** Operational: the head is the user's workstation. Quiescing is a policy decision, not a recipe change.

**Verifier reasoning:** Initial free memory is 109.0 (head, engine.log.tail:68) vs 111.67 (worker, rank1:56), and the 629 vs 197 s load is verified. Current uptime on spark1 shows 97 users but load avg 1.52 (the reviewer's 7-8 was a snapshot), and spark2 is at 0.23. No per-rank NCCL wait or straggler measurement exists. The decode impact is speculative.

**Verifier corrected impact:** Unquantified. Likely below the ±11% prose noise floor.

## SYS-14: First c=2 request pays a reproducible ~6 s TTFT (one-time warmup/JIT) and several kernels JIT during inference

- kind=perf component=warmup / JIT caches impact=2 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** On every fresh boot the first 2-way concurrent prose run has TTFT ~6.3-6.6 s for both streams. A warmed serve does not. Triton rejection kernels, _prepare_dflash_inputs_kernel and the TileLang mhc_pre kernel (~5 s compile) JIT during live traffic.

**Mechanism:** Some shapes, such as a 2-request mixed prefill and the rejection/sampler kernels for real sampling params, are not covered by startup warmup. Their first use triggers a JIT compile or autotune on the critical path of both TP ranks. The JIT caches live inside the container and are lost on docker rm.

**Evidence:**
- evidence/rebench-20260902T204243Z/bench.txt: prose c=2 run1 ttft 6.286/6.286, runs 2-3 0.367
- git show f2538e5:.../bench.txt (DFlash2-5 boot): c=2 run1 ttft 6.422
- evidence/iter-h-snap/bench.txt: c=2 run1 ttft 6.630; bench-wave2.txt (same serve, warm): 0.403
- engine-rank1.log.tail:82-86 jit_monitor: _compute_local_logits_stats_kernel/_rejection_kernel/_resample_kernel (21:03:54), _prepare_dflash_inputs_kernel (21:06:13), TileLang mhc_pre_big_fuse_with_norm_tilelang (21:06:21); TileLang compile of that kernel takes ~5 s (20:59:28->20:59:33)

**Proposed action:** After wait_ready, have run.sh send warmup traffic before declaring Ready: one c=1 greedy and one sampled short request, one c=2 concurrent pair, and one small image request (vision is on). Mount a host dir for ~/.triton, the TileLang cache and ~/.cache/vllm (the FlashInfer autotune file is already under /root/.cache/vllm) so JIT results survive restarts.

**Est. impact:** Removes the ~6 s first-hit TTFT for the first concurrent users and ~5 s TileLang recompiles. Cuts ~1-2 min of boot JIT with persistent caches (estimate). No steady-state decode change.

**Validation:** Fresh boot with warmup, then bench: c=2 run1 TTFT should be ≤0.5 s and jit_monitor should log no warnings during the bench.

**Risks:** Persistent caches can serve stale kernels across image rebuilds, so key the cache directory by image ID.

**Verifier reasoning:** The phenomenon is real: rebench prose c=2 run1 TTFT 6.286 s, dflash5 6.422 s, H3 note 'c=2 prose TTFT spike 6.6s'. The evidence citation is inverted, though. iter-h-snap/bench.txt (wave1) has c=2 run1 TTFT 0.403, and bench-wave2.txt (second wave, same serve) has 6.630, plus prose c=1 run1 3.635 s. So the spike hit a WARM serve, not the first hit. The jit_monitor warnings cited (rank1:82-86, head:189-230) are timestamped 21:03:54-21:06:21, during smoke/probes, before bench_decode started at 21:07:02. No JIT warning is logged during the 21:07:20 spike. The JIT mechanism is unproven, and swap-in stalls (SYS-2) fit the h-snap case better. Warmup traffic plus persistent caches is still reasonable.

**Verifier corrected claim:** A reproducible ~6 s TTFT hits a c=2 wave. It occurs after fresh boots and also on a warm serve (h-snap wave2), with no JIT logged at that moment. The cause is unknown (untracked compile/autotune, or swap-in).

**Verifier corrected impact:** First-user TTFT only. No steady-state decode change.

## SYS-15: Pipeline or expert parallelism would not beat TP=2 for decode on two Sparks

- kind=methodology component=parallelism layout impact=2 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** TP=2 is the right layout. PP=2 serializes the two nodes' weight reads, roughly doubling c=1 step latency, and is not implemented for Glm5Next (mHC post/comb state and DFlash aux layers 5-42 would have to cross stages). EP=2 gives the same average expert bytes per rank, but the busier rank gates each step (~+10% on the expert term), and all2all dispatch and combine replaces one all-reduce over RoCE. AGENTS.md also forbids copying NVIDIA's --enable-expert-parallel.

**Mechanism:** Decode is bandwidth-bound per node. TP makes both nodes read in parallel with ~100 small all-reduces. PP makes the nodes take turns (with async batch-queue overlap only at c≥2). EP replaces all-reduce with a latency-bound 2-way all2all and adds imbalance.

**Evidence:**
- models/glm5next/nvidia/model.py:725-733 'PP is gated off for GLM5Next (no make_empty_intermediate_tensors)'; post/comb dropped across PP
- draft config target_layer_ids [5,14,24,33,42] span any PP split
- SYS-1 bytes/step: ~25 GiB/rank in parallel under TP; PP needs ~2 x 25 GiB serially at c=1
- expert load imbalance: D≈58 distinct experts/layer split over 2 ranks, binomial sd ≈ 3.8 -> E[max] ≈ 32 vs mean 29 (+10%)
- AGENTS.md 'Never touch: ... --enable-expert-parallel'

**Proposed action:** Record the analysis in evidence/ and keep TP=2. Put the comm effort into SYS-6 (faster all-reduce) instead of layout changes.

**Est. impact:** Avoids an estimated -40-50% (PP) or -5-10% (EP) regression at c=1.

**Validation:** None needed for PP (unsupported). EP could be sanity-checked later with a single boot if a W4A4 EP MoE path appears.

**Risks:** At high concurrency (c≥4) PP with async batch queue could match TP aggregate throughput, but that is not this recipe's occupancy.

**Verifier reasoning:** model.py:725-733 comment: 'PP is gated off for GLM5Next (no make_empty_intermediate_tensors)'. The draft config target_layer_ids are [5,14,24,33,42] (log: aux layers 6,15,25,34,43). AGENTS.md forbids --enable-expert-parallel. A small arithmetic nit: with D≈58 and binomial sd = sqrt(58*0.25) = 3.8, E[max of 2] ≈ 29 + 0.564*3.8 ≈ 31.1, i.e. about +7%, not +10%. The conclusion is unchanged.

## SYS-16: CPU placement ignores GB10's big.LITTLE split (X925 3.9 GHz vs A725 2.8 GHz)

- kind=perf component=CPU scheduling impact=1 confidence=2 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 1)

**Claim:** The containers are not pinned to cores. The EngineCore busy loop, worker launch threads, NCCL proxy threads and API server can land on the A725 efficiency cores (0-4, 10-14) instead of the X925 performance cores (5-9, 15-19). Steady-state decode hides most CPU time (async plus FULL graphs), but prefill/TTFT paths (piecewise/eager, tokenization, MM processing) and the NCCL proxy do not.

**Mechanism:** The NCCL proxy thread services every RoCE all-reduce (GDR 0). Running it on a slow core adds per-op latency across ~100 ops/step. The Python-heavy prefill and scheduling paths run ~1.4x+ slower on A725 (frequency alone).

**Evidence:**
- lscpu -e: CPUs 0-4,10-14 max 2808 MHz (A725); 5-9,15-19 max 3900 MHz (X925); governor performance
- run.sh:311-323 docker run has no --cpuset-cpus
- engine.log.tail: 'Reducing Torch threads from 20 to 1 for serving'
- tonyd2wild GLM 2x Spark README uses --cpuset-cpus

**Proposed action:** Opt-in CPUSET (default unset) forwarded to both ranks, e.g. CPUSET=5-9,15-19. Measure prose c=1/c=2 decode and TTFT. If neutral on decode but better on TTFT, keep it for TTFT.

**Est. impact:** Decode +0-3% (mostly NCCL proxy latency); short-prompt TTFT -5-10% (estimate, unverified).

**Validation:** Single knob A/B with bench_decode (decode + TTFT) and loadavg captured.

**Risks:** Restricting to 10 cores can starve tokenization or MM preprocessing under load. Other host workloads get pushed onto the A725 cores.

**Verifier reasoning:** lscpu confirms CPUs 0-4 and 10-14 at 2808 MHz, 5-9 and 15-19 at 3900 MHz, governor performance. run.sh:311-323 has no --cpuset-cpus. The cited 'tonyd2wild README uses --cpuset-cpus' is not in the scratchpad README copy (grep cpuset: no match). The impact is speculative and far below the bench noise floor.

**Verifier corrected impact:** Decode likely <1%, not measurable with the current n=3 prose bench.

## Open questions
- What does NCCL actually pick for 64-128 KiB all-reduces on this pair (protocol, algo, channels), and is GDR 0 or 1? One boot with NCCL_DEBUG=INFO is needed, which requires the env passthrough from SYS-8.
- Does passwordless sudo exist on spark1/spark2? The read-only check was refused in this isolated session. If not, maybe_drop_caches has never run.
- What is the actual distinct-expert count per MoE layer per verify step on prose vs structured? A router-trace probe would tighten the SYS-1 byte model (independence gives 58/layer; the DFlash2-5 vs DFlash2-7 step delta suggests fewer).
- Is the GB10 GPU clock healthy on both nodes? tonyd2wild saw GPUs stuck at 611-890 MHz / ~14 W after watchdog reboots, which would look like a bandwidth or straggler problem. It needs a GPU-side read, which was not allowed here.
- Is the per-request KV footprint of 26.7% of the pool driven by the Mamba 'align' prefix-cache mode plus 7 spec slots, or by one-block-per-group granularity at block 4608? This decides whether a bigger pin or a prefix-cache fix (tony's #18) is the better memory lever.
- How much UMA do the NCCL communicators (TP, EP and world groups, pynccl plus torch NCCL) and FlashInfer/Marlin workspaces take per rank? Suspected 0.3-1 GiB; not measured.
- Is the head's 3.2x slower weight load caused by page-cache reclaim of the previous DeepSeek serve, by disk contention, or by interactive load? Capture vmstat and iostat during the next boot.

## Verifier: missed issues
- The c=2 bench cell is not a real two-user test, and the reviewer used it as if it were. bench_decode.py:34-36 sends the same prompt to every stream with temperature 0, so c=2 streams are greedy-identical and share all routed experts. This makes the published c=2 numbers optimistic for a bandwidth-bound MoE. It is also the cleanest evidence against SYS-1: the step still grows from 115 ms to 131 ms (structured 60.41/7.92) and 150 ms (prose 16.64/2.50) with no extra expert bytes. Fix: add per-stream salted or different prompts before using c=2 for system decisions.
- Noise floor. Prose c=1 runs 23.40/18.75/21.18 (sd about 11%, n=3) make every 1-5% system knob in SYS-2/6/13/16 undetectable. A two-sample test for a 3% effect at this sd needs about 200 runs per arm. Structured c=1 has about 2-4% spread (67.13/67.64/70.23) because acceptance is pinned near 7.8 per step. System knobs should be ranked on step time, either structured tok/s ÷ acceptance or engine spec-decode metrics, not on prose tok/s, whose variance is mostly acceptance.
- SYS-7's prefill arithmetic used the logger's 10 s window average (engine.log.tail:385, 4438 tok/s). The measured needle prefill is 1443 tok/s (needle-20480-c2.txt: 44,386 tokens, TTFT 30.75 s) and 884 tok/s (needle-8192.txt). Any prefill or TTFT estimate built on 4438 tok/s overstates comm share by about 3x.
- The h-snap wave2 TTFT stalls (prose c=1 run1 3.635 s, c=2 run1 6.630 s on a warm serve, after wave1's 0.403 s) are the strongest in-repo sign of swap or reclaim stalls. The reviewer attributed the c=2 one to JIT instead. No bench captures /proc/vmstat or PSI deltas, so the swap hypothesis cannot be tested from evidence/. Every future bench should record them.
- Tony's measured NVFP4-attention/MLP analogue (tony_nvfp4_attn.md: TP4 measured x1.14-1.30, TP2 predicted x1.21, frees 5.21 GiB/rank) is the best evidence for a bytes-per-step lever. It uses the Marlin W4A16 path that already runs on sm_121 here (rank1 log line 35). The reviewer's FP8 path depends on an unverified sm_121 FP8 GEMM in a _C built for 12.0.
- Tony's README also warns to use /health rather than /v1/models for liveness ('returns 200 from config alone, with a dead engine behind it'). run.sh wait_ready and PR #12's abort loop both key on /v1/models, and doctor/ready decisions after boot should use /health.
- The SYS-2 citation of tony's guidance is inverted. Per tony, swap OFF kills the worker during Marlin repack, and DEFAULT swappiness with swap on causes the mid-load UVM livelock. Any host-tuning proposal should cite it that way. The load-time risk (livelock during the 181-190 GiB stream through ~20 GiB free) matters more than the steady-state decode effect.
