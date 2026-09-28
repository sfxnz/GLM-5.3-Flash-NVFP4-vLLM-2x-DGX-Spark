# Dimension: open-prs

## Reviewer summary

I reviewed PRs #12, #9, #8 and #3 and the uncommitted b12x worktree against the #11 head (29954c8). I read the diffs, the evidence files, safetensors headers, and the vLLM/FlashInfer source shipped in v11. #12 cannot merge as written. Its 16 GiB MemAvailable tripwire, active while the API is still loading, would stop every healthy boot: MemAvailable is already 17.16 GiB at draft load, before the 4.14 GiB KV cache and CUDA graphs are allocated, and it sits at 5.0–5.7 GiB once the API is ready. Its 20 GiB start floor has the opposite problem on busy Sparks: this serve needs about 105–110 GiB per node, so a start with 20–100 GiB free would pass the check and then OOM the node. #9's revert of DFlash2-5 happened for the wrong reason (a stale lint literal), but the outcome was right. The +14% prose gain is measured against a low parity baseline (19.23). Against the published 21.2 it is +3.6%, inside noise, while structured falls 17.6% at c=1 and 31% at c=2, and the DFlash2-5 rebench failed its probe gate. Dividing tok/s by acceptance gives about 115–119 ms per verify step at k=7 and 105–108 ms at k=5. That is consistent with step time being set by the bytes of distinct experts read, and it points to an acceptance-adaptive k rather than a fixed k=5. #8 keeps the bench request body the same but fails on merge: it deletes files that #11 and #12 edit, and it carries the env-{} lint case that threw away H1. Rebase it onto #11 before #12 and #9. #3 matches the nvidia pack's shapes (31 BF16 o_proj tensors, 2.68 GB) but conflicts with the current tree, reuses the existing local `glm53-sm121-v12` tag (the nvrtc-dev image), fetches the donor from an unpinned `main` (now 745aac2, not the validated e90ef415), and would serve an uncensored model on 0.0.0.0 without auth. Close it in this form. The b12x worktree diagnosed the failure correctly. Three small Python-level changes (FlashInfer silu clamp, the dispatch that drops the limit, vLLM plumbing plus the oracle clamp set) would unblock it, but decode is already close to the bandwidth limit, so the expected decode gain is at most ~5–10%. Along the way I found that the nvidia pack's MTP layer 45 stores BF16 experts (13.5 GiB) while `hf_quant_config.json` does not exclude that layer, so `SPEC=mtp` as a rollback on the nvidia pin is probably broken. Recommended merge order: #11 (after one measured boot) → #8 rebased → #12 with fixes → #9 evidence only (drop a76474e) → b12x evidence and guard PR → #3 closed or reworked as a pinned opt-in overlay.

## PR12-1: #12 wait-phase 16 GiB MemAvailable tripwire would stop every healthy boot

- kind=pr-review component=run.sh wait_ready/wait_uma_or_abort (PR #12) impact=5 confidence=5 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** wait_uma_or_abort calls abort_load, which runs ./stop.sh on both ranks, whenever MemAvailable < UMA_ABORT_GIB (16) while /v1/models is not up. On a healthy boot MemAvailable is already near 17 GiB after the main weights and falls to about 5–10 GiB during KV allocation and CUDA graph capture, all before the API server starts. The tripwire therefore fires in the last few minutes of every normal boot. The PR text assumes the drop to about 8 GiB happens after ready. It does not, and the 8 GiB figure is also stale (5.0–5.7 GiB measured).

**Mechanism:** psutil 'Available RAM' is MemAvailable. KV (4.14 GiB), draft weights (2.18 GiB), CUDA graph pools and activation workspaces are all allocated inside 'init engine', before uvicorn starts. 17.16 − 2.18 − 4.14 ≈ 10.8 GiB, already below 16 GiB, and it keeps falling to ≈5.7 GiB. The nvidia pack is not larger per rank under dflash2 (layer 45 is skipped, model.py:820, and the dense MLP drops 0.84→0.24 GiB), so the LibertAI numbers carry over.

**Evidence:**
- gh pr diff 12: run.sh wait_ready hunk (@@ -361,29 +478,40) calls wait_uma_or_abort "" and wait_uma_or_abort "$WORKER_HOST" every 5 s before the API is up; abort_kib=$((UMA_ABORT_GIB*1024*1024))
- evidence/rebench-20260902T204243Z/engine.log.tail:45 'Checkpoint size: 2.18 GiB. Available RAM: 17.16 GiB' (draft load, after 'Model loading took 90.67 GiB' at :52)
- engine.log.tail:68 reserved 4.14 GiB KV; :94-97 graph capture; :102 'init engine (profile, create kv cache, warmup model) took 277.93 s'; :146-147 APIServer 'Application startup complete' comes after all of it
- opt-b12x worktree evidence/iter-b12x-A-marlin/postboot.txt: spark1 available 5.7Gi right after ready; postboot-spark2.txt: 9.8Gi
- evidence/rebench-20260902T204243Z/free-after-bench.txt:2 available 5 (GiB), swap used 6
- evidence/oom-20260831/diagnosis.txt:10-11 avail 18.21/17.50 GiB at draft load, even on the failed boot

**Proposed action:** Do not merge #12 until this is fixed. (a) Drop the static 16 GiB floor, or set it below the healthy minimum with margin (2–3 GiB; healthy minimum is 5.0 GiB after load). (b) Better: trip on real distress rather than a level. /proc/pressure/memory 'full avg10' above a threshold, rising SwapTotal−SwapFree while MemAvailable is under 3 GiB, or kernel OOM-killer lines in dmesg. (c) Log MemAvailable per poll into run.log so the next boot records the real curve. (d) Fix the README/AGENTS text: after a healthy boot MemAvailable is ~5–6 GiB on spark1 and ~10 GiB on spark2, not ~8.

**Est. impact:** As written, recipe availability drops to 0%: each boot is killed about 15–18 minutes in (ready_seconds 1095 in rebench-dflash5) after spending 90.67 GiB of weight load. With the fix, the guard still catches a real OOM spiral, like the cutlass JIT case that went from ~18 GiB to 0.

**Validation:** GPU-free: unit-test wait_uma_or_abort with an injectable meminfo path (add MEMINFO=${MEMINFO:-/proc/meminfo}) replaying the curve 114→17.16→10.8→5.7 GiB and assert no abort. On a later exclusive Spark slot, run a read-only sampler (awk MemAvailable every 5 s on both nodes) during a normal ./run.sh and record the minimum before ready.

**Risks:** Setting the floor too low (<2 GiB) lets a true spiral reach NV_ERR_NO_MEMORY before abort. PSI thresholds need one calibration boot.

**Verifier reasoning:** `gh pr diff 12` (md5 ab1e5762… matches the head fa8c305) shows `wait_uma_or_abort` running every 5 s inside `wait_ready` before `/v1/models` answers. It reads MemAvailable on the head and, through `ssh_worker`, on spark2, then calls `abort_load` → `./stop.sh` when the value is below UMA_ABORT_GIB×1024² (16 GiB). engine.log.tail:45 shows 'Available RAM: 17.16 GiB' at draft load (20:55:36). Line 68 reserves 4.14 GiB of KV at 20:57:56. Lines 94-98 capture graphs, line 102 ends init at 277.93 s, and line 147 'Application startup complete' comes after all of that. The KV reservation alone takes MemAvailable to about 13 GiB, below 16, minutes before the API is up. After ready, the b12x-A postboot files show 5.7Gi on spark1 and 9.8Gi on spark2, free-after-bench shows 5 GiB, and oom-20260831 shows 17.50/18.21 GiB at draft load. Both nodes therefore cross 16 GiB during every normal boot. One small note: 17.16 GiB is printed after the draft params are allocated, so also subtracting the 2.18 GiB draft double-counts it. The conclusion does not change.

## PR12-2: #12 20 GiB start floor passes when a co-tenant holds most of UMA, which is the dangerous case

- kind=pr-review component=run.sh refuse_low_uma (PR #12) impact=5 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The start guard only refuses below 20 GiB MemAvailable. This serve needs about 105–110 GiB per node (116 GiB used at steady state). A start with 30–100 GiB available would pass, then OOM mid-load. GB10 OOMs here have been global (CONSTRAINT_NONE) and killed unrelated processes. The guard is therefore missing exactly in the case the task describes (Sparks busy with another workload). Also, page cache from the mmap'd safetensors is not a false-refuse risk: MemAvailable already counts reclaimable file pages.

**Mechanism:** Required ≈ 90.67 (weights) + 2.18 (draft) + 4.14 (KV) + ~6–10 GiB (CUDA ctx, graphs, NCCL/IB buffers, activations, mm profiling) ≈ 103–107 GiB, plus ~5 GiB left free at steady state. MemAvailable = MemFree + reclaimable file LRU + slab, so cached safetensors pages count as available and do not cause false refusals. vm.swappiness=60 with 15 GiB swap can briefly inflate MemAvailable as a co-tenant is swapped out, which is another source of false passes.

**Evidence:**
- gh pr diff 12: UMA_RESERVE_GIB default 20; refuse_low_uma compares kib < UMA_RESERVE_GIB*1024*1024
- opt-b12x evidence/iter-b12x-A-marlin/preboot.txt: idle spark1 available 114Gi (buff/cache 7.4Gi); postboot.txt used 115Gi; preboot-spark2.txt 117Gi available
- evidence/oom-20260831/diagnosis.txt:12-14 'constraint=CONSTRAINT_NONE global_oom', invokers dockerd, cudafe++, gsd-media-keys
- engine.log.tail:52 'Model loading took 90.67 GiB'; :68 KV 4.14 GiB; :45 draft 2.18 GiB

**Proposed action:** Set the default UMA_RESERVE_GIB from a derived need, e.g. need_gib = weights_per_rank(90.7) + draft(2.2) + KV(4.14) + overhead(10) ≈ 107. Compute it in run.sh from KV_CACHE_MEMORY and SPEC so a knob change moves the floor, and refuse below it unless FORCE_UNSAFE_UMA=1. Keep a separate message for 'this recipe's own container is still running' (see PR12-3).

**Est. impact:** Stops a start that would otherwise global-OOM a co-tenant. The ~20 GiB→~107 GiB change only affects starts that could not have succeeded anyway, so no healthy start is lost (idle nodes show 114–117 GiB).

**Validation:** GPU-free test with fake meminfo values of 30, 100 and 114 GiB: expect refuse, refuse, pass. Check the overhead term against the next boot's MemAvailable curve (PR12-1 sampler).

**Risks:** If the overhead term is overestimated, a start that would have fit on an idle node near 110 GiB is refused. Record the measured minimum and tune it.

**Verifier reasoning:** The diff sets UMA_RESERVE_GIB default 20 and `refuse_low_uma` compares against it. Idle nodes read 114Gi (spark1) and 117Gi (spark2) available. After boot, spark1 reads 115Gi used with swap at 7.3Gi, up from 340Mi. The serve therefore consumes about 108 GiB of MemAvailable plus about 7 GiB of swap on spark1, so a 20 GiB floor lets through almost any start that cannot fit. oom-20260831 records CONSTRAINT_NONE global_oom, with dockerd, cudafe++ and gsd-media-keys among the invokers. One overstatement: `refuse_foreign_serve` already refuses any co-tenant container that has GPU/IB device requests. The floor is the only guard only for non-Docker or CPU-only co-tenants, so this is not quite 'exactly the case described'. The proposed ~107 GiB floor is well supported, though idle MemAvailable of 114 GiB leaves only about 7 GiB of margin.

**Verifier corrected claim:** The 20 GiB start floor is far below the ~108 GiB of MemAvailable this serve consumes per node, measured as 114→5.7 GiB plus 7 GiB of swap on spark1. It lets through starts that cannot fit whenever the co-tenant is not a GPU/IB Docker container, which refuse_foreign_serve already catches.

**Verifier corrected impact:** High, not maximal: it prevents global OOMs caused by non-Docker or CPU-heavy co-tenants. GPU-container co-tenants are already refused by refuse_foreign_serve.

## PR12-3: #12 watcher/refuse logic: SSH blips kill a healthy load, env grep refuses GPU-less CUDA-image containers, non-Docker GPU users pass, re-run behaviour changes, worker can be orphaned; tests only grep text

- kind=pr-review component=run.sh refuse_foreign_serve/wait_ready/start_local + tests/test_recipe_ops.py (PR #12) impact=4 confidence=4 effort=M needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** (1) Each 5 s poll makes 2 SSH calls to spark2 (docker ps, then meminfo) with ConnectTimeout=5 and no retry. One transient SSH failure goes to abort_load ('Worker container ... exited early' or 'Could not read MemAvailable') and stops both ranks. That is ~960 chances over a 40-minute wait. (2) refuse_foreign_serve greps 'gpu|nvidia|infiniband' across .Config.Env. Every image built on an NVIDIA CUDA base carries PATH=/usr/local/nvidia/bin, NVIDIA_VISIBLE_DEVICES=all and NVIDIA_REQUIRE_CUDA, so a CPU-only helper container from such an image is refused. (3) GPU users outside Docker (bare-metal python/vLLM, containerd/k8s) are not detected. (4) refuse_low_uma runs before stop_local, so re-running ./run.sh while this recipe is up (~5 GiB available) now refuses with a misleading 'first NVIDIA load' message, where it used to docker rm -f and restart. (5) Head-side refusals inside start_local 0 run after the worker was already started and not stopped. The worker keeps ~90 GiB loaded and waits for NCCL for up to VLLM_ENGINE_READY_TIMEOUT_S=3600. (6) The tests assert source substrings. VALIDATE_ONLY exits at run.sh:123-128, before any new guard, so CI never runs refuse_low_uma, refuse_foreign_serve or the watcher.

**Mechanism:** Level-triggered, single-sample checks over SSH turn transient faults into hard kills. Grepping env matches the image lineage, not device use. Guards added after the worker start have no rollback path.

**Evidence:**
- gh pr diff 12: ssh_worker uses 'ssh -o BatchMode=yes -o ConnectTimeout=5'; wait_ready calls ssh_worker docker ps and mem_available_kib per iteration and aborts on the first failure
- docker history glm53-sm121-v12 (local): ENV NVIDIA_VISIBLE_DEVICES=all, PATH=/usr/local/nvidia/bin..., NVIDIA_REQUIRE_CUDA=... (inherited by any CUDA-based image)
- run.sh:238-241 (#11) start_local order maybe_drop_caches→stop_local; #12 inserts refuse_foreign_serve/refuse_low_uma before stop_local
- run.sh:381-394 (#11) worker started via ssh, sleep 25, then start_local 0 (#12 adds refusals there with no worker cleanup)
- run.sh:253 VLLM_ENGINE_READY_TIMEOUT_S=3600
- tests/test_recipe_ops.py (PR #12): test_refuse_foreign_serve_* / test_wait_ready_* use assertIn on function bodies; test_validate_only_* runs VALIDATE_ONLY=1, which exits before the guards (run.sh:123-128)

**Proposed action:** Merge after fixes: (a) abort only after N=3 consecutive failed SSH/meminfo reads, and only on 'container gone' confirmed twice. (b) Detect GPU use from .HostConfig.DeviceRequests/Devices/Runtime (and Privileged) only, not Env. Add a host-level check for non-Docker GPU users (e.g. 'fuser /dev/nvidia0 /dev/nvidiactl' or /proc/*/maps scan; no nvidia-smi, per AGENTS.md). (c) Run stop_local for this recipe's own container name (with a log line) before refuse_low_uma, or detect it and say 'run ./stop.sh first'. (d) On the head, add 'trap ... ./stop.sh' between the worker start and ready so any head exit stops rank 1. (e) Make meminfo/docker injectable (MEMINFO path, DOCKER=fake script) and add behavioural tests: floor pass/refuse, 3-strike SSH, CUDA-base env not refused, own container handled.

**Est. impact:** Removes a false-kill risk that scales with boot length (~960 SSH polls per boot) and a false-refuse class. Stops a ~90 GiB orphaned rank 1 that could otherwise sit for up to 60 minutes.

**Validation:** GPU-free: bash unit tests with PATH-shimmed ssh/docker that fail once or twice then succeed (expect no abort), plus a fake container whose env has NVIDIA_VISIBLE_DEVICES but no DeviceRequests (expect no refuse). Confirm on Sparks only with a read-only 'docker ps' during the next slot.

**Risks:** Device-only detection misses a container started with --privileged and no device flags, so check Privileged too.

**Verifier reasoning:** Each point checks out against the diff. (1) `ssh_worker` uses ConnectTimeout=5 with no retry. Per iteration, `ssh_worker docker ps|grep` failing leads to abort 'Worker container ... exited early', and a failed meminfo read leads to abort. 480 iterations × 2 = 960 is the 40-minute maximum; a normal boot (~18 min) is about 440 calls. (2) `docker image inspect glm53-sm121-v11` Env contains PATH=/usr/local/nvidia/bin, NVIDIA_VISIBLE_DEVICES=all and NVIDIA_REQUIRE_CUDA, and the grep covers `.Config.Env`. (3) Only `docker ps` is enumerated. (4) `refuse_foreign_serve`/`refuse_low_uma` are inserted before `stop_local` in start_local and at the top of the ORCHESTRATE block, so a re-run while this recipe is up (~5 GiB available) refuses. (5) This holds, but the risk is low: the head re-runs checks that passed about 25 s earlier. Pre-existing ensure_image/ensure_weights failures in start_local 0 already orphan rank 1 the same way. (6) The tests are assertIn on function bodies, and VALIDATE_ONLY exits at run.sh:123-128 before any guard.

**Verifier corrected impact:** Point (5) is mostly pre-existing (ensure_image/ensure_weights failures orphan rank 1 too), and #12 adds only a small race window to it. The SSH false-kill risk is about 440 polls on a typical 18-minute boot, not 960.

## PR9-1: #9 DFlash2-5 was reverted for the wrong reason, but reverting was right; drop a76474e

- kind=methodology component=PR #9 hillclimb H1 (NUM_SPECULATIVE_TOKENS=5) impact=4 confidence=5 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** iterate.sh reverted H1 only because recipe_lint's expect_case {env: {}} encoded the old ladder. On the evidence, H1 still does not meet the AGENTS.md bar ('beat noise and do not regress another cell'). (a) The +2.74 (+14%) is measured against the parity baseline 19.23, which itself sits 9.2% below the published 21.2 for the same config. Against 21.2, H1 (21.97) is +3.6% and the DFlash2-5 rebench (21.76) is +2.6%, both inside the 8% band. The per-run ranges overlap: H1 19.86/21.97/23.62 vs published 18.75/21.18/23.40. (b) The same-boot DFlash2-5 rebench shows structured c=1 67.6→55.73 (−17.6%) and c=2 per-stream 60.4→41.56 (−31%), aggregate 120.8→81.3 (−33%). (c) That rebench failed its probe gate (needle-8192 refused as injection). Hand-applied commit a76474e also leaves README/AGENTS/run.sh comments claiming DFlash2-7 ('Default is the trained block (7 of 8)', '1/2/4 + 6/12' hardcoded) and a decode table from a different default. Prose c=2 does improve (16.6→19.62, +18%), which is real signal for adaptive k.

**Mechanism:** The hillclimb loop gated on prose c=1 only (phases=['prose'] in bench.log). It used a same-night baseline 9% low, and its 8% band is narrower than the ruler's own ±11–22% 3-run spread, so a keep was likely from noise alone. Structured, where DFlash2-7's extra slots are accepted (7.84 of 8), was not measured by the loop.

**Evidence:**
- PR #9 evidence/iter-H1-20260903-20260903T010419Z/verdict.json: reason 'render or lint failed after keep', before_c1 19.23 after_c1 21.97
- iter-H1 lint.log: 'FAIL VALIDATE_ONLY=1 {} did not print "cudagraph_capture_sizes":[1,2,4,8,16]'
- iter-H1 bench.log: prose c=1 per-run 21.97/23.62/19.86, acceptance_len 2.308
- evidence/rebench-parity-20260903T005500Z/bench.json (copy of parity-20260902T225109Z): prose c=1 19.23 acc 2.292; structured c=1 66.55 acc 7.844
- PR #8 evidence/parity-20260902T225109Z/PARITY-RESULT.md: reference prose c=1 runs 23.40/18.75/21.18; parity 19.2 (−9.2%)
- PR #9 evidence/rebench-dflash5-20260903T045815Z/bench.json: prose c=1 21.76 (acc 2.354), c=2 19.62; structured c=1 55.73 (acc 6.0), c=2 41.56 (agg 81.34); summary.json result=fail probes=false
- recipe.yaml measured rows (#11): prose c=1 21.2, c=2 16.6; README structured 67.6/60.4
- git diff origin/agent/kit origin/agent/hillclimb-20260903 -- recipe.yaml run.sh: NUM_SPECULATIVE_TOKENS 7→5 and hardcoded '1/2/4 + 6/12'

**Proposed action:** Disposition for #9: merge after fixes as evidence only. Drop a76474e (the DFlash2-5 default and README) and 0e90782's ledger rows for the 'apply'. Keep the H1–H4 receipts, the rebench-dflash5 receipts, c0c4af3 (move it into #8, see PR8-1) and e898a80 (needle passphrase payload). Change the decision.tsv H1 verdict to 'reverted: inside noise vs published 21.2; structured −17.6%/−31%'. Rebase onto main after #8 lands. The loop should gate on both phases and compare against the published row, not a same-night copy.

**Est. impact:** Keeps structured throughput from dropping ~18% (c=1) and ~33% (c=2 aggregate) for a prose change that is not statistically real (+2.6–3.6% vs published).

**Validation:** GPU-free: recompute from the committed bench.json files (done above). Any future k change needs ≥9 runs per cell, both phases, same boot as its baseline (see PR9-2).

**Risks:** If the maintainer re-scopes 'published = prose only' so that structured never vetoes, k=5 becomes defensible for prose c=2 (+18%). That is a policy decision; state it in AGENTS.md either way.

**Verifier reasoning:** iter-H1 verdict.json reason is 'render or lint failed after keep: render: wrote README.md result=fail'. lint.log shows 'FAIL VALIDATE_ONLY=1 {} did not print cudagraph_capture_sizes [1,2,4,8,16]', from #8's `expect_cases {env: {}}`. Recomputed from the committed files: the published run is prose c=1 21.18 and parity is 19.23, which is 9.2% lower. H1 at 21.97 is +3.7% over published and dflash5 at 21.76 is +2.7%. Structured c=1 goes 67.64→55.73 (−17.6%), structured c=2 per-stream 60.41→41.56 (−31.2%) and aggregate 120.76→81.34 (−32.6%). Prose c=2 goes 16.64→19.62 (+17.9%). rebench-dflash5 summary.json has probes_pass false, and needle-8192 is a refusal. a76474e hand-edits README/recipe.yaml to 'capture ladder 1/2/4 + 6/12' and leaves the run.sh comment 'Default is the trained block (7 of 8)'. The structured comparison is cross-boot, but it also holds against the k=7 parity boot (−16.3% at c=1, −25.1% at c=2).

## PR9-2: Step time, not tok/s, is the stable signal: k=7 ≈115–119 ms/step, k=5 ≈105–108 ms/step, consistent with a distinct-expert-bytes model

- kind=methodology component=bench ruler / speculative decoding (evidence from #8/#9) impact=4 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Dividing decode tok/s by acceptance_len gives verify steps/s, which is nearly independent of phase: k=7 prose 19.23/2.292=8.39 and 21.2/2.43=8.72, structured 66.55/7.844=8.48 and 67.6/7.84=8.62 (115–119 ms/step). k=5 prose 21.97/2.308=9.52 and 21.76/2.354=9.24, structured 55.73/6.0=9.29 (105–108 ms). MTP-4: 19.22/2.175=8.84 (113 ms). Removing 2 verify slots saves ~10 ms/step. A bandwidth model predicts ~14.5 ms at peak BW: E[distinct experts] with 8 vs 6 independent tokens = 288(1−(280/288)^n) = 58.1 vs 44.8. The difference, 13.3 experts × 42 layers × 7.08 MB/expert/rank = 3.95 GB/rank, costs 14.5 ms at 273 GB/s. Acceptance itself varies 2.17–2.43 across boots at fixed k=7 and a greedy prompt, which explains much of the ±11–22% tok/s noise.

**Mechanism:** At c=1 each verify step streams the weights of every distinct expert hit by the k+1 tokens, plus BF16 attention (≥5.7 GiB/rank), shared expert, lm_head and drafter. Total ≈ 26–27 GB/rank/step ≈ 96–100 ms at 273 GB/s, so the observed 115 ms is ~85% of peak. Step cost therefore grows with verify width, while prose accepts only ~2.3 tokens. Correlated routing across consecutive positions makes the independence estimate an upper bound.

**Evidence:**
- PR #9 iter-H1/H2/H3 and rebench-dflash5 bench.json acceptance_len values (H2 2.175, H3 2.168, H1 2.308)
- parity bench.json (PR #8/#9): prose 19.23 acc 2.292, structured 66.55 acc 7.844
- safetensors headers (nvidia 09b04e5): routed_experts 159.47 GiB / (42 MoE layers × 288) = 13.50 MiB per expert, so 6.75 MiB (7.08 MB) per rank at TP=2; attn 11.32 GiB BF16; shared_expert 1.97 GiB; lm_head 1.18 GiB (scratchpad/agents/prs/hdr.py output)
- kit/bench_decode.py (PR #8): spec counters are read once before and once after all runs of a (phase,c) cell, so there is no per-run acceptance

**Proposed action:** (1) In the kit bench, snapshot spec counters per run and report steps_per_s = decode/acceptance_len and ms_per_step next to tok/s. Gate kernel-level changes on ms_per_step (it removes acceptance variance). Raise runs to ≥9 for prose (DSV41 uses 9-run medians). (2) Treat the k result as evidence for acceptance-adaptive speculation (per-request k by running acceptance, or k=3–5 for low-acceptance streams and 7 for high) instead of a fixed default. (3) Report per-step byte budgets so the planning agent can rank byte cuts, e.g. FP8 attention weights (~3 GB/rank/step ≈ 11 ms) against MoE kernel swaps.

**Est. impact:** A better ruler: step-rate spread across the k=7 boots is 8.39–8.72 (±2%) against ±10% for tok/s, so changes of ~3–4% become detectable instead of ~20%. Model-based estimate for adaptive k: prose at k≈4–5 gains +5–10% step rate over k=7 while structured keeps k=7 (no −17.6%/−31% regression).

**Validation:** On the next exclusive slot, one boot, same session: 9 runs per cell at k=3,5,7 (one knob per boot). Plot ms_per_step against distinct-expert estimate × 42 × 7.08 MB and check the linear fit. Compare acceptance per run to size the noise split.

**Risks:** Distinct-expert counts are an estimate (routing across consecutive tokens is correlated). Adaptive k needs DFlash2 backport support for per-request k, and CUDA-graph sizes must cover each (k+1)×seqs.

**Verifier reasoning:** The c=1 arithmetic is correct. Step times are 21.18/2.426→114.6 ms, 19.23/2.292→119.2, 67.64/7.844→116.0, 66.55/7.844→117.9, H1 21.97/2.308→105.0, dflash5 21.76/2.354→108.2 and 55.73/6.0→107.7, and MTP 19.22/2.175→113.2 ms. Distinct experts are 288(1−(280/288)^n) = 58.1 for n=8 and 44.8 for n=6; 13.3×42×7.08 MB = 3.95 GB, which is 14.5 ms at 273 GB/s. The header numbers (159.47 GiB / 12096 = 13.50 MiB per expert) are verified. Problems: (a) The evidence mislabels H2's 2.175 as 'fixed k=7'. H2 is SPEC=mtp. (b) The '±2% step-rate spread across k=7 boots' leaves out H3: MAX_NUM_SEQS=1 at k=7 gives 17.40/2.168 = 8.03 steps/s, or 124.6 ms, so the k=7 spread is 114.6–124.6 ms (about ±4%). (c) The model breaks at c=2. In the same k=5 boot, structured c=2 runs 41.56/5.98 = 144 ms per step against prose c=2 at 19.62/2.573 = 131 ms, even though the structured streams are identical and share experts. k=5 structured c=2 (144 ms) is also slower per step than k=7 structured c=2 (60.41/7.92 = 131 ms published; 142 ms parity). (d) The observed k7→k5 saving of 7–10 ms is about 50–70% of the predicted 14.5 ms, and fixed costs (NCCL all-reduces over RoCE, mHC sinkhorn, KDA recurrence, drafter) are not modelled. The direction (step cost grows with verify width at c=1) is supported. 'Consistent with a distinct-expert-bytes model' and the ±2% ruler claim are not proven.

**Verifier corrected claim:** At c=1, verify step time is 105–108 ms at k=5, 114.6–124.6 ms at k=7 (including H3) and 113 ms with MTP-4. That is directionally consistent with the cost of verify width, but c=2 data contradicts a pure distinct-expert-bytes model: same-boot k=5 structured c=2 runs at 144 ms/step vs prose c=2 at 131 ms. The step-rate spread at fixed k=7 is about ±4%, not ±2%.

**Verifier corrected impact:** ms_per_step with per-run acceptance is still a better ruler than tok/s, at roughly ±4% instead of ±10–20%. The adaptive-k gain of +5–10% is a model estimate with weak validation.

## PR9-3: Needle evidence shows chain-of-thought in content with thinking off at 8k–16k context, and a safety refusal; probes still pass

- kind=quality component=quality probes (kit/probes/needle.py, thinking_off) as used in #8/#9 impact=3 confidence=4 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** In the H1 needle-8192 (16,400 prompt tokens) with enable_thinking=false, content starts 'The user is asking me to repeat a secret code that they've embedded in their message...'. That is reasoning in content, the leak AGENTS.md forbids, but the probe passes because the needle substring appears. In rebench-dflash5 the same probe got a refusal ('classic prompt injection / data exfiltration pattern'). #9's e898a80 switches the needle to a passphrase payload, which avoids the refusal but hides the leak. thinking_off.py only tests a 24-token 'PING' prompt.

**Mechanism:** The local chat_template seeds <think></think> when thinking is off, but on long, odd prompts the model still writes meta-reasoning in plain text. Substring-hit probes cannot see this.

**Evidence:**
- PR #9 evidence/iter-H1-20260903-20260903T010419Z/needle-8192.txt: '--- response content The user is asking me to repeat a secret code ...' hit=1 verdict=PASS
- PR #9 evidence/rebench-dflash5-20260903T045815Z/needle-8192.txt: hit=0 'I can't do that ... prompt injection' verdict=FAIL
- PR #9 kit/probes/needle.py diff: new default payload 'passphrase'; is_hit only checks words present
- PR #9 rebench-dflash5 thinking_off.txt: prompt_tokens 24, content 'PING' PASS
- AGENTS.md: 'Thinking-off smoke must not start content with chain-of-thought'

**Proposed action:** Add a content-shape check to needle.py and thinking_off.py: fail or warn when content starts with meta-reasoning markers ('The user is asking', 'Let me', 'I need to', '<think>') or when completion length far exceeds 'answer only'. Add a long-context thinking-off case (≥16k tokens). Keep both payloads: passphrase for window/recall, code for the refusal/robustness signal. Run on the nvidia pack before #11 is published, since its template/quant differences may change this.

**Est. impact:** Output quality: catches a user-visible CoT leak regression. It occurred in 1 of 2 long-context 8k needle transcripts reviewed, so it is not rare.

**Validation:** GPU-free: re-score all committed needle-*.txt with the new check (expect H1 needle-8192 flagged). On Sparks: run the long thinking-off probe on the #11 nvidia pin.

**Risks:** Heuristic markers can false-flag legitimate answers. Keep it a WARN until tuned on ~20 transcripts.

**Verifier reasoning:** The H1 needle-8192 content does begin 'The user is asking me to repeat a secret code...' and still passes on a substring hit. rebench-dflash5 needle-8192 is a refusal ('classic prompt injection / data exfiltration pattern'), recorded as hit=0 FAIL. I checked all five committed needle-8192 transcripts with content. H1 has a meta-reasoning opener. H2 answers first and then adds commentary ('This appears to be a needle-in-a-haystack test'), which is not CoT. H3 ('NEEDLECODE-7F3A91C2') and parity ('The secret code is ...') are clean, and dflash5 is the refusal. The leak-like start therefore occurs in 1 of 5 transcripts, not 1 of 2. Whether the H1 opener counts as chain-of-thought or just a verbose preamble is debatable. The recommendation to add a shape check to the probes is sound.

**Verifier corrected claim:** One of five committed 8k-needle transcripts (H1) starts content with meta-reasoning while thinking is off, and one (dflash5) refuses. Substring-hit probes cannot detect the first case.

**Verifier corrected impact:** Low to moderate quality signal (1/5 transcripts). Worth a WARN-level content-shape check and a long-context thinking-off probe on the nvidia pin.

## PR8-1: #8 kit: same bench request body, better evidence format, but conflicts with #11/#12, carries the env-{} lint bug that lost H1, and leaves the vision probe off

- kind=pr-review component=kit/ (PR #8), recipe.yaml bench/probes/lint impact=3 confidence=5 effort=M needs_gpu=False
- **verdict: confirmed** (corrected confidence 5)

**Claim:** Bench semantics are unchanged: same PHASES prompts, temperature 0, max_tokens 200, 3 runs, c=1,2, and chat_template_kwargs {enable_thinking:false}, now read from recipe.yaml. Two changes: a wave now fails on any failed stream or completion_tokens==0, and it writes bench.json with acceptance. The served name comes from recipe.yaml, so it follows #11's nvidia name automatically. Merging as-is would still break: (1) it deletes root bench_decode.py and .cursor/skills/verify-glm53-flash/*, which #11 modifies (bench default phase=prose and model name; recipe-lint.py +50 vision checks; new vision-smoke.md) and #12 modifies (recipe-lint.py +43 UMA checks, serve-start-stop.md), giving modify/delete conflicts, and both PRs' lint checks would vanish. (2) lint.expect_cases[0] {env: {}} tests defaults, which is the bug that auto-reverted H1 (#9 c0c4af3 fixes it, but only in #9). (3) kit/probes/vision.py exists but recipe.yaml probes: does not enable it, while #11 adds a separate root smoke_vision.py. (4) It is vendored at forge 6f40808; #9 re-vendors 6285f70 (redact.py, needle payload), so #8 alone is already stale.

**Mechanism:** The PRs were developed on parallel bases (main vs #11 head). A generic lint that reads defaults from env {} breaks every default-moving keep.

**Evidence:**
- PR #8 diff kit/bench_decode.py: only the chat_template_kwargs source, Tee/bench.json, and wave failure handling changed; PHASES and max_tokens unchanged
- PR #8 evidence/parity-20260902T225109Z/PARITY-RESULT.md: 'the kit's bench_decode.py sends the same request body as the pre-kit bench_decode.py'
- PR #8 recipe.yaml lint.expect_cases: '{env: {}, expect: "cudagraph_capture_sizes":[1,2,4,8,16]}'
- PR #9 iter-H1 lint.log failure caused by that case; c0c4af3 fix
- gh pr view 11 files: bench_decode.py, .cursor/skills/verify-glm53-flash/scripts/recipe-lint.py (+50), features/vision-smoke.md; gh pr view 12 files: recipe-lint.py (+43)
- PR #8 kit/probes/vision.py and run-all.sh list 'vision'; recipe.yaml probes list lacks it
- PR #9 diff: kit/* headers 6f40808 → 6285f70, new kit/redact.py

**Proposed action:** Merge after fixes, rebased onto #11 (and before #12): port #11's vision/LANGUAGE_MODEL_ONLY refuse checks and #12's (fixed) guard checks into recipe.yaml lint: (refuse_cases/contains); fold c0c4af3 (pin env in expect_cases) and e898a80 (re-vendor 6285f70) into #8; enable '- vision: {image: solid-red, expect: red}' and delete root smoke_vision.py, or make it call the kit probe; set README/AGENTS verify commands to kit paths; keep the prose-only publication rule from #11 (the kit's default --phase both is fine for evidence). Add per-run acceptance and ms_per_step (PR9-2).

**Est. impact:** One ruler across recipes, bench.json with acceptance for every run, and no loss of #11/#12 checks. Adding per-run acceptance takes the ruler from ±10% to ±2% resolution on the step metric, which the kernel-level plan needs.

**Validation:** GPU-free: after rebase, python3 kit/recipe_lint.py . and kit/render.py --check --strict pass; VALIDATE_ONLY refuse cases for LANGUAGE_MODEL_ONLY=1, MAX_MODEL_LEN=1048576 and MOE_BACKEND=flashinfer_cutlass still fire; kit/sync.sh --check matches the forge SHA.

**Risks:** Vendored kit drift across recipes; pin the forge SHA in one place. The wave-fails-on-any-stream change can turn a flaky c=2 run into a failed bench, which is correct but noisier to operate.

**Verifier reasoning:** pr8.diff deletes .cursor/skills/verify-glm53-flash/* (including scripts/recipe-lint.py and features/serve-start-stop.md) and adds kit/probes/vision.py. recipe.yaml `probes:` lists smoke/count/thinking_off/tool_call/hermes_two_turn/needle with no vision entry. `lint.expect_cases[0]` is `{env: {}, expect: [1,2,4,8,16]}`, the case that failed iter-H1's lint. `gh pr view 11` files include bench_decode.py, recipe-lint.py and vision-smoke.md; #12 touches recipe-lint.py and serve-start-stop.md; the kit header is @6f40808. Nuance: `gh pr view 8` reports MERGEABLE against main today. The modify/delete conflicts appear only after #11/#12 land or on a rebase onto #11, so 'merging as-is would break' is about merge order, not current mergeability. #8's lint also hardcodes README `contains: LibertAIDAI/GLM-5.3-Flash-NVFP4` and a `glm53-sm121-v11` build line, which need updating for #11's nvidia pin.

## PR3-1: #3 ABLIT o_proj transplant: shapes fit the nvidia pack, but conflicting, tag-colliding, unpinned, safety-sensitive; close in this form

- kind=pr-review component=PR #3 (ablit/, docker/Dockerfile.sm121-v12, run.sh) impact=3 confidence=4 effort=M needs_gpu=True
- **verdict: confirmed** (corrected confidence 4)

**Claim:** Compatibility: the nvidia pack keeps self_attn BF16 (excluded in hf_quant_config), and o_proj shapes match what the runtime slices. KDA layers [4096,8192] (64 MiB), MLA layers 15,19,...,43 plus MTP layer 45 [4096,16384] (128 MiB). Layers 15–45 give 22×64 + 9×128 MiB = 2.68 GB, matching the PR's '~2.7 GB'. RowParallel input-dim slicing is right for TP=2, and the hook anchor matches v11 model.py:926-928 and mtp.py:438-440. Blockers: (1) the base predates recipe.yaml/render and #11. It hand-edits generated run.sh blocks, still pins the aa28e1f snapshot path and LibertAI name, and is CONFLICTING. (2) It renames the default image to glm53-sm121-v12, but a different local glm53-sm121-v12 already exists (v11 + cuda-nvrtc-dev, used in the cutlass OOM), so a new build would silently replace or collide with it. (3) fetch_transplant.py defaults to REVISION='main'. Donor main is now 745aac2 (modified 2026-09-19), not the validated e90ef415. The SHA256 'verification' checks against a manifest built from the same download (trust on first use) with no hashes committed. (4) The donor card does not disclose which weights 'CRACK' edited, so transplanting only o_proj 15–45 may be partial and has unmeasured efficacy. (5) Safety: the donor reports 0% HarmBench refusals, and run.sh serves on 0.0.0.0 with no API key. (6) Quality: donor MMLU 85.28 vs 86.16 base (−0.88 pp). The DFlash2 drafter was trained on stock hidden states, so acceptance may drop (output stays lossless, speed may not). Licensing is fine: donor MIT, nvidia pack MIT, base MIT; the drafter is still CC BY-NC-ND.

**Mechanism:** A load-time in-memory weight swap after load_weights. Mechanically it is sound for BF16 RowParallel o_proj. The risk is in provenance, tagging, merge state and policy.

**Evidence:**
- gh pr view 3: mergeable CONFLICTING, base main at 351db8c era; diff run.sh SNAPSHOT=...aa28e1f..., IMAGE default glm53-sm121-v12
- docker history glm53-sm121-v12: top layer 2026-08-31 'apt-get install cuda-nvrtc-dev-13-0' (existing, different image)
- evidence/oom-20260831/diagnosis.txt:5 image=glm53-sm121-v12
- safetensors headers nvidia 09b04e5: layers.14/16/44 o_proj BF16 [4096,8192]; layers.15/45 o_proj BF16 [4096,16384]; hf_quant_config.json exclude 'layers.N.self_attn*'
- HF API dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4: license MIT, base zai-org/GLM-5.3-Flash, sha 745aac2ff0f1, lastModified 2026-09-19; card: MMLU 85.28 vs 86.16, HarmBench 0% refusals, modified layers not disclosed
- nvidia snapshot README.md header: license mit
- v11src model.py:926-928 and mtp.py:436-440 match the PR's MODEL_TAIL/MTP_TAIL anchors

**Proposed action:** Close as-is. If the owner wants the option: open a new PR on top of #11 and #8 that (a) builds a separate tag (glm53-sm121-v11-ablit), leaving v12 alone; (b) goes through recipe.yaml with ABLIT default 0 and a README warning; (c) pins ABLIT_DONOR_REVISION to a full SHA and commits the 31 expected sha256/nbytes, verified before any container stop; (d) refuses ABLIT=1 unless the serve binds 127.0.0.1 or an --api-key is set; (e) records a same-boot A/B (prose/structured steps/s, acceptance, count probe, MMLU-mini). Merge last, never on the default path.

**Est. impact:** Prevents a silent image-tag collision and an unpinned supply chain. Quality delta is about −0.9 pp MMLU (donor-reported); the DFlash2 acceptance effect is unknown (could lower prose tok/s by a few %).

**Validation:** GPU-free: compare donor header ranges (HTTP Range on headers only) to confirm dtype/shape for all 31 keys at the pinned SHA. On Sparks: A/B boot with ABLIT=0/1 on the same image and compare acceptance_len and ms_per_step.

**Risks:** Policy and reputational risk of serving an uncensored model on a shared tailnet. Maintenance: the patch anchors break on the next image rebase.

**Verifier reasoning:** Headers of nvidia 09b04e5: layers.14/44 o_proj are BF16 [4096,8192] and layers.15/45 are BF16 [4096,16384]. Layers 15–45 split into 22 KDA layers (64 MiB each) and 9 MLA layers (15,19,…,43 plus 45; 128 MiB each), totalling 2560 MiB = 2.68 GB. `gh pr view 3`: base main, CONFLICTING. The diff sets IMAGE default glm53-sm121-v12, `REVISION = os.environ.get('ABLIT_DONOR_REVISION','main')`, and the aa28e1f LibertAI snapshot. The local glm53-sm121-v12 (sha256:aca349f0…, 2026-08-31) is v11 plus a cuda-nvrtc-dev layer. The HF API shows the donor at sha 745aac2ff0f1, lastModified 2026-09-19, license MIT. The PR body says it was validated at e90ef415. The donor card states MMLU 85.28 ('within ~0.9 pt') and 0% HarmBench-320 refusals, and says the MTP head is 'also CRACK'd', but does not list the edited tensors. run.sh:281 binds --host 0.0.0.0 with no --api-key. Extra point: the PR body says its prototype logged 'transplanted layers 15–44', so layer 45 was never exercised under DFlash2.

## MTP-NV-1: On the nvidia pin, SPEC=mtp is probably not a working rollback: layer-45 experts are BF16 (13.5 GiB) but not excluded from NVFP4

- kind=correctness component=#11 rollback claim / #9 H2 / #3 layer-45 transplant; vllm glm5next mtp.py + modelopt impact=3 confidence=3 effort=S needs_gpu=True
- **verdict: plausible** (corrected confidence 4)

**Claim:** In the nvidia pack, layer 45 (the MTP block) stores routed experts as plain BF16 (e.g. experts.0.gate_proj.weight BF16 [2048,4096]; 13.50 GiB total, vs 3.80 GiB NVFP4 in LibertAI). hf_quant_config.json exclude_modules lists layers.0–44 attention/gate/shared but not layer 45. The MTP module is built with the full vllm_config.quant_config (mtp.py:44,224), and modelopt.py has no MTP special case. With SPEC=mtp, vLLM would therefore create NVFP4 FusedMoE params (packed U8 plus scales) for layer 45 and receive BF16 tensors. Likely outcome: a shape/missing-scale error at load. Even if it loaded, that is +4.85 GiB/rank against ~5 GiB headroom. #9's H2 (MTP-4 prose 19.22) was measured on the LibertAI pack and does not transfer to the nvidia pin.

**Mechanism:** ModelOpt left the MTP block unquantized but did not list it in exclude_modules. vLLM applies NVFP4 to every non-excluded linear/MoE.

**Evidence:**
- safetensors headers nvidia 09b04e5: 'mtp:routed_experts 13.50 GiB [BF16]'; layers.45.mlp.experts.0.{gate,up}_proj.weight BF16 [2048,4096], down_proj BF16 [4096,2048]; layers.44 experts U8 + F8_E4M3 scales
- LibertAI caca4e6 headers: 'mtp:routed_experts 3.80 GiB [F32, F8_E4M3, U8]'
- nvidia hf_quant_config.json:10-143 exclude_modules has no layers.45 entry
- v11src vllm/models/glm5next/nvidia/mtp.py:44 quant_config = vllm_config.quant_config; :436-438 raises if MTP layer weights missing
- v11src vllm/model_executor/layers/quantization/modelopt.py:139-237 exclusion is wildcard-only, no mtp/nextn handling
- AGENTS.md (#11): 'SPEC=mtp rolls back to MTP-4'

**Proposed action:** Before #11 publishes 'SPEC=mtp rolls back', either (a) make run.sh refuse SPEC=mtp when MODEL is the nvidia pack (FORCE override) with the reason above, or (b) add an image patch mapping 'model.language_model.layers.45.*' into exclude_modules for this pack (then budget the +4.85 GiB/rank BF16 experts; it likely will not fit next to the 4.14 GiB KV pin), or (c) point the MTP rollback at the LibertAI pin only. Mark #9 H2 evidence as LibertAI-only.

**Est. impact:** Avoids a ~17-minute failed boot, or an OOM on UMA, the first time someone uses the documented rollback. Memory: +4.85 GiB/rank if loaded as BF16 (6.75 vs 1.9 GiB), more than the ~5 GiB steady-state headroom.

**Validation:** GPU-free: trace FusedMoE weight_loader for a modelopt NVFP4 layer handed a BF16 [2048,4096] tensor (expect a shape assert). Then one boot with SPEC=mtp on the nvidia pin in an exclusive slot, with the PR12 sampler running.

**Risks:** If the fork's glm5next MTP loader has an unlisted fallback that de-quantizes layer 45 to BF16, it boots, and the memory risk is what remains.

**Verifier reasoning:** Verified statically, and stronger than the reviewer stated. Headers show layers.45.mlp.experts.* as BF16 (13.50 GiB) and layers.44 experts as U8+F8 scales. Neither hf_quant_config.json nor config.json `quantization_config.ignore` (132 entries) has any layers.45 entry. The nvidia exclude entries are per-layer names ('model.language_model.layers.N.self_attn*', '.mlp.gate', '.mlp.shared_experts*'). mtp.py:226 builds the MTP module under prefix 'model' → 'model.layers.45.*', and only the main model has the 'model.language_model.'→'language_model.model.' mapper (multimodal.py:354, model.py:1042). None of the nvidia exclusions can match any MTP-layer module, so modelopt.py `get_quant_method` would give NVFP4 methods to MTP attention, shared experts and gate as well as the routed experts, all of which are BF16 in the checkpoint. LibertAI caca4e6 config.json uses prefix-agnostic wildcards ('*.self_attn.q_proj', …), which explains why H2 (MTP) worked on that pack. The exact failure mode (shape assert vs missing scale vs silent misload) was not traced through the FusedMoE weight_loader and needs a boot to confirm.

**Verifier corrected claim:** On the nvidia pin, SPEC=mtp is very likely broken. The whole MTP layer 45 (routed experts 13.5 GiB, plus attention, shared expert and gate) is BF16 in the checkpoint, but the pack's per-layer 'model.language_model.layers.N.*' exclusions have no layer-45 entry and cannot match the MTP module's 'model.layers.45.*' prefix. vLLM would therefore build NVFP4 params for BF16 tensors. LibertAI's wildcard excludes do not have this problem, so H2 does not transfer.

**Verifier corrected impact:** The documented rollback fails at load, or needs a patch plus about +4.85 GiB/rank if loaded as BF16. Fix #11's AGENTS/README text or add a refuse-guard before publishing.

## B12X-1: b12x worktree diagnosis is accurate; exact patch to unblock flashinfer_b12x for GLM's clamped SiLU

- kind=perf component=FlashInfer 0.6.18 SM12x fused MoE + vLLM FlashInferB12xExperts/oracle (image layer) impact=3 confidence=5 effort=M needs_gpu=True
- **verdict: confirmed** (corrected confidence 5)

**Claim:** Verified in v11 source: (1) the oracle NVFP4_BACKENDS_WITH_CLAMP omits FLASHINFER_B12X and raises for an explicit request when swiglu_limit is set (nvfp4.py:191-199, 259-269), which is the logged ValueError. (2) vLLM FlashInferB12xExperts never passes swiglu_limit to B12xMoEWrapper (flashinfer_b12x_moe.py:242-251). (3) FlashInfer's gated_activation_f32 clamps only in the swigluoai branch. The silu branch ignores `limit` (moe_activation.py:114-129), while normalize_swiglu_limit_for_activation already accepts a limit for gated silu (:36-48). (4) The direct-micro dispatch sets swiglu_limit=None for any activation other than swigluoai (moe_dispatch.py:1370-1376). The static, micro and dynamic kernels already carry self.swiglu_limit into gated_activation_f32 (moe_static_kernel.py:368,1826; moe_micro_kernel.py:390,1907; _moe_dynamic/gated.py:769,1643,2075). The nvidia pack has the same swiglu_limit=10.0 on silu (config.json:269), so it is blocked in the same way.

**Mechanism:** The kernel math supports a clamp. Only the silu path is not wired, and the vLLM oracle correctly refuses to run the unclamped path.

**Evidence:**
- v11src vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:177-179 (B12X excluded from auto: 'CUTLASS SM121 MMA op guard'), :191-199, :259-269
- v11src vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:242-251 (no swiglu_limit), :123-125 (a2_gscale forced to 1.0)
- v11src flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_activation.py:36-48, 114-133
- v11src flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py:1370-1376
- v11src vllm/model_executor/layers/fused_moe/utils.py:476-478 reference semantics: gate=min(gate,L); up=clamp(up,-L,L)
- opt-b12x evidence/iter-b12x-B-flashinfer/diagnosis.txt; nvidia config.json:269 swiglu_limit 10.0, hidden_act silu

**Proposed action:** Build a Python-only image layer on v11 (no nvcc; CuteDSL JITs at runtime): (a) moe_activation.py, silu branch: `if limit is not None: g = fmin_f32(g, Float32(limit)); u = fmax_f32(fmin_f32(u, Float32(limit)), Float32(-limit))`, matching vLLM _swiglu_limit_torch. (b) moe_dispatch.py:1370-1376: keep swiglu_limit when activation in GATED_MOE_ACTIVATIONS (null only alpha/beta), so it also enters the launch/compile cache keys. (c) flashinfer_b12x_moe.py: pass `swiglu_limit=self.moe_config.swiglu_limit` (only if activation==SILU) to B12xMoEWrapper. (d) nvfp4.py: add FLASHINFER_B12X to NVFP4_BACKENDS_WITH_CLAMP. Add a CPU/GPU numerics unit test against _swiglu_limit_torch for inputs above ±10. Also commit the worktree's evidence and the refuse guard as a small PR rebased on #11 (it currently conflicts on AGENTS/README MoE lines and still names caca4e6); guard text: 'refused until the clamp patch image exists'.

**Est. impact:** Unblocks the native SM12x W4A4 fused MoE path (single fused launch for dispatch, FC1, act, FC2 and reduce). Decode upside is capped by bandwidth (see B12X-2); prefill/TTFT upside is potentially large because FP4 tensor cores replace Marlin dequant (unmeasured).

**Validation:** Numerics first (one GPU, small shapes; after the other workload frees the Sparks): compare b12x silu+clamp against the Marlin reference for one layer on random inputs with a pre-activation range of ±40, max abs diff within FP4 tolerance. Then a single boot FORCE_UNSAFE_MOE=1 MOE_BACKEND=flashinfer_b12x on the nvidia pin with the PR12 sampler: count probe 200/200, thinking-off, ms_per_step (9 runs) against the same-session Marlin baseline.

**Risks:** Upstream says B12X is excluded from auto-select pending a 'CUTLASS SM121 MMA op guard'. The Qwen recipe refuses b12x citing Xid reports on sm_121. CuteDSL compile after 90.67 GiB weights may spike memory (cf. cutlass JIT OOM).

**Verifier reasoning:** In v11src: oracle/nvfp4.py:177-179 excludes B12X from auto selection (CUTLASS SM121 MMA guard). NVFP4_BACKENDS_WITH_CLAMP at :191-199 lacks FLASHINFER_B12X, and :259-269 raises for an explicit request when swiglu_limit is set. flashinfer_b12x_moe.py:242-251 constructs B12xMoEWrapper without swiglu_limit, although b12x_moe.py:282/371 accepts it. moe_activation.py:114-129 clamps only in the swigluoai_uninterleave branch. normalize_swiglu_limit_for_activation (:36-48) accepts a limit for any gated activation. moe_dispatch.py:1370-1376 is the only site that nulls swiglu_limit for non-swigluoai. static/micro kernels pass self.swiglu_limit into the activation (moe_static_kernel.py:368/1826, moe_micro_kernel.py:390/1907). The patch plan is coherent. The risks (Xid/SM121 guard, JIT memory) are correctly flagged.

## B12X-2: Expected value of b12x on decode is small (≤~5–10%), trades W4A16 accuracy for W4A4, and adds per-layer workspaces on a ~5 GiB headroom

- kind=perf component=MoE backend choice (Marlin W4A16 vs b12x W4A4) on 2× GB10 impact=3 confidence=3 effort=M needs_gpu=True
- **verdict: plausible** (corrected confidence 3)

**Claim:** Per-step bytes at c=1, k=7, per rank: MoE ≈ 58 distinct experts × 42 × 7.08 MB ≈ 17.3 GB (independent-routing upper bound). BF16 attention ≥ 6.1 GB, shared expert ≈ 1.06 GB, lm_head ≈ 0.63 GB, drafter ≈ 1.2–2.3 GB. Total ≈ 26–27 GB, i.e. 96–100 ms at 273 GB/s, against 115 ms observed (~85% of peak). b12x reads the same FP4 weight bytes as Marlin, so it can only recover kernel inefficiency: at most ~(115−100)/115 ≈ 13% even at perfect bandwidth, more realistically 0–10%. It also changes numerics: the nvidia pack is calibrated W4A4, and b12x quantizes activations to FP4 in-kernel (FC2 with dynamic scales, since a2_gscale is forced to 1.0), whereas Marlin keeps BF16 activations. Marlin is therefore likely the more accurate of the two, so b12x is a quality risk, not a quality gain. Each of the 42 FusedMoE layers builds its own B12xMoEWrapper with use_cuda_graph=True, which pre-allocates static and dynamic workspaces sized max_num_tokens×top_k rows (b12x_moe.py:391-480). That is a GiB-scale total on nodes with 5.0–5.7 GiB free.

**Mechanism:** Decode at c=1–2 is memory-bandwidth-bound on LPDDR5X. Bytes per step, not math throughput, set the step time. W4A4 only helps compute-bound shapes (prefill).

**Evidence:**
- safetensors category sizes (scratchpad/agents/prs/hdr.py on nvidia 09b04e5): routed 159.47 GiB, attn 11.32 GiB BF16, shared 1.97, lm_head 1.18, visual 1.05
- step-time analysis in PR9-2 (115–119 ms at k=7)
- v11src flashinfer/fused_moe/cute_dsl/b12x_moe.py:391-480 _allocate_buffers (static + dynamic workspace + _moe_output per wrapper); flashinfer_b12x_moe.py:235-251 one wrapper per experts instance
- v11src flashinfer_b12x_moe.py:116-125 a2_gscale.fill_(1.0) (dynamic FC2 activation quant)
- opt-b12x evidence/iter-b12x-A-marlin/backend.txt: Marlin 'Weight-only FP4 compression' (W4A16)
- opt-b12x postboot.txt available 5.7Gi (spark1)

**Proposed action:** Rank b12x below byte-reduction levers for decode, such as acceptance-adaptive k (PR9-2) and FP8 attention weights (~3 GB/rank/step ≈ 11 ms ≈ 10%). Pursue b12x mainly for prefill/TTFT, and only after the clamp patch (B12X-1). Gate it on count/needle/quality probes and on MemAvailable after ready ≥ 3 GiB. If workspaces are too large, share one wrapper/workspace across layers (layers run sequentially) in the vLLM class.

**Est. impact:** Decode +0–10% (upper bound 13%). Prefill possibly large (unmeasured). Quality: small degradation risk from FP4 activations. Memory: GiB-scale workspaces (estimate) against ~5 GiB headroom.

**Validation:** GPU-free: compute the workspace bytes from allocate_sm120_moe_workspace shape formulas for max_num_tokens=2048, top_k=8, k=4096, n=1024, E=288, and multiply by 42. On Sparks: same-session A/B Marlin vs b12x with ms_per_step (decode), TTFT on an 8k prompt (prefill), count/needle probes, and a small accuracy set.

**Risks:** The byte model assumes independent routing and peak 273 GB/s. If Marlin's small-M efficiency is worse than estimated, the decode gain could be larger.

**Verifier reasoning:** The directional point holds: decode at c=1–2 is bandwidth-bound and b12x reads the same FP4 bytes. The '≤13% upper bound' is not a real bound. It assumes the independent-routing byte count (an upper bound on bytes) and the 273 GB/s theoretical peak. If routing across consecutive positions is correlated, bytes per step are lower and kernel headroom is larger. If the achievable LPDDR5X bandwidth is about 220–240 GB/s, headroom is near zero. The measured k7→k5 delta (7–10 ms vs 14.5 predicted) suggests the byte model overestimates. The quality framing misses that FlashInfer's B12xMoEWrapper has a W4A16 mode: activation_precision='bf16' selects quant_mode='w4a16' (b12x_moe.py:119-151, 254-257, and the w4a16 branch at :403-419). vLLM's wrapper just never passes it, so W4A4 is not forced. The nvidia pack is also calibrated for W4A4, since input_scale is present. The workspace concern is valid in kind: per-layer wrapper, static+dynamic workspaces sized max_num_tokens(2048)×top_k at b12x_moe.py:402-482. 'GiB-scale' is not computed.

**Verifier corrected claim:** b12x's decode gain is probably small, but it is not bounded at 13%: the byte model is unvalidated and the gain depends on routing correlation and achievable bandwidth. The accuracy concern can be avoided with FlashInfer's W4A16 mode (activation_precision='bf16'), which needs one more vLLM plumbing line. Workspace memory per wrapper × 42 layers still has to be computed.

**Verifier corrected impact:** Decode about 0–10% (estimate, weakly bounded). Prefill upside unmeasured. Quality risk avoidable via the w4a16 mode. Memory impact unquantified.

## ORDER-1: Recommended disposition and merge order relative to #11

- kind=pr-review component=repo PR queue (#3, #8, #9, #11, #12, b12x worktree) impact=3 confidence=4 effort=S needs_gpu=False
- **verdict: confirmed** (corrected confidence 4)

**Claim:** The PRs sit on three different bases (main, #11 head cursor/nvidia-nvfp4-vision-35dc, agent/kit) and overlap on run.sh generated blocks, recipe.yaml, README/AGENTS, CI and the .cursor verify skill. Merging in any order other than below produces modify/delete conflicts and silently drops guard/lint checks.

**Mechanism:** Stacked and parallel agent branches without a single integration base.

**Evidence:**
- gh pr view 12: base cursor/nvidia-nvfp4-vision-35dc (= #11 head 29954c8)
- gh pr view 9: base agent/kit (#8); git log origin/agent/kit..origin/agent/hillclimb-20260903 shows 10 commits incl. a76474e
- gh pr view 8: base main, deletes .cursor/skills/verify-glm53-flash/* and bench_decode.py
- gh pr view 3: base main (pre-recipe.yaml), CONFLICTING
- opt-b12x worktree: git status shows uncommitted run.sh/recipe.yaml/AGENTS/README edits on 63a433a (pre-#11)

**Proposed action:** 1) #11: merge after one measured exclusive boot (prose c=1/2, vision smoke, count, thinking-off) and after resolving the SPEC=mtp rollback text (MTP-NV-1). 2) #8: rebase onto #11 and fold in c0c4af3 + e898a80 + vision probe + ported #11 lint (PR8-1). 3) #12: rebase onto #8 after fixing the tripwire/floor/watcher (PR12-1..3) and convert its lint to recipe.yaml lint:. 4) #9: evidence only, drop a76474e and README change, rebase (PR9-1). 5) b12x worktree: commit evidence + refuse guard as a small PR on top; the kernel patch image goes separately (B12X-1). 6) #3: close; reopen only as a pinned opt-in overlay after 1–4 (PR3-1).

**Est. impact:** Avoids at least 4 modify/delete conflicts (bench_decode.py, recipe-lint.py ×2, serve-start-stop.md) and loss of the vision/UMA lint checks. Keeps one reproducible ruler for the performance plan.

**Validation:** After each rebase: python3 kit/render.py --check --strict, python3 kit/recipe_lint.py ., python3 -m unittest discover -s tests, VALIDATE_ONLY refuse cases. All GPU-free.

**Risks:** The #11 measured boot needs an exclusive slot. Until then #8/#12/#9 stay stacked on an unmeasured pin.

**Verifier reasoning:** `gh pr view`: #12 base cursor/nvidia-nvfp4-vision-35dc (MERGEABLE), #9 base agent/kit (MERGEABLE), #8 base main (MERGEABLE against main), #3 base main (CONFLICTING). `git log agent/kit..agent/hillclimb-20260903` lists 10 commits including a76474e and c0c4af3. The b12x worktree has uncommitted edits to AGENTS/README/recipe.yaml/run.sh/evidence plus untracked evidence dirs. The modify/delete overlaps between #8 and #11/#12 are real (see PR8-1). The order is sensible. One caveat: #8 is independently mergeable today, and the conflicts only arise relative to #11/#12.

## Open questions
- Is the co-tenant workload on the Sparks Docker-based (visible to #12's refuse_foreign_serve) or bare-metal/containerd? That decides whether a host-level GPU-user check (fuser /dev/nvidia*) is needed.
- Policy question for the owner: with #11's 'published decode score is prose only', may a structured-cell regression still veto a keep (AGENTS.md 'regresses another cell')? The answer flips the DFlash2-5 verdict for prose c=2 (+18%).
- Does the fork's glm5next MTP loader have any path that de-quantizes or skips NVFP4 for BF16 layer-45 experts in the nvidia pack? Needs a GPU-free trace of the FusedMoE weight_loader, or one boot, to confirm MTP-NV-1.
- What is the real size of the per-layer B12xMoEWrapper workspaces for max_num_tokens (2048 under DFlash2), top_k=8, E=288 at TP=2? Computable from allocate_sm120_moe_workspace formulas, not yet done.
- Is the forge kit SHA 6285f70 (redact.py, needle payload) the intended vendoring point for #8, and who owns syncing it across recipes?
- Does the owner want an uncensored option at all? If yes, which donor SHA was validated (the PR says e90ef415; HF main is now 745aac2), and should ABLIT=1 require a localhost bind or an API key?
- Why does acceptance_len vary 2.17–2.43 at fixed k=7 on the same greedy prompt across boots (nondeterministic target numerics vs completion-length differences)? This affects how much of the ruler noise the step metric removes.

## Verifier: missed issues
- The step-time model contradicts the c=2 data. In the same k=5 boot (PR #9 rebench-dflash5 bench.json), structured c=2 runs at 41.56/5.98 = 6.95 steps/s (144 ms) and prose c=2 at 19.62/2.573 = 7.63 steps/s (131 ms), although the structured streams are identical and should share experts. k=5 structured c=2 (144 ms) is also slower per step than k=7 structured c=2: 60.41/7.92 = 131 ms published (evidence/rebench-20260902T204243Z/bench.json) and 55.50/7.88 = 142 ms parity. So the −31% structured c=2 regression is not only the 6-vs-8 acceptance cap (which predicts about −24%). Something else in the k=5 c=2 path, such as the 12-token CUDA-graph bucket or the drafter or scheduling, is slower. Any adaptive-k proposal needs a same-boot c=2 check.
- H3 (k=7, MAX_NUM_SEQS=1; iter-H3 bench.json prose c=1 17.40, acc 2.168) gives 124.6 ms/step. That widens the k=7 c=1 step range to 114.6–124.6 ms and undercuts the '±2% ruler' claim. The PR9-2 evidence also mislabels H2's 2.175 as fixed-k=7 acceptance; H2 is SPEC=mtp (iter-H2 verdict.json change 'SPEC=mtp').
- The MTP-NV-1 root cause is broader than the experts. The nvidia pack excludes by per-layer names under 'model.language_model.layers.N.*' (config.json quantization_config.ignore, 132 entries; hf_quant_config.json). The MTP module is built under 'model.layers.45.*' (mtp.py:226), and only the main model has the language_model mapper (multimodal.py:354). So no exclusion applies to any layer-45 module: attention, shared expert, gate and experts are all BF16 in the checkpoint but would get NVFP4 methods. LibertAI caca4e6 uses prefix-agnostic wildcards ('*.self_attn.q_proj', …), which is why H2 worked there. This is a concrete fix target: an image patch adding 'model.layers.45*' to exclude_modules, or a run.sh refusal.
- FlashInfer's B12xMoEWrapper supports a W4A16 mode: activation_precision='bf16' selects quant_mode='w4a16' (v11src flashinfer/fused_moe/cute_dsl/b12x_moe.py:119-151, 254-257, w4a16 branch at 403-419). vLLM's FlashInferB12xExperts (flashinfer_b12x_moe.py:242-251) never passes it. A b12x trial could keep Marlin-equivalent BF16-activation numerics, which removes B12X-2's main quality objection, or run an A/B of W4A4 against W4A16 on the same kernel.
- PR #12's watcher also runs `wait_uma_or_abort` on spark2, whose healthy post-ready MemAvailable is 9.8 GiB (b12x-A postboot-spark2.txt). That is below the 16 GiB tripwire on both nodes independently, so fixing only the head-side threshold would not be enough.
- PR #8's recipe.yaml lint `contains:` hardcodes README strings 'LibertAIDAI/GLM-5.3-Flash-NVFP4' and 'glm53-sm121-v11' plus the `{env: {}}` default-ladder case (pr8.diff recipe.yaml hunk). A rebase onto #11 (nvidia pin as default) needs those lint expectations rewritten, not only the #11/#12 lint checks ported.
