## Diagnostic-boot safety envelope and attribution: the watcher kill rules and the D12 auto-mode proof do not match the observed failure dynamics

# Gap 1 spec: memory watcher and kernel attribution for the first nvidia boot

**Verdict.** The watcher proposed in the plan would fire after memory is already exhausted. Its attribution grep would also miss the suspected compiler. The kernel choice is logged 12.4 to 13.6 minutes before the collapse, so attribution should gate the boot at construct. Don't run D12 live on the pair; measure the compile on its own instead (section 5).

## 1. Collapse dynamics (all six proofs; confirmed from the proof tables)

| run | last pre-cliff → kill sample (GiB, s1/s2) | interval | rate (GiB/s) s1/s2 | interval before | abort → mem back |
|---|---|---|---|---|---|
| stock | 10.33/9.83 → 0.59/0.45 | 4 s | 2.44/2.35 | 1.16/1.34 | 11 s |
| novideo | 15.68/15.21 → 5.01/2.81 | 4 s | 2.67/3.10 | 0.81/1.11 | 11 s |
| skipmm | 16.77/16.45 → 6.46/9.77 | 4 s | 2.58/1.67 | 1.44/1.72 | 10 s |
| eager | 17.45/16.94 → 2.08/3.57 | 4 s | **3.84**/3.34 | 0.24/0.31 | 11 s |
| langonly | 17.96/17.32 → 10.43/7.35 | 3 s | 2.51/3.32 | 0.56/0.75 | 13 s |
| batch1024 | 17.70/17.04 → 6.39/4.37 | 4 s | 2.83/3.17 | 0.38/0.55 | 20 s |

- **Stock proof table error.** It labels 00:46:51 as the "last sample ≥8" (`glm53-nvidia-spark-proof.md:37`), but 10.33 GiB at :54 (:75) is later and also ≥8.
- **Why the cadence was 2–4 s.** `POLL_S=2` plus ssh and `curl --max-time 2` run serially (`uma_watch.sh:14,28,80`). The idle `uma.tsv` shows about 2.3 s per sample.
- **Why the kill was slow.** The run.pid kill includes a `sleep 1` (`uma_watch.sh:41-48`). `stop.sh` is serial: local `rm -f`, then an ssh probe with `ConnectTimeout=5`, then the remote `rm` (`stop.sh:8-15,23-27`).
- **Old watcher required a floor above normal dips.** With a 4 s sample and about 2 s kill, it needed F > 3.84 × 6 ≈ 23 GiB. That is above the lowest normal load-phase dip (15.35, skipmm head Marlin). The stock run stopped at 0.45 GiB, which was luck.
- **New floor (estimated).** Rate r = 5 GiB/s (observed max 3.84 plus 30% margin). Sample period Δ = 0.5 s. Kill-to-growth-stop L ≤ 1 s, which needs a dry-run to confirm. Reserve 2.5 GiB. F = 5 × 1.5 + 2.5 = **10 GiB**. That leaves 5.35 GiB margin to the lowest normal dip.
- **The proposed rules fire too late.** "MemAvailable <1.5 GiB for 5 s": at ≥2.4 GiB/s the 5-second hold ends after exhaustion. The 08-31 global OOM (`diagnosis.txt:12-16`) is this failure class. "PSI full avg10 >20%" is a 10-second moving average. Full stalls only start near zero, because buff/cache was just 0.6–2.2 GiB in the proofs. Neither rule fires before exhaustion (derived). Log the raw PSI `full total` delta instead; `/proc/pressure/memory` exists on spark1.
- **Floors must depend on phase.** After a successful LibertAI boot and bench, available memory was **5 GiB with 6 GiB swap used** (`rebench.../free-after-bench.txt`). A fixed 10 GiB floor would kill a good boot.

## 2. Watcher spec (both nodes)

- **Placement.** Run a node-local watcher on spark1 and spark2. Each reads `/proc/meminfo` every 0.5 s. Each kills its own rank and cross-kills the peer over a pre-opened ssh ControlMaster.
- **Kill action.** Run `docker kill -s KILL glm53-flash-nvfp4` in parallel on both nodes, not `stop.sh`. Use kill, not `rm`: the container journal was lost to `rm` in skipmm (:25), eager (:26) and batch1024 (:29). Snapshot the logs, then remove the container.
- **Phases and rules** (phase boundaries come from each rank's log):
  - **P0**, from start to `Model loading took`: hard floor 10 GiB.
  - **P1**, from `Model loading took` to `Initial free memory … reserved` (`gpu_worker.py:492`; rank1 log:56): this is the danger phase. Kill on the 10 GiB floor. Also kill on a sustained slope: ≥0.75 GiB drop in each of two consecutive 0.5 s intervals, while below 14 GiB. The stock encoder profile, which is normal, bottomed at 14.53.
  - **P2**, from the KV line to ready: floor 3 GiB. The watcher alone cannot protect this phase; rely on the attribution gates below.
- **Process census, every 1 s.** Read the processes in the container's cgroup `cgroup.procs`. Tag each by `/proc/<pid>/cwd` and argv:
  - FlashInfer ninja runs with `cwd=<ws>/cached_ops/<module>` (`cpp_ext.py:370-374`; `core.py:337-341`; `env.py:162`).
  - Any nvcc not under `cached_ops` is tagged `other-nvcc (tilelang?)`.
  - **Kill immediately** if a module name matches `fp4_gemm_cutlass_sm120|fp4|cutlass`.
  - Log per-tag RSS and the cgroup's `memory.current`. If `memory.current` stays flat while MemAvailable falls, the growth is GPU/UMA memory rather than compiler memory. That split is not verified yet; record it on this boot.
- **Kernel follower.** Run `journalctl -k -f`, grepping `NVRM|Xid|NV_ERR_NO_MEMORY|oom-kill`. `dmesg_restrict=1`, but the user is in `adm`.

## 3. JIT paths in v11 (confirmed)

- **FlashInfer nvcc builds run as a ninja subprocess and log nothing at INFO.**
  - `build_and_load` → `build` → `run_ninja` contains no info log (`core.py:300-321,412-427`); the only log there is an AOT-load warning (:405).
  - No `-j` is passed unless `MAX_JOBS` is set (`cpp_ext.py:346-365`), so ninja uses nproc+2 = 22 on this 20-core host. `nvcc --threads=1` (:94-112).
  - The SM120 FP4 GEMM is **17 TUs**: 1 base plus 8 tile shapes × 2 dtypes (`jit/gemm/core.py:256-290`).
  - No AOT cache: jit-cache was uninstalled (`Dockerfile.sm121-v8:67`), and the cache directory is not bind-mounted (`run.sh:293-296`). Every boot compiles cold.
- **Correction to the gap statement: grepping `flashinfer.jit` will not show this compile.** That tag only prints autotuner lines (rank1 log:68,71) and `Compiling CuTe-DSL kernel` from the cached CuTe path (`cute_dsl_core.py:177`). Set `FLASHINFER_JIT_VERBOSE=1` (`core.py:421`); ninja `-v` output then reaches the log (`cpp_ext.py:372`). Also set `MAX_JOBS=2` as a safety cap.
- **b12x compiles CuTe-DSL in-process** (`cute.compile`, `gemm_mm_fp4_cute_dsl.py:210`). No child process appears, so only the memory rules catch it.
- **TileLang: unknown.** Its source is not in v11src and it is not installed on the host (`pip show` found nothing). The only trace on the host is the DSv4.1 cache (`~/.cache/vllm-dsv41-flash-exl3/tilelang/0.1.12`, with cubins). An nvcc subprocess is the upstream default, but that is not verified here.

## 4. Attribution (confirmed unless marked)

- **The kernel is picked at construct.** `ModelOptNvFp4LinearMethod.__init__` calls `init_nvfp4_linear_kernel()` (`modelopt.py:1104-1107`, via `:1026`). It logs `Using %s for NVFP4 GEMM` (`linear/__init__.py:1073`, or `:1033` on the forced path).
- **When that line appears.** It lands between `Loading model from scratch` and `Filesystem type…`, 43–64 s after the containers are up. That is **12.4–13.6 min before the collapse** in all six runs.
- **Only the dense MLP is affected.** The checkpoint has exactly 9 non-expert NVFP4 tensors: gate/up/down in layers 0–2 (computed from `model.safetensors.index.json`).
- **Timing fits (estimated).** Layer 0 is mHC → KDA attention (BF16) → mHC → dense NVFP4 MLP. The collapse starts 2–7 s after `mhc_post` compiles.
- **Auto-selection picks `FlashInferCutlassNvFp4LinearKernel`, not b12x.** Estimated strong; the fork's `_C` is not inspected.
  - Priority order is CuteDsl, then FlashInferCutlass, then B12x (`:500-512`).
  - CuteDsl needs the sm_10x family (`nvfp4/flashinfer.py:38`).
  - Cutlass needs `cutlass_fp4_supported()`, capability ≥100 and FlashInfer (`:113-117`; `nvfp4_utils.py:56-61`).
  - Upstream v0.30.0 returns true on sm_12x when `ENABLE_NVFP4_SM120` is built (`nvfp4_scaled_mm_entry.cu:71-86`). That flag is set from `"12.0f"` ∩ CUDA_ARCHS when CUDA ≥13.0 (`CMakeLists.txt:1005-1022`). The image has CUDA 13.0.1 and an arch list that includes 12.0.
  - The v0.25.0 log on sm_121 shows exactly this choice (`…-117.log:24`).
- **Activation quant does not trigger a second JIT.** `scaled_fp4_quant(backend="flashinfer-cutlass")` goes to the prebuilt `_C.scaled_fp4_quant.out`, because "trtllm" is not in the backend name (`_custom_ops.py:1572,1589`). Only the `backend="cutlass"` FP4 GEMM builds (`gemm_base.py:1822-1840`).

**Procedure**
1. Boot with `--linear-backend marlin`. Narrower alternative: `VLLM_DISABLED_KERNELS=FlashInferCutlassNvFp4LinearKernel,FlashInferB12xNvFp4LinearKernel,CutlassNvFp4LinearKernel` (`:1047`).
2. Abort on either rank if the selection line is not `MarlinNvFp4LinearKernel`, or if `Filesystem type` appears without it.
3. Keep the census rule: any `cached_ops/fp4_gemm*` means kill.

## 5. D12: don't run it live

Measure the compile in isolation instead. On one node, with no model loaded and about 117 GiB free, run `get_gemm_sm120_module_cutlass_fp4()` in a standalone container. Do it twice, with `MAX_JOBS` unset and with `MAX_JOBS=2`, and record the peak drop in MemAvailable.

- If the unset-MAX_JOBS peak is above about 18 GiB (the leftover after weights), causation is shown without an OOM on the pair.
- The Marlin-pinned boot is the one-knob differential that tests the fix.
- The built `cached_ops` can later be bind-mounted so an auto-select boot does not compile at all.

All of this is GPU-free reading. Kill latency L and the `memory.current` split need a dry-run before the boot.

## Boot time and host memory policy are unplanned prerequisites for a ~60 slot-hour, boot-bound programme

# GAP 2: boot-time levers and pre-slot host policy (read-only findings)

**Bottom line:** A boot is about 16.5–18 min. About 70% of that is the head rank loading weights (629–785 s against 190–243 s on the worker). The unused layer 45 is probably not the main cost. The largest levers are on the host: spark1 contention and swap, and slow page-fault reads. Persisting compile caches is a second, independent lever. Set swappiness to 0 before the first calibration boot and hold it constant.

## Boot anatomy, head critical path (confirmed from logs)
In the LibertAI rebench `evidence/rebench-20260902T204243Z/engine.log.tail`:
- Head weights: 629 s (l.38). The worker took 197 s (`engine-rank1.log.tail:33`).
- Repack and draft: about 47 s (l.52).
- `init engine … took 277.93 s` (compile, autotune, TileLang, graphs).
- Ready at 21:01:03.

On the nvidia pack, head `Loading weights took` 679–733 s against worker 232–243 s total, across all six proofs (e.g. `glm53-nvidia-spark-proof.md:70-72`).

The head is also slow in the DSv4.1 sibling: 328 s against 180 s. So this is **host-specific to spark1, not pack-specific**.

## Answers to (A): load path
1. **Layer-45 skip (confirmed):** The iterator calls `f.get_tensor(name)` on every key (`weight_utils.py:958-963`). `should_skip_weight` only filters EP experts (`ep_weight_filter.py:70-81`). The drop happens later, in `glm5next/nvidia/model.py:818-820`.
   - Header parse: layer 45 is **13.84 GiB of BF16 per rank**, 7.3% of 190.40 GiB. It sits in shards 1–3 (3.83 / 5.01 / 5.01 GiB), mixed with backbone tensors.
   - Estimated: safetensors ≥0.4 with torch backs `safe_open(pt)` with a lazy `UntypedStorage.from_file` mmap. If so, skipped tensors are mostly never faulted in and the saving is about 0 s.
   - Upper bound if they are read: 13.84 / 0.26 GiB/s = **≤53 s on the head**, ≤17 s on the worker. I could not check the safetensors version in the image.
   - Any filter must depend on `SPEC=dflash2`, because the MTP rollback needs layer 45.
2. **`sharded_state` exists** (`model_loader/__init__.py:59-61`). Blockers:
   - (a) `process_weights_after_loading` runs unconditionally after load (`base_loader.py:80`), so Marlin would repack already-repacked tensors.
   - (b) The save is post-processing (`sharded_state_loader.py:194`), but the load copies into a freshly initialised `state_dict()` (l.136-155), so shapes and names mismatch.
   - (c) It bypasses glm5next's own load logic: FP8 indexer wk fusion and kv_a NoPE padding (`model.py:803-828`).
   - (d) The draft inherits the target load config (`config/speculative.py:216-218`), so DFlash2 would need its own shards.
   - (e) It needs one successful boot first (L1), plus about 90 GiB per node of disk. spark1 has 394 GiB free.
   - Verdict: **not near-term.**
3. **`enable_multithread_load`** (`default_loader.py:84-126, 278-285` → `weight_utils.py:966-993`):
   - It ignores `local_expert_ids` and the sort order.
   - `as_completed()` turns the generator into a set, so all 33 shards are submitted at once despite the comment at l.978.
   - Each worker thread runs `load_file(device="cpu")`.
   - If tensors are mmap views, the gain is about 0. The DSv4.1 finding is that H2D `copy_` pins mmap source pages (`flags.md:680-683`), so more shard dicts held at once means more pinning.
   - If tensors are real copies, 8 × 5.5–8.3 GiB = 44–66 GiB of anonymous memory, against about 19 GiB of headroom at load (`Available RAM 18.84`). That is an OOM.
   - Load order also becomes random from boot to boot. **Do not use.**
4. **spark1 disk (confirmed):**
   - `/dev/nvme0n1p2` ext4 `rw,relatime`, 3.7T, 89% used, Samsung MZALC4T0HBL1.
   - `read_ahead_kb=128`, scheduler `none`.
   - `filefrag`: 27, 15 and 13 extents for 8.3, 5.5 and 5.5 GiB shards, so **not fragmented**. Being 89% full is unlikely to be the cause.
   - The stronger candidates (estimated):
     - Reads come from single-threaded, 128 KiB page faults, so any extra latency multiplies load time.
     - Swap writes go to the same NVMe (`/swap.img`).
     - spark1 is a busy desktop: Chrome is using about 330% CPU, `uptime` shows 113 users, and load average is about 4.

## Ranked boot-time levers (seconds on the head critical path)
| # | Lever | Code? | Estimated saving per boot | Risk |
|---|---|---|---|---|
| 1 | Quiet spark1 during load, swappiness 0, empty swap at slot start | none (sudo) | 0–~490 s (gap to worker ~240 s); unproven | Low. Needs a same-slot A/B with `vmstat 1` / `iostat -x` / `/proc/<TP0>/io` on both nodes |
| 2 | Persist compile caches (`/root/.cache/vllm`: torch_compile and autotune; TileLang, Triton and FlashInfer caches). Today only the HF cache is mounted (`run.sh:293`). Autotune logs "0 from previous config" | run.sh mount, keyed by image ID, under `~/projects/data/` | 60–150 s of the 278 s init | Stale cache if not keyed by image; first boot per config still cold |
| 3 | `read_ahead_kb` 128 → 4096 on both nodes | none (sudo, reversible) | 0–30% of load | More page-cache churn inside ~19 GiB headroom |
| 4 | Bounded next-shard `posix_fadvise(WILLNEED)` (≤1 shard ahead) | ~10-line image patch | Could approach disk bandwidth; worker perhaps ~100 s | Page-cache pressure; patch upkeep |
| 5 | Layer-45 name filter before `get_tensor` | iterator patch | 0–53 s | Would break `SPEC=mtp` unless conditional |
| – | Stock `--safetensors-load-strategy prefetch` | – | Likely negative | Only prefetches this rank's slice of the file list (`weight_utils.py:765-771`), so each node warms only half the shards. It also prefetches everything with no limit, 8 threads ahead of the reader, well past ~19 GiB of headroom (warns only, l.897-905) |
| – | multithread load, sharded_state, fastsafetensors | – | – | Rejected (see above) |

## Answers to (B): host policy, spark1 (confirmed readings)
- `swappiness=60`, `min_free_kbytes=45167`, `watermark_scale_factor=10`, `lru_gen/enabled=0x0003` (MGLRU on).
- Swap is a 16 GiB `/swap.img` with **3.5 GiB used**. `MemAvailable` is 113 GiB, of which 95 GiB is `Cached`.
- Nothing in `/etc/sysctl.d` sets these values.
- Tony's notes say swappiness 0 is mandatory and does not survive a reboot (`tony_readme.md:246-250`). They also say keep swap **on**, because with swap off the worker dies during Marlin repack (`:375-377`).
- The nvidia proof shows swap rising from 151 MiB to 1.6 GiB (`spark-proof.md:92`).
- Estimated: with MGLRU, 0 means file-only reclaim, which is stricter than 1. Use **0**, the only value tested on GB10. Fall back to 1 only if a boot dies silently during repack.
- spark2's values are unread.

**Recommendation:** fix these before the first calibration boot and hold them constant. Log them as host precondition H0 in `trail.tsv`, not as an A/B knob. Earlier receipts were taken at 60, so do not mix them with the new calibration. Leave the watermarks at their current values and record them; there is no local evidence for changing them.

## Pre-slot checklist (both nodes)
1. Save a snapshot to `evidence/<run>/host-before-<node>.txt`: the vm values above, `/proc/swaps`, `read_ahead_kb`, `lru_gen`, `free -h`, `uptime`, top processes by RSS.
2. Confirm DeepSeek is down (`docker ps`) and `MemAvailable` is at least 115 GiB.
3. Confirm `sudo -n true` works. Otherwise `maybe_drop_caches` silently does nothing (`run.sh:177-181`), and spark1's 95 GiB of cache would give a partly warm first boot.
4. Run `sudo sysctl -w vm.swappiness=0` (runtime only).
5. Run `sudo swapoff -a && sudo swapon -a` once, which clears the 3.5 GiB. Keep swap on.
6. Close Chrome and pause heavy agent jobs on spark1.
7. Check raw disk parity with `dd if=<shard> of=/dev/null bs=16M iflag=direct` (CPU only, about 10 s).
8. For each boot, record:
   - `SwapFree` before and after.
   - `vmstat 1`.
   - `/proc/<TP pid>/io` `read_bytes`. This settles whether layer 45 is actually read.
   - `Loading weights took` on each rank.

**Restore for the co-tenant:** run `sudo sysctl -w vm.swappiness=60` and set `read_ahead_kb` back to 128 if it was changed. Verify with `cat` on both nodes and note it in `trail.tsv`. Nothing is persisted, so a reboot also restores the defaults.

The only file I created is the header-parse script `/tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-GLM-5-3-Flash-NVFP4-vLLM-2x-DGX-Spark/29f0f0df-0b0e-4224-a895-0ea5d735b7de/scratchpad/agents/gap2_hdr.py`.

## Statistical feasibility of the plan's gates: the nvidia-flip non-inferiority margin, the Tier-1 pass rule and ruler-v2 step counting

# GAP 3: whether the plan's statistical gates can be met, and corrected versions

## Summary
All three gates fail as written.
- **Flip gate:** the "+2% non-inferiority, same session" test needs 15–29 boots per arm.
- **Tier-1 rule:** even two identical packs fail it with probability 93–100%.
- **ITL step counting:** this part holds with async scheduling on (checked in the code), but `num_drafts` is the better counter.

Scripts are in `/tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-GLM-5-3-Flash-NVFP4-vLLM-2x-DGX-Spark/29f0f0df-0b0e-4224-a895-0ea5d735b7de/scratchpad/agents/gap3/` (`steps.py`, `power.py`, `disc.py`).

## 1. step_ms from the receipts (confirmed arithmetic)
step_ms = acceptance / tok_s. Receipts record acceptance once per cell, not per run. So per-run prose step_ms mixes in text variation. Structured cells are clean because structured acceptance is deterministic (7.8441558 in 4 sessions).

| | Structured step_ms (cells) | Within-boot per-run CV | Between-boot CV |
|---|---|---|---|
| k=7 | 116.4, 120.9 (swap wave), 114.8, 122.8, 121.1; parity 117.9 | 1.7–2.4%, with a run-1 warm-up trend | 3.0% over 3 boots (2.9% by cell) |
| k=5 | 107.8, 105.5, 107.9, 107.5 | 0.2–0.6% | about 1.0% over 4 boots |

- **Prose k=7:** cells 115.9–127.9 ms, CV 4.1%. Per-run CV is 4.5–11%, mostly acceptance noise. Prose step runs 1–6% above structured in the same boot.
- **k=5 vs k=7:** step is 107.5 vs about 115–118 ms, so k=7 costs about 8–10% more per step at similar prose acceptance (2.31–2.35 vs 2.29–2.43).
- **What varies:** almost all of the noise is between boots or waves (UMA/swap state), not between runs in a boot. So the boot is the unit that has to be replicated.
- **Uncertainty (estimate):** with 3 boots (df=2), the 95% CI on σ_B runs from 0.5× to 6.3× the point value. σ_B must be measured again in slot 1.

**Boots per arm for non-inferiority** (one-sided α=0.05, power 80%, true difference 0). Each cell is z with σ known from pooled calibration / Student t:

| σ_B | margin +2% | +3% | +5% |
|---|---|---|---|
| 1.5% (swap gate works) | 7 / 8 | 4 / 5 | 2 / 3 |
| 2.2% | 15 / 16 | 7 / 8 | 3 / 4 |
| 3.0% | 28 / 29 | 13 / 14 | 5 / 6 |

**What each boot count can resolve at σ_B = 2.2%** (t / z):

| Boots per arm | Non-inferiority margin | Two-sided MDE |
|---|---|---|
| 2 | 8.8 / 5.5% | 11.8 / 6.2% |
| 3 | 5.5 / 4.5% | 6.7 / 5.0% |
| 4 | 4.4 / 3.9% | 5.2 / 4.4% |

At σ_B = 1.5%, 4 boots per arm gives a 3.0% margin and a 3.6% MDE.

## 2. Tier 1 (arithmetic confirmed; discordance rates estimated)
Paired 95% half-width is 1.96·√(q/n), in pp, at discordance q:

| Task | n | q=5% | q=10% | q=20% |
|---|---|---|---|---|
| IFEval | 150 | 3.6 | 5.1 | 7.2 |
| GSM8K | 200 | 3.1 | 4.4 | 6.2 |
| MMLU-Pro | 280 | 2.6 | 3.7 | 5.2 |
| BFCL | 120 | 4.0 | 5.7 | 8.0 |
| MMMU | 60 | 5.7 | 8.0 | 11.3 |
| ChartQA | 100 | 4.4 | 6.2 | 8.8 |
| OCRBench | 100 | 4.4 | 6.2 | 8.8 |

**Measured discordance on DeepSeek-V4.1 (near-lossless config changes, greedy, thinking off).** Paired runs from `.worktrees/perf-review-0924/results/2026-09-24-review/`: `quality-baseline/full.json`, `s12…/stock_full.json` and `s13…/quality_full.json`.
- GSM8K-100: 2/100 discordant.
- GSM8K-think-40: 0/40.
- MMLU-228: 1.8–2.6%.

A pack swap will be higher. I assumed IFEval 10%, GSM8K 5%, MMLU-Pro and BFCL 8%, ChartQA and OCRBench 10%, MMMU 20% (estimates).

**The plan's rule** fails a task if McNemar shows a regression or the paired lower bound is below −5pp. With identical packs, the per-task false-fail rate is 12–86% (MMMU 86%, ChartQA 65%). Family-wise it is **93–100%**, which is worse than the 20–35% first estimated, because of the lower-bound test.

**Corrected rule:**
- **Pooled test:** fail if the item-weighted paired Δ over all 1,010 items is below **−2.0pp** (SE 0.93pp).
  - False-fail rate 1.6% (7.8% if q=20%).
  - Catches a uniform −3pp regression 86% of the time and −4pp 98%.
- **Per-stratum guard:** use 5 strata (IFEval, GSM8K, MMLU-Pro, BFCL, and Vision = ChartQA + OCRBench + MMMU, n=260). Fail a stratum if it has a one-sided McNemar p < 0.05/5 **and** Δ ≤ −5pp.
  - Family false-fail ≤ 5%.
  - Vision stratum catches a −10pp break 74–87% of the time. MMMU alone (n=60) catches only 15%.
- **Total false-fail:** about 6.5% (about 12% at q=20%).
- **Scope:** Tier 1 catches gross breaks. Sub-point drift from the pack swap is Tier 0's job (NLL / top-1 agreement).

## 3. Step counting with async scheduling (confirmed by reading code; not validated live)
- **Async is on:** `config/vllm.py:1184-1233` turns it on automatically for DFlash.
- **One output per step:** `core.py:704` pops exactly one batch future per step. `EngineCoreOutputs.timestamp` is set once per step (`v1/engine/__init__.py:264`). The scheduler emits one output per request per step (`scheduler.py:1909`). So ITL count equals steps even with async on.
- **Drift risk:** outputs from a finished or aborted request are skipped (`scheduler.py:1759`). Client-side SSE chunks can merge (`RequestOutputCollector`, `output_processor.py:50`), so client chunk counts are not per-step.
- **Better counter:** `vllm:spec_decode_num_drafts`. It goes up by 1 per request per verify step (`observe_draft`, `spec_decode/metrics.py:42`; `scheduler.py:1797`). `bench_decode.py` already reads it, but per cell.
- **Formulas:** with one request at a time at c=1, take counter deltas per request.
  - step_ms = `request_decode_time` sum / Δnum_drafts
  - acc_len = (completion_tokens − 1) / Δnum_drafts. This uses emitted tokens. `num_accepted` overcounts when output is cut off at `max_tokens`.
  - Cross-check: ITL count − Δnum_drafts should be 0 or 1 per request.

## 4. Forced-length output (confirmed by reading code)
- `logit_bias.py:90-104,226-239` sets EOS and all stop IDs to −inf while pos+1 < min_len, for each verify position through `expanded_idx_mapping`.
- Drafts are not masked. After the natural EOS point, drafted EOS tokens are always rejected, and the text drifts off-distribution. The bias direction is unknown.
- Use min_tokens = max_tokens. With `ignore_eos`, the model emits EOS and keeps going.
- min_tokens also routes every step through the FP32 logits-processing path (`sampler.py:220`). I estimate the cost below 0.1 ms.

**Slot-1 validation:** run 16 panel prompts natural (max_tokens 2048, twice for an A/A baseline) and forced (512).
- Require natural length L ≥ 576 for every prompt, or replace the prompt.
- Paired per-prompt acceptance: |Δ| must be at most the A/A spread.
- step_ms forced vs natural: within 1%.
- Diagnostic: force the legacy ~98-token prompt to 512 tokens to size the bias.

## Corrected gates and slot-hours (estimates)
Timing basis: boot 18m12s and 18m17s in the receipts, so about 20 minutes with teardown. Fast gate about 7 minutes, Tier 0 15 minutes, Tier 1 about 2.75 hours per config.

- **Flip to the nvidia pack.** The packs cannot share a session, so compare interleaved boots in one slot window (L N N L L N N L), with identical flags on both arms. Apply `--linear-backend` to both arms and log the NVFP4 GEMM line on both. The flip needs all of:
  - the pack boots to `/v1/models`;
  - the step_ms upper 95% bound is at most **+5%** (4 boots per arm; 5–6 if σ_B ≥ 2.5%);
  - the paired acc_len ratio lower bound is at least −3%;
  - Tier 0 passes, and the Tier-1 rule above passes.

  A +2% margin would need 16 boots per arm, about 7 extra slot-hours.
- **Keep or revert a lossless knob.** Use step_ms with 3 boots per arm and a known-σ z test; MDE is about 5% at σ_B 2.2% and 3.4% at 1.5%.
  - Keep if the lower bound > 0 and Δ ≥ +3%.
  - Revert if the upper bound < +1%.
  - Anything else is "inconclusive", never "reverted".
  - Knobs that change acceptance use paired per-prompt acc_len.
  - Moving the published prose score to the new ruler has to be recorded in `trail.tsv`.
- **Budget:**

| Step | Slot-hours |
|---|---|
| Slot-1 calibration: 4 A/A boots, full panel, forced/natural check, Tier 0 ×2 | ≈ 3.1 |
| Flip: 4 nvidia boots, Tier 0, Tier 1 ×2 (reuses calibration boots as the L arm) | ≈ 7.6 |
| Flip total | ≈ 10.7 (≈ 12.5 without reuse) |
| Each lossless knob | ≈ 3.0 (about 2.1 when a control arm is shared across 2 knobs) |

## Feasibility of the largest kernel lever (K1: FP8 W8A16 for the BF16 non-MoE weights), and the alternative the plan discarded (NVFP4-W4A16 attention, which has a measured TP4 analogue)

# GAP 4: K1 feasibility (FP8 W8A16) compared with NVFP4-W4A16 and MXFP8

Paths: `v11` is `scratchpad/v11src/vllm`. `up` is upstream vLLM at `487ecf187`. The image's `_version.py` is `0.1.dev20051+g487ecf187`, and `gh api` shows that commit is on upstream, dated 2026-08-25. I saved the upstream `CMakeLists.txt`, `generate_kernels.py` and `utils.cmake` as `scratchpad/agents/gap4_*`. C means confirmed from code or logs. E means estimated or inferred.

## 1. How to turn on W8A16 in v11
- **The gate.** `MarlinFP8ScaledMMLinearKernel.is_supported` refuses cc>=89 unless `VLLM_TEST_FORCE_FP8_MARLIN=1` (v11 `kernels/linear/scaled_mm/marlin.py:46-55`) (C). The env var has one other effect: it forces the NVFP4 and FP8 MoE oracles to Marlin (`fused_moe/oracle/nvfp4.py:274`). That is harmless here because we already use `--moe-backend marlin` (C).
- **Online FP8 methods** (`online/base.py:66-72`):
  - Per-tensor (`online/fp8.py:158-257`) picks Marlin, but it uses one amax for a whole fused layer (`:214`). That is not acceptable for the merged KDA `in_proj`, which carries q/k/v plus gates.
  - Block-128 (`:259-336`) routes through the block-kernel list, where DeepGEMM and CUTLASS come ahead of Marlin (`__init__.py:407-417`). Those are W8A8.
  - PTPC per-channel is the right layout, but `:381-389` raises if Marlin is chosen, because PTPC means per-token activation FP8 and Marlin is weight-only (C).
  - Its `process_weights_after_loading` already writes weights as (K,N) with an [N,1] scale. That is exactly what Marlin's non-block path accepts (`marlin_utils_fp8.py:123-190`). So a small subclass that drops the refusal (or calls `init_wfp8_a16_linear_kernel`, `__init__.py:945`) is enough (C).
- **Online quant can't be switched on from the CLI.** `OnlineQuantizationConfig` is model-level. The ModelOpt config already owns the model, so a per-layer wrapper config is needed (C).
- **The op.** `ops.marlin_gemm(b_q_type=float8_e4m3fn)` (`marlin_utils_fp8.py:83-101`). It runs W8A16 only; `:80-81` raises for W8A8 (C).
- **The three BF16 forcings:**
  - `kda.py:168-174` strips `quant_config`.
  - `model.py:331` builds MLA with `None`.
  - `model.py:1203-1240` only acts on `float8_e4m3fn` tensors on disk. It does nothing for online quantization from BF16. For **offline** FP8 packs it is a silent trap: an `.o_proj.` weight that has `weight_scale` rather than `weight_scale_inv` gets buffered and never loaded (`:1228-1240`, C by reading the code). Community FP8 packs are therefore unsafe.
  - Shared experts and `lm_head` already receive `quant_config` (`model.py:216-220`, `:948`). They are BF16 only because the checkpoint's ignore list excludes them (C).

## 2. Is the FP8 Marlin kernel built for sm_121?
- `up CMakeLists.txt:617-622`: with CUDA 13 or later, `MARLIN_BF16_ARCHS` is `"8.0+PTX;9.0+PTX;12.0f"`. `utils.cmake:402-411` turns the image's `12.0` into `12.0f`, a family cubin that runs on sm_121 (C).
- `generate_kernels.py:87-93` generates FP8 b-type kernels with group -1 and 128. `:112-118` generates MXFP8. Both go into the same file, `sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu`, which is compiled with `MARLIN_BF16_ARCHS` (`CMake:687-695`) (C).
- **Consequence: FP8 and MXFP8 are present or missing together. NVFP4 sits in a different file (`fe2m1f`).**
- The image env matches: `TORCH_CUDA_ARCH_LIST=8.0 … 12.0`, CUDA 13.0.1. `VLLM_BUILD_COMMIT` is unknown, so whether the binary was built from this tree is inferred (E, high confidence).
- **Correction to system-uma-comm-MISSED-5:** `rank1.log.tail:30,35` is `marlin_utils_fp4.py:354`, i.e. **MoE** Marlin (a different library), not dense `marlin_gemm`. No dense Marlin kernel has ever run in this lab. Dense Marlin NVFP4 has run on sm_121 only in Tony's TP4 test, which uses the same day-0 base image (E on image lineage).
- **What would settle it** (needs owner OK):
  - Extract the library: `docker create` then `docker cp …/vllm/_C_stable_libtorch.abi3.so`.
  - `cuobjdump --list-elf` should show sm_120 ELF entries.
  - `cuobjdump -sass -arch sm_120 … | grep 'Function :' | grep Marlin | grep -c 2814749767172868` counts FP8 b-type instances (the `kFE4M3fn` id). Compare with `562949953487106` (`kFE2M1f`).
  - Better still: a 1-node microbenchmark of `ops.marlin_gemm` against `F.linear` at M=8 and 16 on the real shapes. That answers both "is the kernel there" and "what bandwidth does it actually reach".

## 3. Memory spike on GB10 while quantizing at load (E)
- The online path materialises one layer at a time from the meta device. The fp32 scratch is chunked at 64 MB or less (`online/fp8.py:84-96`).
- The biggest per-rank target layer is about 64-100 MiB BF16. Scratch is about 3× the FP8 size, so the peak is roughly the final footprint plus 0.4 GiB or less.
- Resident weights fall from 90.36 to about 87.05 GiB/rank. There is no nvcc or JIT step, because Marlin is compiled ahead of time. By contrast, FlashInfer's CUTLASS path showed 8.7 GB of compiler (`cicc`) memory (`tony_readme.md:35`). Against about 20 GiB free, this is safe.
- Risk: the layer-by-layer loader only processes a layer once every element has loaded. `fused_qkv_a` carries NoPE padding that is never loaded, so exclude it for now.

## 4. Where the "corruption correlation" claim comes from
- The only source is `tony_readme.md:9,17-25,31` ("ModelOpt builds that quantize attention…") plus vLLM #54150. That issue is open with the root cause unresolved.
- EXT-10 (confirmed) contradicts the attention link: LibertAI `caca4e6` keeps every `self_attn` in BF16 (11.31 GiB). So the corrupting pack does not quantize attention.
- Tony's own launcher guard would also refuse his NVFP4-attention build (`:31`).
- **So the appendix's reason for discarding NVFP4 attention has no support.** The real open risk is quality that nobody has measured.

| per rank, same scope (KDA+MLA q_b/o/wq_b+shared = 6.62 GiB) | FP8-W8A16 (MarlinFP8) | NVFP4-W4A16 (Marlin FP4) | MXFP8 (MarlinMxfp8) |
|---|---|---|---|
| Bytes saved per step, and resident memory freed | 3.31 GiB | 4.76 GiB | 3.20 GiB |
| Upper bound at 273/220 GB/s (on a ~116 ms step) | 13.0/16.2 ms, +13-16% | 18.7/23.2 ms, +19-25% (Tony TP4 measured ×1.14-1.30) | 12.6/15.6 ms, +12-16% |
| Kernel on sm_121 | Built per source; has never run anywhere we know | Dense kernel ran in Tony's TP4 test | Same file as FP8. Auto-selection trap: `FlashInferCutlassMxfp8.is_supported` accepts cc>=100, so it wins on sm_121 (`mxfp8/flashinfer.py:19-28`, `__init__.py:480-485`), which is the FlashInfer JIT class that OOM'd spark2. Needs `--linear-backend marlin` (`:266-271`) |
| Patches | Env var + ~15-line per-channel W8A16 method + wrapper config at `kda.py:172`, `model.py:331`, `:216-220`. No new checkpoint, Python only | 2-line un-forcing + **new offline checkpoint**. There is no online NVFP4 linear method (`online/base.py:66-72`). Needs the `W4A16_NVFP4` / mixed-precision config (`modelopt.py:1024-1032,2248-2256`) and about 10 GiB of shards on each node | Online method already exists (`online/base.py:70`) + wrapper + `--linear-backend marlin` |
| Quality evidence | Z.ai's own FP8 release quantizes MLA and shared experts but keeps KDA BF16. Void-Z: GSM8K +1.1, IFBench -2.4 | Needles pass to 450K. One digit dropped at 131K. No KL or benchmarks. NVIDIA and LibertAI both left `self_attn` unquantized at 4-bit | Used by other Spark lanes through B12X, not Marlin. No published accuracy numbers found |
| Verdict | **GO first**, once the kernel check passes | **GO second**: biggest upside and a kernel path already shown to run | **NO-GO as the main path.** Keep as a quality fallback for KDA `in_proj` |

If FP8 Marlin turns out to be missing from the library, MXFP8 is missing too, and K1 becomes an image rebuild (XL). NVFP4 then becomes the lever.

## 5. Recommendation
1. **Phase 0 (a few GPU minutes on one node):** the `cuobjdump` check plus the Marlin microbenchmark. Continue only if Marlin FP8 reaches at least 0.85× the bytes-per-second of cuBLAS BF16.
2. **Drafter FP8:** `"quantization":"fp8"` in the speculative config plus the env var. The target model's output stays lossless and only acceptance can move (E; `config/speculative.py:114,924`). This proves the kernel works end to end.
3. **Target, one knob at a time:** shared experts, then MLA q_b/o/wq_b, then KDA `o_proj`, then KDA `in_proj`.
4. **Keep in BF16:** the indexer, `f_b`/`g_b`, the router, mHC fn, embed, `lm_head`, `kv_b`, vision, and `fused_qkv_a`.

**Tier-0 quality bar per stage (proposed thresholds):**
- Count-200 greedy output stays lossless, the thinking-off smoke test passes, and `smoke_vision` passes.
- Teacher-forced top-1 agreement with the BF16 run is at least 99% for FP8 and at least 98% for NVFP4, over 2,000 or more positions (prose, code, CJK, tool calls). Mean top-20 KL is at most 1e-3 for FP8 and 3e-3 for NVFP4.
- Zero U+FFFD across 3 runs of about 50 CJK prompts.
- Tool-call JSON parses at least 98% of the time.
- Needles at 8192 and 20480 pass, plus one at 128K or more for the KDA stages.
- DFlash2 acceptance stays within ±3%.
- Judge speed on ms/step over 10 or more runs, not on the 3-run prose median (L3).