# Grounded hypotheses (read before each attempt)

H1. DFlash2-6 (num_speculative_tokens=6, graphs 7/14/21/28). Mechanism: only untested slot; pos 4 still accepts on mixed traffic. Risk: KDA copies squeeze c=4. Expected: small prose c=1 gain or none.

H2. `--max-num-batched-tokens 4096` (or 3072). Mechanism: engine warns 2048 is suboptimal with extra draft slots. Expected: better admission / slightly higher decode, needed later for 1M prefill.

H3. DFlash2-7 at max-num-seqs=2. Mechanism: full trained block (block_size 8). Tony reported 46.9 tok/s on code with 7 slots. Our c=4 occupancy is the reason we truncated to 5. Expected: higher c=1 decode, worse c=4.

H4. NVFP4 packed MLA KV (`nvfp4_ds_mla`) + max-model-len 1048576. Mechanism: ~1.8× denser KV is the only published 2× GB10 path that actually needles 1M. Expected: context pass, decode drop toward ~22.

H5. Occupancy lanes in run.sh (LANE=decode|long). Mechanism: first-principles of two targets that do not share a KV format. Decode lane keeps fp8+DFlash2. Long lane is NVFP4 KV + 1M + max-num-seqs=2.

H6. Thinking-off template leak. Mechanism: smoke content still contains chain-of-thought; drowzeys documented stock template ignoring enable_thinking. Not a tok/s hill; correctness of the published smoke.

H-cutlass. `MOE_BACKEND=flashinfer_cutlass` on `glm53-sm121-v12` so Marlin-blind `input_scale` tensors get used. Mechanism: LibertAI 2026-08-30 calibrated MoE input scales; Marlin never reads them. Result: spark2 global OOM during `cudafe++` after 90.67 GiB weights (~18 GiB left). Do not retry. `run.sh` refuses non-marlin unless `FORCE_UNSAFE_MOE=1`.

H-snap. Pin `caca4e6` on v11 + marlin (one change vs `aa28e1f`). Mechanism: same layout, newer NVFP4 shards. Marlin still ignores `input_scale`, so this is a quality/decode check of the rest of the checkpoint, not a MoE-scale win.

Do not stack. One change, measure, keep or revert.
