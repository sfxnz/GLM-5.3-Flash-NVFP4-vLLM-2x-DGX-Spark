# r3: research, bug fixes, measured decode and quality (2026-11-02)

Branch `opt/r3-research-decode`, stacked on PR #16 (`opt/r2-prefill-decode`, 95e1880). Baseline is PR #16's
defaults: prose A 31.23 tok/s (step 69.3 ms, acceptance 2.17), H 21.46 per stream, prefill 1593 / 1562 tok/s at
32k / 128k, checkpoint `09b04e5`.

## Done when (all measured on the Sparks)

1. The new defaults beat PR #16 on prose A past cross-boot noise (`kit/compare.py` KEEP over three plain boots), H
   does not regress, or quality improves measurably (Tier 1 pooled at or above the `09b04e5` runs and the thinking-on
   subset no worse) while A and H stay inside noise.
2. Nothing regresses: cell E within -5% at 32k and 128k, count-200, thinking-off probe, needles, vision, Tier 0 against
   its own reference (a new one when the checkpoint changes), Tier 1 pooled >= -2.0 against the `09b04e5` runs.
3. Each bug fix has a before and after on a live serve.
4. `recipe.yaml` measured rows come from three plain `./run.sh` boots; `kit/render.py --check` and the unit tests pass.

## Rigor

High. One knob per boot, the boot is the unit, keep or revert, rows in `evidence/decision.tsv` and
`evidence/trail.tsv`. No bench while a download, rsync or microbench runs on either node.

## Phases

- P1 research, read-only, in parallel: upstream vLLM / FlashInfer / Humming against our image
  (`research/upstream.md`), community fast lanes and drafters (`research/community.md`), our own bugs
  (`research/bugs.md`). Humming vs Marlin MoE microbench on spark1 (`moe-microbench/`).
- P2 baseline B0: plain `./run.sh` today (`evidence/r2-tools/final.sh` with Tier 0), plus the thinking-on subset
  (`quality/tier1.py run --tasks gsm8k,mmlu_pro --limit 50 --chat-kwargs '{"enable_thinking": true}'
  --think-budget 8192`) for reasoning length and accuracy.
- P3 X1, `SNAPSHOT_REV=1d00738`: NVIDIA's routed-expert re-export that fixes uncalibrated `weight_scale_2` on about a
  third of the experts. Tier 0 `record` of a new reference, the thinking-on subset paired against B0, fast gate,
  cell E, then full Tier 1 against the three `09b04e5` runs (857, 856, 863 of 1010).
- P4 decode and bug units chosen from P1, highest expected value first, one per boot.
- P5 three plain boots of the final defaults, Tier 0 and Tier 1, `recipe.yaml` rows, README render, PR on top of #16.
