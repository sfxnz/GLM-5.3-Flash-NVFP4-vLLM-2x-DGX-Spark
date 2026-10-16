# v14 image layer: decode-step patches

`Dockerfile.sm121-v14` builds on `glm53-sm121-v13` and adds one Python-only patch layer, `patch_v14_decode.py`. Each switch is a `GLM53_*` environment variable and is off in the image. With none set, v14 runs v13's code.

```bash
docker build -f docker/Dockerfile.sm121-v14 -t glm53-sm121-v14 docker
GLM53_V13_SRC=/path/to/v13src python3 -m unittest docker/test_v14_decode.py -v   # CPU tests
```

`GLM53_V13_SRC` is the directory that holds the `vllm/` tree copied out of `glm53-sm121-v13`. Build once on the head and ship it (`docker save glm53-sm121-v14 | ssh spark2 docker load`) so both nodes run the same image ID.

## Switches

| Env var | Changes | Why | Numerics |
|---|---|---|---|
| `GLM53_ROUTER_DEDUPE=1` | `Glm5NextMoE.forward` skips its own `self.gate(hidden_states)` when the MoE runner holds the same gate | The model passes `gate=self.gate` to `FusedMoEFactory`. The runner's `_forward_impl` then recomputes `router_logits` and overwrites the argument before reading it. The model-side call is dead work: one 4096×288 GEMM plus a split-K reduce per MoE layer, 42 per step (~0.72 ms per verify step in the E5 F3 trace) | Exact. The logits the runner uses are unchanged |
| `GLM53_HEADGATE_MM=1` | The DSA indexer head-gate runs `torch.mm(x_bf16, W_bf16, out_dtype=float32)` instead of `torch.mm(x.float(), W_fp32)` | At M=8 cuBLAS picks a 2-block fp32 gemv (`gemmSN_NN<float>`, ~72 µs for 0.5 MB, 11 layers per step). The operands are bf16 at the source, so their fp32 casts are exact. Microbench (spark1): M=8 41 → 21 µs, M=1152 169 → 17 µs, M=4608 1135 → 166 µs | Same products with fp32 accumulation; only the summation order differs (max relative difference 4e-6) |
| `GLM53_LATE_MLA_PLAN=1` | `build_attn_metadata` builds the KV-cache group of the SM90 sparse-MLA builder last (not during graph capture) | That builder replans with `seq_lens[:num_reqs].cpu()`, which waits for the previous step's drafter graph. The four KDA builders that followed it read host tensors only, so ~2.1 ms per verify step of host work ran while the GPU idled (E5 F3 trace). Built last, that host work overlaps the drafter graph | Exact. Every group gets the same metadata; only the host build order changes |

`run.sh` turns `GLM53_ROUTER_DEDUPE` and `GLM53_HEADGATE_MM` on through the `DECODE_FIXES` knob (see the recipe README). `GLM53_LATE_MLA_PLAN` stays off: r2 B6 measured no decode change (A 31.11 vs 31.37 tok/s, step 69.9 vs 69.7 ms). The idle analysis behind it ran on a census-on trace, whose `.cpu()` already stalls the host, so the overlap it predicted may not exist in a default serve. Set it through `EXTRA_ENV` to try again.
