#!/usr/bin/env python3
r"""Per-rank BF16 vs Marlin FP8 W8A16 vs Marlin NVFP4 W4A16 GEMM bench.

Covers GLM53_FP8_W8A16 and GLM53_NVFP4_W4A16. Shapes come from the safetensors
headers of the nvidia target pack and the DFlash2 drafter, sharded as one TP
rank sees them. Each distinct GEMM is timed at M = 1, 8, 16, 32 rows (c=1
verify at k=7 is M=8, c=2 is M=16) inside a CUDA graph, which is how the serve
replays verify steps and also proves both paths are capture-safe. Both paths
quantize the same (checkpoint or --random) weights with the patch's own code.

  # CPU only, no torch: bytes per group per rank per step
  python3 tools/bench_fp8_marlin.py --report-bytes

  # GPU, exclusive slot (no serve up), in glm53-sm121-v13 (needs patch_v13_fp8).
  # The image's entrypoint is `vllm serve`, so replace it with python3. Mount
  # the repo and the HF cache (E1 mounted it at /hf: evidence/e1-microbench/).
  docker run --rm --gpus all --entrypoint python3 \
    -v "$PWD":/work -w /work -v ~/.cache/huggingface:/hf:ro -e HF_HUB_CACHE=/hf/hub \
    glm53-sm121-v13 tools/bench_fp8_marlin.py --json fp8-bench.json

--ckpt and --draft default to the snapshots of recipe.yaml's model.revision
and model.draft_revision under $HF_HUB_CACHE (else $HF_HOME/hub). Pass --draft
to bench another drafter revision.

Pass: for every group, at each --gate-m, the count-weighted FP8 time is at most
--max-ratio (0.6) of BF16, i.e. >= 0.85x of the ideal 2x byte saving, and the
NVFP4 time is at most --max-ratio-nvfp4 (0.8) of FP8; the FP8 output stays
within --max-fro-err of BF16, and the NVFP4 output within --max-fro-err-nvfp4
of a BF16 GEMM on its own dequantized weights (both catch layout bugs, not
quantization noise). Exit status 0 = pass, 1 = fail.
"""

import argparse
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

RECIPE = Path(__file__).resolve().parent.parent / "recipe.yaml"
GROUPS = ("draft", "shared", "mla", "kda_o", "kda_in", "lm_head")


def recipe_pin(key: str) -> str:
    """A pinned sha from recipe.yaml, the source of truth (`key: &key <sha>`; no PyYAML)."""
    m = re.search(rf"^\s*{key}: &{key} ([0-9a-f]{{40}})\s*$", RECIPE.read_text(encoding="utf-8"), re.M)
    if m is None:
        raise SystemExit(f"{RECIPE}: no `{key}: &{key} <40-hex sha>` line")
    return m.group(1)


def hub_dir() -> Path:
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"
    )
    return Path(os.path.expanduser(hub))


def read_headers(snapshot: Path) -> dict[str, tuple[Path, int, str, list[int]]]:
    """name -> (file, absolute data offset, dtype, shape) from every shard."""
    out = {}
    for f in sorted(snapshot.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                start = 8 + n + meta["data_offsets"][0]
                out[name] = (f, start, meta["dtype"], meta["shape"])
    return out


def marlin_padded(n: int, k: int) -> tuple[int, int]:
    """marlin_padded_nk(n, k, group_size=-1) from the v11 image."""
    up = lambda x, m: (x + m - 1) // m * m  # noqa: E731
    return min(((up(n, 64), up(k, 128)), (up(n, 128), up(k, 64))),
               key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


def build_gemms(ckpt: Path, draft: Path, tp: int, rank: int) -> list[dict]:
    """Per-rank GEMMs per group: parts are (tensor, mode) with mode col|row|rep."""
    cfg = json.loads((ckpt / "config.json").read_text())
    layers = cfg.get("text_config", cfg)["num_hidden_layers"]  # excludes MTP layer
    t_hdr, d_hdr = read_headers(ckpt), read_headers(draft)
    tpre = "model.language_model.layers.{}."

    def target_layers(suffix):
        return [i for i in range(layers) if tpre.format(i) + suffix in t_hdr]

    kda = target_layers("self_attn.k_proj.weight")
    mla = target_layers("self_attn.q_b_proj.weight")
    shared = target_layers("mlp.shared_experts.gate_proj.weight")
    nd = json.loads((draft / "config.json").read_text())["num_hidden_layers"]
    specs = [
        ("kda_in", "in_proj_qkvbfg_a", t_hdr, tpre, kda,
         [("self_attn.q_proj", "col"), ("self_attn.k_proj", "col"),
          ("self_attn.v_proj", "col"), ("self_attn.b_proj", "col"),
          ("self_attn.f_a_proj", "rep"), ("self_attn.g_a_proj", "rep")], 1),
        ("kda_o", "o_proj", t_hdr, tpre, kda, [("self_attn.o_proj", "row")], 1),
        ("mla", "q_b_proj", t_hdr, tpre, mla, [("self_attn.q_b_proj", "col")], 1),
        ("mla", "o_proj", t_hdr, tpre, mla, [("self_attn.o_proj", "row")], 1),
        ("shared", "gate_up_proj", t_hdr, tpre, shared,
         [("mlp.shared_experts.gate_proj", "col"), ("mlp.shared_experts.up_proj", "col")], 1),
        ("shared", "down_proj", t_hdr, tpre, shared,
         [("mlp.shared_experts.down_proj", "row")], 1),
        # Read twice per step: target verify and DFlash2's candidate top-k.
        ("lm_head", "lm_head", t_hdr, "", [None], [("lm_head", "col")], 2),
        ("draft", "qkv_proj", d_hdr, "layers.{}.", range(nd),
         [("self_attn.q_proj", "col"), ("self_attn.k_proj", "col"),
          ("self_attn.v_proj", "col")], 1),
        ("draft", "o_proj", d_hdr, "layers.{}.", range(nd), [("self_attn.o_proj", "row")], 1),
        ("draft", "gate_up_proj", d_hdr, "layers.{}.", range(nd),
         [("mlp.gate_proj", "col"), ("mlp.up_proj", "col")], 1),
        ("draft", "down_proj", d_hdr, "layers.{}.", range(nd), [("mlp.down_proj", "row")], 1),
        ("draft", "kernel_projection", d_hdr, "layers.{}.", range(nd),
         [("attention_conv.kernel_projection", "rep")], 2),  # attention_conv + mlp_conv
        ("draft", "fc", d_hdr, "", [None], [("fc", "rep")], 1),
        ("draft", "hidden_projection", d_hdr, "", [None],
         [("candidate_selector.hidden_projection", "rep")], 1),
    ]
    gemms = []
    for group, name, hdr, prefix, idx, parts, reads in specs:
        first = idx[0]
        full = []
        for suffix, mode in parts:
            tname = (prefix.format(first) if first is not None else "") + suffix + ".weight"
            _, _, dtype, shape = hdr[tname]
            if dtype != "BF16" or len(shape) != 2:
                raise SystemExit(f"{tname}: expected 2-D BF16, got {dtype} {shape}")
            full.append((tname, mode, shape))
        n = sum(s[0] // tp if m == "col" else s[0] for _, m, s in full)
        k = {s[1] // tp if m == "row" else s[1] for _, m, s in full}
        if len(k) != 1:
            raise SystemExit(f"{group}/{name}: mixed K {k}")
        gemms.append(dict(group=group, name=name, n=n, k=k.pop(), count=len(idx) * reads,
                          parts=[(hdr[t][0], hdr[t][1], s, m) for t, m, s in full],
                          tp=tp, rank=rank))
    return gemms


def report_bytes(gemms: list[dict]) -> None:
    """Weight bytes streamed per rank per step: BF16, Marlin FP8 (e4m3 + BF16
    per-channel scale) and Marlin NVFP4 (E2M1 + e4m3 per 16 + fp32 global),
    both on Marlin's padded (n, k)."""
    mib = 2**20
    print(f"{'group':8} {'gemm':18} {'n x k (per rank)':>18} {'GEMMs/step':>10} "
          f"{'BF16 MiB':>9} {'FP8 MiB':>8} {'NVFP4 MiB':>9} {'FP8 saves':>9} "
          f"{'NVFP4 saves':>11} {'vs FP8':>8}")
    tot = [0, 0, 0]
    for g in GROUPS:
        gb = gf = g4 = 0
        for m in (x for x in gemms if x["group"] == g):
            pn, pk = marlin_padded(m["n"], m["k"])
            b = m["n"] * m["k"] * 2 * m["count"]
            f = (pn * pk + pn * 2) * m["count"]
            f4 = (pn * pk // 2 + pn * pk // 16 + 4) * m["count"]
            gb, gf, g4 = gb + b, gf + f, g4 + f4
            print(f"{g:8} {m['name']:18} {m['n']:>8} x {m['k']:<7} {m['count']:>10} "
                  f"{b / mib:9.1f} {f / mib:8.1f} {f4 / mib:9.1f} {(b - f) / mib:9.1f} "
                  f"{(b - f4) / mib:11.1f} {(f - f4) / mib:8.1f}")
        print(f"{g:8} {'= group total':18} {'':>18} {'':>10} {gb / mib:9.1f} {gf / mib:8.1f} "
              f"{g4 / mib:9.1f} {(gb - gf) / mib:9.1f} {(gb - g4) / mib:11.1f} "
              f"{(gf - g4) / mib:8.1f}  (NVFP4 saves {(gb - g4) / 1e9:.3f} GB/step vs BF16, "
              f"{(gf - g4) / 1e9:.3f} vs FP8)")
        tot = [tot[0] + gb, tot[1] + gf, tot[2] + g4]
    for label, t in (("FP8", tot[1]), ("NVFP4", tot[2])):
        print(f"all groups: BF16 {tot[0] / 1e9:.3f} GB -> {label} {t / 1e9:.3f} GB per rank "
              f"per step, saves {(tot[0] - t) / 1e9:.3f} GB = {(tot[0] - t) / 205e6:.1f} ms "
              f"at 205 GB/s, {(tot[0] - t) / 273e6:.1f} ms at 273 GB/s")


def load_rank_weight(m: dict, torch):
    """Rank-local (n, k) BF16 weight assembled from the checkpoint shards."""
    rows = []
    for path, start, (r, c), mode in m["parts"]:
        with open(path, "rb") as fh:
            if mode == "col":
                rr = r // m["tp"]
                fh.seek(start + m["rank"] * rr * c * 2)
                t = torch.frombuffer(bytearray(fh.read(rr * c * 2)), dtype=torch.bfloat16)
                rows.append(t.view(rr, c))
            else:
                fh.seek(start)
                t = torch.frombuffer(bytearray(fh.read(r * c * 2)), dtype=torch.bfloat16)
                t = t.view(r, c)
                if mode == "row":
                    cc = c // m["tp"]
                    t = t[:, m["rank"] * cc : (m["rank"] + 1) * cc]
                rows.append(t)
    return torch.cat(rows, 0).contiguous()


def bench(gemms: list[dict], args) -> int:
    import torch
    import torch.nn.functional as F
    from vllm.model_executor.layers.quantization.glm53_fp8_w8a16 import (
        quantize_layer_to_marlin_fp8,
    )

    nvfp4 = None
    if not args.no_nvfp4:
        try:
            from vllm.model_executor.layers.quantization.glm53_fp8_w8a16 import (
                dequantize_nvfp4,
                quantize_layer_to_marlin_nvfp4,
                quantize_nvfp4,
            )
            nvfp4 = (quantize_nvfp4, dequantize_nvfp4, quantize_layer_to_marlin_nvfp4)
        except ImportError as e:
            print(f"NVFP4 path unavailable ({e}); skipping that column")

    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(0)

    def time_ms(fn) -> float:
        """ms per fn() call, replayed from a CUDA graph."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(args.iters):
            g.replay()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) / args.iters

    rows, ok = [], True
    for m in gemms:
        if m["group"] not in args.groups:
            continue
        n, k = m["n"], m["k"]
        w = (torch.randn(n, k) * 0.02).to(torch.bfloat16) if args.random \
            else load_rank_weight(m, torch)
        w = w.to(dev)
        wbytes = n * k * 2
        copies = max(1, min(16, math.ceil(args.rotate_mb * 2**20 / wbytes)))
        bf16_ws = [w.clone() for _ in range(copies)]
        fp8_layers = []
        for i in range(copies):
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
            quantize_layer_to_marlin_fp8(layer)
            fp8_layers.append(layer)
        lay0 = fp8_layers[0]
        fp8_bytes = lay0.weight.numel() * lay0.weight.element_size() \
            + lay0.weight_scale.numel() * lay0.weight_scale.element_size()
        fp4_layers, w4ref = [], None
        if nvfp4 is not None:
            # BF16 GEMM on the dequantized NVFP4 weights isolates kernel/layout
            # error from quantization error.
            w4ref = nvfp4[1](*nvfp4[0](w)).to(torch.bfloat16)
            for _ in range(copies):
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
                nvfp4[2](layer)
                fp4_layers.append(layer)
            lay4 = fp4_layers[0]
            fp4_bytes = sum(p.numel() * p.element_size() for p in
                            (lay4.weight, lay4.weight_scale, lay4.weight_global_scale))
        for M in args.m:
            x = torch.randn(M, k, device=dev).to(torch.bfloat16)
            ref_y = F.linear(x, w).float()
            y8 = fp8_layers[0].quant_method.apply(fp8_layers[0], x).float()
            diff = (y8 - ref_y).abs()
            max_abs = float(diff.max())
            max_rel = max_abs / max(float(ref_y.abs().max()), 1e-30)
            fro = float((y8 - ref_y).norm() / ref_y.norm().clamp_min(1e-30))
            t16 = time_ms(lambda: [F.linear(x, c) for c in bf16_ws]) / copies
            t8 = time_ms(lambda: [lay.quant_method.apply(lay, x) for lay in fp8_layers]) / copies
            t4 = fro4 = quant4 = None
            if fp4_layers:
                y4 = lay4.quant_method.apply(lay4, x).float()
                ref4 = F.linear(x, w4ref).float()
                fro4 = float((y4 - ref4).norm() / ref4.norm().clamp_min(1e-30))
                quant4 = float((y4 - ref_y).norm() / ref_y.norm().clamp_min(1e-30))
                t4 = time_ms(lambda: [lay.quant_method.apply(lay, x)
                                      for lay in fp4_layers]) / copies
            row = dict(group=m["group"], gemm=m["name"], n=n, k=k, count=m["count"], M=M,
                       bf16_us=t16 * 1e3, fp8_us=t8 * 1e3, ratio=t8 / t16,
                       bf16_gbs=wbytes / t16 / 1e6, fp8_gbs=fp8_bytes / t8 / 1e6,
                       nvfp4_us=None if t4 is None else t4 * 1e3,
                       nvfp4_gbs=None if t4 is None else fp4_bytes / t4 / 1e6,
                       max_abs_err=max_abs, max_rel_err=max_rel, fro_rel_err=fro,
                       nvfp4_fro_err=fro4, nvfp4_vs_bf16_fro_err=quant4,
                       weights="random" if args.random else "checkpoint")
            rows.append(row)
            nv = "" if t4 is None else (
                f" nvfp4 {t4 * 1e3:8.1f}us {row['nvfp4_gbs']:6.1f}GB/s ratio {t4 / t16:.3f}"
                f" (fro {fro4:.2e} vs dequant, {quant4:.3f} vs bf16)")
            print(f"{m['group']:8} {m['name']:18} {n:>6}x{k:<6} M={M:<3} "
                  f"bf16 {t16 * 1e3:8.1f}us {row['bf16_gbs']:6.1f}GB/s  "
                  f"fp8 {t8 * 1e3:8.1f}us {row['fp8_gbs']:6.1f}GB/s  ratio {t8 / t16:.3f}"
                  f"  err max_abs {max_abs:.2e} max_rel {max_rel:.2e} fro {fro:.2e}{nv}",
                  flush=True)
            if fro > args.max_fro_err:
                ok = False
                print(f"FAIL {m['group']}/{m['name']} M={M}: fro_rel_err {fro:.3f} "
                      f"> {args.max_fro_err} (layout/kernel error)")
            if fro4 is not None and fro4 > args.max_fro_err_nvfp4:
                ok = False
                print(f"FAIL {m['group']}/{m['name']} M={M}: NVFP4 fro_rel_err {fro4:.3f} "
                      f"vs dequantized weights > {args.max_fro_err_nvfp4} (layout/kernel error)")
        del bf16_ws, fp8_layers, fp4_layers, w4ref, w
        torch.cuda.empty_cache()

    print("\ncount-weighted per step (one rank):")
    summary = []
    for g in args.groups:
        for M in args.m:
            sel = [r for r in rows if r["group"] == g and r["M"] == M]
            if not sel:
                continue
            t16 = sum(r["bf16_us"] * r["count"] for r in sel)
            t8 = sum(r["fp8_us"] * r["count"] for r in sel)
            has4 = all(r["nvfp4_us"] is not None for r in sel)
            t4 = sum(r["nvfp4_us"] * r["count"] for r in sel) if has4 else None
            gate = M in args.gate_m
            passed = t8 / t16 <= args.max_ratio
            passed4 = None if t4 is None else t4 / t8 <= args.max_ratio_nvfp4
            ok &= (passed and passed4 is not False) or not gate
            summary.append(dict(group=g, M=M, bf16_ms=t16 / 1e3, fp8_ms=t8 / 1e3,
                                ratio=t8 / t16, saved_ms=(t16 - t8) / 1e3,
                                nvfp4_ms=None if t4 is None else t4 / 1e3,
                                nvfp4_ratio=None if t4 is None else t4 / t16,
                                nvfp4_vs_fp8=None if t4 is None else t4 / t8,
                                nvfp4_saved_ms=None if t4 is None else (t16 - t4) / 1e3,
                                gated=gate, passed=passed, nvfp4_passed=passed4))
            verdict = ("PASS" if passed else "FAIL") if gate else "info"
            nv = "" if t4 is None else (
                f"  nvfp4 {t4 / 1e3:7.2f} ms ratio {t4 / t16:.3f} ({t4 / t8:.3f} of fp8) "
                f"saves {(t16 - t4) / 1e3:6.2f} ms/step  "
                f"{('PASS' if passed4 else 'FAIL') if gate else 'info'}")
            print(f"  {g:8} M={M:<3} bf16 {t16 / 1e3:7.2f} ms  fp8 {t8 / 1e3:7.2f} ms  "
                  f"ratio {t8 / t16:.3f}  saves {(t16 - t8) / 1e3:6.2f} ms/step  {verdict}{nv}")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            rows=rows, summary=summary, passed=ok, max_ratio=args.max_ratio,
            max_ratio_nvfp4=args.max_ratio_nvfp4), indent=1))
    print(f"\n{'PASS' if ok else 'FAIL'}: FP8/BF16 time ratio <= {args.max_ratio} and "
          f"NVFP4/FP8 <= {args.max_ratio_nvfp4} at M in {args.gate_m} for every group, "
          f"fro_rel_err <= {args.max_fro_err} (FP8 vs BF16) and "
          f"<= {args.max_fro_err_nvfp4} (NVFP4 vs its dequantized weights)")
    return 0 if ok else 1


def main() -> int:
    ints = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=hub_dir()
                   / f"models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/{recipe_pin('revision')}")
    p.add_argument("--draft", type=Path, default=hub_dir()
                   / f"models--incoai--GLM-5.3-Flash-DFlash2/snapshots/{recipe_pin('draft_revision')}")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--groups", type=lambda s: s.split(","), default=list(GROUPS))
    p.add_argument("--m", type=ints, default=[1, 8, 16, 32], help="GEMM rows to time")
    p.add_argument("--gate-m", type=ints, default=[8, 16], help="rows the pass rule uses")
    p.add_argument("--iters", type=int, default=50, help="graph replays per timing")
    p.add_argument("--rotate-mb", type=int, default=256,
                   help="rotate weight copies so each timing streams >= this much (beats L2)")
    p.add_argument("--max-ratio", type=float, default=0.6)
    p.add_argument("--max-fro-err", type=float, default=0.1)
    p.add_argument("--max-ratio-nvfp4", type=float, default=0.8,
                   help="NVFP4/FP8 time ratio the pass rule allows")
    p.add_argument("--max-fro-err-nvfp4", type=float, default=0.02,
                   help="NVFP4 output vs BF16 GEMM on the dequantized NVFP4 weights")
    p.add_argument("--random", action="store_true", help="random weights, skip ckpt reads")
    p.add_argument("--no-nvfp4", action="store_true")
    p.add_argument("--report-bytes", action="store_true", help="CPU only: bytes table")
    p.add_argument("--json", help="write rows and summary here")
    args = p.parse_args()
    bad = set(args.groups) - set(GROUPS)
    if bad:
        p.error(f"unknown groups {sorted(bad)}")
    gemms = build_gemms(args.ckpt, args.draft, args.tp, args.rank)
    if args.report_bytes:
        report_bytes(gemms)
        return 0
    return bench(gemms, args)


if __name__ == "__main__":
    sys.exit(main())
