#!/usr/bin/env python3
r"""Bit-level GPU check of the GLM53_WQ_DEQUANT_MIN_M dequant on checkpoint weights.

tools/bench_fp8_marlin.py --dequant compares the dequantized weight with its
reference by value (`!=` on BF16), for the first layer of each GEMM on rank 0.
This compares the int16 bit patterns instead, and sorts every mismatch into
sign-of-zero only (+0 vs -0), NaN, or a value difference. The Marlin tensors
come from the patch's own swap functions (the real ops.gptq_marlin_repack on
the GPU), exactly as the serve builds them.

  --all-layers kda_in   checks every layer of that group (the served one) on
                        each --ranks rank, in the modes of --all-layers-modes;
  every other --groups GEMM is checked on its first layer, rank 0, all modes.

  docker run --rm --gpus all --entrypoint python3 -v "$PWD":/work -w /work \
    -v ~/.cache/huggingface:/hf:ro -e HF_HUB_CACHE=/hf/hub glm53-sm121-v13 \
    evidence/e6-prefill/tools/dq_bits_gpu.py --json OUT.json

Exit 0 when no GEMM has a value or NaN mismatch in the INT8 mode (the served
mode for kda_in), 1 otherwise. Sign-of-zero counts are reported, not gated.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.bench_fp8_marlin import (  # noqa: E402
    build_gemms,
    hub_dir,
    load_rank_weight,
    read_headers,
    recipe_pin,
)


def layer_parts(m, hdr, names, prefix, i):
    """The GEMM m's parts for layer i (build_gemms keeps the first layer only)."""
    parts = []
    for path, start, shape, mode in m["parts"]:
        name = names[path, start]  # (file, offset) is unique in the headers
        first = name[len("model.language_model.layers."):].split(".", 1)
        t = prefix.format(i) + first[1]
        parts.append((hdr[t][0], hdr[t][1], hdr[t][3], mode))
    return dict(m, parts=parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups", default="kda_in,kda_o,mla,shared")
    p.add_argument("--modes", default="int8,int4,fp8,nvfp4")
    p.add_argument("--all-layers", default="kda_in")
    p.add_argument("--all-layers-modes", default="int8")
    p.add_argument("--ranks", default="0,1")
    p.add_argument("--int-group-size", type=int, default=128)
    p.add_argument("--json")
    a = p.parse_args()

    import torch
    from vllm.model_executor.layers.quantization import glm53_fp8_w8a16 as q

    ckpt = hub_dir() / f"models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/{recipe_pin('revision')}"
    draft = hub_dir() / f"models--incoai--GLM-5.3-Flash-DFlash2/snapshots/{recipe_pin('draft_revision')}"
    hdr = read_headers(ckpt)
    names = {(v[0], v[1]): t for t, v in hdr.items()}
    cfg = json.loads((ckpt / "config.json").read_text())
    nl = cfg.get("text_config", cfg)["num_hidden_layers"]
    tpre = "model.language_model.layers.{}."
    kda = [i for i in range(nl) if tpre.format(i) + "self_attn.k_proj.weight" in hdr]
    dev = torch.device("cuda", 0)
    gs = a.int_group_size

    def swap_and_ref(w, mode):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
        if mode in q.INT_BITS:
            bits = q.INT_BITS[mode]
            ref = q.dequantize_int(*q.quantize_int(w, bits, gs, q.INT_CLIP_RATIOS[mode]), bits)
            q.quantize_layer_to_marlin_int(layer, mode, gs)
        elif mode == "fp8":
            ref = q.dequantize_per_channel(*q.quantize_per_channel(w))
            q.quantize_layer_to_marlin_fp8(layer)
        else:
            ref = q.dequantize_nvfp4(*q.quantize_nvfp4(w))
            q.quantize_layer_to_marlin_nvfp4(layer)
        return layer, ref.to(torch.bfloat16)

    def check(m, rank, mode, label):
        n, k = m["n"], m["k"]
        if mode in q.INT_BITS and k % gs:
            return None
        w = load_rank_weight(dict(m, rank=rank), torch).to(dev)
        layer, ref = swap_and_ref(w, mode)
        q.ensure_workspace(n * k, dev)
        out = q.dequantize_marlin(layer, q.dequant_spec(layer, mode, gs))
        torch.cuda.synchronize()
        bad = out.view(torch.int16) != ref.view(torch.int16)
        zero = bad & (out == 0) & (ref == 0)
        nan = bad & (out.isnan() | ref.isnan())
        val = bad & ~zero & ~nan
        r = dict(label=label, group=m["group"], gemm=m["name"], rank=rank, mode=mode, n=n, k=k,
                 elements=n * k, bit_mismatch=int(bad.sum()), sign_of_zero=int(zero.sum()),
                 ref_minus0=int((zero & torch.signbit(ref)).sum()), nan=int(nan.sum()),
                 value_diff=int(val.sum()), value_equal=int((out != ref).sum()) == 0)
        print(json.dumps(r), flush=True)
        del w, layer, ref, out
        return r

    t0 = time.time()
    gemms = [m for m in build_gemms(ckpt, draft, 2, 0) if m["group"] in a.groups.split(",")]
    rows = []
    for m in gemms:
        for mode in a.modes.split(","):
            r = check(m, 0, mode, f"{m['group']}/{m['name']} first layer")
            if r:
                rows.append(r)
        if m["group"] in a.all_layers.split(","):
            for rank in (int(x) for x in a.ranks.split(",")):
                for i in kda if m["group"].startswith("kda") else []:
                    ml = layer_parts(m, hdr, names, tpre, i)
                    for mode in a.all_layers_modes.split(","):
                        r = check(ml, rank, mode, f"{m['group']} layer {i}")
                        if r:
                            rows.append(r)
    torch.cuda.empty_cache()
    summ = {}
    for r in rows:
        s = summ.setdefault(f"{r['group']}/{r['mode']}", dict(checks=0, elements=0, bit_mismatch=0,
                                                              sign_of_zero=0, nan=0, value_diff=0))
        for key in ("elements", "bit_mismatch", "sign_of_zero", "nan", "value_diff"):
            s[key] += r[key]
        s["checks"] += 1
    print("\nsummary (group/mode: checks, elements, bit mismatches = sign-of-zero + NaN + value):")
    for key, s in summ.items():
        print(f"  {key:14} {s['checks']:4} checks {s['elements']:>12} el  bits {s['bit_mismatch']:>8}"
              f" = zero {s['sign_of_zero']:>8} + nan {s['nan']} + value {s['value_diff']}")
    served_ok = all(r["value_diff"] == 0 and r["nan"] == 0 for r in rows if r["mode"] == "int8")
    int_bits_ok = all(r["bit_mismatch"] == 0 for r in rows if r["mode"] in q.INT_BITS)
    value_ok = all(r["value_diff"] == 0 and r["nan"] == 0 for r in rows)
    print(f"\nINT8/INT4 bit-equal: {int_bits_ok}; every mode value-equal: {value_ok}; "
          f"{len(rows)} checks in {time.time() - t0:.0f} s")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(rows=rows, summary=summ, int_bits_ok=int_bits_ok,
                                                value_ok=value_ok), indent=1))
    return 0 if served_ok else 1


if __name__ == "__main__":
    sys.exit(main())
