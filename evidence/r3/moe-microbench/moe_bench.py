#!/usr/bin/env python3
r"""Routed-expert MoE microbench for GLM-5.3-Flash NVFP4 at TP=2: vLLM's Marlin
NVFP4 MoE path against its Humming path (indexed GEMM, the default, and grouped).

Each backend gets its own RoutedExperts layers, built by vLLM's FusedMoEFactory
under a VllmConfig whose kernel_config.moe_backend names the backend (what
--moe-backend sets), loaded with identical random ModelOpt NVFP4 checkpoint
tensors (E2M1 codes, E4M3 scale per 16, FP32 weight_scale_2 per expert),
converted by the serve's ModelOptNvFp4FusedMoE.process_weights_after_loading,
and called through RoutedExperts.forward_modular with fixed top-8 routing: the
path a boot runs after the router. Shapes are one TP=2 rank: 288 experts,
w13 [2*1024, 4096], w2 [4096, 1024], BF16 activations.

--layers distinct layers (own weights, own routing) run back to back in every
timed call, so no expert stays in L2. Rows up to --graph-max-m are timed as
CUDA graph replays (how the serve runs decode and verify), larger rows eagerly
(prefill chunks). Every repeat times every backend, in an order that reverses
between repeats. Routing is uniform random top-8 with a fixed seed; a case
like 8p24 draws each row's top-8 from one 24-expert pool per layer, about the
expert count a prose c=1 verify step touches. GB/s counts the checkpoint bytes
of the experts the routing touches, against a 1 GiB copy and sum measured
first. Layer 0's outputs are compared with each other and with an FP32
reference on the dequantized weights.

  docker run --rm --gpus all --ipc=host -v "$PWD":/w -w /w --entrypoint python3 \
    -e HUMMING_CACHE_DIR=/tmp/humming-cache glm53-sm121-v14 \
    evidence/r3/moe-microbench/moe_bench.py --json run.json

  # CPU, no torch: merge runs into the table and the boot decision
  python3 evidence/r3/moe-microbench/moe_bench.py --table run-a.json run-b.json
"""

import argparse
import functools
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

E, TOPK, HIDDEN, INTER = 288, 8, 4096, 1024
W13_N = 2 * INTER
WEIGHTS_PER_EXPERT = W13_N * HIDDEN + HIDDEN * INTER
EXPERT_BYTES = WEIGHTS_PER_EXPERT // 2 + WEIGHTS_PER_EXPERT // 16 + 3 * 4
FLOPS_PER_ROW_EXPERT = 2 * WEIGHTS_PER_EXPERT
BACKENDS = {
    "marlin": ("marlin", None),
    "humming": ("humming", "indexed"),
    "humming-grouped": ("humming", "grouped"),
}
QUANT = {"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8",
                          "group_size": 16, "exclude_modules": []}}
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
COMPILERS = ("nvrtc_compile", "nvcc", "ptxas", "cicc", "cc1plus", "cudafe++", "ninja", "c++", "g++")
MAX_FRO_ERR = 0.02
BOOT_MARGIN = 0.05


def meminfo_kib(key: str) -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    raise KeyError(key)


class MemWatch:
    """Every 0.1 s: lowest MemAvailable per phase; per compiler name (processes
    visible in this container) the pid count and peak RSS (VmHWM); and the most
    compiler processes alive at once with their summed RSS."""

    def __init__(self):
        self.phase = "start"
        self.low: dict[str, int] = {}
        self.compilers: dict[str, dict] = {}
        self.concurrent = {"max_procs": 0, "max_rss_sum_kib": 0}
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            avail, phase = meminfo_kib("MemAvailable"), self.phase
            self.low[phase] = min(self.low.get(phase, avail), avail)
            alive, rss_sum = 0, 0
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    status = Path(f"/proc/{pid}/status").read_text()
                except OSError:
                    continue
                fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
                name = fields.get("Name", "").strip()
                if name not in COMPILERS:
                    continue
                hwm = int(fields.get("VmHWM", "0 kB").split()[0])
                alive += 1
                rss_sum += int(fields.get("VmRSS", "0 kB").split()[0])
                c = self.compilers.setdefault(name, {"pids": set(), "peak_rss_kib": 0, "phases": set()})
                c["pids"].add(pid)
                c["peak_rss_kib"] = max(c["peak_rss_kib"], hwm)
                c["phases"].add(phase)
            self.concurrent["max_procs"] = max(self.concurrent["max_procs"], alive)
            self.concurrent["max_rss_sum_kib"] = max(self.concurrent["max_rss_sum_kib"], rss_sum)
            time.sleep(0.1)

    def report(self) -> dict:
        return {"low_mem_available_kib": dict(self.low), "compilers_concurrent": dict(self.concurrent),
                "compilers": {k: {"pids": len(v["pids"]), "peak_rss_kib": v["peak_rss_kib"],
                                  "phases": sorted(v["phases"])} for k, v in self.compilers.items()}}


def humming_cache() -> Path:
    return Path(os.environ.get("HUMMING_CACHE_DIR", os.path.expanduser("~/.humming/cache/")))


def cubins() -> list[Path]:
    root = humming_cache()
    return sorted(root.rglob("kernel.cubin")) if root.exists() else []


def make_ckpt(layer: int, dev, torch) -> dict:
    """Random ModelOpt-format NVFP4 tensors for one layer, the same for every
    backend and for the reference (seeded by layer). Effective weight RMS ~0.02."""
    g = torch.Generator(device=dev)
    g.manual_seed(1000 + layer)
    codes = lambda *s: torch.randint(0, 256, s, generator=g, device=dev, dtype=torch.uint8)  # noqa: E731
    scales = lambda *s: (torch.rand(s, generator=g, device=dev) * 432 + 16).to(torch.float8_e4m3fn)  # noqa: E731
    gscale = lambda: (torch.rand(E, generator=g, device=dev) * 0.4 + 0.8) * 2.9e-5  # noqa: E731
    w13_gs = gscale()
    return {
        "w13_weight": codes(E, W13_N, HIDDEN // 2),
        "w13_weight_scale": scales(E, W13_N, HIDDEN // 16),
        "w13_weight_scale_2": torch.stack((w13_gs, w13_gs), 1),
        "w2_weight": codes(E, HIDDEN, INTER // 2),
        "w2_weight_scale": scales(E, HIDDEN, INTER // 16),
        "w2_weight_scale_2": gscale(),
        "w13_input_scale": torch.ones(E, 2, device=dev),
        "w2_input_scale": torch.ones(E, device=dev),
    }


def parse_case(s: str) -> tuple[int, int | None]:
    """'8' is 8 rows routed uniformly over all experts; '8p24' draws each row's
    top-8 from one 24-expert pool per layer (correlated verify rows)."""
    rows, _, pool = s.partition("p")
    return int(rows), int(pool) if pool else None


def make_routing(m: int, pool: int | None, layer: int, dev, torch, ids_dtype):
    g = torch.Generator(device=dev)
    g.manual_seed(m * 1000 + (pool or 0) * 100_000 + layer)
    experts = torch.randperm(E, generator=g, device=dev)[:pool] if pool else torch.arange(E, device=dev)
    pick = torch.rand(m, experts.numel(), generator=g, device=dev).topk(TOPK, dim=1).indices
    ids = experts[pick].to(ids_dtype)
    w = torch.rand(m, TOPK, generator=g, device=dev) + 0.1
    return (w / w.sum(1, keepdim=True)).float(), ids


def dequant(packed, scale, gscale, torch):
    """[N, K/2] uint8 (low nibble first) x [N, K/16] E4M3 x FP32 -> [N, K] FP32."""
    lut = torch.tensor(E2M1, device=packed.device)
    codes = torch.stack((packed & 0xF, packed >> 4), -1).flatten(-2).long()
    return lut[codes] * scale.float().repeat_interleave(16, -1) * gscale


def reference(ck: dict, x, w, ids, torch):
    """FP32 MoE on the dequantized checkpoint: sum_k w_k * (silu(g) * u) W2^T."""
    import torch.nn.functional as F

    y = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    xf = x.float()
    for e in ids.unique().tolist():
        rows, slots = (ids == e).nonzero(as_tuple=True)
        w13 = dequant(ck["w13_weight"][e], ck["w13_weight_scale"][e], ck["w13_weight_scale_2"][e, 0], torch)
        w2 = dequant(ck["w2_weight"][e], ck["w2_weight_scale"][e], ck["w2_weight_scale_2"][e], torch)
        gu = xf[rows] @ w13.T
        h = F.silu(gu[:, :INTER]) * gu[:, INTER:]
        y.index_add_(0, rows, (h @ w2.T) * w[rows, slots, None])
    return y


def errors(y, ref) -> dict:
    d = (y.float() - ref.float())
    return {"max_rel": float(d.abs().max() / ref.abs().max().clamp_min(1e-30)),
            "fro": float(d.norm() / ref.norm().clamp_min(1e-30))}


def vllm_config(moe_backend: str, args):
    from vllm.config import CompilationConfig, DeviceConfig, KernelConfig, VllmConfig
    from vllm.config.scheduler import SchedulerConfig

    return VllmConfig(
        kernel_config=KernelConfig(moe_backend=moe_backend),
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_num_batched_tokens),
        compilation_config=CompilationConfig(cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24]),
        device_config=DeviceConfig(device="cuda"),
    )


def build_backend(name: str, args, dev, torch) -> dict:
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    moe_backend, gemm_type = BACKENDS[name]
    if gemm_type is None:
        os.environ.pop("VLLM_HUMMING_MOE_GEMM_TYPE", None)
    else:
        os.environ["VLLM_HUMMING_MOE_GEMM_TYPE"] = gemm_type
    cfg = vllm_config(moe_backend, args)
    qc = ModelOptNvFp4Config.from_config(QUANT)
    t0, before = time.perf_counter(), torch.cuda.memory_allocated()
    runners, layers = [], []
    with set_current_vllm_config(cfg), torch.device(dev):
        for i in range(args.layers):
            runner = FusedMoEFactory(
                num_experts=E, top_k=TOPK, hidden_size=HIDDEN, intermediate_size=INTER,
                params_dtype=torch.bfloat16, quant_config=qc, tp_size=1, dp_size=1, pcp_size=1,
                prefix=f"{name}.layers.{i}.mlp.experts", activation="silu", renormalize=True)
            layer = runner.routed_experts
            for k, v in make_ckpt(i, dev, torch).items():
                getattr(layer, k).data.copy_(v)
            layer.quant_method.process_weights_after_loading(layer)
            runners.append(runner)
            layers.append(layer)
            torch.cuda.empty_cache()
    qm = layers[0].quant_method
    schema = getattr(layers[0], "input_schemas", {}).get("w13")
    return {
        "cfg": cfg, "runners": runners, "layers": layers,
        "ids_dtype": qm.topk_indices_dtype or torch.int32,
        "info": {"nvfp4_backend": qm.nvfp4_backend.value,
                 "experts_cls": type(qm.moe_kernel.fused_experts).__name__,
                 "humming_input_schema": None if schema is None else repr(schema),
                 "gemm_type_env": os.environ.get("VLLM_HUMMING_MOE_GEMM_TYPE"),
                 "build_s": time.perf_counter() - t0,
                 "weight_bytes_per_layer": (torch.cuda.memory_allocated() - before) // args.layers},
    }


def run_layers(be: dict, x, routes, outs) -> None:
    from vllm.forward_context import set_forward_context

    with set_forward_context(None, be["cfg"], num_tokens=x.shape[0]):
        for j, (layer, (w, ids)) in enumerate(zip(be["layers"], routes)):
            outs[j] = layer.forward_modular(x, w, ids)


def capture(fn, torch):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def elapsed_ms(step, iters: int, torch) -> float:
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        step()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / iters


def dram_ceiling(dev, torch) -> dict:
    """Today's DRAM ceilings on 1 GiB buffers: a device copy (read + write bytes
    counted) and a sum (read only, the MoE weight stream's access pattern)."""
    src = torch.empty(2**30, dtype=torch.uint8, device=dev)
    dst = torch.empty_like(src)
    f32 = src.view(torch.float32)
    f32.sum()
    dst.copy_(src)
    copy = [2 * 2**30 / elapsed_ms(lambda: dst.copy_(src), 10, torch) / 1e6 for _ in range(5)]
    read = [2**30 / elapsed_ms(lambda: f32.sum(), 10, torch) / 1e6 for _ in range(5)]
    del src, dst, f32
    a = torch.randn(4608, HIDDEN, device=dev, dtype=torch.bfloat16)
    b = torch.randn(HIDDEN, W13_N, device=dev, dtype=torch.bfloat16)
    a @ b
    mm = [2 * 4608 * HIDDEN * W13_N / elapsed_ms(lambda: a @ b, 20, torch) / 1e9 for _ in range(5)]
    del a, b
    torch.cuda.empty_cache()
    return {"median_gbs": statistics.median(copy), "runs_gbs": copy,
            "read_median_gbs": statistics.median(read), "read_runs_gbs": read,
            "bf16_gemm_tflops": statistics.median(mm), "bf16_gemm_runs_tflops": mm}


def bench(args) -> int:
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.v1.worker.workspace import init_workspace_manager

    watch = MemWatch()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    meta = {"argv": sys.argv, "torch": torch.__version__, "device": props.name,
            "capability": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
            "l2_bytes": getattr(props, "L2_cache_size", None),
            "humming_cache_dir": str(humming_cache()), "layers": args.layers,
            "expert_bytes": EXPERT_BYTES, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mem_available_kib_start": meminfo_kib("MemAvailable")}
    for pkg in ("vllm", "humming-kernels", "flashinfer-python"):
        try:
            from importlib.metadata import version
            meta[pkg] = version(pkg)
        except Exception as e:  # noqa: BLE001
            meta[pkg] = f"missing ({e})"
    print(json.dumps(meta), flush=True)

    with set_current_vllm_config(vllm_config("marlin", args)):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl",
                                     distributed_init_method=f"tcp://127.0.0.1:{args.port}")
        initialize_model_parallel(1, 1)
    init_workspace_manager(dev)

    watch.phase = "dram-ceiling"
    ceiling = dram_ceiling(dev, torch)
    print(f"DRAM ceiling: copy {ceiling['median_gbs']:.1f} GB/s (read+write), "
          f"sum {ceiling['read_median_gbs']:.1f} GB/s (read); dense BF16 GEMM 4608x4096x2048 "
          f"{ceiling['bf16_gemm_tflops']:.1f} TFLOP/s", flush=True)

    built, build_errors = {}, {}
    for name in args.backends:
        watch.phase = f"build {name}"
        c0 = len(cubins())
        try:
            built[name] = build_backend(name, args, dev, torch)
        except Exception as e:  # noqa: BLE001
            build_errors[name] = f"{type(e).__name__}: {e}"
            print(f"BUILD FAIL {name}: {build_errors[name]}", flush=True)
            continue
        built[name]["info"]["cubins_compiled_at_build"] = len(cubins()) - c0
        print(f"built {name}: {json.dumps(built[name]['info'])}", flush=True)
    names = list(built)
    after_build = meminfo_kib("MemAvailable")

    samples, summary, first_calls, errs = [], [], [], []
    gx = torch.Generator(device=dev)
    for case in args.m:
        m, pool = parse_case(case)
        gx.manual_seed(7 + m)
        x = torch.randn(m, HIDDEN, generator=gx, device=dev).to(torch.bfloat16)
        graphed = m <= args.graph_max_m
        iters = args.iters if graphed else args.prefill_iters
        steps, outs, eager0, routes_by = {}, {}, {}, {}
        for name in names:
            be = built[name]
            routes = [make_routing(m, pool, i, dev, torch, be["ids_dtype"]) for i in range(args.layers)]
            routes_by[name] = routes
            outs[name] = [None] * args.layers
            fn = functools.partial(run_layers, be, x, routes, outs[name])
            watch.phase = f"first-call {name} M={case}"
            c0, avail0, t0 = len(cubins()), meminfo_kib("MemAvailable"), time.perf_counter()
            try:
                fn()
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"RUN FAIL {name} M={case}: {type(e).__name__}: {e}", flush=True)
                errs.append({"case": case, "backend": name, "error": f"{type(e).__name__}: {e}"})
                continue
            first_s = time.perf_counter() - t0
            eager0[name] = outs[name][0].clone()
            fc = {"case": case, "backend": name, "first_call_s": first_s,
                  "cubins_compiled": len(cubins()) - c0,
                  "mem_available_kib_before": avail0,
                  "low_mem_available_kib": watch.low.get(watch.phase)}
            if graphed:
                watch.phase = f"capture {name} M={case}"
                g = capture(fn, torch)
                g.replay()
                torch.cuda.synchronize()
                fc["graph_vs_eager_max_abs"] = float((outs[name][0].float() - eager0[name].float()).abs().max())
                steps[name] = g.replay
            else:
                steps[name] = fn
            first_calls.append(fc)
            print(f"first call {json.dumps(fc)}", flush=True)
        order = [n for n in names if n in steps]
        watch.phase = f"timing M={case}"
        for rep in range(args.reps):
            for name in (order if rep % 2 == 0 else order[::-1]):
                ms = elapsed_ms(steps[name], iters, torch) / args.layers
                samples.append({"case": case, "M": m, "backend": name, "rep": rep, "ms": ms})
        ids_any = routes_by[order[0]] if order else []
        uniq = statistics.mean(int(ids.unique().numel()) for _, ids in ids_any) if ids_any else 0
        watch.phase = f"reference M={case}"
        ref = None
        if order:
            ck = make_ckpt(0, dev, torch)
            w0, ids0 = routes_by[order[0]][0]
            ref = reference(ck, x, w0, ids0.long(), torch)
            del ck
            torch.cuda.empty_cache()
        for name in order:
            ms = [s["ms"] for s in samples if s["case"] == case and s["backend"] == name]
            med = statistics.median(ms)
            row = {"case": case, "M": m, "pool": pool, "backend": name,
                   "mode": "graph" if graphed else "eager",
                   "iters": iters, "reps": len(ms), "median_ms": med, "min_ms": min(ms),
                   "max_ms": max(ms), "unique_experts": uniq,
                   "gbs": uniq * EXPERT_BYTES / med / 1e6,
                   "tflops": m * TOPK * FLOPS_PER_ROW_EXPERT / med / 1e9,
                   "vs_ref": errors(eager0[name], ref)}
            if name != "marlin" and "marlin" in eager0:
                row["vs_marlin"] = errors(eager0[name], eager0["marlin"])
            summary.append(row)
            print(f"M={case:<5} {name:16} {row['mode']:5} median {med:8.3f} ms  "
                  f"[{min(ms):.3f}, {max(ms):.3f}]  {row['gbs']:6.1f} GB/s  "
                  f"{row['tflops']:6.2f} TFLOP/s  uniq {uniq:6.1f}  "
                  f"vs ref max_rel {row['vs_ref']['max_rel']:.2e} fro {row['vs_ref']['fro']:.2e}"
                  + (f"  vs marlin fro {row['vs_marlin']['fro']:.2e}" if "vs_marlin" in row else ""),
                  flush=True)
        del steps, outs, eager0, ref
        torch.cuda.empty_cache()

    cubin_files = cubins()
    result = {
        "meta": meta, "dram_ceiling": ceiling,
        "backends": {n: b["info"] for n, b in built.items()}, "build_errors": build_errors,
        "run_errors": errs, "first_calls": first_calls, "samples": samples, "summary": summary,
        "mem": {**watch.report(), "mem_available_kib_after_build": after_build,
                "mem_available_kib_end": meminfo_kib("MemAvailable"),
                "torch_max_allocated_bytes": torch.cuda.max_memory_allocated()},
        "humming_cubins": {"count": len(cubin_files),
                           "bytes": sum(p.stat().st_size for p in cubin_files)},
    }
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))
    print(f"compilers seen: {json.dumps(result['mem']['compilers'])}", flush=True)
    print(f"humming cubins: {json.dumps(result['humming_cubins'])}", flush=True)
    return 0 if not build_errors and not errs else 1


def speedup(run: dict, r: dict) -> float | None:
    """Marlin median / this median - 1 within one run; positive is faster than Marlin."""
    base = next((s for s in run["summary"] if s["case"] == r["case"] and s["backend"] == "marlin"), None)
    return None if base is None or r["backend"] == "marlin" else base["median_ms"] / r["median_ms"] - 1


def table(paths: list[str]) -> int:
    runs = []
    for p in paths:
        run = json.loads(Path(p).read_text())
        run["label"] = Path(p).stem.removeprefix("run-")
        runs.append(run)
    groups: dict[tuple, list] = {}
    for run in runs:
        for r in run["summary"]:
            key = (r["M"], r["pool"] or 0, run["meta"]["humming-kernels"], list(BACKENDS).index(r["backend"]))
            groups.setdefault(key, []).append((r, run))
    pct = lambda v: f"{v:+.1%}"  # noqa: E731
    print(f"All runs, per layer per rank (one routed-expert call, TP=2 shard). median ms = median of the "
          f"runs' medians (each run: {runs[0]['meta']['layers']} rotating layers, 7 alternating repeats); "
          f"run range = min-max of the runs' medians.\nGB/s = touched-expert checkpoint bytes / median. "
          f"vs marlin = Marlin / this - 1 inside each run (positive is faster), median [min, max] over "
          f"runs. Errors are the worst over runs, layer 0 vs an FP32 reference on the dequantized "
          f"weights (max rel = max abs / max |ref|; fro = relative Frobenius) and vs Marlin's output.\n")
    print(f"{'M':>5} {'backend':16} {'humming':8} {'runs':>4} {'median ms':>10} {'run range ms':>17} "
          f"{'GB/s':>6} {'uniq':>5} {'vs marlin [min, max]':>24} {'max rel err':>11} {'fro err':>8} "
          f"{'fro vs marlin':>13}")
    for key in sorted(groups):
        items = groups[key]
        r0 = items[0][0]
        meds = [r["median_ms"] for r, _ in items]
        med = statistics.median(meds)
        sp = [s for s in (speedup(run, r) for r, run in items) if s is not None]
        sp_txt = f"{pct(statistics.median(sp))} [{pct(min(sp))}, {pct(max(sp))}]" if sp else ""
        vm = [r["vs_marlin"]["fro"] for r, _ in items if "vs_marlin" in r]
        print(f"{r0['case']:>5} {r0['backend']:16} {key[2]:8} {len(items):>4} {med:10.3f} "
              f"{min(meds):8.3f}-{max(meds):<8.3f} {r0['unique_experts'] * EXPERT_BYTES / med / 1e6:6.1f} "
              f"{r0['unique_experts']:5.1f} {sp_txt:>24} "
              f"{max(r['vs_ref']['max_rel'] for r, _ in items):11.2e} "
              f"{max(r['vs_ref']['fro'] for r, _ in items):8.2e} {f'{max(vm):.2e}' if vm else '':>13}")
    print(f"\nBoot rule (decode, M=8): Humming faster than Marlin by more than {BOOT_MARGIN:.0%} in the median "
          f"over runs and faster in every run, outputs agree (fro <= {MAX_FRO_ERR}) in every run, and JIT "
          f"memory safe (see the per-run lines). 8p24 and 4608 are reported the same way for context.")
    for key in sorted(k for k in groups if k[3] != 0 and (k[0], k[1]) in ((8, 0), (8, 24), (4608, 0))):
        items = groups[key]
        sp = [s for s in (speedup(run, r) for r, run in items) if s is not None]
        agree = all(max(r["vs_ref"]["fro"], r.get("vs_marlin", {}).get("fro", 1.0)) <= MAX_FRO_ERR
                    for r, _ in items)
        ok = statistics.median(sp) > BOOT_MARGIN and min(sp) > 0 and agree
        print(f"  M={items[0][0]['case']:<5} {items[0][0]['backend']:16} humming {key[2]}: median "
              f"{pct(statistics.median(sp))} over {len(sp)} runs [{pct(min(sp))}, {pct(max(sp))}]; outputs "
              f"agree: {agree} -> {'passes the speed and accuracy legs' if ok else 'fails'}")
    print("\nPer run:")
    for run in runs:
        hv, mem, ceil = run["meta"]["humming-kernels"], run["mem"], run["dram_ceiling"]
        mar8 = next((s["median_ms"] for s in run["summary"] if s["case"] == "8" and s["backend"] == "marlin"), 0)
        print(f"{run['label']} ({run['meta']['start']}): marlin M=8 {mar8:.3f} ms")
        fcs = [f for f in run["first_calls"] if f["backend"] != "marlin" and f["low_mem_available_kib"]]
        drop = max((f["mem_available_kib_before"] - f["low_mem_available_kib"] for f in fcs), default=0)
        jit_s = sum(f["first_call_s"] for f in run["first_calls"] if f["cubins_compiled"])
        nvrtc = mem["compilers"].get("nvrtc_compile", {})
        print(f"  DRAM ceiling copy {ceil['median_gbs']:.1f} GB/s (read+write), sum "
              f"{ceil['read_median_gbs']:.1f} GB/s (read), dense BF16 GEMM at M=4608 "
              f"{ceil['bf16_gemm_tflops']:.1f} TFLOP/s")
        print(f"  humming {hv}: {run['humming_cubins']['count']} cubins "
              f"({run['humming_cubins']['bytes'] / 2**20:.1f} MiB) in {jit_s:.1f} s of first calls "
              f"plus build; nvrtc_compile peak RSS {nvrtc.get('peak_rss_kib', 0) / 2**10:.0f} MiB per "
              f"process, at most {mem['compilers_concurrent']['max_procs']} compilers at once "
              f"({mem['compilers_concurrent']['max_rss_sum_kib'] / 2**10:.0f} MiB summed); largest "
              f"MemAvailable drop in a Humming first call {drop / 2**20:.2f} GiB; build/run errors "
              f"{len(run['build_errors']) + len(run['run_errors'])}")
    print(f"\nPer run detail:\n{'M':>5} {'backend':16} {'run':18} {'mode':5} {'median ms':>10} "
          f"{'rep range ms':>17} {'GB/s':>6} {'TFLOP/s':>7} {'vs marlin':>9} {'max rel err':>11} "
          f"{'fro err':>8} {'fro vs marlin':>13}")
    rows = sorted(((r["M"], r["pool"] or 0, list(BACKENDS).index(r["backend"]), run["label"], r, run)
                   for run in runs for r in run["summary"]), key=lambda t: t[:4])
    for *_, r, run in rows:
        s = speedup(run, r)
        vm = r.get("vs_marlin", {}).get("fro")
        print(f"{r['case']:>5} {r['backend']:16} {run['label']:18} {r['mode']:5} {r['median_ms']:10.3f} "
              f"{r['min_ms']:8.3f}-{r['max_ms']:<8.3f} {r['gbs']:6.1f} {r['tflops']:7.2f} "
              f"{'' if s is None else pct(s):>9} {r['vs_ref']['max_rel']:11.2e} {r['vs_ref']['fro']:8.2e} "
              f"{'' if vm is None else f'{vm:.2e}':>13}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=lambda s: s.split(","), default="1,2,4,8,8p24,16,4608".split(","),
                   help="cases: rows per call, 'NpK' draws top-8 from a K-expert pool per layer")
    p.add_argument("--backends", type=lambda s: s.split(","), default=list(BACKENDS))
    p.add_argument("--layers", type=int, default=5, help="distinct layers per timed call (beats L2)")
    p.add_argument("--reps", type=int, default=7, help="timed repeats per backend, alternating order")
    p.add_argument("--iters", type=int, default=50, help="graph replays per timed repeat")
    p.add_argument("--prefill-iters", type=int, default=4, help="eager calls per timed repeat")
    p.add_argument("--graph-max-m", type=int, default=24, help="largest M timed as a CUDA graph")
    p.add_argument("--max-num-batched-tokens", type=int, default=4624, help="the serve's value")
    p.add_argument("--port", type=int, default=29517)
    p.add_argument("--json", help="write meta, samples, summary, JIT and memory here")
    p.add_argument("--table", nargs="+", metavar="JSON", help="CPU only: merge runs into the table")
    args = p.parse_args()
    if args.table:
        return table(args.table)
    bad = set(args.backends) - set(BACKENDS)
    if bad:
        p.error(f"unknown backends {sorted(bad)}")
    return bench(args)


if __name__ == "__main__":
    sys.exit(main())
