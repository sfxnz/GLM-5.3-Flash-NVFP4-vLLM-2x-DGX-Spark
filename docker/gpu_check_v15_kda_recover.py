"""GPU check for GLM53_KDA_RECOVER (v15): bitwise equality, graph capture, timing.

    docker run --rm --gpus all --entrypoint python3 -v "$PWD":/work -w /work \\
        glm53-sm121-v15 docker/gpu_check_v15_kda_recover.py [--layers 34] [--iters 50]
"""

import argparse
import importlib
import importlib.util
import random
import statistics
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.third_party.flash_linear_attention.ops import kda as fla_kda

H, K, Q, LB, ALIGN = 32, 128, 8, -5.0, 1152
P = H * K
WIDTH = 3 * P + H + 2 * K
BF = torch.bfloat16
DEV = "cuda"


def load_module():
    try:
        return importlib.import_module("vllm.models.glm5next.nvidia.ops.glm53_kda_recover")
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location(
        "patch_v15_kda_recover", Path(__file__).resolve().parent / "patch_v15_kda_recover.py"
    )
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    mod = types.ModuleType("glm53_kda_recover")
    exec(compile(patch.MODULE, "glm53_kda_recover.py", "exec"), mod.__dict__)
    print("module: from patch_v15_kda_recover.py (image has no v15 patch)")
    return mod


R = load_module()


def make_layer(blocks):
    return SimpleNamespace(
        A_log=(torch.rand(1, 1, H, 1, device=DEV) * 2.0 - 1.0).float(),
        dt_bias=(torch.randn(P, device=DEV) * 0.5).float(),
        kda_lower_bound=LB,
        local_num_heads=H,
        head_dim=K,
        kv_cache=(None, torch.randn(blocks, H, K, K, device=DEV) * 0.05),
    )


def make_inputs(t):
    projected = torch.randn(t, WIDTH, device=DEV).to(BF)
    qkv = projected[:, : 3 * P]
    beta = projected[:, 3 * P : 3 * P + H].unsqueeze(0)
    g = (torch.randn(1, t, H, K, device=DEV) * 2.0).to(BF)
    return qkv, g, beta


def verify(layer, pool, qkv, g, beta, cu, idx, nacc, out):
    q, k, v = qkv.split(P, dim=-1)

    def rearr(x):
        return x.reshape(1, -1, H, K)

    fla_kda.fused_recurrent_kda(
        q=rearr(q),
        k=rearr(k),
        v=rearr(v),
        g=g,
        beta=beta,
        initial_state=pool,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        ssm_state_indices=idx,
        num_accepted_tokens=nacc,
        out=out,
        sigmoid_beta=True,
        a_log=layer.A_log,
        g_bias=layer.dt_bias,
        compute_gate=True,
        lower_bound=layer.kda_lower_bound,
    )


def same(a, b):
    view = torch.int32 if a.dtype == torch.float32 else torch.int16
    return a.shape == b.shape and torch.equal(a.view(view), b.view(view))


class Pair:
    def __init__(self, nreq, blocks, align):
        self.layer = make_layer(blocks)
        self.a = self.layer.kv_cache[1]
        self.b = self.a.clone()
        self.rs = R.Recover([self.layer], Q, nreq, nreq * Q, align, BF, DEV)
        self.nreq = nreq

    def step(self, S, nacc):
        t = self.nreq * Q
        qkv, g, beta = make_inputs(t)
        cu = torch.arange(0, t + 1, Q, dtype=torch.int32, device=DEV)
        out_a = torch.zeros(1, t, H, K, dtype=BF, device=DEV)
        out_b = torch.zeros_like(out_a)
        verify(self.layer, self.a, qkv, g, beta, cu, S, nacc, out_a)
        R.store_records(self.layer.glm53_rs_records, qkv, g, beta, P)
        fields = self.rs.prepare(S, nacc, None)
        verify(
            self.layer, self.b, qkv, g, beta, cu,
            fields["glm53_rs_verify_idx"], fields["glm53_rs_nacc"], out_b,
        )
        return same(out_a, out_b), SimpleNamespace(
            num_spec_decodes=self.nreq,
            spec_state_indices_tensor=S,
            spec_query_start_loc=cu,
            **fields,
        )

    def commit(self, md, n, num_computed):
        self.layer.kv_cache = (None, self.b)
        idx_mapping = torch.arange(self.nreq, dtype=torch.int32, device=DEV)
        R.commit_step({"x": md}, idx_mapping, n, num_computed)


def chain_none(rng, nreq, steps):
    pair = Pair(nreq, nreq * Q + 2, 0)
    S = torch.tensor(rng.sample(range(1, nreq * Q + 2), nreq * Q), dtype=torch.int32, device=DEV).view(nreq, Q)
    nacc = torch.ones(nreq, dtype=torch.int32, device=DEV)
    ok = True
    for _ in range(steps):
        out_ok, md = pair.step(S, nacc.clone())
        n = torch.tensor([rng.randint(1, Q) for _ in range(nreq)], dtype=torch.int32, device=DEV)
        pair.commit(md, n, None)
        ok &= out_ok
        for r in range(nreq):
            blk = int(S[r, int(n[r]) - 1])
            ok &= same(pair.a[blk], pair.b[blk])
        nacc = n
    return ok


def chain_align(rng, nreq, steps):
    width = steps * Q // ALIGN + 3 + Q
    blocks = nreq * width + 1
    pair = Pair(nreq, blocks, ALIGN)
    flat = rng.sample(range(1, blocks), nreq * width)
    bt = [flat[r * width : (r + 1) * width] for r in range(nreq)]
    computed = [max(1, ALIGN - rng.randint(1, 3 * Q)) for _ in range(nreq)]
    state_idx = [(c - 1) // ALIGN for c in computed]
    nacc = [1] * nreq
    cached = set()
    ok = True
    for _ in range(steps):
        for r in range(nreq):
            new = (computed[r] + Q - 1) // ALIGN
            if state_idx[r] != new:
                src = bt[r][state_idx[r] + nacc[r] - 1]
                for pool in (pair.a, pair.b):
                    pool[bt[r][new]] = pool[src]
                nacc[r] = 1
            state_idx[r] = new
        S = torch.tensor([bt[r][state_idx[r] : state_idx[r] + Q] for r in range(nreq)], dtype=torch.int32, device=DEV)
        out_ok, md = pair.step(S, torch.tensor(nacc, dtype=torch.int32, device=DEV))
        ok &= out_ok
        n = [rng.randint(1, Q) for _ in range(nreq)]
        new_computed = [computed[r] + n[r] for r in range(nreq)]
        pair.commit(
            md,
            torch.tensor(n, dtype=torch.int32, device=DEV),
            torch.tensor(new_computed, dtype=torch.int32, device=DEV),
        )
        for r in range(nreq):
            nacc[r] = n[r]
            running = new_computed[r] - n[r] + 1
            aligned = new_computed[r] // ALIGN * ALIGN
            if aligned >= running:
                bias = aligned - running
                dest = aligned // ALIGN - 1
                if state_idx[r] == dest:
                    nacc[r] = 1
                if not (state_idx[r] == dest and bias == 0):
                    for pool in (pair.a, pair.b):
                        pool[bt[r][dest]] = pool[bt[r][state_idx[r] + bias]]
                cached.add(bt[r][dest])
            computed[r] = new_computed[r]
        for blk in cached:
            ok &= same(pair.a[blk], pair.b[blk])
        for r in range(nreq):
            blk = bt[r][state_idx[r] + nacc[r] - 1]
            ok &= same(pair.a[blk], pair.b[blk])
    return ok, len(cached)


def every_n(rng):
    ok = True
    for n in range(1, Q + 1):
        pair = Pair(1, Q + 2, 0)
        S = torch.tensor([rng.sample(range(1, Q + 2), Q)], dtype=torch.int32, device=DEV)
        nacc = torch.tensor([rng.randint(1, Q)], dtype=torch.int32, device=DEV)
        out_ok, md = pair.step(S, nacc)
        pair.commit(md, torch.tensor([n], dtype=torch.int32, device=DEV), None)
        ok &= out_ok and same(pair.a[int(S[0, n - 1])], pair.b[int(S[0, n - 1])])
    return ok


def graphs_match(rng, nreq):
    pair = Pair(nreq, nreq * Q + 2, 0)
    layer = pair.layer
    t = nreq * Q
    qkv, g, beta = make_inputs(t)
    S = torch.tensor(rng.sample(range(1, nreq * Q + 2), nreq * Q), dtype=torch.int32, device=DEV).view(nreq, Q)
    nacc = torch.tensor([rng.randint(1, Q) for _ in range(nreq)], dtype=torch.int32, device=DEV)
    n = torch.tensor([rng.randint(1, Q) for _ in range(nreq)], dtype=torch.int32, device=DEV)
    cu = torch.arange(0, t + 1, Q, dtype=torch.int32, device=DEV)
    fields = pair.rs.prepare(S, nacc, None)
    md = SimpleNamespace(num_spec_decodes=nreq, spec_state_indices_tensor=S, spec_query_start_loc=cu, **fields)
    idx_mapping = torch.arange(nreq, dtype=torch.int32, device=DEV)
    out = torch.zeros(1, t, H, K, dtype=BF, device=DEV)
    pool0 = pair.b.clone()

    def fwd():
        R.store_records(layer.glm53_rs_records, qkv, g, beta, P)
        verify(layer, pair.b, qkv, g, beta, cu, fields["glm53_rs_verify_idx"], fields["glm53_rs_nacc"], out)

    def commit():
        R.commit_step({"x": md}, idx_mapping, n, None)

    layer.kv_cache = (None, pair.b)
    fwd()
    commit()
    eager_out, eager_pool = out.clone(), pair.b.clone()
    pair.b.copy_(pool0)
    out.zero_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fwd()
        commit()
    torch.cuda.current_stream().wait_stream(stream)
    pair.b.copy_(pool0)
    g_fwd, g_commit = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_fwd):
        fwd()
    with torch.cuda.graph(g_commit):
        commit()
    pair.b.copy_(pool0)
    out.zero_()
    g_fwd.replay()
    g_commit.replay()
    torch.cuda.synchronize()
    return same(out, eager_out) and same(pair.b, eager_pool)


def time_step(layers_n, nreq, iters, rng):
    t = nreq * Q
    blocks = nreq * Q + 2
    layers = [make_layer(blocks) for _ in range(layers_n)]
    rs = R.Recover(layers, Q, nreq, t, 0, BF, DEV)
    qkv, g, beta = make_inputs(t)
    S = torch.tensor(rng.sample(range(1, blocks), nreq * Q), dtype=torch.int32, device=DEV).view(nreq, Q)
    nacc = torch.full((nreq,), 3, dtype=torch.int32, device=DEV)
    n = torch.full((nreq,), 3, dtype=torch.int32, device=DEV)
    cu = torch.arange(0, t + 1, Q, dtype=torch.int32, device=DEV)
    out = torch.zeros(1, t, H, K, dtype=BF, device=DEV)
    fields = rs.prepare(S, nacc, None)
    md = SimpleNamespace(num_spec_decodes=nreq, spec_state_indices_tensor=S, spec_query_start_loc=cu, **fields)
    idx_mapping = torch.arange(nreq, dtype=torch.int32, device=DEV)

    def stock():
        for layer in layers:
            verify(layer, layer.kv_cache[1], qkv, g, beta, cu, S, nacc, out)

    def recover_fwd():
        for layer in layers:
            R.store_records(layer.glm53_rs_records, qkv, g, beta, P)
            verify(layer, layer.kv_cache[1], qkv, g, beta, cu, fields["glm53_rs_verify_idx"], fields["glm53_rs_nacc"], out)

    def commit():
        R.commit_step({"x": md}, idx_mapping, n, None)

    graphs = {}
    for name, fn in (("stock", stock), ("recover_fwd", recover_fwd), ("commit", commit)):
        fn()
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            fn()

    def ms(fn):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return statistics.median(samples)

    res = {
        "stock (graph)": ms(graphs["stock"].replay),
        "recover verify+records (graph)": ms(graphs["recover_fwd"].replay),
        "recover commit (graph)": ms(graphs["commit"].replay),
        "recover commit (eager, as served)": ms(commit),
    }
    res["recover total (graph+graph)"] = res["recover verify+records (graph)"] + res["recover commit (graph)"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=34)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name()}")
    all_ok = True
    for trim in (False, True):
        fla_kda._GLM53_KDA_TRIM = trim
        tag = f"KDA_TRIM={int(trim)}"
        checks = {"every accepted count 0..7": every_n(rng)}
        for c in (1, 2):
            checks[f"chain none c={c}"] = chain_none(rng, c, args.steps)
            ok, cached = chain_align(rng, c, args.steps)
            checks[f"chain align c={c} ({cached} boundary blocks)"] = ok
            checks[f"graph replay == eager c={c}"] = graphs_match(rng, c)
        for name, ok in checks.items():
            print(f"[{tag}] {'OK  ' if ok else 'FAIL'} {name}")
            all_ok &= ok
    fla_kda._GLM53_KDA_TRIM = False
    state_mib = H * K * K * 4 / 2**20
    for c in (1, 2):
        res = time_step(args.layers, c, args.iters, rng)
        print(f"timing c={c}, {args.layers} layers, median of {args.iters} (ms per step):")
        for name, value in res.items():
            print(f"  {name:36s} {value:7.3f}")
        saved = res["stock (graph)"] - res["recover total (graph+graph)"]
        print(f"  saved per step (graph+graph)        {saved:7.3f}")
        print(
            f"  state bytes per layer and request: stock {(1 + Q) * state_mib:.0f} MiB, "
            f"recover {3 * state_mib:.0f} MiB (+{state_mib:.0f} MiB per block boundary)"
        )
    print("ALL BITWISE EQUAL" if all_ok else "MISMATCH")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
