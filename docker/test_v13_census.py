"""CPU tests for docker/patch_v13_census.py and tools/census_report.py.

    GLM53_V11_SRC=/path/to/v11/site-packages python3 docker/test_v13_census.py

GLM53_V11_SRC is the directory that contains the vllm/ package shipped in
glm53-sm121-v11 (read-only; the tests patch a temporary copy). Without it the
patch-apply tests skip. Nothing here needs torch, a GPU or docker: the census
recorder is exercised with a numpy stand-in for its device buffer.
"""

import ast
import hashlib
import importlib.util
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("GLM53_V11_SRC", "/nonexistent")) / "vllm"
HAVE_SRC = (SRC / "__init__.py").is_file()
RUNNER = "v1/worker/gpu/model_runner.py"
MODULE = "v1/worker/gpu/glm53_expert_census.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


patch = load("patch_v13_census", HERE / "patch_v13_census.py")
report = load("census_report", HERE.parent / "tools" / "census_report.py")


class FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args)

    info_once = warning_once = info


def census_namespace() -> dict:
    """Exec the census module's pure parts (no torch, no vllm imports)."""
    tree = ast.parse(patch.CENSUS_MODULE)
    keep = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    env = {"np": np, "os": os, "json": __import__("json"), "Path": Path,
           "atexit": types.SimpleNamespace(register=lambda f: f),
           "init_logger": lambda name: FakeLogger(),
           "torch": types.SimpleNamespace(Tensor=object)}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "census", "exec"), env)  # noqa: S102
    return env


class FakeTensor:
    """The slice of the torch API ExpertCensus.commit uses, over numpy."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def __getitem__(self, idx):
        return FakeTensor(self.a[idx])

    def index_select(self, dim, index):
        return FakeTensor(np.take(self.a, index.a, axis=dim))

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def tree_digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))}


@unittest.skipUnless(HAVE_SRC, "set GLM53_V11_SRC to the dir holding the v11 vllm/")
class ApplyTests(unittest.TestCase):
    """The patch reads only vllm/__init__.py and the V2 model runner, so a
    copy of those two files is the whole tree it can touch."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v13census-"))
        self.root = self.tmp / "vllm"
        for rel in ("__init__.py", RUNNER):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / rel, self.root / rel)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run([sys.executable, str(HERE / "patch_v13_census.py"), str(self.root)],
                              capture_output=True, text=True, check=False)

    def test_every_anchor_occurs_once_in_v11(self):
        src = (SRC / RUNNER).read_text()
        for what, old, new in patch.RUNNER_EDITS:
            self.assertEqual(src.count(old), 1, what)
            self.assertEqual(src.count(new), 0, what)
        self.assertFalse((SRC / MODULE).exists())

    def test_apply_twice_is_idempotent(self):
        before = tree_digest(self.root)
        first = self.run_patch()
        self.assertEqual(first.returncode, 0, first.stderr)
        after = tree_digest(self.root)
        self.assertEqual({k for k in after if before.get(k) != after[k]}, {RUNNER, MODULE})
        second = self.run_patch()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("(0 files written)", second.stdout)
        self.assertEqual(tree_digest(self.root), after)

    def test_refuses_on_drift_and_writes_nothing(self):
        path = self.root / RUNNER
        path.write_text(path.read_text().replace("get_offloader().post_init()", "pass"))
        before = tree_digest(self.root)
        res = self.run_patch()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("refusing", res.stderr)
        self.assertEqual(tree_digest(self.root), before, "no partial writes")

    def test_touched_files_compile(self):
        self.assertEqual(self.run_patch().returncode, 0)
        for rel in (RUNNER, MODULE):
            py_compile.compile(str(self.root / rel), cfile=str(self.tmp / "pyc" / (rel.replace("/", "_") + "c")),
                               doraise=True)

    def test_hooks_are_guarded(self):
        """Every runner hook sits behind `self._glm53_census is not None`."""
        self.assertEqual(self.run_patch().returncode, 0)
        text = (self.root / RUNNER).read_text()
        self.assertEqual(text.count("self._glm53_census."), 2)
        self.assertEqual(text.count("if not dummy_run and self._glm53_census is not None:"), 1)
        self.assertEqual(text.count("        if self._glm53_census is not None:\n"), 1)


class PatchStaticTests(unittest.TestCase):
    def test_other_v13_patches_leave_the_runner_alone(self):
        """Dockerfile order is misc, fp8, census: the anchors see v11 text."""
        for name in ("patch_v13_misc.py", "patch_v13_fp8.py"):
            self.assertNotIn(RUNNER, (HERE / name).read_text(), name)

    def test_dockerfile_runs_census_patch_before_compileall(self):
        text = (HERE / "Dockerfile.sm121-v13").read_text()
        self.assertIn('python3 /tmp/patch_v13_census.py "$VLLM_ROOT"', text)
        self.assertLess(text.index("patch_v13_fp8.py"), text.index("patch_v13_census.py"))
        self.assertLess(text.index("patch_v13_census.py"), text.index("compileall"))


class CensusModuleTests(unittest.TestCase):
    def setUp(self):
        self.env = census_namespace()
        self.tmp = Path(tempfile.mkdtemp(prefix="v13census-"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_off_when_unset(self):
        old = os.environ.pop("GLM53_EXPERT_CENSUS", None)
        try:
            # Returns before touching the model, the config or torch.
            self.assertIsNone(self.env["maybe_create_census"](None, 0, None, None))
        finally:
            if old is not None:
                os.environ["GLM53_EXPERT_CENSUS"] = old

    def test_env_int(self):
        f = self.env["_env_int"]
        name = "GLM53_EXPERT_CENSUS_STEPS"
        old = os.environ.pop(name, None)
        try:
            self.assertEqual(f(name, 2000, 1), 2000)
            os.environ[name] = "30"
            self.assertEqual(f(name, 2000, 1), 30)
            for bad in ("0", "-2", "x", "1.5"):
                os.environ[name] = bad
                with self.assertRaises(ValueError):
                    f(name, 2000, 1)
        finally:
            os.environ.pop(name, None)
            if old is not None:
                os.environ[name] = old

    def make_census(self, steps, skip, layers=(3, 5), top_k=2, max_tokens=16):
        census = object.__new__(self.env["ExpertCensus"])
        census.__dict__.update(
            out_dir=self.tmp, rank=0, steps=steps, skip=skip, layer_ids=list(layers),
            _layer_index=FakeTensor(np.array(layers)),
            buf=FakeTensor(np.zeros((max_tokens, max(layers) + 1, top_k), np.int32)),
            step=0, chunk=0, _staged=None, _uids={}, _topk=[], _seg=[])
        return census

    @staticmethod
    def batch(req_ids, qsl, ndraft, ncomputed, prefill):
        return types.SimpleNamespace(
            req_ids=req_ids, num_reqs=len(req_ids), num_tokens=int(qsl[-1]),
            query_start_loc_np=np.asarray(qsl, np.int32),
            num_draft_tokens_per_req=None if ndraft is None else np.asarray(ndraft, np.int32),
            num_computed_tokens_np=np.asarray(ncomputed, np.int32),
            is_prefilling_np=np.asarray(prefill, bool))

    def step(self, census, batch, fill, sampled):
        """One real step: the graph fills the buffer, then stage + commit."""
        a = census.buf.a
        a[:] = 0
        a[: batch.num_tokens] = np.broadcast_to(fill, a.shape)[: batch.num_tokens]
        census.stage(batch)
        census.commit(FakeTensor(np.asarray(sampled)))

    def test_window_segments_and_report_roundtrip(self):
        census = self.make_census(steps=3, skip=1)
        fill = np.arange(16 * 6 * 2).reshape(16, 6, 2) % 288
        # step 0: prefill chunk of "a" (skipped by SKIP=1)
        self.step(census, self.batch(["a"], [0, 5], None, [0], [1]), fill, [0])
        # steps 1-3: c=2 verify, "a" and "b", 1 anchor + 3 drafts each
        for i in range(3):
            self.step(census, self.batch(["a", "b"], [0, 4, 8], [3, 3], [5 + i, 9], [0, 0]),
                      fill, [2, 1])
        # step 4: past the window, not recorded
        self.step(census, self.batch(["a"], [0, 4], [3], [8], [0]), fill, [4])
        self.assertEqual(census.step, 4)
        files = sorted(self.tmp.glob("census-rank0-*.npz"))
        self.assertEqual([p.name for p in files], ["census-rank0-0000.npz"])
        with np.load(files[0]) as z:
            topk, seg = z["topk"], z["seg"]
        self.assertEqual(topk.shape, (24, 2, 2))
        self.assertEqual(topk.dtype, np.uint16)
        np.testing.assert_array_equal(topk[:8], fill[:8][:, [3, 5]])
        np.testing.assert_array_equal(seg[:2], [[1, 0, 0, 4, 3, 2, 5, 0], [1, 1, 4, 4, 3, 1, 9, 0]])
        self.assertEqual(census._uids, {"a": 0, "b": 1})  # ids start at the window

        (self.tmp / "census-rank0.json").write_text(
            '{"rank": 0, "layers": [3, 5], "top_k": 2, "num_experts": 288}')
        meta, t, s = report.load_census(self.tmp, 0)
        self.assertEqual(list(s[:, 2]), [0, 4, 8, 12, 16, 20])  # global row offsets
        rep = report.analyze(meta, t, s)
        self.assertEqual(rep["step_kinds"], {"verify": 3, "prefill": 0, "mixed": 0, "decode": 0})
        self.assertEqual(rep["verify_blocks"], 6)

    def test_host_layout_mismatch_skips_step(self):
        census = self.make_census(steps=5, skip=0)
        b = self.batch(["a"], [0, 8], [7], [3], [0])
        b.num_tokens = 6  # adaptive verification: host qsl is an upper bound
        self.step(census, b, 1, [3])
        self.assertEqual(census._seg, [])
        self.assertEqual(census.step, 1)


def synth(blocks: list[list[np.ndarray]], sampled: list[list[int]], layers: int = 1):
    """blocks[step][req] = rows [n, K] (same ids on every layer) -> meta, topk, seg."""
    topk, seg, row = [], [], 0
    for st, (reqs, samp) in enumerate(zip(blocks, sampled)):
        for r, (rows, ns) in enumerate(zip(reqs, samp)):
            rows = np.repeat(np.asarray(rows)[:, None, :], layers, axis=1)
            seg.append([st, r, row, len(rows), len(rows) - 1, ns, 100, 0])
            topk.append(rows)
            row += len(rows)
    meta = {"rank": 0, "top_k": 8, "num_experts": 288}
    return meta, np.concatenate(topk).astype(np.uint16), np.asarray(seg, np.int32)


class ReportMathTests(unittest.TestCase):
    A = np.arange(0, 8)
    B = np.arange(8, 16)
    C = np.arange(16, 24)

    def test_independent_distinct(self):
        d = report.independent_distinct
        self.assertAlmostEqual(d(1), 8.0)
        self.assertAlmostEqual(d(8), 58.11, places=2)
        self.assertAlmostEqual(d(6), 44.79, places=2)
        self.assertAlmostEqual(d(4), 30.69, places=2)

    def test_distinct_identical_and_disjoint(self):
        same = np.stack([self.A] * 5)[:, None, :]
        self.assertEqual(report.distinct_per_layer(same).tolist(), [8])
        disjoint = np.stack([self.A, self.B, self.C])[:, None, :]
        self.assertEqual(report.distinct_per_layer(disjoint).tolist(), [24])

    def test_random_routing_matches_independent_expectation(self):
        rng = np.random.default_rng(0)
        rows = np.stack([np.stack([rng.choice(288, 8, replace=False) for _ in range(8)])
                         for _ in range(2000)])  # [blocks, n, K]
        for n in (2, 4, 8):
            d = np.mean([report.distinct_per_layer(b[:n, None, :])[0] for b in rows])
            self.assertAlmostEqual(d / report.independent_distinct(n), 1.0, delta=0.01)

    def test_position_stats(self):
        rows = np.stack([self.A, self.A, np.r_[self.A[:4], self.B[:4]], self.C])[:, None, :]
        overlap, new = report.position_stats(rows)
        np.testing.assert_allclose(overlap, [1, 1, 0.5, 0])
        np.testing.assert_allclose(new, [8, 0, 4, 8])

    def test_prefix_dup_positions_and_bytes(self):
        # One verify step, c=1: anchor A, draft1 A, draft2 B (rejected): nsampled 2.
        meta, topk, seg = synth([[np.stack([self.A, self.A, self.B])]], [[2]], layers=42)
        rep = report.analyze(meta, topk, seg, gbps=250.0)
        pre = {r["n"]: r for r in rep["prefix"]}
        self.assertEqual(pre[1]["distinct"], 8)
        self.assertEqual(pre[2]["distinct"], 8)
        self.assertEqual(pre[2]["dup"], 0.5)
        self.assertEqual(pre[3]["distinct"], 16)
        self.assertAlmostEqual(pre[3]["dup"], 1 - 16 / 24, places=4)
        self.assertEqual([p["accept_rate"] for p in rep["positions"]], [None, 1.0, 0.0])
        s = rep["step"]
        self.assertEqual(s["distinct_per_layer"], 16)
        self.assertAlmostEqual(s["moe_gb_per_rank"], 16 * 42 * 7_077_900 / 1e9, places=4)
        self.assertAlmostEqual(s["moe_ms_at_gbps"], 16 * 42 * 7_077_900 / 250e9 * 1e3, places=3)
        # SD-1 oracle keeps rows 0-1 (A only): 8 experts/layer saved, no token lost.
        o = rep["sd1"]["oracle"]
        self.assertAlmostEqual(o["saved_gb"], 8 * 42 * 7_077_900 / 1e9, places=4)
        self.assertEqual(o["saved_frac_moe"], 0.5)
        cut = {c["keep_rows"]: c for c in rep["sd1"]["cut"]}
        self.assertEqual(cut[1]["tokens_kept_frac"], 0.5)  # 1 of 2 emitted tokens
        self.assertEqual(cut[2]["tokens_kept_frac"], 1.0)
        self.assertEqual(cut[3]["saved_gb"], 0.0)

    def test_c2_union_and_cross_request_sharing(self):
        # Two requests share anchor experts A; step union is A|B|C = 24, per-request sum 32.
        meta, topk, seg = synth([[np.stack([self.A, self.B]), np.stack([self.A, self.C])]], [[1, 2]])
        rep = report.analyze(meta, topk, seg)
        self.assertEqual(rep["step"]["distinct_per_layer"], 24)
        self.assertAlmostEqual(rep["step"]["cross_request_shared_gb"], 8 * 7_077_900 / 1e9, places=4)
        # Oracle: req0 keeps A, req1 keeps A,C -> 16 of 24 read, 8 saved.
        self.assertAlmostEqual(rep["sd1"]["oracle"]["saved_frac_moe"], 8 / 24, places=4)


@unittest.skipUnless(importlib.util.find_spec("ijson"), "tools/step_buckets.py needs ijson")
class StepBucketsTests(unittest.TestCase):
    def test_synthetic_trace(self):
        import gzip
        import json

        sb = load("step_buckets", HERE.parent / "tools" / "step_buckets.py")
        ev, corr = [], iter(range(1, 10**6))

        def launch(ts, kernels):  # one CUDA call (e.g. cudaGraphLaunch) -> kernels
            c = next(corr)
            ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": ts, "dur": 1,
                       "args": {"correlation": c}})
            for kts, dur, name in kernels:
                ev.append({"ph": "X", "cat": "kernel", "name": name, "ts": kts, "dur": dur,
                           "args": {"correlation": c, "grid": [4, 1, 1]}})

        for step in range(3):  # the third step is dropped as possibly cut
            t = step * 1000.0
            ev.append({"ph": "X", "cat": "user_annotation", "ts": t, "dur": 100,
                       "name": "execute_context_0(0)_generation_1(8)"})
            launch(t + 10, [(t + 20, 300, "marlin_moe_wna16::Marlin<x>"),
                            (t + 320, 100, "ncclDevKernel_AllReduce_Sum_bf16"),
                            (t + 420, 50, "fused_recurrent_kda_fwd_kernel"),
                            (t + 470, 30, "nvjet_tst_64x8_64x16_4x1_v_bz_TNT")])
            launch(t + 150, [(t + 520, 40, "cutlass_lm_head_gemm")])
            launch(t + 160, [(t + 560, 20, "rejection_greedy_sample_kernel")])
            launch(t + 170, [(t + 600, 200, "dflash_draft_graph_kernel")])
        path = Path(tempfile.mkdtemp(prefix="v13sb-")) / "t.pt.trace.json.gz"
        try:
            with gzip.open(path, "wt") as fh:
                json.dump({"traceEvents": ev}, fh)
            rep = sb.analyze(sb.extract(str(path)))
        finally:
            shutil.rmtree(path.parent)
        ms = {r["bucket"]: r["ms"] for r in rep["buckets"]}
        self.assertEqual(rep["steps"], 2)
        self.assertEqual(ms["routed_moe"], 0.3)
        self.assertEqual(ms["nccl"], 0.1)
        self.assertEqual(ms["kda"], 0.05)
        self.assertEqual(ms["bf16_gemm"], 0.03)
        self.assertEqual(ms["lm_head_logits"], 0.04)
        self.assertEqual(ms["sampler_rejection"], 0.02)
        self.assertEqual(ms["drafter"], 0.2)
        self.assertEqual(rep["wall_ms"], 1.0)  # first kernel to next step's first kernel
        self.assertEqual(rep["busy_ms"], 0.74)
        self.assertEqual(rep["idle_ms"], 0.26)
        self.assertEqual(rep["bucket_sum_ms"], 0.74)
        self.assertEqual(sb.module_guess(34.0), "kda")
        self.assertEqual(sb.module_guess(84.0), "shared_expert x2")
        self.assertEqual(sb.module_guess(33.5), "?")


if __name__ == "__main__":
    unittest.main()
