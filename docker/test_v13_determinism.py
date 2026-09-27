"""CPU tests for docker/patch_v13_determinism.py (no GPU, no docker).

    GLM53_V11_SRC=/path/to/v11src python3 docker/test_v13_determinism.py

GLM53_V11_SRC is the directory holding the vllm/ package tree shipped in
glm53-sm121-v11 (read-only; the tests patch a temporary copy). Without it the
source tests skip. The lm_head test needs torch; the kernel test needs torch
and triton and runs the Triton CPU interpreter (TRITON_INTERPRET=1).
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
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("GLM53_V11_SRC", "/nonexistent")) / "vllm"
HAVE_SRC = (SRC / "__init__.py").is_file()
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_TORCH_TRITON = HAVE_TORCH and importlib.util.find_spec("triton") is not None
ENV = "GLM53_DETERMINISTIC_MLA_INDEX"
SPARSE_UTILS = "v1/attention/backends/mla/sparse_utils.py"
INDEXER = "model_executor/layers/sparse_attn_indexer_kpool.py"
LOGITS = "model_executor/layers/logits_processor.py"

spec = importlib.util.spec_from_file_location("patch_v13_determinism", HERE / "patch_v13_determinism.py")
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


def tree_digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))}


def patched_text(rel: str) -> str:
    src, applied = patch.plan_edits((SRC / rel).read_text(), patch.FILE_EDITS[rel])
    assert applied == len(patch.FILE_EDITS[rel]), rel
    return src


@unittest.skipUnless(HAVE_SRC, "set GLM53_V11_SRC to the dir holding the v11 vllm/")
class ApplyTests(unittest.TestCase):
    """The patch reads vllm/__init__.py and the files it edits, so copies of
    those are the whole tree it can touch."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v13det-"))
        self.root = self.tmp / "vllm"
        for rel in ("__init__.py", *patch.FILE_EDITS):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / rel, self.root / rel)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run([sys.executable, str(HERE / "patch_v13_determinism.py"), str(self.root)],
                              capture_output=True, text=True, check=False)

    def test_every_anchor_occurs_once_in_v11(self):
        for rel, edits in patch.FILE_EDITS.items():
            src = (SRC / rel).read_text()
            for what, old, new in edits:
                self.assertEqual(src.count(old), 1, what)
                self.assertEqual(src.count(new), 0, what)

    def test_apply_twice_is_idempotent(self):
        before = tree_digest(self.root)
        first = self.run_patch()
        self.assertEqual(first.returncode, 0, first.stderr)
        after = tree_digest(self.root)
        self.assertEqual({k for k in after if before.get(k) != after[k]}, set(patch.FILE_EDITS))
        second = self.run_patch()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("(0 files written)", second.stdout)
        self.assertEqual(tree_digest(self.root), after)

    def test_refuses_on_drift_and_writes_nothing(self):
        path = self.root / SPARSE_UTILS
        path.write_text(path.read_text().replace("tok = tl.load(ti_ptr)  # int32", "tok = tl.load(ti_ptr)"))
        before = tree_digest(self.root)
        res = self.run_patch()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("refusing", res.stderr)
        self.assertEqual(tree_digest(self.root), before, "no partial writes")

    def test_touched_files_compile(self):
        self.assertEqual(self.run_patch().returncode, 0)
        for rel in patch.FILE_EDITS:
            py_compile.compile(str(self.root / rel), cfile=str(self.tmp / "pyc" / (rel.replace("/", "_") + "c")),
                               doraise=True)

    def test_switch_is_read_from_env_only(self):
        self.assertEqual(self.run_patch().returncode, 0)
        for rel in (SPARSE_UTILS, INDEXER):
            text = (self.root / rel).read_text()
            self.assertEqual(text.count(f'os.environ.get("{ENV}") == "1"'), 1, rel)


@unittest.skipUnless(HAVE_SRC, "set GLM53_V11_SRC to the dir holding the v11 vllm/")
class PoolSortTests(unittest.TestCase):
    """_glm53_canonical_pools, exec'd from the patched indexer with a numpy
    stand-in for torch.sort."""

    def helper(self, on: bool):
        tree = ast.parse(patched_text(INDEXER))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_glm53_canonical_pools")
        torch = types.SimpleNamespace(
            Tensor=np.ndarray, sort=lambda t, dim: types.SimpleNamespace(values=np.sort(t, axis=dim)))
        env = {"torch": torch, "_GLM53_DET_MLA_INDEX": on}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), INDEXER, "exec"), env)  # noqa: S102
        return env["_glm53_canonical_pools"]

    def test_off_returns_the_input(self):
        pools = np.array([[7, -1, 3]], np.int32)
        self.assertIs(self.helper(False)(pools), pools)

    def test_on_sorts_each_row_and_keeps_the_set(self):
        pools = np.array([[9, 2, -1, 5], [4, 1, 8, 3], [-1, -1, 6, 0]], np.int32)
        out = self.helper(True)(pools)
        np.testing.assert_array_equal(out, [[-1, 2, 5, 9], [1, 3, 4, 8], [-1, -1, 0, 6]])
        for a, b in zip(out, pools):
            self.assertEqual(sorted(a), sorted(b))
        # Any permutation of a row's pools gives the same result.
        rng = np.random.default_rng(0)
        np.testing.assert_array_equal(self.helper(True)(rng.permuted(pools, axis=1)), out)


@unittest.skipUnless(HAVE_SRC and HAVE_TORCH, "needs GLM53_V11_SRC and torch")
class HeadDtypeTests(unittest.TestCase):
    """LogitsProcessor._apply_head, exec'd from v11 and from the patched file,
    with head_dtype float32 on a ModelOpt-style lm_head (LOGITS_FP32=1)."""

    class Embedding:  # UnquantizedEmbeddingMethod
        pass

    class Linear:  # UnquantizedLinearMethod
        pass

    class Marlin:  # a quantized method (e.g. GLM53_FP8 lm_head)
        pass

    def apply_head(self, text):
        import torch
        import torch.nn.functional as F

        cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "LogitsProcessor")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_head")
        env = {"torch": torch, "F": F, "UnquantizedEmbeddingMethod": self.Embedding, "VocabParallelEmbedding": object,
               "current_platform": types.SimpleNamespace(is_cuda=lambda: False, is_rocm=lambda: False)}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "logits_processor", "exec"), env)  # noqa: S102
        return env["_apply_head"]

    def call(self, text, method, head_dtype):
        """_apply_head(self, lm_head, hidden, None) and the fp32 reference."""
        import torch
        import torch.nn.functional as F

        linear = types.ModuleType("vllm.model_executor.layers.linear")
        linear.UnquantizedLinearMethod = self.Linear
        stubs = {n: types.ModuleType(n) for n in ("vllm", "vllm.model_executor", "vllm.model_executor.layers")}
        stubs[linear.__name__] = linear
        gen = torch.Generator().manual_seed(0)
        hidden = torch.randn(3, 8, generator=gen).to(torch.bfloat16)
        weight = torch.randn(5, 8, generator=gen).to(torch.bfloat16)
        method.apply = lambda layer, x, bias=None: F.linear(x, layer.weight)
        lm_head = types.SimpleNamespace(quant_method=method, weight=weight)
        with mock.patch.dict(sys.modules, stubs):
            out = self.apply_head(text)(types.SimpleNamespace(head_dtype=head_dtype), lm_head, hidden, None)
        return out, F.linear(hidden.float(), weight.float())

    def test_v11_rejects_the_modelopt_lm_head(self):
        import torch

        with self.assertRaisesRegex(ValueError, "unquantized lm_head"):
            self.call((SRC / LOGITS).read_text(), self.Linear(), torch.float32)

    def test_patched_accepts_it_and_returns_fp32(self):
        import torch

        text = patched_text(LOGITS)
        for method in (self.Linear(), self.Embedding()):
            out, want = self.call(text, method, torch.float32)
            self.assertEqual(out.dtype, torch.float32)
            torch.testing.assert_close(out, want)
        with self.assertRaisesRegex(ValueError, "unquantized lm_head"):
            self.call(text, self.Marlin(), torch.float32)

    def test_default_head_dtype_takes_the_v11_path(self):
        import torch

        for text in ((SRC / LOGITS).read_text(), patched_text(LOGITS)):
            out, _ = self.call(text, self.Marlin(), torch.bfloat16)
            self.assertEqual(out.dtype, torch.bfloat16)


class PatchStaticTests(unittest.TestCase):
    def test_other_v13_patches_leave_these_files_alone(self):
        for name in ("patch_v13_misc.py", "patch_v13_fp8.py", "patch_v13_census.py"):
            text = (HERE / name).read_text()
            for rel in patch.FILE_EDITS:
                self.assertNotIn(rel, text, name)

    def test_dockerfile_runs_the_patch_before_compileall(self):
        text = (HERE / "Dockerfile.sm121-v13").read_text()
        self.assertIn('python3 /tmp/patch_v13_determinism.py "$VLLM_ROOT"', text)
        self.assertLess(text.index("v13.stamp"), text.index("patch_v13_determinism.py"))
        self.assertLess(text.index("patch_v13_determinism.py"), text.index("compileall"))


KERNEL_SCRIPT = textwrap.dedent(
    r"""
    import importlib.util, sys, types
    import torch, triton, triton.language as tl

    tu = types.ModuleType("vllm.triton_utils")
    tu.tl, tu.triton = tl, triton
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm.triton_utils"] = tu

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    orig, new = load("su_v11", sys.argv[1]), load("su_v13", sys.argv[2])

    class GridSpy:
        # Records each launch grid: the interpreter runs tiles in order, so
        # only the grid shows whether a row still has racing tiles.
        def __init__(self, kernel):
            self.kernel, self.grids = kernel, []

        def __getitem__(self, grid):
            self.grids.append(grid)
            return self.kernel[grid]

    spy = new._convert_req_index_to_global_index_kernel = GridSpy(new._convert_req_index_to_global_index_kernel)
    gen = torch.Generator().manual_seed(0)
    BS, NB, NREQ = 64, 40, 3  # block size, blocks per request, requests

    def rand(n, hi):
        return torch.randint(0, hi, (n,), generator=gen, dtype=torch.int32)

    def case(width, rows, used):
        # Rows like the indexer's: `used` distinct positions in shuffled order
        # with interior -1 holes, then the -1 tail up to the buffer width.
        tok = torch.full((rows, width), -1, dtype=torch.int32)
        for r in range(rows):
            seq = int(torch.randint(used, NB * BS, (1,), generator=gen))
            tok[r, :used] = torch.randperm(seq, generator=gen)[:used].to(torch.int32)
            tok[r, torch.randperm(used, generator=gen)[: used // 10]] = -1
        bt = torch.stack([torch.randperm(NREQ * NB, generator=gen)[:NB] for _ in range(NREQ)]).to(torch.int32)
        return rand(rows, NREQ), bt, tok

    def reference(req, bt, tok, ws_req=None, ws_start=None):
        # Stable compaction: the valid slots of each row in column order.
        out = torch.full_like(tok, -1)
        cnt = torch.zeros(tok.shape[0], dtype=torch.int32)
        for r in range(tok.shape[0]):
            t = tok[r][tok[r] >= 0]
            if ws_req is not None and ws_req[r] >= 0:
                vals = ws_start[ws_req[r]] + t
            else:
                vals = bt[req[r], t // BS] * BS + t % BS
            out[r, : len(vals)] = vals
            cnt[r] = len(vals)
        return out, cnt

    def convert(mod, on, *args, **kw):
        mod._GLM53_DET_MLA_INDEX = on
        return mod.triton_convert_req_index_to_global_index(*args, BLOCK_SIZE=BS, **kw)

    def same(a, b):
        a, b = (a,) if torch.is_tensor(a) else a, (b,) if torch.is_tensor(b) else b
        return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))

    for width, used in ((2176, 2051), (2048, 2048)):
        req, bt, tok = case(width, rows=6, used=used)
        kw = dict(NUM_TOPK_TOKENS=width, return_valid_counts=True)
        v11 = convert(orig, False, req, bt, tok, **kw)
        off = convert(new, False, req, bt, tok, **kw)
        on = convert(new, True, req, bt, tok, **kw)
        assert spy.grids[-2:] == [(6, 1 if width == 2048 else 17), (6, 1)], spy.grids
        assert same(off, v11), f"{width}: off differs from v11"
        assert same(on, reference(req, bt, tok)), f"{width}: on is not the column-order compaction"
        # The interpreter runs v11's 17 tiles in order, so v11 lands on the
        # same order; on the GPU the tiles race.
        assert same(on, v11), f"{width}: on differs from in-order v11"
        # Without valid counts there is no compaction and nothing changes.
        plain = dict(NUM_TOPK_TOKENS=width)
        assert same(convert(new, True, req, bt, tok, **plain), convert(orig, False, req, bt, tok, **plain))

    # Prefill-workspace rows (FlashMLA sparse path) through the padded lanes.
    req, bt, tok = case(2176, rows=4, used=2051)
    ws_req = torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
    ws_start = torch.tensor([0, 5000], dtype=torch.int32)
    kw = dict(NUM_TOPK_TOKENS=2176, return_valid_counts=True, HAS_PREFILL_WORKSPACE=True,
              prefill_workspace_request_ids=ws_req, prefill_workspace_starts=ws_start)
    on = convert(new, True, req, bt, tok, **kw)
    assert same(on, reference(req, bt, tok, ws_req, ws_start)), "prefill workspace rows"
    assert same(convert(new, False, req, bt, tok, **kw), convert(orig, False, req, bt, tok, **kw))
    print("sparse index kernel: off == v11; on == column-order compaction (2176 padded to 4096, 2048)")
    """
)


@unittest.skipUnless(HAVE_SRC and HAVE_TORCH_TRITON, "needs GLM53_V11_SRC, torch, triton")
class KernelInterpreterTest(unittest.TestCase):
    def test_padded_single_tile_matches_stable_compaction(self):
        with tempfile.TemporaryDirectory(prefix="v13det-") as tmp:
            new = Path(tmp) / "sparse_utils_v13.py"
            new.write_text(patched_text(SPARSE_UTILS))
            res = subprocess.run([sys.executable, "-c", KERNEL_SCRIPT, str(SRC / SPARSE_UTILS), str(new)],
                                 capture_output=True, text=True, check=False,
                                 env={**os.environ, "TRITON_INTERPRET": "1"})
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            print("\n  " + res.stdout.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
