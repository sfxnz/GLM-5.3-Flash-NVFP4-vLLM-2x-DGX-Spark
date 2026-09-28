"""CPU tests for docker/patch_v13_misc.py (no GPU, no docker).

    GLM53_V11_SRC=/path/to/v11src python3 docker/test_v13_misc.py

GLM53_V11_SRC must point at the directory holding the vllm/ package tree
shipped in glm53-sm121-v11, the same variable test_v13_fp8.py reads. The old
GLM53_V11SRC (the vllm/ dir itself) still works when GLM53_V11_SRC is unset.
The tree is read-only; the tests patch temporary copies. Without it, the patch-apply
tests skip. Tests that need torch skip without it. The KDA kernel test also
needs triton and runs the Triton CPU interpreter (TRITON_INTERPRET=1).
"""

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
_V11_SRC = os.environ.get("GLM53_V11_SRC")
SRC = Path(_V11_SRC) / "vllm" if _V11_SRC else Path(os.environ.get("GLM53_V11SRC", "/nonexistent"))
HAVE_SRC = (SRC / "__init__.py").is_file()
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_TRITON = importlib.util.find_spec("triton") is not None
# The hub cache as huggingface_hub finds it: HF_HUB_CACHE, else $HF_HOME/hub, else
# ~/.cache/huggingface/hub (/home/sfxnz/... on the host, /root/... in the v11 container).
HF_HUB_CACHE = Path(
    os.environ.get("HF_HUB_CACHE")
    or Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface") / "hub"
)
NV_CKPT = (
    HF_HUB_CACHE / "models--nvidia--GLM-5.3-Flash-NVFP4"
    / "snapshots/09b04e5e74bca08ca8549fc736d4cdd8624bfde3"
)

spec = importlib.util.spec_from_file_location(
    "patch_v13_misc", HERE / "patch_v13_misc.py"
)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

ENV_VARS = (
    "GLM53_ROUTER_FP32",
    "GLM53_INDEXER_WS_FACTOR",
    "GLM53_MHC_WARMUP",
    "GLM53_KDA_TRIM",
    "GLM53_SKIP_MTP_WEIGHTS",
    "GLM53_DFLASH_PREFIX_CACHE_FIX",
)


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def extract(path: Path, names: set[str], env: dict) -> dict:
    """Exec only the named top-level functions/assignments of ``path``."""
    tree = ast.parse(path.read_text())
    keep = [
        n
        for n in tree.body
        if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
        or (
            isinstance(n, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(t, ast.Name) and t.id in names
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
            )
        )
    ]
    mod = ast.Module(body=keep, type_ignores=[])
    exec(compile(mod, str(path), "exec"), env)  # noqa: S102
    missing = {n for n in names if n not in env}
    assert not missing, f"{missing} not found in {path}"
    return env


class FakeLogger:
    def __init__(self):
        self.lines = []

    def info_once(self, msg, *args):
        self.lines.append(msg % args)

    info = info_once


@unittest.skipUnless(HAVE_SRC, "set GLM53_V11_SRC to the dir holding the v11 vllm/ tree")
class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v13misc-"))
        self.root = self.tmp / "vllm"
        shutil.copytree(SRC, self.root, ignore=shutil.ignore_patterns("__pycache__"))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self):
        return subprocess.run(
            [sys.executable, str(HERE / "patch_v13_misc.py"), str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_apply_twice_is_idempotent(self):
        before = tree_digest(self.root)
        first = self.run_patch()
        self.assertEqual(first.returncode, 0, first.stderr)
        after_first = tree_digest(self.root)
        changed = {k for k in after_first if before.get(k) != after_first[k]}
        self.assertEqual(
            changed,
            set(patch.FILE_EDITS) | set(patch.NEW_FILES),
            "exactly the declared files change",
        )
        second = self.run_patch()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("(0 files written)", second.stdout)
        self.assertEqual(tree_digest(self.root), after_first)

    def test_every_edit_anchor_occurs_once_in_v11(self):
        for rel, edits in patch.FILE_EDITS.items():
            src = (SRC / rel).read_text()
            for what, old, new in edits:
                self.assertEqual(src.count(old), 1, what)
                self.assertEqual(src.count(new), 0, what)

    def test_refuses_on_drift_and_writes_nothing(self):
        target = self.root / "v1/core/kv_cache_coordinator.py"
        target.write_text(
            target.read_text().replace(
                "eagle_verified.clear()", "eagle_verified = set()"
            )
        )
        before = tree_digest(self.root)
        res = self.run_patch()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("refusing", res.stderr)
        self.assertEqual(tree_digest(self.root), before, "no partial writes")

    def test_touched_files_compile(self):
        self.assertEqual(self.run_patch().returncode, 0)
        import py_compile

        for rel in list(patch.FILE_EDITS) + list(patch.NEW_FILES):
            py_compile.compile(
                str(self.root / rel),
                cfile=str(self.tmp / "pyc" / (rel.replace("/", "_") + "c")),
                doraise=True,
            )

    def test_behaviour_is_env_gated(self):
        """Each GLM53_* var is read, and only through os.environ lookups."""
        self.assertEqual(self.run_patch().returncode, 0)
        text = "\n".join(
            (self.root / rel).read_text()
            for rel in list(patch.FILE_EDITS) + list(patch.NEW_FILES)
        )
        for var in ENV_VARS:
            self.assertIn(f'os.environ.get("{var}"', text, var)

    # --- pure logic, extracted from the patched sources ---------------------

    def patched(self, rel: str) -> Path:
        self.assertEqual(self.run_patch().returncode, 0)
        return self.root / rel

    def test_indexer_ws_factor(self):
        path = self.patched("v1/attention/backends/mla/indexer.py")
        log = FakeLogger()
        env = extract(
            path,
            {"get_max_prefill_buffer_size"},
            {"os": os, "logger": log, "VllmConfig": object},
        )
        fn = env["get_max_prefill_buffer_size"]
        cfg = types.SimpleNamespace(
            model_config=types.SimpleNamespace(max_model_len=327680)
        )
        old = os.environ.pop("GLM53_INDEXER_WS_FACTOR", None)
        try:
            self.assertEqual(fn(cfg), 327680 * 40)  # unset: stock
            self.assertEqual(log.lines, [])
            os.environ["GLM53_INDEXER_WS_FACTOR"] = "1"
            self.assertEqual(fn(cfg), 327680)
            self.assertIn("GLM53_INDEXER_WS_FACTOR=1", log.lines[0])
            # 327680 entries x 132 B: 1.61 GiB at 40, 41 MiB at 1.
            self.assertAlmostEqual(327680 * 40 * 132 / 2**30, 1.611, places=3)
            for bad in ("0", "-1", "abc", "1.5"):
                os.environ["GLM53_INDEXER_WS_FACTOR"] = bad
                with self.assertRaises(ValueError):
                    fn(cfg)
        finally:
            os.environ.pop("GLM53_INDEXER_WS_FACTOR", None)
            if old is not None:
                os.environ["GLM53_INDEXER_WS_FACTOR"] = old

    def load_ep_filter(self):
        path = self.patched("model_executor/model_loader/ep_weight_filter.py")
        src = path.read_text()
        if importlib.util.find_spec("regex") is None:
            src = src.replace("import regex as re", "import re")
        mod = types.ModuleType("ep_weight_filter")
        exec(compile(src, str(path), "exec"), mod.__dict__)  # noqa: S102
        return mod

    def test_skip_prefix_filter(self):
        f = self.load_ep_filter()
        name = "model.language_model.layers.45.mlp.experts.3.down_proj.weight"
        self.assertEqual(f.SKIP_NAME_PREFIXES, ())
        self.assertFalse(f.should_skip_weight(name, None))  # default: v11
        f.SKIP_NAME_PREFIXES = ("model.language_model.layers.45.",)
        self.assertTrue(f.should_skip_weight(name, None))
        self.assertFalse(
            f.should_skip_weight(
                "model.language_model.layers.44.self_attn.o_proj.weight", None
            )
        )
        self.assertFalse(
            f.should_skip_weight("model.language_model.layers.4.x.weight", None)
        )
        # EP expert filtering is unchanged.
        f.SKIP_NAME_PREFIXES = ()
        self.assertTrue(
            f.should_skip_weight("m.layers.1.mlp.experts.7.w1.weight", {1, 2})
        )
        self.assertFalse(
            f.should_skip_weight("m.layers.1.mlp.experts.7.w1.weight", {7})
        )

    @unittest.skipUnless(
        (NV_CKPT / "model.safetensors.index.json").is_file(), "no nvidia ckpt"
    )
    def test_skip_prefix_matches_exactly_the_nvidia_mtp_layer(self):
        f = self.load_ep_filter()
        cfg = json.loads((NV_CKPT / "config.json").read_text())["text_config"]
        n_layers, n_mtp = cfg["num_hidden_layers"], cfg["num_nextn_predict_layers"]
        # Same tuple the patched Glm5NextModel.__init__ builds.
        f.SKIP_NAME_PREFIXES = tuple(
            f"{root}layers.{n_layers + i}."
            for i in range(n_mtp)
            for root in ("model.language_model.", "model.", "")
        )
        wmap = json.loads((NV_CKPT / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        skipped = {k for k in wmap if f.should_skip_weight(k, None)}
        self.assertEqual(skipped, {k for k in wmap if ".layers.45." in k})
        self.assertEqual(len(skipped), 889)
        nbytes = 0
        for shard in sorted({wmap[k] for k in skipped}):
            with open(NV_CKPT / shard, "rb") as fh:
                header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
            for k in skipped & set(header):
                start, end = header[k]["data_offsets"]
                nbytes += end - start
        print(f"\n  MTP bytes not read per rank: {nbytes / 2**30:.2f} GiB")
        self.assertGreater(nbytes / 2**30, 13.0)

    def test_draft_group_predicate(self):
        path = self.patched("v1/core/kv_cache_coordinator.py")

        class SlidingWindowSpec:
            pass

        class KpoolTailSpec(SlidingWindowSpec):
            pass

        class Uniform:
            def __init__(self, s):
                self.kv_cache_specs = {"model.layers.0.x": s}

        env = extract(
            path, {"_glm53_is_draft_swa_spec"}, {"SlidingWindowSpec": SlidingWindowSpec}
        )
        pred = env["_glm53_is_draft_swa_spec"]
        self.assertTrue(pred(SlidingWindowSpec()))
        self.assertTrue(pred(Uniform(SlidingWindowSpec())))
        self.assertFalse(pred(KpoolTailSpec()))
        self.assertFalse(pred(Uniform(KpoolTailSpec())))
        self.assertFalse(pred(Uniform(object())))

    def test_mhc_warmup_covers_every_specialisation(self):
        path = self.patched("model_executor/warmup/glm5next_mhc_warmup.py")
        env = extract(
            path,
            {"select_mhc_token_sizes", "_cdiv", "_BLOCK", "_SMALL_FMA_MAX_TOKENS"},
            {"Callable": __import__("collections.abc").abc.Callable},
        )
        select = env["select_mhc_token_sizes"]

        # compute_num_split(64, 4 * 4096, grid) on GB10 (48 SMs), as in
        # kernels/mhc/tilelang_kernels.py.
        def num_split(grid, n_sms=48, k=4 * 4096):
            return max(min(n_sms // grid, (-(-k // 64)) // 4), 1)

        def key(t, deep_gemm):
            big = num_split(-(-t // 64)) if deep_gemm else (1, t < 128, t >= 1024)
            return ((t < 8) if t <= 16 else None, big)

        for deep_gemm in (True, False):
            for max_tokens in (1, 16, 100, 2048, 8192):
                sizes = select(max_tokens, num_split, deep_gemm)
                reachable = {key(t, deep_gemm) for t in range(1, max_tokens + 1)}
                covered = [key(t, deep_gemm) for t in sizes]
                self.assertEqual(set(covered), reachable, (deep_gemm, max_tokens))
                self.assertEqual(
                    len(covered), len(set(covered)), "no duplicate compiles"
                )
        sizes = select(2048, num_split, True)
        print(f"\n  mHC warmup sizes @2048 on 48 SMs: {sizes}")
        self.assertEqual(len({num_split(-(-t // 64)) for t in sizes}), 12)

    @unittest.skipUnless(HAVE_TORCH, "torch not installed")
    def test_kda_trim_predicate_on_glm_views(self):
        import torch

        path = self.patched("third_party/flash_linear_attention/ops/kda.py")
        env = extract(path, {"_glm53_kda_trim_ok"}, {"_GLM53_KDA_TRIM": True})
        ok = env["_glm53_kda_trim_ok"]
        H, D, n = 16, 128, 8  # per-rank KDA heads at TP=2, 8 verify tokens
        proj = torch.empty(n, 3 * H * D + H + 2 * D, dtype=torch.bfloat16)
        qkv, beta_raw = proj[:, : 3 * H * D], proj[:, 3 * H * D : 3 * H * D + H]
        q, k, v = (t.reshape(1, -1, H, D) for t in qkv.split(H * D, dim=-1))
        beta = beta_raw.unsqueeze(0)
        self.assertFalse(q.is_contiguous() or beta.is_contiguous())
        self.assertTrue(ok(q, k, v, beta))
        self.assertTrue(
            ok(q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous())
        )
        env["_GLM53_KDA_TRIM"] = False
        self.assertFalse(ok(q, k, v, beta))  # default off
        env["_GLM53_KDA_TRIM"] = True
        qb = torch.empty(2, n, H, D)
        self.assertFalse(ok(qb, qb, qb, torch.empty(2, n, H)))  # B > 1
        self.assertFalse(ok(q.transpose(-1, -2), k, v, beta))  # inner dim strided
        self.assertFalse(ok(q, k, v, torch.empty(1, n, H, D)))  # head-wise beta


KERNEL_SCRIPT = textwrap.dedent(
    r"""
    import importlib, sys, types
    import torch, triton, triton.language as tl

    pkgroot = sys.argv[1]
    sys.path.insert(0, pkgroot)
    tu = types.ModuleType("vllm.triton_utils")
    tu.tl, tu.triton = tl, triton
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm.triton_utils"] = tu
    orig = importlib.import_module("fla_orig.fused_recurrent")
    new = importlib.import_module("fla_new.fused_recurrent")

    torch.manual_seed(0)
    H = HV = 2
    K = V = 16
    N, L = 3, 4  # sequences, verify tokens per sequence
    T = N * L
    W = 3 * H * K + H + 5  # merged projection row: q|k|v|beta|other
    proj = torch.randn(T, W).to(torch.bfloat16)
    qkv = proj[:, : 3 * H * K]
    q, k, v = (t.reshape(1, T, H, K) for t in qkv.split(H * K, dim=-1))
    beta = proj[:, 3 * H * K : 3 * H * K + H].unsqueeze(0)
    g = torch.randn(1, T, H, K).to(torch.bfloat16)
    a_log = torch.randn(H) * 0.1
    g_bias = torch.randn(H * K) * 0.1
    cu = torch.arange(0, T + 1, L, dtype=torch.int32)
    slots = 1 + torch.arange(N * L, dtype=torch.int32).reshape(N, L)
    acc = torch.tensor([1, 3, 2], dtype=torch.int32)
    state0 = torch.randn(1 + N * L, HV, V, K)

    def run(mod, q, k, v, beta, extra):
        o = torch.empty(1, T, HV, V, dtype=torch.bfloat16)
        st = state0.clone()
        mod.fused_recurrent_gated_delta_rule_fwd_kernel[(1, V // 8, N * HV)](
            q=q, k=k, v=v, g=g, beta=beta, o=o, h0=st, ht=st,
            cu_seqlens=cu, ssm_state_indices=slots, num_accepted_tokens=acc,
            scale=K ** -0.5, N=N, T=T, B=1, H=H, HV=HV, K=K, V=V, BK=K, BV=8,
            stride_init_state_token=st.stride(0),
            stride_final_state_token=st.stride(0),
            stride_indices_seq=slots.stride(0), stride_indices_tok=1,
            IS_BETA_HEADWISE=False, USE_QK_L2NORM_IN_KERNEL=True,
            INPLACE_FINAL_STATE=True, IS_KDA=True, SIGMOID_BETA=True,
            a_log=a_log, g_bias=g_bias, COMPUTE_GATE=True, SAFE_GATE=True,
            LOWER_BOUND=-5.0, num_warps=1, num_stages=3, **extra,
        )
        return o, st

    dense = [t.contiguous() for t in (q, k, v, beta)]
    ref_o, ref_st = run(orig, *dense, {})
    off = dict(stride_q_tok=0, stride_k_tok=0, stride_v_tok=0,
               stride_beta_tok=0, STRIDED_QKVB=False)
    o1, st1 = run(new, *dense, off)
    on = dict(stride_q_tok=q.stride(1), stride_k_tok=k.stride(1),
              stride_v_tok=v.stride(1), stride_beta_tok=beta.stride(1),
              STRIDED_QKVB=True)
    o2, st2 = run(new, q, k, v, beta, on)
    assert not q.is_contiguous() and q.stride(1) == W, q.stride()
    for name, (a, b) in {
        "off.o": (o1, ref_o), "off.state": (st1, ref_st),
        "trim.o": (o2, ref_o), "trim.state": (st2, ref_st),
    }.items():
        assert torch.equal(a, b), name
    assert ref_o.abs().sum() > 0 and not torch.equal(ref_st, state0)
    print("KDA kernel bit-exact: trim-off and trim-on (token stride %d) vs v11" % W)
    """
)


@unittest.skipUnless(
    HAVE_SRC and HAVE_TORCH and HAVE_TRITON, "needs GLM53_V11_SRC, torch, triton"
)
class KdaKernelInterpreterTest(unittest.TestCase):
    def test_strided_kernel_is_bit_exact(self):
        with tempfile.TemporaryDirectory(prefix="v13kda-") as tmp:
            tmp = Path(tmp)
            rel = "third_party/flash_linear_attention/ops/fused_recurrent.py"
            src, applied = patch.plan_edits(
                (SRC / rel).read_text(), patch.FILE_EDITS[rel]
            )
            self.assertEqual(applied, len(patch.FILE_EDITS[rel]))
            for pkg, text in (("fla_orig", (SRC / rel).read_text()), ("fla_new", src)):
                d = tmp / pkg
                d.mkdir()
                (d / "__init__.py").write_text("")
                # Same ops as fla op.py (FLA_USE_FAST_OPS unset); jit-wrapped so
                # the interpreter resolves them.
                (d / "op.py").write_text(
                    "import triton\nimport triton.language as tl\n\n"
                    "@triton.jit\ndef exp(x):\n    return tl.exp(x)\n\n"
                    "@triton.jit\ndef log(x):\n    return tl.log(x)\n"
                )
                (d / "fused_recurrent.py").write_text(text)
            res = subprocess.run(
                [sys.executable, "-c", KERNEL_SCRIPT, str(tmp)],
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "TRITON_INTERPRET": "1"},
            )
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            print("\n  " + res.stdout.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
