"""CPU tests for patch_v13_fp8.py (no GPU, no vLLM install needed).

Runs against a copy of the vLLM python tree shipped in glm53-sm121-v11, taken
from $GLM53_V11_SRC (a directory holding vllm/). Skips when that is absent.
The pure-torch quantizer tests run only when torch is importable.

    GLM53_V11_SRC=/path/to/v11src python3 -m unittest docker/test_v13_fp8.py -v
"""

import ast
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = HERE / "patch_v13_fp8.py"
SRC = Path(os.environ.get("GLM53_V11_SRC", "")) / "vllm"
MODULE = "model_executor/layers/quantization/glm53_fp8_w8a16.py"
TOUCHED = [MODULE, "model_executor/model_loader/utils.py", "envs.py"]

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None


def run_patch(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PATCH), str(root)], capture_output=True, text=True
    )


def digest(root: Path) -> dict[str, str]:
    return {
        rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
        for rel in TOUCHED
        if (root / rel).exists()
    }


@unittest.skipUnless(SRC.is_dir(), "set GLM53_V11_SRC to the v11 vLLM source")
class PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "vllm"
        shutil.copytree(SRC, cls.root, ignore=shutil.ignore_patterns("__pycache__"))
        cls.pristine = digest(cls.root)
        cls.first = run_patch(cls.root)
        cls.patched = digest(cls.root)
        cls.second = run_patch(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_applies_then_is_idempotent(self):
        self.assertEqual(self.first.returncode, 0, self.first.stderr)
        self.assertIn("3 file(s) written", self.first.stdout)
        self.assertEqual(self.second.returncode, 0, self.second.stderr)
        self.assertIn("0 file(s) written", self.second.stdout)
        self.assertEqual(self.patched, digest(self.root))
        self.assertNotIn(MODULE, self.pristine)
        for rel in TOUCHED[1:]:
            self.assertNotEqual(self.pristine[rel], self.patched[rel], rel)

    def test_touched_files_compile(self):
        for rel in TOUCHED:
            subprocess.run(
                [sys.executable, "-m", "py_compile", str(self.root / rel)], check=True
            )

    def test_hook_is_env_gated(self):
        text = (self.root / "model_executor/model_loader/utils.py").read_text()
        fn = next(
            n
            for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef) and n.name == "process_weights_after_loading"
        )
        last = fn.body[-1]
        self.assertIsInstance(last, ast.If)
        self.assertEqual(
            ast.unparse(last.test),
            "_os.environ.get('GLM53_FP8_W8A16', '').strip() or "
            "_os.environ.get('GLM53_NVFP4_W4A16', '').strip()",
        )
        self.assertIn("apply_glm53_fp8_w8a16(model, target_device)", ast.unparse(last))

    def test_compile_factor_only_when_set(self):
        text = (self.root / "envs.py").read_text()
        self.assertEqual(text.count("factors[_glm53] = "), 1)
        self.assertIn('for _glm53 in ("GLM53_FP8_W8A16", "GLM53_NVFP4_W4A16"):', text)
        self.assertIn('if os.getenv(_glm53, "").strip():', text)

    def test_refuses_drift_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vllm"
            for rel in TOUCHED[1:]:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(SRC / rel, root / rel)
            (root / MODULE).parent.mkdir(parents=True, exist_ok=True)
            utils = root / "model_executor/model_loader/utils.py"
            utils.write_text(
                utils.read_text().replace("set_torchao_reload_attrs(model, model_config)", "pass")
            )
            before = digest(root)
            proc = run_patch(root)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("refusing", proc.stderr)
            self.assertEqual(before, digest(root))

    def test_refuses_foreign_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vllm"
            shutil.copytree(self.root, root, ignore=shutil.ignore_patterns("__pycache__"))
            (root / MODULE).write_text("# someone else's file\n")
            proc = run_patch(root)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("refusing", proc.stderr)

    def test_referenced_classes_and_attributes_exist(self):
        """The name-based site table must match the v11 model code."""
        src = {
            "kda": (SRC / "models/glm5next/nvidia/kda.py").read_text(),
            "attn": (SRC / "models/glm5next/nvidia/attention.py").read_text(),
            "model": (SRC / "models/glm5next/nvidia/model.py").read_text(),
            "dflash": (SRC / "model_executor/models/qwen3_dflash.py").read_text(),
            "dflash2": (SRC / "model_executor/models/qwen3_dflash2.py").read_text(),
            "vocab": (SRC / "model_executor/layers/vocab_parallel_embedding.py").read_text(),
        }
        needles = [
            ("kda", "class Glm5NextLinearAttention("),
            ("kda", "self.in_proj_qkvbfg_a = "),
            ("kda", "self.o_proj = RowParallelLinear("),
            ("attn", "class Glm5NextMLAAttention("),
            ("attn", "self.q_b_proj = ColumnParallelLinear("),
            ("attn", "self.o_proj = RowParallelLinear("),
            ("model", "class Glm5NextMoE("),
            ("model", "self.shared_experts = Glm5NextMLP("),
            ("model", "self.gate_up_proj = MergedColumnParallelLinear("),
            ("model", "self.down_proj = RowParallelLinear("),
            ("model", "class Glm5NextForCausalLM("),
            ("model", "class Glm5NextForConditionalGeneration("),
            ("model", "self.lm_head = ParallelLMHead("),
            ("model", "quant_config=None,  # MLA projections are BF16 in checkpoint"),
            ("dflash", "class DFlashQwen3ForCausalLM("),
            ("dflash", "self._fused_kv_weight = torch.cat(kv_weights, dim=0)"),
            ("dflash", "self.model._build_fused_kv_buffers()"),
            ("dflash2", "class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):"),
            ("vocab", "class ParallelLMHead(VocabParallelEmbedding):"),
        ]
        for key, needle in needles:
            self.assertIn(needle, src[key], f"{key}: {needle}")

    def test_marlin_api_matches(self):
        text = (SRC / "model_executor/layers/quantization/utils/marlin_utils_fp8.py").read_text()
        tree = ast.parse(text)
        fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        prep = [a.arg for a in fns["prepare_fp8_layer_for_marlin"].args.args]
        self.assertEqual(prep, ["layer", "size_k_first", "input_dtype"])
        gemm = [a.arg for a in fns["apply_fp8_marlin_linear"].args.args]
        for arg in ("input", "weight", "weight_scale", "workspace", "size_n", "size_k", "bias"):
            self.assertIn(arg, gemm)
        # Per-channel path: one scale per output row, cast to the layer dtype.
        self.assertIn("scales = layer.weight_scale.to(layer.orig_dtype)", text)
        self.assertIn("scales = scales.view(1, part_size_n)", text)
        utils = (SRC / "model_executor/utils.py").read_text()
        self.assertRegex(utils, r"def replace_parameter\(\s*layer")
        linear = (SRC / "model_executor/layers/linear.py").read_text()
        self.assertEqual(len(re.findall(r"self\.quant_method\.apply\(self, ", linear)), 3)
        logits = (SRC / "model_executor/layers/logits_processor.py").read_text()
        self.assertIn("return lm_head.quant_method.apply(", logits)

    def test_marlin_nvfp4_api_matches(self):
        """quantize_layer_to_marlin_nvfp4 feeds prepare_fp4_layer_for_marlin a
        ModelOpt-shaped layer; these are the v11 facts it relies on."""
        text = (SRC / "model_executor/layers/quantization/utils/marlin_utils_fp4.py").read_text()
        fns = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        self.assertEqual([a.arg for a in fns["prepare_fp4_layer_for_marlin"].args.args],
                         ["layer", "input_dtype"])
        gemm = [a.arg for a in fns["apply_fp4_marlin_linear"].args.args]
        for arg in ("input", "weight", "weight_scale", "weight_global_scale", "workspace",
                    "size_n", "size_k", "bias"):
            self.assertIn(arg, gemm)
        for needle in (
            'is_nvfp4 = hasattr(layer, "weight_global_scale")',
            "param_dtype = layer.params_dtype",
            "assert layer.weight.shape == (part_size_n, part_size_k // 2)",
            "qweight = layer.weight.view(torch.int32).T.contiguous()",
            "weight_scale = layer.weight_scale.T.contiguous()",
            "weight_global_scale = layer.weight_global_scale.to(torch.float32)",
            # Why block scales must stay e4m3 normals (>= 2^-6):
            "marlin_scales[marlin_scales < 2] = 0",
            # Packing reference: element 2j is the low nibble.
            "fp4_weight2 = fp4_weight << 4",
            "[fp4_weight_part_2.unsqueeze(2), fp4_weight_part_1.unsqueeze(2)], 2",
        ):
            self.assertIn(needle, text)
        kernel = (SRC / "model_executor/kernels/linear/nvfp4/marlin.py").read_text()
        self.assertIn("weight_global_scale=layer.weight_global_scale,", kernel)


def load_module(root: Path):
    spec = importlib.util.spec_from_file_location("glm53_fp8_w8a16", root / MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def module_constants() -> dict:
    """Literal top-level constants of the installed module, read without torch."""
    spec = importlib.util.spec_from_file_location("patch_v13_fp8", PATCH)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    out = {}
    for node in ast.parse(patch.MODULE_SRC).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            for t in node.targets:
                names = t.elts if isinstance(t, ast.Tuple) else [t]
                values = value if isinstance(t, ast.Tuple) else [value]
                for n, v in zip(names, values):
                    out[n.id] = v
    return out


# numpy reference of quantize_nvfp4, step for step (same fp32 ops and order).
C = module_constants()
if np is not None:
    GRID = np.array(C["E2M1_GRID"], np.float32)
    MIDS = np.array(C["E2M1_MIDS"], np.float32)


def e4m3_encode(x):
    """float32 in the e4m3 normal range [2^-6, 448] -> code, round half to even."""
    f, e = np.frexp(x)
    m = np.rint((f * 2 - 1) * 8)
    carry = m == 8
    return ((e + 6 + carry) * 8 + np.where(carry, 0, m)).astype(np.uint8)


def e4m3_decode(code):
    """Any finite e4m3fn code (subnormals included) -> float32."""
    c = code.astype(np.int32)
    e, m = (c >> 3) & 15, c & 7
    mag = np.where(e == 0, np.ldexp(m / 8, -6), np.ldexp(1 + m / 8, e - 7))
    return np.where(c & 0x80, -mag, mag).astype(np.float32)


def np_quantize_nvfp4(w, steps=None, chunk_rows=256):
    """Returns (packed uint8 (N, K/2), scale codes uint8 (N, K/16), global fp32)."""
    steps = C["NVFP4_CODE_STEPS"] if steps is None else steps
    n, k = w.shape
    g = np.maximum(np.abs(w).max() / np.float32(6 * C["FP8_MAX"]), np.float32(1e-30))
    nib = np.empty((n, k), np.uint8)
    codes = np.empty((n, k // 16), np.uint8)
    for i in range(0, n, chunk_rows):
        wb = w[i : i + chunk_rows].reshape(-1, k // 16, 16)
        s0 = np.abs(wb).max(-1) / (np.float32(6) * g)
        base = e4m3_encode(np.clip(s0, np.float32(2**-6), np.float32(C["FP8_MAX"])))
        sign = (wb < 0).astype(np.uint8) << 3
        best = None
        for step in steps:
            code = np.clip(base.astype(np.int32) + step, C["E4M3_MIN_NORMAL_CODE"],
                           C["E4M3_MAX_CODE"]).astype(np.uint8)
            s = (e4m3_decode(code) * g)[..., None]
            x = wb / s
            idx = np.searchsorted(MIDS, np.abs(x), side="left").astype(np.uint8)
            err = np.square(np.copysign(GRID[idx], x) * s - wb).sum(-1)
            q = idx | sign
            if best is None:
                best, bcode, bq = err, code, q
            else:
                better = err < best
                best = np.where(better, err, best)
                bcode = np.where(better, code, bcode)
                bq = np.where(better[..., None], q, bq)
        nib[i : i + chunk_rows] = bq.reshape(-1, k)
        codes[i : i + chunk_rows] = bcode
    return nib[:, 0::2] | (nib[:, 1::2] << 4), codes, g


def np_dequantize_nvfp4(packed, codes, g):
    n = packed.shape[0]
    nib = np.stack([packed & 15, packed >> 4], -1).reshape(n, -1)
    v = np.where(nib & 8, -GRID[nib & 7], GRID[nib & 7])
    return (v.reshape(n, -1, 16) * (e4m3_decode(codes) * g)[..., None]).reshape(n, -1)


def bf16_bits_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


CKPT = Path(os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))) \
    / "models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/09b04e5e74bca08ca8549fc736d4cdd8624bfde3"
REAL_ROWS = 256  # rows read per tensor: 2-8 MiB each, dropped from page cache after
REAL_TENSORS = {  # group -> checkpoint tensor (layer 5 is KDA, layer 3 is MLA)
    "kda_in": "model.language_model.layers.5.self_attn.q_proj.weight",
    "kda_o": "model.language_model.layers.5.self_attn.o_proj.weight",
    "mla": "model.language_model.layers.3.self_attn.q_b_proj.weight",
    "shared": "model.language_model.layers.3.mlp.shared_experts.down_proj.weight",
    "lm_head": "lm_head.weight",
}


def read_headers(snapshot: Path) -> dict:
    spec = importlib.util.spec_from_file_location(
        "bench_fp8_marlin", HERE.parent / "tools/bench_fp8_marlin.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench.read_headers(snapshot)


def read_rows(headers: dict, name: str, rows: int):
    path, start, dtype, (r, c) = headers[name]
    assert dtype == "BF16", (name, dtype)
    nbytes = min(rows, r) * c * 2
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = os.pread(fd, nbytes, start)
        os.posix_fadvise(fd, start, nbytes, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return bf16_bits_to_f32(np.frombuffer(buf, np.uint16).reshape(-1, c))


def synthetic_weight(n=300, k=512, seed=0):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((n, k)) * 0.02).astype(np.float32)
    w[3, 7] = 1.5  # outlier: sets the global scale
    w[5] *= 1e-2  # small row: block scales well inside the e4m3 range
    w[6] *= 1e-5  # tiny row: block scales clamp to the e4m3 normal floor
    w[9] = 0.0  # all-zero row
    return bf16_bits_to_f32((w.view(np.uint32) >> 16).astype(np.uint16))  # truncate to bf16


@unittest.skipIf(np is None, "numpy not importable")
class Nvfp4ReferenceTest(unittest.TestCase):
    """numpy-only: NVFP4 format facts, and the quantizer's error on real weights."""

    def test_e2m1_grid_matches_marlin_bit_trick(self):
        # rand_marlin_weight_nvfp4_like decodes a nibble as e4m3 bits
        # (sign << 4 | magnitude << 2) times 2^6.
        nib = np.arange(16, dtype=np.uint8)
        as_e4m3 = ((nib & 8) << 4) | ((nib & 7) << 2)
        want = np.where(nib & 8, -GRID[nib & 7], GRID[nib & 7])
        np.testing.assert_array_equal(e4m3_decode(as_e4m3) * 64, want)

    def test_e4m3_encode_roundtrips_normals_and_rounds_half_even(self):
        codes = np.arange(C["E4M3_MIN_NORMAL_CODE"], C["E4M3_MAX_CODE"] + 1, dtype=np.uint8)
        np.testing.assert_array_equal(e4m3_encode(e4m3_decode(codes)), codes)
        self.assertEqual(float(e4m3_decode(np.uint8(C["E4M3_MAX_CODE"]))), 448.0)
        mid = (e4m3_decode(np.array([8, 9], np.uint8)).sum() / 2).astype(np.float32)
        self.assertEqual(int(e4m3_encode(mid)), 8)  # tie -> even mantissa

    def test_marlin_scale_format_keeps_normal_block_scales(self):
        """nvfp4_marlin_process_scales: half(s) * 2^7, zero if < 2, keep the top
        byte of (bits << 1). Lossless for e4m3 normals, zero for subnormals."""
        codes = np.arange(1, C["E4M3_MAX_CODE"] + 1, dtype=np.uint8)
        s = e4m3_decode(codes).astype(np.float16) * np.float16(2**7)
        s[s < 2] = 0
        top = ((s.view(np.uint16) << 1) >> 8).astype(np.int32)  # S0E5M3 byte
        back = np.where(top == 0, 0, np.ldexp(1 + (top & 7) / 8, (top >> 3) - 15 - 7))
        normal = codes >= C["E4M3_MIN_NORMAL_CODE"]
        np.testing.assert_array_equal(back[normal], e4m3_decode(codes[normal]))
        self.assertTrue((back[~normal] == 0).all())

    def test_roundtrip_error_bounds(self):
        w = synthetic_weight()
        packed, codes, g = np_quantize_nvfp4(w)
        self.assertEqual(packed.shape, (300, 256))
        self.assertEqual(codes.shape, (300, 32))
        self.assertTrue(((codes >= 8) & (codes <= 0x7E)).all())
        self.assertEqual(g.dtype, np.float32)
        deq = np_dequantize_nvfp4(packed, codes, g)
        self.assertTrue((deq[9] == 0).all())
        # Elementwise: at most a 2-code-step clip (6 * 1.125^2 * 1.0625) of the
        # chosen block scale, else half the widest E2M1 gap.
        s = np.repeat(e4m3_decode(codes) * g, 16, axis=1)
        self.assertTrue((np.abs(deq - w) <= 2.1 * s).all())
        self.assertLess(rel_err(deq, w), 0.09)
        # The global scale puts the largest block scale on the e4m3 maximum.
        base_codes = np_quantize_nvfp4(w, steps=(0,))[1]
        self.assertEqual(int(base_codes.max()), C["E4M3_MAX_CODE"])
        # A row 1e-2 below the outlier keeps NVFP4 relative precision. Blocks
        # below amax / (6 * 448 * 64) sit on the scale floor: only the
        # absolute bound above holds for them (row 6).
        self.assertLess(rel_err(deq[5], w[5]), 0.1)
        self.assertTrue((codes[6] == C["E4M3_MIN_NORMAL_CODE"]).all())

    def test_search_never_loses_to_amax_over_6(self):
        w = synthetic_weight(seed=2)
        base = np_dequantize_nvfp4(*np_quantize_nvfp4(w, steps=(0,)))
        best = np_dequantize_nvfp4(*np_quantize_nvfp4(w))
        blk = lambda d: np.square(d - w).reshape(300, -1, 16).sum(-1)  # noqa: E731
        self.assertTrue((blk(best) <= blk(base)).all())
        self.assertLess(rel_err(best, w), rel_err(base, w))

    def test_chunking_is_exact(self):
        w = synthetic_weight(n=257, k=256, seed=1)
        a = np_quantize_nvfp4(w)
        b = np_quantize_nvfp4(w, chunk_rows=7)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    @unittest.skipUnless(CKPT.is_dir(), f"checkpoint not at {CKPT}")
    def test_real_checkpoint_error(self):
        """Relative Frobenius error on the first REAL_ROWS rows of one tensor
        per group. Measured 2026-09-27 on 09b04e5: 0.083-0.086 (FP8 per-channel
        is 0.025-0.029 on the same tensors; amax/6 alone is 0.092-0.095)."""
        headers = read_headers(CKPT)
        for group, name in REAL_TENSORS.items():
            w = read_rows(headers, name, REAL_ROWS)
            base = rel_err(np_dequantize_nvfp4(*np_quantize_nvfp4(w, steps=(0,))), w)
            best = rel_err(np_dequantize_nvfp4(*np_quantize_nvfp4(w)), w)
            print(f"\n  {group:8} {name} {w.shape}: amax/6 {base:.4f}, search {best:.4f}",
                  end="")
            self.assertLess(best, 0.09, name)
            self.assertLess(best, base * 0.95, name)


@unittest.skipUnless(SRC.is_dir(), "set GLM53_V11_SRC to the v11 vLLM source")
@unittest.skipIf(torch is None, "torch not importable")
class QuantizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name) / "vllm"
        (root / MODULE).parent.mkdir(parents=True)
        for rel in TOUCHED[1:]:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(SRC / rel, root / rel)
        proc = run_patch(root)
        assert proc.returncode == 0, proc.stderr
        cls.m = load_module(root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parse_groups(self):
        self.assertEqual(self.m.parse_groups(""), [])
        self.assertEqual(self.m.parse_groups(" , "), [])
        self.assertEqual(
            self.m.parse_groups("draft, shared,draft,lm_head"), ["draft", "shared", "lm_head"]
        )
        self.assertEqual(set(self.m.parse_groups(",".join(self.m.GROUPS))), set(self.m.GROUPS))
        with self.assertRaises(ValueError):
            self.m.parse_groups("draft,kda")

    def weight(self, n=300, k=512, seed=0):
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(n, k, generator=g) * 0.02
        w[3, 7] = 1.5  # row outlier
        w[5] *= 1e-4  # tiny row: exercises e4m3 subnormals
        w[9] = 0.0  # all-zero row
        return w.to(torch.bfloat16)

    def test_per_channel_error_bound(self):
        w = self.weight()
        q, s = self.m.quantize_per_channel(w)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.dtype, torch.bfloat16)
        self.assertEqual(tuple(s.shape), (w.shape[0],))
        self.assertTrue(torch.isfinite(q.float()).all())
        self.assertTrue((s.float() > 0).all())
        deq = self.m.dequantize_per_channel(q, s)
        wf = w.float()
        err = (deq - wf).abs()
        # e4m3: 3 mantissa bits -> half-ULP 2^-4 relative for normals, and
        # half the smallest subnormal (2^-10) times the scale near zero.
        bound = torch.maximum(wf.abs() * 2**-4, s.float().unsqueeze(1) * 2**-10)
        self.assertTrue((err <= bound * 1.0001).all(), float((err / bound).max()))
        self.assertTrue((deq[9] == 0).all())
        # Row max maps to (at most) the e4m3 max, scaled back within 1 bf16 ULP.
        amax = wf.abs().amax(dim=1)
        nz = amax > 0
        self.assertTrue(
            torch.allclose(deq.abs().amax(dim=1)[nz], amax[nz], rtol=2**-7, atol=0)
        )
        rel = (deq - wf).norm() / wf.norm()
        self.assertLess(float(rel), 0.03)

    def test_chunking_is_exact(self):
        w = self.weight(n=257, k=256, seed=1)
        q1, s1 = self.m.quantize_per_channel(w)
        q2, s2 = self.m.quantize_per_channel(w, chunk_elems=3 * 256)
        q3, s3 = self.m.quantize_per_channel(w, chunk_elems=1)
        self.assertTrue(torch.equal(s1, s2) and torch.equal(s1, s3))
        for q in (q2, q3):
            self.assertTrue(torch.equal(q1.view(torch.uint8), q.view(torch.uint8)))

    def test_collect_sites(self):
        nn = torch.nn

        def cls(name, base=nn.Module):
            return type(name, (base,), {})

        Linear = cls("LinearBase")
        kda = cls("Glm5NextLinearAttention")()
        kda.in_proj_qkvbfg_a, kda.o_proj, kda.f_b_proj = Linear(), Linear(), Linear()
        mla = cls("Glm5NextMLAAttention")()
        mla.q_b_proj, mla.o_proj, mla.kv_b_proj = Linear(), Linear(), Linear()
        mla.fused_qkv_a_proj = Linear()
        moe = cls("Glm5NextMoE")()
        moe.shared_experts = nn.Module()
        moe.shared_experts.gate_up_proj = Linear()
        moe.shared_experts.down_proj = Linear()
        moe.gate = Linear()
        target = cls("Glm5NextForConditionalGeneration")()
        target.layers = nn.ModuleList([kda, mla, moe])
        target.lm_head = cls("ParallelLMHead")()
        target.embed_tokens = cls("VocabParallelEmbedding")()

        def names(model, groups):
            kind, sites = self.m._collect_sites(model, groups, Linear)
            return kind, sorted((g, n) for g, n, _ in sites)

        kind, got = names(target, list(self.m.GROUPS))
        self.assertEqual(kind, "target")
        self.assertEqual(got, sorted([
            ("kda_in", "layers.0.in_proj_qkvbfg_a"), ("kda_o", "layers.0.o_proj"),
            ("mla", "layers.1.q_b_proj"), ("mla", "layers.1.o_proj"),
            ("shared", "layers.2.shared_experts.gate_up_proj"),
            ("shared", "layers.2.shared_experts.down_proj"),
            ("lm_head", "lm_head"),
        ]))
        self.assertEqual(names(target, ["draft"]), ("target", []))
        self.assertEqual(names(target, ["kda_o"])[1], [("kda_o", "layers.0.o_proj")])

        draft = cls("DFlash2Qwen3ForCausalLM", cls("DFlashQwen3ForCausalLM"))()
        draft.fc, draft.lm_head = Linear(), cls("ParallelLMHead")()
        self.assertEqual(names(draft, ["draft", "lm_head"]), ("draft", [("draft", "fc")]))
        self.assertEqual(names(draft, ["mla"]), ("draft", []))
        self.assertEqual(names(cls("Glm5NextMTP")(), ["mla"]), (None, []))

    def test_scale_survives_marlin_exponent_bias(self):
        """Marlin folds 2^120 into BF16 scales; realistic scales must stay finite."""
        s = self.m.quantize_per_channel(self.weight())[1]
        self.assertTrue(torch.isfinite(s * 2.0**120).all())

    def test_selected_modes(self):
        env = {"GLM53_FP8_W8A16": "draft,shared", "GLM53_NVFP4_W4A16": " kda_o, mla "}
        self.assertEqual(self.m.selected_modes(env), {
            "draft": "fp8", "shared": "fp8", "kda_o": "nvfp4", "mla": "nvfp4"})
        self.assertEqual(self.m.selected_modes({}), {})
        with self.assertRaisesRegex(ValueError, "both"):
            self.m.selected_modes({"GLM53_FP8_W8A16": "mla", "GLM53_NVFP4_W4A16": "mla"})
        with self.assertRaisesRegex(ValueError, "GLM53_NVFP4_W4A16: unknown"):
            self.m.selected_modes({"GLM53_NVFP4_W4A16": "attn"})

    @unittest.skipIf(np is None, "numpy not importable")
    def test_nvfp4_matches_numpy_reference(self):
        w = synthetic_weight()
        packed, scale, g = self.m.quantize_nvfp4(torch.from_numpy(w).to(torch.bfloat16))
        rp, rc, rg = np_quantize_nvfp4(w)
        self.assertEqual((packed.dtype, scale.dtype, g.dtype),
                         (torch.uint8, torch.float8_e4m3fn, torch.float32))
        self.assertEqual(float(g), float(rg))
        # Block SSE sums may round differently, so allow rare tie flips.
        codes = scale.view(torch.uint8).numpy()
        self.assertGreater((codes == rc).mean(), 0.999)
        self.assertGreater((packed.numpy() == rp).mean(), 0.999)
        deq = self.m.dequantize_nvfp4(packed, scale, g).numpy()
        np.testing.assert_allclose(deq, np_dequantize_nvfp4(packed.numpy(), codes, rg),
                                   rtol=1e-6, atol=0)
        self.assertLess(rel_err(deq, w), 0.09)

    def test_nvfp4_chunking_is_exact(self):
        w = self.weight(n=257, k=256, seed=1)
        a = self.m.quantize_nvfp4(w)
        for chunk in (3 * 256, 1):
            b = self.m.quantize_nvfp4(w, chunk_elems=chunk)
            self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[2], b[2]))
            self.assertTrue(torch.equal(a[1].view(torch.uint8), b[1].view(torch.uint8)))
        with self.assertRaises(ValueError):
            self.m.quantize_nvfp4(torch.zeros(4, 24, dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
