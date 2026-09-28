"""CPU tests for patch_v13_fp8.py (no GPU, no vLLM install needed).

Runs against a copy of the vLLM python tree shipped in glm53-sm121-v11, taken
from $GLM53_V11_SRC (a directory holding vllm/). Skips when that is absent.
The pure-torch quantizer tests run only when torch is importable, and the
GLM53_WQ_DEQUANT_MIN_M tests also need triton (they use its CPU interpreter).

    GLM53_V11_SRC=/path/to/v11src python3 -m unittest docker/test_v13_fp8.py -v
"""

import ast
import hashlib
import importlib.util
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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
try:
    import triton
except ImportError:
    triton = None


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
            "any((_os.environ.get(_v, '').strip() for _v in ('GLM53_FP8_W8A16', "
            "'GLM53_NVFP4_W4A16', 'GLM53_INT8_W8A16', 'GLM53_INT4_W4A16')))",
        )
        self.assertIn("apply_glm53_fp8_w8a16(model, target_device)", ast.unparse(last))

    def test_compile_factor_only_when_set(self):
        text = (self.root / "envs.py").read_text()
        self.assertEqual(text.count("factors[_glm53] = "), 1)
        loop = next(n for n in ast.walk(ast.parse(text))
                    if isinstance(n, ast.For) and ast.unparse(n.target) == "_glm53")
        self.assertEqual(ast.literal_eval(loop.iter), (
            "GLM53_FP8_W8A16", "GLM53_NVFP4_W4A16", "GLM53_INT8_W8A16",
            "GLM53_INT4_W4A16", "GLM53_INT_GROUP_SIZE"))
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

    def test_marlin_gptq_api_matches(self):
        """quantize_layer_to_marlin_int repeats MarlinLinearKernel's GPTQ path
        (no act-order, no zero points); these are the v11 facts it relies on."""
        text = (SRC / "model_executor/layers/quantization/utils/marlin_utils.py").read_text()
        fns = {n.name: [a.arg for a in n.args.args]
               for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(fns["marlin_padded_nk"], ["size_n", "size_k", "group_size"])
        self.assertEqual(fns["marlin_pad_qweight"],
                         ["qweight", "size_n", "size_k", "padded_n", "padded_k"])
        self.assertEqual(fns["marlin_pad_scales"],
                         ["scales", "size_n", "size_k", "padded_n", "padded_k", "group_size"])
        self.assertEqual(fns["marlin_permute_scales"],
                         ["s", "size_k", "size_n", "group_size", "is_a_8bit"])
        self.assertEqual(fns["marlin_make_workspace_new"],
                         ["device", "max_blocks_per_sm", "existing"])
        self.assertEqual(fns["apply_gptq_marlin_linear"], [
            "input", "weight", "weight_scale", "weight_zp", "g_idx", "g_idx_sort_indices",
            "workspace", "wtype", "output_size_per_partition", "input_size_per_partition",
            "is_k_full", "input_global_scale", "bias", "use_fp32_reduce", "input_dtype"])
        self.assertIn("MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]", text)
        self.assertIn("res = [scalar_types.uint4b8, scalar_types.uint8b128]", text)
        # The padded K/N the GEMM runs with come back from the repacked shape.
        self.assertIn("padded_n, padded_k = marlin_repacked_nk(weight, wtype.size_bits)", text)
        ops = (SRC / "_custom_ops.py").read_text()
        repack = next(n for n in ast.parse(ops).body
                      if isinstance(n, ast.FunctionDef) and n.name == "gptq_marlin_repack")
        self.assertEqual([a.arg for a in repack.args.args],
                         ["b_q_weight", "perm", "size_k", "size_n", "num_bits", "is_a_8bit"])
        kernel = (SRC / "model_executor/kernels/linear/mixed_precision/marlin.py").read_text()
        for needle in (
            # GPTQ layout: (K / pack, N) int32, packed along K.
            "permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)",
            "x.data.contiguous(), size_n, size_k, padded_n, padded_k",
            "perm=layer.g_idx_sort_indices,",
            "num_bits=c.weight_type.size_bits,",
            "x.data = marlin_permute_scales(",
            "padded_n, padded_k = marlin_padded_nk(size_n, size_k, c.group_size)",
            "setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))",
            "is_k_full=self.is_k_full,",
        ):
            self.assertIn(needle, kernel)

    def test_gptq_scalar_types(self):
        """quantize_int stores q + 2**(bits - 1) in [0, 2**bits): the uint4b8 /
        uint8b128 encodings (scalar_type.py is stdlib-only)."""
        spec = importlib.util.spec_from_file_location("v11_scalar_type", SRC / "scalar_type.py")
        st = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = st  # dataclasses resolves the module by name
        try:
            spec.loader.exec_module(st)
        finally:
            del sys.modules[spec.name]
        for t, bits in ((st.scalar_types.uint8b128, 8), (st.scalar_types.uint4b8, 4)):
            self.assertEqual((t.size_bits, t.bias), (bits, 2 ** (bits - 1)))
            self.assertEqual((t.min(), t.max()), (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1))


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


def v11_functions(rel: str, names: set, env: dict) -> dict:
    """Exec the named top-level functions of a v11 source file into env."""
    tree = ast.parse((SRC / rel).read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == names, names
    exec(compile(ast.Module(body, []), str(SRC / rel), "exec"), env)
    return env


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


DRAFT_CKPT = CKPT.parents[2] / (
    "models--incoai--GLM-5.3-Flash-DFlash2/snapshots/7d74cdd881ed7e32c31175984a67823127b66cfe")


@unittest.skipUnless(SRC.is_dir(), "set GLM53_V11_SRC to the v11 vLLM source")
@unittest.skipUnless(CKPT.is_dir() and DRAFT_CKPT.is_dir(), "checkpoints not in the HF cache")
class IntShapeTest(unittest.TestCase):
    def test_every_gemm_fits_gptq_marlin_at_tp2(self):
        """Per-rank GEMM shapes from the safetensors headers at TP=2: every K is
        a multiple of both INT group sizes, so row-parallel shards split on
        group boundaries and no layer falls back to BF16; marlin_padded_nk
        never pads K, and pads N only for kda_in (12576 -> 12608, as FP8 does)."""
        spec = importlib.util.spec_from_file_location(
            "bench_fp8_marlin", HERE.parent / "tools/bench_fp8_marlin.py")
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        quiet = type("Logger", (), {"warning_once": staticmethod(lambda *a, **k: None)})()
        env = v11_functions("utils/math_utils.py", {"round_up"}, {})
        env = v11_functions("model_executor/layers/quantization/utils/marlin_utils.py",
                            {"marlin_padded_nk"}, {**env, "math": __import__("math"),
                                                   "logger": quiet})
        gemms = bench.build_gemms(CKPT, DRAFT_CKPT, 2, 0)
        self.assertEqual({m["group"] for m in gemms}, set(bench.GROUPS))
        padded = set()
        for m in gemms:
            for gs in (64, 128):
                self.assertEqual(m["k"] % gs, 0, m["name"])
                pn, pk = env["marlin_padded_nk"](m["n"], m["k"], gs)
                self.assertEqual(pk, m["k"], m["name"])
                if pn != m["n"]:
                    padded.add((m["group"], m["name"], m["n"], pn))
        self.assertEqual(padded, {("kda_in", "in_proj_qkvbfg_a", 12576, 12608)})


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
        env = {"GLM53_INT8_W8A16": "kda_in,lm_head", "GLM53_INT4_W4A16": "draft",
               "GLM53_FP8_W8A16": "shared"}
        self.assertEqual(self.m.selected_modes(env), {
            "kda_in": "int8", "lm_head": "int8", "draft": "int4", "shared": "fp8"})
        with self.assertRaisesRegex(ValueError, "mla set in both GLM53_NVFP4_W4A16 and "
                                                "GLM53_INT8_W8A16"):
            self.m.selected_modes({"GLM53_INT8_W8A16": "mla", "GLM53_NVFP4_W4A16": "mla"})
        with self.assertRaisesRegex(ValueError, "both GLM53_INT8_W8A16 and GLM53_INT4_W4A16"):
            self.m.selected_modes({"GLM53_INT8_W8A16": "draft", "GLM53_INT4_W4A16": "draft"})

    def test_int_group_size(self):
        self.assertEqual(self.m.int_group_size({}), 128)
        self.assertEqual(self.m.int_group_size({"GLM53_INT_GROUP_SIZE": " 64 "}), 64)
        for bad in ("32", "256", "-1", "abc"):
            with self.assertRaisesRegex(ValueError, "GLM53_INT_GROUP_SIZE"):
                self.m.int_group_size({"GLM53_INT_GROUP_SIZE": bad})

    def test_dequant_groups(self):
        env = "GLM53_WQ_DEQUANT_GROUPS"
        for unset in ({}, {env: ""}, {env: " , "}):
            self.assertEqual(self.m.dequant_groups(unset), ["kda_in"])
        self.assertEqual(self.m.dequant_groups({env: " kda_o, kda_in,kda_o "}), ["kda_o", "kda_in"])
        self.assertEqual(self.m.dequant_groups({env: ",".join(self.m.GROUPS)}), list(self.m.GROUPS))
        for bad in ("kda", "kda_in,attn", "KDA_IN"):
            with self.assertRaisesRegex(ValueError, f"{env}: unknown group"):
                self.m.dequant_groups({env: bad})

    def test_int_pack_layout_matches_vllm_pack_rows(self):
        """qweight is vLLM's GPTQ layout: pack_rows of the biased ints, which
        MarlinLinearKernel hands to gptq_marlin_repack."""
        env = v11_functions("model_executor/layers/quantization/utils/quant_utils.py",
                            {"get_pack_factor", "pack_rows"}, {"numpy": np, "torch": torch})
        w = self.weight(n=192, k=512, seed=3)
        for bits in (8, 4):
            for gs in (64, 128):
                qweight, scale = self.m.quantize_int(w, bits, gs)
                self.assertEqual((qweight.dtype, tuple(qweight.shape)),
                                 (torch.int32, (512 * bits // 32, 192)))
                self.assertEqual((scale.dtype, tuple(scale.shape)), (torch.bfloat16, (512 // gs, 192)))
                s = scale.float().repeat_interleave(gs, 0)  # (K, N)
                q = torch.round(w.float().T / s).clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
                ref = env["pack_rows"](q.int() + 2 ** (bits - 1), bits, 512, 192)
                self.assertTrue(torch.equal(qweight, ref), (bits, gs))

    def test_int8_roundtrip_bound(self):
        w = self.weight()
        for gs in (64, 128):
            qweight, scale = self.m.quantize_int(w, 8, gs)
            deq = self.m.dequantize_int(qweight, scale, 8)
            s = scale.float().repeat_interleave(gs, 0).T
            err = (deq - w.float()).abs()
            # Half a step; the BF16-rounded scale can clip amax by < 127 * 2^-9 steps.
            self.assertTrue((err <= 0.5 * s * (1 + 2**-7)).all(), float((err / s).max()))
            self.assertTrue((deq[9] == 0).all())
            amax = w.float().abs().unflatten(1, (-1, gs)).amax(-1).T
            self.assertTrue(torch.allclose(scale.float(), amax / 127, rtol=2**-8, atol=1e-30))
            q = (qweight.unsqueeze(1) >> torch.arange(0, 32, 8).view(1, -1, 1)) & 255
            self.assertTrue(((q >= 1) & (q <= 255)).all())  # symmetric: -128 unused
        # Gaussian weights (no outlier row): INT8 g128 ~0.0066, FP8 per-channel ~0.026.
        g = (torch.randn(256, 1024, generator=torch.Generator().manual_seed(4)) * 0.02)
        g = g.to(torch.bfloat16)
        rel = lambda d: float((d - g.float()).norm() / g.float().norm())  # noqa: E731
        int8 = rel(self.m.dequantize_int(*self.m.quantize_int(g, 8, 128), 8))
        fp8 = rel(self.m.dequantize_per_channel(*self.m.quantize_per_channel(g)))
        self.assertLess(int8, 0.008)
        self.assertLess(int8, 0.3 * fp8)

    def test_int4_clip_search_never_loses_to_amax(self):
        w = self.weight(seed=2)
        base = self.m.dequantize_int(*self.m.quantize_int(w, 4, 128), 4)
        qweight, scale = self.m.quantize_int(w, 4, 128, self.m.INT_CLIP_RATIOS["int4"])
        best = self.m.dequantize_int(qweight, scale, 4)
        blk = lambda d: (d - w.float()).square().unflatten(1, (-1, 128)).sum(-1)  # noqa: E731
        self.assertTrue((blk(best) <= blk(base)).all())
        self.assertLess(float(blk(best).sum()), float(blk(base).sum()))
        q = (qweight.unsqueeze(1) >> torch.arange(0, 32, 4).view(1, -1, 1)) & 15
        self.assertTrue((q == 0).any())  # clipping reaches -8

    def test_int_chunking_is_exact(self):
        w = self.weight(n=257, k=256, seed=1)
        for bits, ratios in ((8, (1.0,)), (4, self.m.INT_CLIP_RATIOS["int4"])):
            a = self.m.quantize_int(w, bits, 64, ratios)
            for chunk in (3 * 256, 1):
                b = self.m.quantize_int(w, bits, 64, ratios, chunk_elems=chunk)
                self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
        with self.assertRaises(ValueError):
            self.m.quantize_int(torch.zeros(4, 192, dtype=torch.bfloat16), 8, 128)

    def test_compact_reallocates_after_empty_cache(self):
        from unittest import mock

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.int32), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.ones(3), requires_grad=False)
        param, ptrs = layer.weight, [layer.weight.data_ptr(), layer.weight_scale.data_ptr()]
        calls = []
        with mock.patch.object(torch.cuda, "empty_cache",
                               side_effect=lambda: calls.append(layer.weight.data_ptr())):
            self.m._compact(layer)
        self.assertEqual(calls, [ptrs[0]])  # emptied before the copies
        self.assertIs(layer.weight, param)
        self.assertNotEqual(layer.weight.data_ptr(), ptrs[0])
        self.assertNotEqual(layer.weight_scale.data_ptr(), ptrs[1])
        self.assertTrue(torch.equal(layer.weight, torch.arange(12, dtype=torch.int32)))

    @unittest.skipIf(np is None, "numpy not importable")
    @unittest.skipUnless(CKPT.is_dir(), f"checkpoint not at {CKPT}")
    def test_real_checkpoint_int_error(self):
        """Relative Frobenius error, first REAL_ROWS rows of one tensor per group.
        Measured 2026-09-28 on 09b04e5: INT8 g128 0.0066-0.0073 (FP8 per-channel
        0.024-0.029), INT4 g128 amax 0.120-0.137, clip search 0.104-0.119 (NVFP4
        0.083-0.086): uniform INT4 does not beat NVFP4."""
        headers = read_headers(CKPT)
        for group, name in REAL_TENSORS.items():
            w = torch.from_numpy(read_rows(headers, name, REAL_ROWS)).to(torch.bfloat16)
            rel = lambda d: float((d - w.float()).norm() / w.float().norm())  # noqa: E731
            e8 = rel(self.m.dequantize_int(*self.m.quantize_int(w, 8, 128), 8))
            e4 = rel(self.m.dequantize_int(*self.m.quantize_int(w, 4, 128), 4))
            e4c = rel(self.m.dequantize_int(
                *self.m.quantize_int(w, 4, 128, self.m.INT_CLIP_RATIOS["int4"]), 4))
            print(f"\n  {group:8} {name} {tuple(w.shape)}: int8 {e8:.4f}, int4 {e4:.4f}, "
                  f"int4 clip {e4c:.4f}", end="")
            self.assertLess(e8, 0.008, name)
            self.assertLess(e4c, 0.12, name)
            self.assertLess(e4c, e4 * 0.95, name)

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


def positive_bf16(shape, gen):
    """Positive BF16 scales spread over 2^-20 .. 2^4."""
    e = torch.randint(-20, 4, shape, generator=gen).float()
    return (torch.exp2(e) * (1 + torch.rand(shape, generator=gen))).to(torch.bfloat16)


@unittest.skipUnless(SRC.is_dir(), "set GLM53_V11_SRC to the v11 vLLM source")
@unittest.skipIf(torch is None or np is None or triton is None, "needs torch, numpy, triton")
class DequantTest(unittest.TestCase):
    """GLM53_WQ_DEQUANT_MIN_M. The Triton dequant runs in the CPU interpreter
    on Marlin tensors that v11's Python references build: marlin_weights is
    what vLLM's own repack test holds gptq_marlin_repack to, and the scale
    steps are the functions prepare_*_for_marlin call. Every element must be
    the BF16 rounding of the patch's reference dequant, bit for bit (FP8 and
    NVFP4 up to the sign of zero)."""

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
        with mock.patch.dict(os.environ, {"TRITON_INTERPRET": "1"}):
            cls.m = load_module(root)
        quiet = types.SimpleNamespace(warning_once=lambda *a, **k: None)
        v = {"np": np, "torch": torch, "math": math, "logger": quiet, "GPTQ_MARLIN_TILE": 16}
        utils = "model_executor/layers/quantization/utils/"
        for rel, names in (
            ("utils/math_utils.py", {"round_up"}),
            (utils + "quant_utils.py", {"get_pack_factor"}),
            (utils + "marlin_utils.py", {"get_scale_perms", "marlin_permute_scales",
                                         "marlin_padded_nk", "marlin_pad_qweight",
                                         "marlin_pad_scales"}),
            (utils + "marlin_utils_test.py", {"marlin_permute_weights", "marlin_weights",
                                              "get_weight_perm"}),
            (utils + "marlin_utils_fp8.py", {"fp8_fused_exponent_bias_into_scales",
                                             "pack_fp8_to_int32"}),
            (utils + "marlin_utils_fp4.py", {"_nvfp4_compute_scale_factor",
                                             "nvfp4_marlin_process_scales",
                                             "nvfp4_marlin_process_global_scale"}),
        ):
            v11_functions(rel, names, v)
        cls.v = v
        cls.perms = mock.patch.object(
            cls.m, "_marlin_perms", lambda: (v["get_weight_perm"], v["get_scale_perms"]()))
        cls.perms.start()

    @classmethod
    def tearDownClass(cls):
        cls.perms.stop()
        cls.tmp.cleanup()

    def setUp(self):
        self.m._WORKSPACE = None

    def repack(self, gptq, n, k, pn, pk, bits):
        """gptq_marlin_repack on CPU: marlin_weights of the padded, unpacked GPTQ matrix."""
        qw = self.v["marlin_pad_qweight"](gptq, n, k, pn, pk)
        shifts = torch.arange(0, 32, bits, dtype=torch.int32)
        q = ((qw.unsqueeze(1) >> shifts.view(1, -1, 1)) & (2**bits - 1)).flatten(0, 1)
        return self.v["marlin_weights"](q, pk, pn, bits, self.v["get_weight_perm"](bits))

    def layer(self, mode, parts, n, k, gs=None):
        """A swapped layer, built with the steps of the patch's swap functions."""
        v, layer = self.v, torch.nn.Module()
        layer.output_size_per_partition, layer.input_size_per_partition = n, k
        if mode in ("int8", "int4"):
            qweight, scale = parts
            pn, pk = v["marlin_padded_nk"](n, k, gs)
            layer.weight = self.repack(qweight, n, k, pn, pk, self.m.INT_BITS[mode])
            s = v["marlin_pad_scales"](scale, n, k, pn, pk, gs)
            layer.weight_scale = v["marlin_permute_scales"](s, pk, pn, gs)
        elif mode == "fp8":  # prepare_fp8_layer_for_marlin(size_k_first=False)
            q, scale = parts
            pn, pk = v["marlin_padded_nk"](n, k, -1)
            gptq = v["pack_fp8_to_int32"](q, False).T.contiguous()
            layer.weight = self.repack(gptq, n, k, pn, pk, 8)
            s = v["marlin_pad_scales"](scale.view(1, n), n, k, pn, pk, -1)
            s = v["marlin_permute_scales"](s, pk, pn, -1)
            layer.weight_scale = v["fp8_fused_exponent_bias_into_scales"](s)
        else:  # prepare_fp4_layer_for_marlin, params_dtype BF16
            packed, scale, g = parts
            pn, pk = v["marlin_padded_nk"](n, k, 16)
            layer.weight = self.repack(packed.view(torch.int32).T.contiguous(), n, k, pn, pk, 4)
            s = v["marlin_pad_scales"](scale.T.contiguous().to(torch.bfloat16), n, k, pn, pk, 16)
            s = v["marlin_permute_scales"](s, pk, pn, 16)
            layer.weight_scale, sf = v["nvfp4_marlin_process_scales"](s, a_dtype=torch.bfloat16)
            glob = v["nvfp4_marlin_process_global_scale"](g.float(), torch.bfloat16)
            layer.weight_global_scale = glob / sf
        return layer

    def check(self, mode, parts, ref, n, k, gs=None):
        layer = self.layer(mode, parts, n, k, gs)
        out = torch.empty(n, k, dtype=torch.bfloat16)
        self.m.dequantize_marlin(layer, self.m.dequant_spec(layer, mode, gs), out)
        ref = ref.to(torch.bfloat16)
        if mode in ("fp8", "nvfp4"):
            # E6: in the v11 image the kernel's -w writes +0 for the -0 codes, so compare values here.
            out, ref = out + 0.0, ref + 0.0
        bad = out.view(torch.int16) != ref.view(torch.int16)
        self.assertFalse(bad.any(), f"{mode} g{gs} {n}x{k}: {int(bad.sum())} elements differ")
        return layer

    def test_tables_invert_the_permutations(self):
        v = self.v
        for bits in (8, 4):
            perm = v["get_weight_perm"](bits)
            tile, _ = self.m.marlin_dequant_tables(perm, list(range(64)))
            j, kk = torch.arange(64), torch.arange(16).unsqueeze(1)
            self.assertTrue(torch.equal(perm[tile.long()], j // 16 * 256 + kk * 16 + j % 16))
        scale_perm, single = v["get_scale_perms"]()
        for sp, nvfp4 in ((scale_perm, False), (single, False), (scale_perm, True)):
            _, scol = self.m.marlin_dequant_tables(v["get_weight_perm"](8), sp, nvfp4)
            cols = torch.arange(len(sp))
            permuted = cols[sp]  # marlin_permute_scales on one chunk
            if nvfp4:
                permuted = permuted.view(-1, 4)[:, [0, 2, 1, 3]].flatten()
            self.assertTrue(torch.equal(permuted[scol.long()], cols))

    def test_int_matches_reference(self):
        """INT8 g128 (the recipe default), INT8 g64 and INT4, with and without N padding."""
        for n, k in ((300, 512), (128, 384)):
            w = torch.from_numpy(synthetic_weight(n, k, seed=n)).to(torch.bfloat16)
            for mode, gs in (("int8", 128), ("int8", 64), ("int4", 128)):
                bits = self.m.INT_BITS[mode]
                parts = self.m.quantize_int(w, bits, gs, self.m.INT_CLIP_RATIOS[mode])
                self.check(mode, parts, self.m.dequantize_int(*parts, bits), n, k, gs)

    def test_fp8_matches_reference(self):
        """Per-channel FP8; K = 80 makes Marlin pad K to 128."""
        for n, k in ((300, 512), (192, 80)):
            w = torch.from_numpy(synthetic_weight(n, k, seed=k)).to(torch.bfloat16)
            q, s = self.m.quantize_per_channel(w)
            self.check("fp8", (q, s), self.m.dequantize_per_channel(q, s), n, k)

    def test_nvfp4_matches_reference(self):
        w = torch.from_numpy(synthetic_weight()).to(torch.bfloat16)
        parts = self.m.quantize_nvfp4(w)
        self.check("nvfp4", parts, self.m.dequantize_nvfp4(*parts), 300, 512)

    def test_every_code_matches_reference(self):
        """Random packed values cover every INT byte and nibble, every finite
        e4m3 code (subnormals and -0 too), and every E2M1 nibble with block
        scales small enough that Marlin rescales them (sf > 1). A crafted
        NVFP4 block pins the fp32 rounding order."""
        gen = torch.Generator().manual_seed(7)
        n, k = 320, 256
        for mode, gs in (("int8", 128), ("int4", 64)):
            bits = self.m.INT_BITS[mode]
            qweight = torch.randint(-(2**31), 2**31, (k * bits // 32, n), generator=gen)
            qweight = qweight.to(torch.int32)
            scale = positive_bf16((k // gs, n), gen)
            ref = self.m.dequantize_int(qweight, scale, bits)
            self.check(mode, (qweight, scale), ref, n, k, gs)
        codes = torch.randint(0, 256, (n, k), generator=gen, dtype=torch.int32)
        codes = torch.where((codes & 0x7F) == 0x7F, codes - 1, codes).to(torch.uint8)  # no NaN
        q, s = codes.view(torch.float8_e4m3fn), positive_bf16((n,), gen)
        self.check("fp8", (q, s), self.m.dequantize_per_channel(q, s), n, k)
        packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.int32)
        packed = packed.to(torch.uint8)
        bcodes = torch.randint(8, 0x5F, (n, k // 16), generator=gen, dtype=torch.int32)
        bscale = bcodes.to(torch.uint8).view(torch.float8_e4m3fn)
        g = torch.tensor(3.1e-4)
        ref = self.m.dequantize_nvfp4(packed, bscale, g)
        layer = self.check("nvfp4", (packed, bscale, g), ref, n, k)
        self.assertLess(float(layer.weight_global_scale), float(g) * 2.0**119)  # sf > 1
        # Rounding order: find a block scale and global scale for which
        # 3 * (bs * g) and (3 * bs) * g round to different BF16 values; the
        # kernel must follow dequantize_nvfp4's order.
        bs = torch.arange(8, 0x7F, dtype=torch.int32).to(torch.uint8)
        bsf = bs.view(torch.float8_e4m3fn).float()
        gscales = torch.rand(4096, generator=gen) * 1e-3 + 1e-4
        a = (3.0 * (bsf * gscales[:, None])).bfloat16()
        b = ((3.0 * bsf) * gscales[:, None]).bfloat16()
        i, j = torch.nonzero(a != b)[0].tolist()
        packed = torch.full((64, 64), 0x55, dtype=torch.uint8)  # E2M1 code 5 = 3.0
        bscale = bs[j].repeat(64, 8).view(torch.float8_e4m3fn)
        g = gscales[i].clone()
        ref = self.m.dequantize_nvfp4(packed, bscale, g)
        self.assertTrue((ref.bfloat16() == a[i, j]).all() and (ref.bfloat16() != b[i, j]).all())
        self.check("nvfp4", (packed, bscale, g), ref, 64, 128)

    def test_apply_switches_at_min_m(self):
        m, F = self.m, torch.nn.functional
        w = torch.from_numpy(synthetic_weight(128, 256)).to(torch.bfloat16)
        parts = m.quantize_int(w, 8, 128)
        ref = m.dequantize_int(*parts, 8).to(torch.bfloat16)
        layer = self.layer("int8", parts, 128, 256, 128)
        inner = layer.quant_method = mock.Mock()
        self.assertEqual(m.install_marlin_dequant([(layer, "int8")], 4, 128), 128 * 256 * 2)
        gen = torch.Generator().manual_seed(1)
        x3 = torch.randn(3, 256, generator=gen).to(torch.bfloat16)
        self.assertIs(layer.quant_method.apply(layer, x3), inner.apply.return_value)
        inner.apply.assert_called_once_with(layer, x3, None)
        x4 = torch.randn(2, 2, 256, generator=gen).to(torch.bfloat16)  # 4 rows
        bias = torch.randn(128, generator=gen).to(torch.bfloat16)
        self.assertTrue(torch.equal(layer.quant_method.apply(layer, x4), F.linear(x4, ref)))
        self.assertTrue(torch.equal(layer.quant_method.apply(layer, x4, bias),
                                    F.linear(x4, ref, bias)))
        self.assertEqual(inner.apply.call_count, 1)
        with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
            layer.quant_method.apply(layer, x4)
        self.assertEqual(inner.apply.call_count, 2)

    def test_one_workspace_sized_for_the_largest_layer(self):
        m = self.m
        layers, refs = [], []
        for n, k in ((128, 256), (192, 512), (256, 512)):
            parts = m.quantize_int(torch.from_numpy(synthetic_weight(n, k)).bfloat16(), 8, 128)
            layers.append(self.layer("int8", parts, n, k, 128))
            layers[-1].quant_method = mock.Mock()
            refs.append(m.dequantize_int(*parts, 8).to(torch.bfloat16))
        small, big, bigger = layers
        self.assertEqual(m.install_marlin_dequant([(small, "int8"), (big, "int8")], 64, 128),
                         192 * 512 * 2)
        ws = m._WORKSPACE
        for layer, ref in zip(layers[:2], refs):
            w = m.dequantize_marlin(layer, layer.quant_method._spec)
            self.assertEqual(w.data_ptr(), ws.data_ptr())
            self.assertTrue(torch.equal(w, ref))
        # A later model (the drafter) grows it once; a smaller one keeps it.
        m.install_marlin_dequant([(small, "int8")], 64, 128)
        self.assertIs(m._WORKSPACE, ws)
        m.install_marlin_dequant([(bigger, "int8")], 64, 128)
        self.assertEqual(m._WORKSPACE.numel(), 256 * 512)

    def test_env_gates_the_wrapper_and_skips_lm_head(self):
        """apply_glm53_fp8_w8a16 on a fake target: MIN_M unset leaves every
        Marlin method as it was, whatever GLM53_WQ_DEQUANT_GROUPS says; set, it
        wraps the swapped layers of the listed groups (default kda_in) but never
        the LM head, and the workspace fits the largest wrapped layer only; a
        bad value of either variable fails before any swap."""
        m, nn = self.m, torch.nn
        plain = type("UnquantizedLinearMethod", (), {})
        fakes = {name: types.ModuleType(name) for name in (
            "vllm", "vllm.logger", "vllm.model_executor", "vllm.model_executor.layers",
            "vllm.model_executor.layers.linear",
            "vllm.model_executor.layers.vocab_parallel_embedding")}
        fakes["vllm.logger"].init_logger = logging.getLogger
        linear_mod = fakes["vllm.model_executor.layers.linear"]
        linear_mod.LinearBase, linear_mod.UnquantizedLinearMethod = nn.Linear, plain
        vocab_mod = fakes["vllm.model_executor.layers.vocab_parallel_embedding"]
        vocab_mod.UnquantizedEmbeddingMethod = plain

        def linear(n, k, cls=nn.Module):
            layer = cls()
            layer.weight = torch.randn(n, k).to(torch.bfloat16)
            layer.quant_method = plain()
            return layer

        def model():
            kda = type("Glm5NextLinearAttention", (nn.Module,), {})()
            kda.in_proj_qkvbfg_a, kda.o_proj = linear(192, 256), linear(128, 128)
            target = type("Glm5NextForConditionalGeneration", (nn.Module,), {})()
            target.layers = nn.ModuleList([kda])
            target.lm_head = linear(512, 128, type("ParallelLMHead", (nn.Module,), {}))
            return target, [kda.in_proj_qkvbfg_a, kda.o_proj], target.lm_head

        swaps = []

        def swap(layer, mode, gs):  # quantize_layer_to_marlin_int, built on CPU
            swaps.append(layer)
            n, k = layer.weight.shape
            built = self.layer(mode, m.quantize_int(layer.weight, 8, gs), n, k, gs)
            layer.weight, layer.weight_scale = built.weight, built.weight_scale
            layer.output_size_per_partition, layer.input_size_per_partition = n, k
            layer.quant_method = ("marlin", n)

        clean = {k: v for k, v in os.environ.items() if not k.startswith("GLM53_")}
        groups = {"GLM53_INT8_W8A16": "kda_in,kda_o,lm_head"}

        def run(extra):
            target, kda, head = model()
            m._WORKSPACE = None
            swaps.clear()
            with mock.patch.dict(sys.modules, fakes), \
                    mock.patch.dict(os.environ, {**clean, **groups, **extra}, clear=True), \
                    mock.patch.object(m, "quantize_layer_to_marlin_int", swap), \
                    mock.patch.object(torch.cuda, "memory_reserved", return_value=0):
                m.apply_glm53_fp8_w8a16(target, torch.device("cuda"))
            return kda, head

        def wrapped(layers):
            """min_m per layer, or None where the Marlin method is untouched."""
            out = []
            for layer in layers:
                qm = layer.quant_method
                if isinstance(qm, m.MarlinDequantMethod):
                    self.assertEqual(qm._inner, ("marlin", layer.output_size_per_partition))
                    out.append(qm._min_m)
                else:
                    self.assertEqual(qm, ("marlin", layer.output_size_per_partition))
                    out.append(None)
            return out

        for off in ({}, {"GLM53_WQ_DEQUANT_GROUPS": "kda_in,kda_o"},
                    {"GLM53_WQ_DEQUANT_GROUPS": "attn"}, {"GLM53_WQ_DEQUANT_MIN_M": ""}):
            with self.subTest(off=off):
                kda, head = run(off)
                self.assertEqual(wrapped(kda + [head]), [None, None, None])
                self.assertIsNone(m._WORKSPACE)
        # kda_in 192 x 256, kda_o 128 x 128, LM head 512 x 128 (never wrapped)
        for dq_groups, want, numel in ((None, [256, None], 192 * 256),
                                       ("kda_in", [256, None], 192 * 256),
                                       ("kda_o", [None, 256], 128 * 128),
                                       ("kda_o,lm_head,draft", [None, 256], 128 * 128),
                                       ("kda_in,kda_o,lm_head", [256, 256], 192 * 256)):
            with self.subTest(dq_groups=dq_groups):
                extra = {"GLM53_WQ_DEQUANT_MIN_M": " 256 "}
                if dq_groups is not None:
                    extra["GLM53_WQ_DEQUANT_GROUPS"] = dq_groups
                kda, head = run(extra)
                self.assertEqual(wrapped(kda + [head]), want + [None])
                self.assertEqual(m._WORKSPACE.numel(), numel)
        for bad in ({"GLM53_WQ_DEQUANT_MIN_M": v} for v in ("0", "-4", "1e3", "abc")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "GLM53_WQ_DEQUANT_MIN_M"):
                run(bad)
            self.assertEqual(swaps, [])
        for bad in ("kda", "kda_in,attn"):
            env = {"GLM53_WQ_DEQUANT_MIN_M": "256", "GLM53_WQ_DEQUANT_GROUPS": bad}
            with self.subTest(bad=bad), \
                    self.assertRaisesRegex(ValueError, "GLM53_WQ_DEQUANT_GROUPS: unknown"):
                run(env)
            self.assertEqual(swaps, [])

    def test_v11_runs_large_m_eagerly(self):
        """Why a Python branch on the row count is safe in v11. GLM-5.3 auto-
        enables breakable CUDA graphs, which turns torch.compile off for the
        target and the drafter. A batch above the largest capture size (16 in
        run.sh) dispatches to NONE and runs eagerly; smaller ones replay at a
        padded capture size, mixed or not. DFlash projects the context through
        fc outside the drafter's graph. The shared experts' aux stream (<= 256
        rows) is ordered against the main stream both ways."""
        cfg = (SRC / "config/vllm.py").read_text()
        auto = cfg[cfg.index("# For model classes don't carry @support_torch_compile"):]
        auto = auto[: auto.index("breakable_cudagraph_enabled = ")]
        self.assertIn('"Glm5NextForConditionalGeneration",', auto)
        self.assertIn('os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"', auto)
        self.assertIn("if breakable_cudagraph_enabled:\n"
                      "            self.compilation_config.mode = CompilationMode.NONE", cfg)
        disp = (SRC / "v1/cudagraph_dispatcher.py").read_text()
        self.assertIn("or num_tokens > max_size", disp)
        self.assertIn("num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]", disp)
        brk = (SRC / "compilation/breakable_cudagraph.py").read_text()
        self.assertIn("if cudagraph_runtime_mode == CUDAGraphMode.NONE:\n"
                      "            return self.runnable(*args, **kwargs)", brk)
        runner = (SRC / "v1/worker/gpu_model_runner.py").read_text()
        self.assertIn("drafter.model = BreakableCUDAGraphWrapper(", runner)
        dflash = (SRC / "model_executor/models/qwen3_dflash.py").read_text()
        self.assertIn("result = self.model.fc(hidden_states)", dflash)
        shared = (SRC / "model_executor/layers/fused_moe/runner/shared_experts.py").read_text()
        for needle in ("<= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
                       "self._stream.wait_stream(current_stream())",
                       "current_stream().wait_stream(self._stream)"):
            self.assertIn(needle, shared)


if __name__ == "__main__":
    unittest.main()
