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
            ast.unparse(last.test), "_os.environ.get('GLM53_FP8_W8A16', '').strip()"
        )
        self.assertIn("apply_glm53_fp8_w8a16(model, target_device)", ast.unparse(last))

    def test_compile_factor_only_when_set(self):
        text = (self.root / "envs.py").read_text()
        self.assertEqual(text.count('factors["GLM53_FP8_W8A16"]'), 1)
        self.assertIn('if os.getenv("GLM53_FP8_W8A16", "").strip():', text)

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


def load_module(root: Path):
    spec = importlib.util.spec_from_file_location("glm53_fp8_w8a16", root / MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


if __name__ == "__main__":
    unittest.main()
