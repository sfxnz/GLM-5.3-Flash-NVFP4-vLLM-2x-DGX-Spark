"""CPU tests for docker/patch_v15_kda_recover.py (GLM53_KDA_RECOVER).

    GLM53_V14_SRC=/path/to/v14src python3 docker/test_v15_kda_recover.py

GLM53_V14_SRC is the directory that holds the vllm/ package copied out of
glm53-sm121-v14 (read-only; the tests patch temporary copies). Without it
only the plan tests run. The interpreter tests also need torch and triton.
They run v14's own fused_recurrent_gated_delta_rule_fwd_kernel and its
fused_recurrent_kda wrappers (lifted unchanged from v14's ops/kda.py) under
the Triton CPU interpreter (TRITON_INTERPRET=1), once with GLM53_KDA_TRIM=0
and once with 1, next to the shipped module. Each compares the stock verify
(per-token state stores, initial state at column num_accepted - 1) with the
recover path (verify without stores, records, commit replay) bit for bit:

  single step  every accepted count 0..7, mamba cache mode none and align
               (block 12, so the 1152-token boundary case recurs), padded
               FULL-graph rows, a mixed batch whose spec rows are not the
               batch rows, and a row with an idx_mapping of -1. Outputs, the
               states a later step reads, and every other block untouched.
  chains       several steps with random acceptance, where the next step reads
               the column the last one left. Align mode runs the V2 pre-copy
               and post-copy rules (preprocess_mamba_align_fused_kernel,
               postprocess_mamba_fused_kernel) on both pools; outputs and
               every cached boundary block must match at each step.
"""

import ast
import importlib.util
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patch_v15_kda_recover as P  # noqa: E402

SRC = Path(os.environ.get("GLM53_V14_SRC", "/nonexistent")) / "vllm"
HAVE_SRC = (SRC / "__init__.py").is_file()
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_TRITON = importlib.util.find_spec("triton") is not None
FLA = "third_party/flash_linear_attention/ops"


class PlanEditsTest(unittest.TestCase):
    def test_refuses_missing_anchor(self):
        with self.assertRaises(SystemExit):
            P.plan_edits("nothing here", [("x", "anchor", "new")])

    def test_already_applied_is_noop(self):
        self.assertEqual(P.plan_edits("a new b", [("x", "old", "new")]), ("a new b", 0))

    def test_module_parses(self):
        ast.parse(P.MODULE)


@unittest.skipUnless(HAVE_SRC, "GLM53_V14_SRC not set")
class SourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v15kr-src-"))
        self.root = self.tmp / "vllm"
        for rel in P.FILE_EDITS:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(SRC / rel, self.root / rel)
        (self.root / P.MODULE_REL).parent.mkdir(parents=True, exist_ok=True)
        (self.root / "__init__.py").touch()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_every_anchor_once(self):
        for rel, edits in P.FILE_EDITS.items():
            src = (self.root / rel).read_text()
            for what, old, _new in edits:
                self.assertEqual(src.count(old), 1, what)

    def test_apply_twice_and_compile(self):
        self.assertEqual(len(P.main(self.root)), 4)
        self.assertEqual(P.main(self.root), [])
        for rel in [*P.FILE_EDITS, P.MODULE_REL]:
            py_compile.compile(str(self.root / rel), doraise=True)

    def test_switch_default_off_keeps_v14_call(self):
        P.main(self.root)
        gdn = (self.root / "v1/attention/backends/gdn_attn.py").read_text()
        hybrid = (self.root / "v1/worker/gpu/model_states/mamba_hybrid.py").read_text()
        kda = (self.root / "models/glm5next/nvidia/kda.py").read_text()
        flag = 'os.environ.get("GLM53_KDA_RECOVER") == "1"'
        self.assertIn(flag, gdn)
        self.assertIn(flag, hybrid)
        self.assertIn(flag, P.MODULE)
        # Off: no builder state, so no metadata field is set and kda.py passes
        # v14's index arguments; the model state never keeps a step.
        self.assertIn("self.glm53_rs = None\n        if _GLM53_KDA_RECOVER:", gdn)
        self.assertIn("spec_state_indices_tensor if rs_index is None else rs_index", kda)
        self.assertIn("num_accepted_tokens\n                    if rs_index is None", kda)
        self.assertIn("if self._glm53_kda_recover:\n            self._glm53_rs_step", hybrid)
        # The conv update keeps the real num_accepted_tokens.
        self.assertIn(
            "conv_state_indices=conv_idx,\n                num_accepted_tokens=num_accepted_tokens,",
            kda,
        )

    def test_commit_runs_before_align_postprocess(self):
        P.main(self.root)
        hybrid = (self.root / "v1/worker/gpu/model_states/mamba_hybrid.py").read_text()
        body = hybrid[hybrid.index("    def postprocess_state(") :]
        self.assertLess(body.index("commit_step("), body.index("run_fused_postprocess_align("))

    @unittest.skipUnless((HERE / "patch_v15_fixes.py").is_file(), "no patch_v15_fixes.py")
    def test_commutes_with_v15_fixes(self):
        spec = importlib.util.spec_from_file_location("patch_v15_fixes", HERE / "patch_v15_fixes.py")
        fixes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixes)
        results = []
        for order in ((P, fixes), (fixes, P)):
            with tempfile.TemporaryDirectory(prefix="v15kr-both-") as tmp:
                root = Path(tmp) / "vllm"
                for rel in {*P.FILE_EDITS, *fixes.FILE_EDITS, *getattr(fixes, "NEW_FILES", {})}:
                    if (SRC / rel).is_file():
                        (root / rel).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(SRC / rel, root / rel)
                (root / P.MODULE_REL).parent.mkdir(parents=True, exist_ok=True)
                (root / "__init__.py").touch()
                for patch in order:
                    patch.main(root)
                for patch in order:
                    self.assertEqual(patch.main(root), [], f"{patch.__name__} rerun wrote files")
                hybrid = root / "v1/worker/gpu/model_states/mamba_hybrid.py"
                py_compile.compile(str(hybrid), doraise=True)
                results.append(hybrid.read_text())
        self.assertEqual(results[0].count("import os\n"), 1)
        self.assertEqual(results[1].count("import os\n"), 1)

    def test_refuses_drift(self):
        path = self.root / "models/glm5next/nvidia/kda.py"
        path.write_text(
            path.read_text().replace(
                "ssm_state_indices=spec_state_indices_tensor,", "ssm_state_indices=x,"
            )
        )
        with self.assertRaises(SystemExit):
            P.main(self.root)

    def test_refuses_foreign_module(self):
        (self.root / P.MODULE_REL).write_text("x = 1\n")
        with self.assertRaises(SystemExit):
            P.main(self.root)


# ---------------------------------------------------------------------------
# Triton CPU interpreter: v14's recurrent kernel and wrappers + the module
# ---------------------------------------------------------------------------
STUBS = {
    "vllm/__init__.py": "",
    "vllm/triton_utils/__init__.py": "import triton\nimport triton.language as tl\n",
    "vllm/utils/__init__.py": "",
    "vllm/utils/torch_utils.py": (
        "def async_tensor_h2d(data, device, dtype=None):\n"
        "    return data.to(device=device, dtype=dtype)\n"
    ),
    "vllm/third_party/__init__.py": "",
    "vllm/third_party/flash_linear_attention/__init__.py": "",
    f"vllm/{FLA}/__init__.py": "",
    # ops/op.py with FLA_USE_FAST_OPS unset (exp = tl.exp, log = tl.log); the
    # real file pulls in the platform layer. The interpreter patches tl.* but
    # not module-level aliases, so the aliases are jit wrappers here. Both
    # paths under test import the same names.
    f"vllm/{FLA}/op.py": textwrap.dedent(
        """\
        from vllm.triton_utils import tl, triton


        @triton.jit
        def exp(x):
            return tl.exp(x)


        @triton.jit
        def log(x):
            return tl.log(x)
        """
    ),
    "vllm/models/__init__.py": "",
    "vllm/models/glm5next/__init__.py": "",
    "vllm/models/glm5next/nvidia/__init__.py": "",
    "vllm/models/glm5next/nvidia/ops/__init__.py": "",
}
KDA_WRAPPERS = ("_glm53_kda_trim_ok", "fused_recurrent_kda_fwd", "fused_recurrent_kda")


def lifted_kda_module(src: str) -> str:
    """v14's ops/kda.py reduced to its recurrent wrappers, unchanged."""
    tree = ast.parse(src)
    parts = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "_GLM53_KDA_TRIM" for t in node.targets
        ):
            parts.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.FunctionDef) and node.name in KDA_WRAPPERS:
            parts.append(ast.get_source_segment(src, node))
    assert len(parts) == 1 + len(KDA_WRAPPERS), "v14 ops/kda.py drifted"
    header = (
        "import os\n\nimport torch\n\n"
        "from vllm.triton_utils import tl, triton\n"
        "from vllm.utils.math_utils import cdiv, next_power_of_2\n\n"
        "from .fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel\n"
    )
    return header + "\n\n" + "\n\n\n".join(parts) + "\n"


SCRIPT = textwrap.dedent(
    r'''
    import random
    import sys
    from types import SimpleNamespace

    import torch

    sys.path.insert(0, sys.argv[1])
    from vllm.third_party.flash_linear_attention.ops import kda as fla_kda
    from vllm.models.glm5next.nvidia.ops import glm53_kda_recover as R

    H, K, Q, BLOCKS, LB = 2, 16, 8, 96, -5.0
    P = H * K
    WIDTH = 3 * P + H + 2 * K + 6  # the merged projection row: q|k|v|b|f_a|g_a
    BF = torch.bfloat16
    rng = random.Random(1234)
    torch.manual_seed(1234)
    assert fla_kda._GLM53_KDA_TRIM == (sys.argv[2] == "1")


    def make_layer():
        return SimpleNamespace(
            A_log=(torch.rand(1, 1, H, 1) * 0.5).float(),
            dt_bias=(torch.randn(P) * 0.1).float(),
            kda_lower_bound=LB,
            local_num_heads=H,
            head_dim=K,
            kv_cache=(None, None),
        )


    def make_inputs(t):
        projected = (torch.randn(t, WIDTH) * 1.5).to(BF)
        qkv = projected[:, : 3 * P]  # row stride WIDTH, as in kda.py
        beta = projected[:, 3 * P : 3 * P + H].unsqueeze(0)
        g = (torch.randn(1, t, H, K) * 2.0).to(BF)
        return qkv, g, beta


    def verify(layer, pool, qkv, g, beta, cu, idx, nacc, out):
        """kda.py's spec-path call, argument for argument."""
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
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
            a.view(torch.int32) if a.dtype == torch.float32 else a.view(torch.int16),
            b.view(torch.int32) if b.dtype == torch.float32 else b.view(torch.int16),
        )


    class Pair:
        """The stock path on pool A, the recover path on pool B."""

        def __init__(self, rows, tokens, align):
            self.layer = make_layer()
            self.rs = R.Recover([self.layer], Q, rows, tokens, align, BF, "cpu")
            self.pool0 = torch.randn(BLOCKS, H, K, K)
            self.a = self.pool0.clone()
            self.b = self.pool0.clone()

        def step(self, S, nacc, lens, mask_cpu=None):
            """One verify on both pools; returns the recover metadata."""
            t = sum(lens)
            qkv, g, beta = make_inputs(max(t, 1))
            qkv, g, beta = qkv[:t], g[:, :t], beta[:, :t]
            cu = torch.tensor([0] + [sum(lens[: i + 1]) for i in range(len(lens))], dtype=torch.int32)
            out_a = torch.zeros(1, t, H, K, dtype=BF)
            out_b = torch.zeros(1, t, H, K, dtype=BF)
            verify(self.layer, self.a, qkv, g, beta, cu, S, nacc, out_a)
            before = self.b.clone()
            R.store_records(self.layer.glm53_rs_records, qkv, g, beta, P)
            fields = self.rs.prepare(S, nacc, mask_cpu)
            verify(self.layer, self.b, qkv, g, beta, cu, fields["glm53_rs_verify_idx"], fields["glm53_rs_nacc"], out_b)
            assert same(out_a, out_b), "verify outputs differ"
            assert same(self.b, before), "recover verify stored a state"
            return SimpleNamespace(
                num_spec_decodes=len(lens),
                spec_state_indices_tensor=S,
                spec_query_start_loc=cu,
                **fields,
            )

        def commit(self, md, idx_mapping, num_sampled, num_computed):
            self.layer.kv_cache = (None, self.b)
            R.commit_step({"layer.0": md, "layer.1": md}, idx_mapping, num_sampled, num_computed)


    def distinct_blocks(n):
        return rng.sample(range(1, BLOCKS), n)


    def read_set(S, r, n, align, new_computed):
        cols = {n - 1}
        if align:
            running = new_computed - n + 1
            aligned = new_computed // align * align
            if aligned >= running:
                cols.add(aligned - running)
        return {int(S[r, c]) for c in cols}


    def single_step(align, n0, mixed):
        """Spec rows 0..2. Pure: a FULL-graph batch, plus a padded row (NULL
        indices, no tokens). Mixed: batch row 0 is a prefill, spec row r is
        batch row r + 1, and spec row 2's request is filtered (idx_mapping -1)."""
        pair = Pair(rows=8, tokens=40, align=align)
        blocks = distinct_blocks(3 * Q)
        S_rows = [blocks[r * Q : (r + 1) * Q] for r in range(3)]
        lens = [Q, Q, rng.randint(1, Q)]
        nacc = [rng.randint(1, Q) for _ in range(3)]
        if not mixed:
            S_rows.append([0] * Q)
            lens.append(0)
            nacc.append(1)
        S = torch.tensor(S_rows, dtype=torch.int32)
        batch = 4
        batch_of = [1, 2, 3] if mixed else [0, 1, 2]
        mask_cpu = torch.tensor([False, True, True, True]) if mixed else None
        md = pair.step(S, torch.tensor(nacc, dtype=torch.int32), lens, mask_cpu)
        md.num_spec_decodes = 3
        spec_n = [min(n0, lens[0]), rng.randint(1, Q), rng.randint(1, lens[2])]
        num_sampled = torch.ones(batch, dtype=torch.int32)
        idx_mapping = torch.arange(batch, dtype=torch.int32)
        if mixed:
            idx_mapping[batch_of[2]] = -1
        computed_pre = [rng.randint(0, 60) for _ in range(batch)]
        if align:
            # Spec row 0 reaches a block boundary at accepted token j - 1.
            computed_pre[batch_of[0]] = align * 3 - rng.randint(1, spec_n[0])
        for r in range(3):
            num_sampled[batch_of[r]] = spec_n[r]
        num_computed = torch.tensor(
            [computed_pre[b] + int(num_sampled[b]) for b in range(batch)], dtype=torch.int32
        )
        pair.commit(md, idx_mapping, num_sampled, num_computed if align else None)
        expect = set()
        crossed = 0
        for r in range(3):
            b = batch_of[r]
            if int(idx_mapping[b]) < 0:
                continue
            got = read_set(S, r, spec_n[r], align, int(num_computed[b]))
            crossed += len(got) > 1
            expect |= got
        for blk in range(BLOCKS):
            if blk in expect:
                assert same(pair.b[blk], pair.a[blk]), ("read state differs", align, n0, mixed, blk)
            else:
                assert same(pair.b[blk], pair.pool0[blk]), ("stray store", align, n0, mixed, blk)
        return crossed


    crossings = 0
    for align in (0, 12):
        for n0 in range(1, Q + 1):
            for mixed in (False, True):
                for _ in range(3):
                    crossings += single_step(align, n0, mixed)
    assert crossings > 0
    print("single step: accepted 0..7 x {none, align} x {pure, mixed}: bit-exact (%d boundary rows)" % crossings)


    def chain_none(steps, nreq):
        pair = Pair(rows=nreq, tokens=nreq * Q, align=0)
        S = torch.tensor(distinct_blocks(nreq * Q), dtype=torch.int32).view(nreq, Q)
        nacc = torch.ones(nreq, dtype=torch.int32)
        seen = set()
        for _ in range(steps):
            md = pair.step(S, nacc.clone(), [Q] * nreq)
            n = torch.tensor([rng.randint(1, Q) for _ in range(nreq)], dtype=torch.int32)
            pair.commit(md, torch.arange(nreq, dtype=torch.int32), n, None)
            for r in range(nreq):
                blk = int(S[r, int(n[r]) - 1])
                assert same(pair.b[blk], pair.a[blk]), "chain state differs"
                seen.add(int(n[r]))
            nacc = n
        return seen


    def chain_align(steps, nreq, B):
        """The V2 align rules from mamba_utils.py / mamba_hybrid.py, applied to
        both pools: pre-copy into the new running block, post-copy of the
        boundary state, num_accepted resets."""
        width = (3 * B + steps * Q) // B + 2 + Q
        pair = Pair(rows=nreq, tokens=nreq * Q, align=B)
        flat = distinct_blocks(nreq * width)
        bt = [flat[r * width : (r + 1) * width] for r in range(nreq)]
        computed = [rng.randint(1, 3 * B) for _ in range(nreq)]
        state_idx = [(c - 1) // B for c in computed]
        nacc = [1] * nreq
        cached = set()
        for _ in range(steps):
            for r in range(nreq):  # preprocess + pre-copy (both pools)
                new = (computed[r] + Q - 1) // B
                if state_idx[r] != new:
                    src = bt[r][state_idx[r] + nacc[r] - 1]
                    for pool in (pair.a, pair.b):
                        pool[bt[r][new]] = pool[src]
                    nacc[r] = 1
                state_idx[r] = new
            S = torch.tensor([bt[r][state_idx[r] : state_idx[r] + Q] for r in range(nreq)], dtype=torch.int32)
            md = pair.step(S, torch.tensor(nacc, dtype=torch.int32), [Q] * nreq)
            n = [rng.randint(1, Q) for _ in range(nreq)]
            new_computed = [computed[r] + n[r] for r in range(nreq)]
            pair.commit(
                md,
                torch.arange(nreq, dtype=torch.int32),
                torch.tensor(n, dtype=torch.int32),
                torch.tensor(new_computed, dtype=torch.int32),
            )
            for r in range(nreq):  # postprocess_mamba_fused_kernel (both pools)
                nacc[r] = n[r]
                running = new_computed[r] - n[r] + 1
                aligned = new_computed[r] // B * B
                if aligned >= running:
                    bias = aligned - running
                    dest = aligned // B - 1
                    if state_idx[r] == dest:
                        nacc[r] = 1
                    if not (state_idx[r] == dest and bias == 0):
                        for pool in (pair.a, pair.b):
                            pool[bt[r][dest]] = pool[bt[r][state_idx[r] + bias]]
                    cached.add(bt[r][dest])
                computed[r] = new_computed[r]
            for blk in cached:
                assert same(pair.b[blk], pair.a[blk]), "cached boundary state differs"
            for r in range(nreq):
                blk = bt[r][state_idx[r] + nacc[r] - 1]
                assert same(pair.b[blk], pair.a[blk]), "running state differs"
        return len(cached)


    seen = set()
    for nreq in (1, 2, 4):
        seen |= chain_none(steps=12, nreq=nreq)
    assert seen == set(range(1, Q + 1)), seen
    cached = sum(chain_align(steps=14, nreq=nreq, B=B) for nreq in (1, 2) for B in (8, 12, 13))
    assert cached > 0
    print("chains: none mode c=1,2,4 and align mode B=8,12,13: bit-exact, %d cached boundary blocks" % cached)
    '''
)


@unittest.skipUnless(
    HAVE_SRC and HAVE_TORCH and HAVE_TRITON, "needs GLM53_V14_SRC, torch, triton"
)
class InterpreterTest(unittest.TestCase):
    def _run(self, trim: str):
        with tempfile.TemporaryDirectory(prefix="v15kr-interp-") as tmp:
            tmp = Path(tmp)
            for rel, text in STUBS.items():
                (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                (tmp / rel).write_text(text)
            shutil.copy2(SRC / "utils/math_utils.py", tmp / "vllm/utils/math_utils.py")
            shutil.copy2(SRC / FLA / "fused_recurrent.py", tmp / "vllm" / FLA / "fused_recurrent.py")
            (tmp / "vllm" / FLA / "kda.py").write_text(
                lifted_kda_module((SRC / FLA / "kda.py").read_text())
            )
            (tmp / "vllm" / P.MODULE_REL).write_text(P.MODULE)
            res = subprocess.run(
                [sys.executable, "-c", SCRIPT, str(tmp), trim],
                capture_output=True,
                text=True,
                check=False,
                env={
                    **os.environ,
                    "TRITON_INTERPRET": "1",
                    "GLM53_KDA_TRIM": trim,
                    "GLM53_KDA_RECOVER": "1",
                },
            )
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            print(f"\n  [GLM53_KDA_TRIM={trim}] " + res.stdout.strip().replace("\n", "\n  "))

    def test_bit_exact_kda_trim_off(self):
        self._run("0")

    def test_bit_exact_kda_trim_on(self):
        self._run("1")


if __name__ == "__main__":
    unittest.main()
