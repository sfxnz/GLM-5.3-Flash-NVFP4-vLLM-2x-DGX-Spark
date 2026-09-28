"""CPU tests for docker/patch_v13_verify.py (GLM53_ADAPTIVE_VERIFY).

    GLM53_V11_SRC=/path/to/v11src python3 docker/test_v13_verify.py

GLM53_V11_SRC is the directory that holds the vllm/ package of glm53-sm121-v11
(read-only; the tests patch temporary copies). Without it the patch-apply tests
skip. The mask, remap and width tests run the shipped module on torch CPU
tensors and skip without torch. The two interpreter tests also need triton and
run v11's own kernels under the Triton CPU interpreter (TRITON_INTERPRET=1):
the end-to-end test runs the input-layout and rejection-sampling kernels and
checks that masking at k=7 emits exactly the tokens that speculation with k'=m
emits; the kpool-ring test runs the indexer's K-pool update kernel over several
verify steps and checks that the tail ring and every committed pool entry
equal k'=m's.
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

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("GLM53_V11_SRC", "/nonexistent")) / "vllm"
HAVE_SRC = (SRC / "__init__.py").is_file()
HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_TRITON = importlib.util.find_spec("triton") is not None
ROUTER = "model_executor/layers/fused_moe/router/base_router.py"
SAMPLER = "v1/worker/gpu/spec_decode/rejection_sampler.py"
RUNNER = "v1/worker/gpu/model_runner.py"
INDEXER = "v1/attention/backends/mla/indexer.py"
INDEXER_MOD = "vllm.v1.attention.backends.mla.indexer"
MODULE = "v1/worker/gpu/spec_decode/glm53_adaptive_verify.py"
K = 7  # recipe default: DFlash2-7


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


patch = load("patch_v13_verify", HERE / "patch_v13_verify.py")


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


class FakeLogger:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args)


def top_level(src: str, names: set[str]) -> str:
    """The top-level defs and annotated assignments of `src` named in `names`,
    as a module that imports torch (the indexer's other imports need a GPU
    build of vllm)."""
    keep = [
        node
        for node in ast.parse(src).body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) in names)
    ]
    assert len(keep) == len(names), [getattr(n, "name", None) for n in keep]
    return "import torch\n\n\n" + "\n\n\n".join(ast.get_source_segment(src, n) for n in keep) + "\n"


TAIL_NAMES = {"compute_kpool_tail_slot_mapping", "GLM53_TAIL_STASH_MASK"}


def patched_tail_slot_source() -> str:
    """compute_kpool_tail_slot_mapping and its hook, exactly as the patch writes them."""
    src, applied = patch.plan_edits((SRC / INDEXER).read_text(), patch.FILE_EDITS[INDEXER])
    assert applied == len(patch.FILE_EDITS[INDEXER])
    return top_level(src, TAIL_NAMES)


def module_namespace() -> dict:
    """Exec the shipped module with vllm.logger stubbed out."""
    tree = ast.parse(patch.MODULE)
    body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("vllm"))
    ]
    log = FakeLogger()
    env = {"__name__": "glm53_adaptive_verify", "init_logger": lambda name: log}
    exec(compile(ast.Module(body=body, type_ignores=[]), MODULE, "exec"), env)  # noqa: S102
    env["LOG"] = log
    return env


# ---------------------------------------------------------------------------
# Patch mechanics
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SRC, "set GLM53_V11_SRC to the dir holding the v11 vllm/")
class ApplyTests(unittest.TestCase):
    """The patch reads vllm/__init__.py and four files; census reads the runner."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v13verify-"))
        self.root = self.tmp / "vllm"
        for rel in ("__init__.py", ROUTER, SAMPLER, RUNNER, INDEXER):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / rel, self.root / rel)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self, name="patch_v13_verify.py"):
        return subprocess.run(
            [sys.executable, str(HERE / name), str(self.root)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_every_anchor_occurs_once_in_v11(self):
        for rel, edits in patch.FILE_EDITS.items():
            src = (SRC / rel).read_text()
            for what, old, new in edits:
                self.assertEqual(src.count(old), 1, what)
                self.assertEqual(src.count(new), 0, what)
        self.assertFalse((SRC / MODULE).exists())

    def test_apply_twice_is_idempotent(self):
        before = tree_digest(self.root)
        first = self.run_patch()
        self.assertEqual(first.returncode, 0, first.stderr)
        after = tree_digest(self.root)
        self.assertEqual(
            {k for k in after if before.get(k) != after[k]},
            {ROUTER, SAMPLER, RUNNER, INDEXER, MODULE},
        )
        second = self.run_patch()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("(0 files written)", second.stdout)
        self.assertEqual(tree_digest(self.root), after)

    def test_census_and_verify_apply_in_either_order(self):
        """Both edit the V2 runner; the Dockerfile runs census first."""
        for order in (("census", "verify"), ("verify", "census")):
            for rel in (RUNNER, MODULE, "v1/worker/gpu/glm53_expert_census.py"):
                (self.root / rel).unlink(missing_ok=True)
            shutil.copy2(SRC / RUNNER, self.root / RUNNER)
            for name in order:
                res = self.run_patch(f"patch_v13_{name}.py")
                self.assertEqual(res.returncode, 0, (order, res.stderr))
            done = tree_digest(self.root)
            for name in order:
                res = self.run_patch(f"patch_v13_{name}.py")
                self.assertIn("(0 files written)", res.stdout, order)
            self.assertEqual(tree_digest(self.root), done)
            text = (self.root / RUNNER).read_text()
            self.assertEqual(text.count("self._glm53_census."), 2)
            self.assertEqual(text.count("self._glm53_av."), 2)

    def test_misc_and_verify_edit_the_indexer_in_either_order(self):
        """Both edit indexer.py; the Dockerfile runs misc first."""
        misc = load("patch_v13_misc", HERE / "patch_v13_misc.py")
        src = (SRC / INDEXER).read_text()
        results = []
        for order in ((misc, patch), (patch, misc)):
            text = src
            for mod in order:
                text, applied = mod.plan_edits(text, mod.FILE_EDITS[INDEXER])
                self.assertEqual(applied, len(mod.FILE_EDITS[INDEXER]))
            for mod in order:
                self.assertEqual(mod.plan_edits(text, mod.FILE_EDITS[INDEXER])[1], 0)
            results.append(text)
        self.assertEqual(results[0], results[1])
        ast.parse(results[0])

    def test_refuses_on_drift_and_writes_nothing(self):
        path = self.root / SAMPLER
        path.write_text(path.read_text().replace("sampled, num_sampled = rejection_sample(", "x = 1"))
        before = tree_digest(self.root)
        res = self.run_patch()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("refusing", res.stderr)
        self.assertEqual(tree_digest(self.root), before, "no partial writes")

    def test_touched_files_compile(self):
        self.assertEqual(self.run_patch().returncode, 0)
        for rel in (ROUTER, SAMPLER, RUNNER, INDEXER, MODULE):
            py_compile.compile(
                str(self.root / rel),
                cfile=str(self.tmp / "pyc" / (rel.replace("/", "_") + "c")),
                doraise=True,
            )

    def test_hooks_are_guarded(self):
        """Every call into the feature sits behind an `is not None` check."""
        self.assertEqual(self.run_patch().returncode, 0)
        runner = (self.root / RUNNER).read_text()
        self.assertEqual(runner.count("self._glm53_av."), 2)
        self.assertEqual(runner.count("if self._glm53_av is not None:\n"), 2)
        self.assertEqual(runner.count("self._glm53_av = maybe_create_adaptive_verify(self)"), 1)
        sampler = (self.root / SAMPLER).read_text()
        self.assertEqual(sampler.count("self.glm53_mask_drafts("), 1)
        self.assertEqual(sampler.count("if self.glm53_mask_drafts is not None:"), 1)
        # Masking comes after apply_sampling_params, which keeps the real ids.
        self.assertLess(
            sampler.index("self.sampler.apply_sampling_params("),
            sampler.index("self.glm53_mask_drafts("),
        )
        router = (self.root / ROUTER).read_text()
        self.assertEqual(router.count("self.glm53_remap("), 1)
        self.assertLess(router.index("self.glm53_remap("), router.index("self.capture_fn(topk_ids)"))
        # The tail-slot mask runs after v11 fills the slots, inside the one
        # function the tail metadata builder calls.
        indexer = (self.root / INDEXER).read_text()
        self.assertEqual(indexer.count("GLM53_TAIL_STASH_MASK[:num_actual_tokens]"), 1)
        self.assertEqual(indexer.count("if GLM53_TAIL_STASH_MASK is not None:\n"), 1)
        fn = indexer[indexer.index("def compute_kpool_tail_slot_mapping(") :]
        fn = fn[: fn.index("\nclass ")]
        self.assertLess(
            fn.index("out[:num_actual_tokens] = own_block"),
            fn.index("if GLM53_TAIL_STASH_MASK is not None:"),
        )


class PatchStaticTests(unittest.TestCase):
    def test_module_parses_and_is_env_gated(self):
        ast.parse(patch.MODULE)
        self.assertIn('if os.environ.get("GLM53_ADAPTIVE_VERIFY") != "1":', patch.MODULE)
        for name in ("GLM53_ADAPTIVE_VERIFY_TAU", "GLM53_ADAPTIVE_VERIFY_MAX"):
            self.assertIn(f'"{name}"', patch.MODULE)

    def test_other_v13_patches_leave_router_and_sampler_alone(self):
        for name in ("patch_v13_misc.py", "patch_v13_fp8.py", "patch_v13_census.py"):
            text = (HERE / name).read_text()
            self.assertNotIn(ROUTER, text, name)
            self.assertNotIn(SAMPLER, text, name)

    def test_dockerfile_runs_verify_patch_after_census_before_compileall(self):
        text = (HERE / "Dockerfile.sm121-v13").read_text()
        self.assertIn('python3 /tmp/patch_v13_verify.py "$VLLM_ROOT"', text)
        self.assertLess(text.index("patch_v13_census.py"), text.index("patch_v13_verify.py"))
        self.assertLess(text.index("patch_v13_verify.py"), text.index("compileall"))


# ---------------------------------------------------------------------------
# Module logic on synthetic tensors (the runner's layout conventions)
# ---------------------------------------------------------------------------
def synthetic_batch(torch, reqs, pad=0):
    """reqs: (req_state, query_len, num_logits) in batch order, as the V2 runner
    lays them out: logit rows are the last num_logits rows of each query, local
    position 0 is the anchor (input_batch.py _combine_sampled_and_draft_tokens)."""
    rows, local, states, q0 = [], [], [], 0
    for state, qlen, nlog in reqs:
        q_end = q0 + qlen
        rows += range(q_end - nlog, q_end)
        local += range(nlog)
        states += [state] * nlog
        q0 = q_end
    return types.SimpleNamespace(
        num_tokens=q0,
        num_tokens_after_padding=q0 + pad,
        num_draft_tokens=sum(n - 1 for _, _, n in reqs),
        logits_indices=torch.tensor(rows, dtype=torch.int64),
        expanded_local_pos=torch.tensor(local, dtype=torch.int32),
        expanded_idx_mapping=torch.tensor(states, dtype=torch.int64),
    )


def scores_with_top_probs(torch, probs, n_cand=8):
    """[R, k, n_cand] scores whose softmax max is probs[r][i] (p >= 1/n_cand)."""
    p = torch.tensor(probs, dtype=torch.float64)
    rest = (1 - p) / (n_cand - 1)
    scores = torch.log(rest).unsqueeze(-1).repeat(1, 1, n_cand)
    scores[..., 3] = torch.log(p)  # any candidate slot may win
    return scores.float()


def top_probs_for_survival(survival):
    """Per-position probabilities whose running product is `survival`."""
    return [s / prev for s, prev in zip(survival, [1.0] + list(survival[:-1]))]


# E0 per-position acceptance (evidence/e0-nvidia-v11/SUMMARY.md), read as survival.
PROSE = [0.640, 0.345, 0.169, 0.065, 0.029, 0.012, 0.003]
CODE = [0.817, 0.657, 0.515, 0.406, 0.322, 0.238, 0.182]


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class ModuleLogicTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.env = module_namespace()
        self.AV = self.env["AdaptiveVerify"]

    def make(self, cap=K, tau=0.0, scores=None, reqs=4, tokens=40):
        return self.AV(tau, cap, scores, reqs, tokens, "cpu")

    def test_verify_width(self):
        torch, width = self.torch, self.env["verify_width"]
        s = scores_with_top_probs(torch, [[0.9, 0.7, 0.5, 0.3, 0.9, 0.9, 0.9]])
        # survival .9 .63 .315 .0945 ...
        self.assertEqual(width(s, 0.1, K).tolist(), [3])
        self.assertEqual(width(s, 0.5, K).tolist(), [2])
        self.assertEqual(width(s, 0.95, K).tolist(), [1])  # never below 1
        self.assertEqual(width(s, 0.0, K).tolist(), [K])
        self.assertEqual(width(s, 0.0, 4).tolist(), [4])  # the MAX cap
        # If the drafter were calibrated to E0, tau=0.1 verifies 3 prose drafts
        # (4 live rows) and all 7 code drafts.
        s = scores_with_top_probs(
            torch, [top_probs_for_survival(PROSE), top_probs_for_survival(CODE)]
        )
        self.assertEqual(width(s, 0.1, K).tolist(), [3, 7])
        self.assertEqual(width(s, 0.3, K).tolist(), [2, 5])

    def test_nan_scores_verify_one(self):
        torch = self.torch
        s = torch.full((1, K, 8), float("nan"))
        self.assertEqual(self.env["verify_width"](s, 0.1, K).tolist(), [1])

    def test_prepare_marks_rows_past_the_width(self):
        torch = self.torch
        av = self.make()
        av.verify_len[torch.tensor([2, 0])] = torch.tensor([3, K], dtype=torch.int32)
        # c=2 verify (states 2 and 0, 1 + 7 rows each) + a 5-token prefill chunk
        # (state 1, one logit) + 3 CUDA-graph padding rows.
        batch = synthetic_batch(torch, [(2, 8, 8), (0, 8, 8), (1, 5, 1)], pad=3)
        av.prepare(batch)
        masked = av.masked[: batch.num_tokens_after_padding]
        self.assertEqual(torch.nonzero(masked).flatten().tolist(), [4, 5, 6, 7])
        self.assertEqual(av.anchor[4:8].tolist(), [0, 0, 0, 0])
        # Unmasked rows keep anchor <= t: the remap gather stays in the batch.
        t = torch.arange(av.anchor.numel())
        self.assertTrue(bool((av.anchor <= t).all()))

        # Next step: one request (state 0, width 1) in a smaller batch.
        av.verify_len[0] = 1
        small = synthetic_batch(torch, [(0, 8, 8)])
        av.prepare(small)
        self.assertEqual(torch.nonzero(av.masked[:8]).flatten().tolist(), [2, 3, 4, 5, 6, 7])
        self.assertTrue(bool((av.anchor <= t).all()))

    def test_no_drafts_masks_nothing(self):
        torch = self.torch
        av = self.make()
        av.masked.fill_(True)
        batch = synthetic_batch(torch, [(0, 30, 1)], pad=2)
        av.prepare(batch)
        self.assertFalse(bool(av.masked[:32].any()))

    def test_remap_reuses_anchor_experts_with_zero_weight(self):
        torch = self.torch
        g = torch.Generator().manual_seed(0)
        av = self.make()
        av.verify_len[torch.tensor([2, 0])] = torch.tensor([2, 5], dtype=torch.int32)
        batch = synthetic_batch(torch, [(2, 8, 8), (0, 8, 8)], pad=0)
        av.prepare(batch)
        n = batch.num_tokens
        ids = torch.stack([torch.randperm(288, generator=g)[:8] for _ in range(n)]).int()
        w = torch.rand(n, 8, generator=g)
        w2, ids2 = av.remap(w, ids)
        masked = av.masked[:n]
        live = ~masked
        self.assertTrue(torch.equal(ids2[live], ids[live]))
        self.assertTrue(torch.equal(w2[live], w[live]))
        self.assertTrue(bool((w2[masked] == 0).all()))
        self.assertTrue(torch.equal(ids2[4:8], ids[0].expand(4, 8)))  # request 0, m=2
        self.assertTrue(torch.equal(ids2[14:16], ids[8].expand(2, 8)))  # request 1, m=5
        for start, m in ((0, 2), (8, 5)):
            block = set(ids2[start : start + 8].flatten().tolist())
            self.assertEqual(block, set(ids[start : start + m + 1].flatten().tolist()))
        self.assertEqual(ids2.dtype, ids.dtype)
        self.assertEqual(w2.dtype, w.dtype)

    def test_draft_mask_is_the_input_mask(self):
        """-1 lands exactly on the masked input rows; rows 0..m keep their ids."""
        torch = self.torch
        av = self.make()
        av.verify_len[torch.tensor([3, 1])] = torch.tensor([4, 1], dtype=torch.int32)
        batch = synthetic_batch(torch, [(3, 8, 8), (1, 8, 8), (0, 3, 1)])
        av.prepare(batch)
        draft = torch.arange(100, 100 + batch.logits_indices.numel(), dtype=torch.int32)
        out = av.mask_drafts(draft, batch.expanded_idx_mapping, batch.expanded_local_pos)
        rejected_rows = batch.logits_indices[out < 0]
        masked_rows = torch.nonzero(av.masked[: batch.num_tokens]).flatten()
        self.assertEqual(rejected_rows.tolist(), masked_rows.tolist())
        # Request 0 (m=4): draft rows 1..4 keep ids, 5..7 are -1.
        self.assertEqual(out[:8].tolist(), [100, 101, 102, 103, 104, -1, -1, -1])
        self.assertEqual(out[8:16].tolist(), [108, 109, -1, -1, -1, -1, -1, -1])
        self.assertEqual(out[16:].tolist(), [116])  # prefill logit untouched
        self.assertEqual(out.dtype, draft.dtype)

    def test_record_writes_widths_by_request_state(self):
        torch = self.torch
        scores = scores_with_top_probs(
            torch, [top_probs_for_survival(PROSE), top_probs_for_survival(CODE)] * 2
        )
        av = self.make(tau=0.1, scores=scores)
        av.record(torch.tensor([3, 1], dtype=torch.int64))
        self.assertEqual(av.verify_len.tolist(), [K, 7, K, 3])
        fixed = self.make(cap=3)
        fixed.record(torch.tensor([0], dtype=torch.int64))  # tau 0: cap only
        self.assertEqual(fixed.verify_len.tolist(), [3, 3, 3, 3])

    # --- maybe_create_adaptive_verify with stand-ins for the vllm classes ---

    def fakes(self):
        class MoERunner:
            pass

        class BaseRouter:
            glm53_remap = None

        class RejectionSampler:
            glm53_mask_drafts = None

        def layer(i):
            m = MoERunner()
            m.router = BaseRouter()
            m._quant_method = types.SimpleNamespace(is_monolithic=False)
            m.layer_name = f"model.layers.{i}.mlp.experts"
            return m

        layers = [layer(i) for i in range(3, 6)]
        runner = types.SimpleNamespace(
            num_speculative_steps=K,
            parallel_config=types.SimpleNamespace(
                pipeline_parallel_size=1,
                data_parallel_size=1,
                prefill_context_parallel_size=1,
                use_ubatching=False,
                use_sequence_parallel_moe=False,
            ),
            speculative_config=types.SimpleNamespace(),
            rejection_sampler=RejectionSampler(),
            model_state=types.SimpleNamespace(num_new_sampled_tokens_per_step=1),
            speculator=types.SimpleNamespace(_selector_scores=self.torch.zeros(4, K, 8)),
            model=types.SimpleNamespace(modules=lambda: [object(), *layers]),
            max_num_reqs=4,
            max_num_tokens=40,
            device="cpu",
        )
        modules = {
            "vllm.model_executor.layers.fused_moe.layer": types.SimpleNamespace(MoERunner=MoERunner),
            "vllm.model_executor.layers.fused_moe.router.base_router": types.SimpleNamespace(
                BaseRouter=BaseRouter
            ),
            "vllm.v1.worker.gpu.spec_decode.rejection_sampler": types.SimpleNamespace(
                RejectionSampler=RejectionSampler
            ),
            INDEXER_MOD: types.SimpleNamespace(GLM53_TAIL_STASH_MASK=None),
        }
        return runner, layers, modules

    def create(self, runner, modules, **env):
        base = {"GLM53_ADAPTIVE_VERIFY": "1"}
        base.update(env)
        with mock.patch.dict(os.environ, base), mock.patch.dict(sys.modules, modules):
            for name in ("GLM53_ADAPTIVE_VERIFY_TAU", "GLM53_ADAPTIVE_VERIFY_MAX"):
                if name not in env:
                    os.environ.pop(name, None)
            return self.env["maybe_create_adaptive_verify"](runner)

    def test_off_when_unset(self):
        runner, layers, modules = self.fakes()
        with mock.patch.dict(os.environ, {"GLM53_ADAPTIVE_VERIFY": "0"}), mock.patch.dict(
            sys.modules, modules
        ):
            self.assertIsNone(self.env["maybe_create_adaptive_verify"](runner))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(self.env["maybe_create_adaptive_verify"](None))
        self.assertTrue(all(lay.router.glm53_remap is None for lay in layers))
        self.assertIsNone(runner.rejection_sampler.glm53_mask_drafts)
        self.assertIsNone(modules[INDEXER_MOD].GLM53_TAIL_STASH_MASK)

    def test_on_binds_every_router_the_sampler_and_the_tail_slots(self):
        runner, layers, modules = self.fakes()
        av = self.create(runner, modules)
        self.assertEqual((av.tau, av.cap), (0.1, K))
        self.assertTrue(all(lay.router.glm53_remap == av.remap for lay in layers))
        self.assertEqual(runner.rejection_sampler.glm53_mask_drafts, av.mask_drafts)
        self.assertIs(modules[INDEXER_MOD].GLM53_TAIL_STASH_MASK, av.masked)
        self.assertIn("tau=0.1 max=7 (k=7)", self.env["LOG"].lines[-1])
        self.assertIn("in 3 MoE layers", self.env["LOG"].lines[-1])
        runner, _, modules = self.fakes()
        av = self.create(
            runner, modules, GLM53_ADAPTIVE_VERIFY_TAU="0", GLM53_ADAPTIVE_VERIFY_MAX="3"
        )
        self.assertEqual((av.tau, av.cap, av.verify_len.tolist()), (0.0, 3, [3, 3, 3, 3]))

    def test_refusals(self):
        cases = [
            ({"pipeline_parallel_size": 2}, {}),
            ({"data_parallel_size": 2}, {}),
            ({"prefill_context_parallel_size": 2}, {}),
            ({"use_ubatching": True}, {}),
            ({"use_sequence_parallel_moe": True}, {}),
            ({}, {"GLM53_ADAPTIVE_VERIFY_TAU": "1.5"}),
            ({}, {"GLM53_ADAPTIVE_VERIFY_TAU": "x"}),
            ({}, {"GLM53_ADAPTIVE_VERIFY_MAX": "0"}),
            ({}, {"GLM53_ADAPTIVE_VERIFY_MAX": "8"}),
            ({}, {"GLM53_ADAPTIVE_VERIFY_TAU": "0"}),  # MAX defaults to k: no-op
        ]
        for par, env in cases:
            runner, _, modules = self.fakes()
            vars(runner.parallel_config).update(par)
            with self.assertRaises(ValueError, msg=(par, env)):
                self.create(runner, modules, **env)
            self.assertIsNone(modules[INDEXER_MOD].GLM53_TAIL_STASH_MASK)
        runner, layers, modules = self.fakes()
        layers[1]._quant_method.is_monolithic = True
        with self.assertRaises(ValueError):
            self.create(runner, modules)
        runner, _, modules = self.fakes()
        runner.speculator = types.SimpleNamespace()  # no DFlash2 selector scores
        with self.assertRaises(ValueError):
            self.create(runner, modules)
        self.assertIsNotNone(
            self.create(runner, modules, GLM53_ADAPTIVE_VERIFY_TAU="0", GLM53_ADAPTIVE_VERIFY_MAX="4")
        )
        runner, _, modules = self.fakes()
        cls = type(runner.rejection_sampler)
        runner.rejection_sampler = type("Custom", (cls,), {})()  # might override _verify
        with self.assertRaises(ValueError):
            self.create(runner, modules)


@unittest.skipUnless(HAVE_SRC and HAVE_TORCH, "needs GLM53_V11_SRC and torch")
class TailSlotTests(unittest.TestCase):
    """compute_kpool_tail_slot_mapping as the patch writes it, against v11's."""

    def setUp(self):
        import torch

        self.torch = torch
        self.v11, self.v13 = {}, {}
        v11_src = top_level((SRC / INDEXER).read_text(), {"compute_kpool_tail_slot_mapping"})
        exec(compile(v11_src, INDEXER, "exec"), self.v11)  # noqa: S102
        exec(compile(patched_tail_slot_source(), INDEXER, "exec"), self.v13)  # noqa: S102
        self.av = module_namespace()["AdaptiveVerify"](0.0, K, None, 4, 40, "cpu")

    def slots(self, env, num_actual_tokens):
        """c=2 verify (tail blocks 5 and 7) + a 5-token prefill chunk (block 9)
        + 3 padding rows, the layout a FULL-graph metadata build sees."""
        torch = self.torch
        pos = [*range(100, 108), *range(37, 45), *range(10, 15), 0, 0, 0]
        return env["compute_kpool_tail_slot_mapping"](
            torch.arange(1000, 1028),  # generic slots; rows past the tokens stay
            torch.tensor([[5, 0], [7, 0], [9, 0]], dtype=torch.int32),
            torch.tensor([0, 8, 16, 21], dtype=torch.int32),
            torch.tensor(pos),
            num_actual_tokens,
            3,
            4,
        )

    def test_off_is_v11(self):
        for n in (0, 21, 24):
            self.assertTrue(self.torch.equal(self.slots(self.v13, n), self.slots(self.v11, n)))

    def test_masked_rows_get_no_tail_slot(self):
        torch, av = self.torch, self.av
        av.verify_len[torch.tensor([2, 0])] = torch.tensor([3, 1], dtype=torch.int32)
        av.prepare(synthetic_batch(torch, [(2, 8, 8), (0, 8, 8), (1, 5, 1)], pad=3))
        self.v13["GLM53_TAIL_STASH_MASK"] = av.masked
        want = self.slots(self.v11, 24)
        want[[4, 5, 6, 7, 10, 11, 12, 13, 14, 15]] = -1  # local_pos > m: 3, then 1
        self.assertTrue(torch.equal(self.slots(self.v13, 24), want))
        self.assertTrue(torch.equal(self.slots(self.v13, 0), self.slots(self.v11, 0)))
        # The next step's prepare() clears the rows: a prefill-only batch is v11's.
        av.prepare(synthetic_batch(torch, [(1, 24, 1)]))
        self.assertTrue(torch.equal(self.slots(self.v13, 24), self.slots(self.v11, 24)))


# ---------------------------------------------------------------------------
# End to end on v11's own kernels (Triton CPU interpreter)
# ---------------------------------------------------------------------------
STUBS = {
    "vllm/__init__.py": "",
    "vllm/logger.py": "import logging\n\n\ndef init_logger(name):\n    return logging.getLogger(name)\n",
    "vllm/utils/__init__.py": "import uuid\n\n\ndef random_uuid():\n    return uuid.uuid4().hex\n",
    "vllm/utils/math_utils.py": "def cdiv(a, b):\n    return -(-a // b)\n",
    # The interpreter does not patch libdevice; a jit-wrapped log1p resolves.
    "vllm/triton_utils/__init__.py": textwrap.dedent(
        """\
        import types

        import triton
        import triton.language as tl

        HAS_TRITON = True


        @triton.jit
        def _log1p(x):
            return tl.where(tl.abs(x) < 1e-4, x * (1.0 - 0.5 * x), tl.log(1.0 + x))


        tldevice = types.SimpleNamespace(log1p=_log1p)
        """
    ),
    "vllm/v1/__init__.py": "",
    "vllm/v1/worker/__init__.py": "",
    "vllm/v1/worker/gpu/__init__.py": "",
    "vllm/v1/worker/gpu/sample/__init__.py": "",
    "vllm/v1/worker/gpu/spec_decode/__init__.py": "",
}
V11_FILES = (
    "v1/worker/gpu/input_batch.py",
    "v1/worker/gpu/sample/gumbel.py",
    "v1/worker/gpu/spec_decode/rejection_sampler_utils.py",
)

E2E_SCRIPT = textwrap.dedent(
    r'''
    import sys
    import zlib
    from collections import Counter
    from types import SimpleNamespace

    import numpy as np
    import torch

    sys.path.insert(0, sys.argv[1])
    from vllm.v1.worker.gpu import input_batch as ib
    from vllm.v1.worker.gpu.spec_decode.glm53_adaptive_verify import AdaptiveVerify
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    K, V, MAX_REQS, MAX_TOKENS, MAX_LEN, PAD = 7, 64, 4, 40, 64, 3
    SENTINEL = V - 1  # the toy target never picks it; a masked row picks nothing else
    GARBAGE = np.full(V, -30.0, np.float32)
    GARBAGE[SENTINEL] = 30.0


    def target_logits(prefix):
        """Causal toy target: the logits after a prefix depend on that prefix only."""
        seed = zlib.crc32(np.asarray(prefix, np.int64).tobytes())
        x = np.random.default_rng(seed).standard_normal(V).astype(np.float32) * 4.0
        x[SENTINEL] = -30.0
        return x


    def experts(prefix):
        seed = zlib.crc32(np.asarray(prefix, np.int64).tobytes()) ^ 0x5EED
        return np.random.default_rng(seed).choice(288, 8, replace=False)


    def step(reqs, n_drafts, temperature, block, draft_logits, av=None):
        """One scheduler step laid out, 'forwarded' and verified the way the
        V2 runner does it. Returns the emitted tokens per request."""
        R = len(reqs)
        idx_mapping = torch.tensor([r["state"] for r in reqs], dtype=torch.int64)
        qlens = [1 + n if r["verify"] else len(r["chunk"]) for r, n in zip(reqs, n_drafts)]
        qsl = np.zeros(MAX_REQS + 1, np.int32)
        qsl[1 : R + 1] = np.cumsum(qlens)
        qsl[R + 1 :] = qsl[R]
        query_start_loc = torch.from_numpy(qsl)
        num_tokens = int(qsl[R])
        num_computed = torch.zeros(MAX_REQS, dtype=torch.int32)
        prefill_len = torch.zeros(MAX_REQS, dtype=torch.int32)
        last_sampled = torch.zeros(MAX_REQS, 1, dtype=torch.int64)
        draft_tokens = torch.zeros(MAX_REQS, K, dtype=torch.int64)
        all_token_ids = torch.zeros(MAX_REQS, MAX_LEN, dtype=torch.int32)
        for r, n in zip(reqs, n_drafts):
            s, ctx = r["state"], r["context"]
            num_computed[s] = len(ctx)
            if r["verify"]:
                prefill_len[s] = r["prompt_len"]
                last_sampled[s, 0] = r["anchor"]
                draft_tokens[s, :n] = torch.tensor(r["drafts"][:n])
            else:
                toks = ctx + r["chunk"]
                prefill_len[s] = len(toks)
                all_token_ids[s, : len(toks)] = torch.tensor(toks, dtype=torch.int32)
        input_ids = torch.zeros(MAX_TOKENS, dtype=torch.int32)
        positions = torch.zeros(MAX_TOKENS, dtype=torch.int64)
        seq_lens = torch.zeros(MAX_REQS, dtype=torch.int32)
        next_prefill = torch.zeros(1, MAX_REQS, dtype=torch.int32)
        ib.prepare_prefill_inputs(
            input_ids, next_prefill, idx_mapping, query_start_loc, all_token_ids,
            prefill_len, num_computed,
        )
        ib.prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed, positions, seq_lens)
        cu = np.zeros(R + 1, np.int32)
        cu[1:] = np.cumsum([1 + n for n in n_drafts])
        cu_num_logits = torch.from_numpy(cu)
        num_logits = int(cu[-1])
        logits_indices = ib.combine_sampled_and_draft_tokens(
            input_ids, idx_mapping, last_sampled, query_start_loc, seq_lens,
            prefill_len, draft_tokens, cu_num_logits, num_logits,
        )
        expanded_idx_mapping, expanded_local_pos = ib.expand_idx_mapping(
            idx_mapping, num_logits, cu_num_logits, K + 1
        )
        batch = SimpleNamespace(
            num_tokens_after_padding=num_tokens + PAD,
            num_draft_tokens=sum(n_drafts),
            logits_indices=logits_indices,
            expanded_local_pos=expanded_local_pos,
            expanded_idx_mapping=expanded_idx_mapping,
        )
        masked = torch.zeros(num_tokens + PAD, dtype=torch.bool)
        if av is not None:
            av.prepare(batch)
            masked = av.masked[: num_tokens + PAD].clone()
        assert not masked[num_tokens:].any(), "padding rows masked"

        # 'Forward': row t sees its request's context plus its query up to t.
        # A masked row outputs garbage that would show up in any token it fed.
        req_of = np.searchsorted(qsl[1 : R + 1], np.arange(num_tokens), side="right")
        prefix = [reqs[b]["context"] + input_ids[qsl[b] : t + 1].tolist()
                  for t, b in enumerate(req_of)]
        rows = logits_indices.tolist()
        logits = torch.from_numpy(np.stack(
            [GARBAGE if masked[t] else target_logits(prefix[t]) for t in rows]
        ))
        if av is not None:  # routing remap on the real layout
            ids = torch.from_numpy(np.stack([experts(p) for p in prefix])).int()
            w = torch.rand(num_tokens, 8)
            w2, ids2 = av.remap(w, ids)
            live = ~masked[:num_tokens]
            assert torch.equal(ids2[live], ids[live]) and torch.equal(w2[live], w[live])
            assert bool((w2[~live] == 0).all())
            for b, r in enumerate(reqs):
                q0, q1 = int(qsl[b]), int(qsl[b + 1])
                got = set(ids2[q0:q1].flatten().tolist())
                want = set(ids[q0:q1][live[q0:q1]].flatten().tolist())
                assert got == want, "a masked row added an expert"

        draft_sampled = input_ids[logits_indices]
        pos = positions[logits_indices]
        if av is not None:
            draft_sampled = av.mask_drafts(draft_sampled, expanded_idx_mapping, expanded_local_pos)
        temp = torch.full((MAX_REQS,), temperature, dtype=torch.float32)
        seeds = torch.tensor([11, 22, 33, 44], dtype=torch.int64)
        sampled, num_sampled = rejection_sample(
            logits, draft_logits, draft_sampled, cu_num_logits, pos, idx_mapping,
            expanded_idx_mapping, expanded_local_pos, temp, seeds, K, None,
            use_fp64=False, use_block_verification=block,
        )
        out = [sampled[b, : int(num_sampled[b])].tolist() for b in range(R)]
        assert all(SENTINEL not in o for o in out), "a masked row was read"
        return out


    def make_requests(rng):
        states = rng.permutation(MAX_REQS)[:3].tolist()
        reqs = []
        for s in states[:2]:
            ctx = rng.integers(0, V - 1, int(rng.integers(3, 9))).tolist()
            anchor = int(rng.integers(0, V - 1))
            drafts = []
            for _ in range(K):
                best = int(np.argmax(target_logits(ctx + [anchor] + drafts)))
                drafts.append(best if rng.random() < 0.8 else int(rng.integers(0, V - 1)))
            reqs.append(dict(verify=True, state=s, context=ctx, prompt_len=2,
                             anchor=anchor, drafts=drafts))
        ctx = rng.integers(0, V - 1, 5).tolist()
        reqs.append(dict(verify=False, state=states[2], context=ctx,
                         chunk=rng.integers(0, V - 1, 4).tolist()))
        return reqs


    rng = np.random.default_rng(0)
    modes = {
        "greedy": (0.0, False, False),
        "sampled": (1.0, False, False),
        "sampled+block": (1.0, True, False),
        "sampled+draft_logits": (1.0, False, True),
        "sampled+block+draft_logits": (1.0, True, True),
    }
    seen = Counter()
    for trial in range(int(sys.argv[2])):
        reqs = make_requests(rng)
        widths = [int(rng.integers(1, K + 1)) for _ in range(2)]
        av = AdaptiveVerify(0.0, K, None, MAX_REQS, MAX_TOKENS, "cpu")
        for r, m in zip(reqs, widths):
            av.verify_len[r["state"]] = m
        dl = torch.randn(MAX_REQS, K, V) * 2
        for name, (temperature, block, with_dl) in modes.items():
            draft_logits = dl if with_dl else None
            got = step(reqs, [K, K, 0], temperature, block, draft_logits, av)
            want = step(reqs, widths + [0], temperature, block, draft_logits)
            assert got == want, (trial, name, widths, got, want)
            for out, m in zip(got, widths):
                seen[(name, "all m accepted" if len(out) == m + 1 else "rejected early")] += 1
    for key in sorted(seen):
        print("%-28s %-15s %d" % (*key, seen[key]))
    both = {name: {k[1] for k in seen if k[0] == name} for name in modes}
    assert all(v == {"all m accepted", "rejected early"} for v in both.values()), both
    print("masked k=7 == truncated k'=m on every trial and mode")
    '''
)


@unittest.skipUnless(
    HAVE_SRC and HAVE_TORCH and HAVE_TRITON, "needs GLM53_V11_SRC, torch, triton"
)
class EndToEndInterpreterTest(unittest.TestCase):
    TRIALS = 12

    def test_masked_k7_emits_what_truncated_k_prime_emits(self):
        with tempfile.TemporaryDirectory(prefix="v13verify-e2e-") as tmp:
            tmp = Path(tmp)
            for rel, text in STUBS.items():
                (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                (tmp / rel).write_text(text)
            for rel in V11_FILES:
                shutil.copy2(SRC / rel, tmp / "vllm" / rel)
            (tmp / "vllm" / MODULE).write_text(patch.MODULE)
            res = subprocess.run(
                [sys.executable, "-c", E2E_SCRIPT, str(tmp), str(self.TRIALS)],
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "TRITON_INTERPRET": "1"},
            )
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            print("\n  " + res.stdout.strip().replace("\n", "\n  "))


# ---------------------------------------------------------------------------
# Across steps: v11's indexer K-pool update kernel (Triton CPU interpreter)
# ---------------------------------------------------------------------------
KPOOL_SCRIPT = textwrap.dedent(
    r'''
    import importlib.util
    import random
    import sys
    import zlib
    from collections import Counter
    from types import SimpleNamespace

    import torch

    root = sys.argv[1]
    sys.path.insert(0, root)
    import glm53_tail_slots as tail  # compute_kpool_tail_slot_mapping as patched
    from vllm.v1.worker.gpu.spec_decode.glm53_adaptive_verify import AdaptiveVerify

    spec = importlib.util.spec_from_file_location("kpool_compress", f"{root}/kpool_compress.py")
    kc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kc)

    K, KPOOL, D, PAGE = 7, 4, 128, 64  # index_kpool = 4 in the pinned 09b04e5 config
    STATES = [3, 1]  # request b: req_state STATES[b], tail block b + 1, pool page b
    R = len(STATES)
    APE = torch.randn(KPOOL, D, generator=torch.Generator().manual_seed(7))


    def vec(key):
        """A row's indexer K and gate: a function of the inputs the row saw."""
        g = torch.Generator().manual_seed(zlib.crc32(repr(key).encode()))
        return torch.randn(2, D, generator=g).to(torch.bfloat16)


    def tail_slots(rows, mask):
        """The tail slots the kpool tail metadata builder hands the indexer."""
        qsl = [0]
        for r in rows:
            qsl.append(qsl[-1] + len(r))
        tail.GLM53_TAIL_STASH_MASK = mask
        flat = tail.compute_kpool_tail_slot_mapping(
            torch.zeros(qsl[-1], dtype=torch.int64),
            torch.arange(1, R + 1, dtype=torch.int32).view(R, 1),
            torch.tensor(qsl, dtype=torch.int32),
            torch.tensor([p for r in rows for p, _ in r]),
            qsl[-1],
            R,
            KPOOL,
        )
        return [flat[qsl[b] : qsl[b + 1]].tolist() for b in range(R)]


    def launch(kv, ring, rows, mask=None):
        """One indexer decode update; short requests padded with -1, as the
        indexer's non-uniform decode scatter pads them."""
        n = max(len(r) for r in rows)
        key = torch.zeros(R, n, 2, D, dtype=torch.bfloat16)
        pos, slot, tslot = (torch.full((R, n), -1, dtype=torch.int32) for _ in range(3))
        for b, (req_rows, req_tslots) in enumerate(zip(rows, tail_slots(rows, mask))):
            for t, ((p, k), ts) in enumerate(zip(req_rows, req_tslots)):
                key[b, t], pos[b, t], tslot[b, t] = vec(k), p, ts
                if p % KPOOL == KPOOL - 1:  # pool-granular slot: the completing row
                    slot[b, t] = b * PAGE + p // KPOOL
        kc.kpool_decode_update_and_maybe_write_cache_batched(
            kv, ring, tslot, key[:, :, 0].contiguous(), key[:, :, 1].contiguous(),
            APE, slot, pos, KPOOL, D, round_scale=True,
        )


    def pool(kv, b, q):
        """Pool q of request b: its fp8 K and its fp32 scale bytes."""
        flat = kv[b].reshape(-1)
        return torch.cat([flat[q * D : (q + 1) * D], flat[PAGE * D + 4 * q : PAGE * D + 4 * q + 4]])


    def run(hist, plan, mode, kv, ring):
        """Verify steps from anchors at `hist`. mode: "k'=m" runs rows 0..m;
        "masked" runs rows 0..k with prepare()'s mask on the tail slots;
        "unsuppressed" runs rows 0..k and lets masked rows stash, as the
        first version of the patch did."""
        P = list(hist)
        av = AdaptiveVerify(0.0, K, None, 4, R * (K + 1), "cpu")
        for s, step in enumerate(plan):
            rows = []
            for b, (m, a) in enumerate(step):
                last = m if mode == "k'=m" else K
                rows.append([
                    (P[b] + j, ("true", b, P[b] + j) if j <= a else
                     ("draft" if j <= m else "masked", s, b, P[b] + j))
                    for j in range(last + 1)
                ])
                av.verify_len[STATES[b]] = m
            n = R * (K + 1)
            av.prepare(SimpleNamespace(
                num_tokens_after_padding=n,
                num_draft_tokens=R * K,
                logits_indices=torch.arange(n),
                expanded_local_pos=torch.arange(K + 1).repeat(R),
                expanded_idx_mapping=torch.tensor(STATES).repeat_interleave(K + 1),
            ))
            launch(kv, ring, rows, av.masked if mode == "masked" else None)
            P = [p + a + 1 for p, (_, a) in zip(P, step)]
        return P


    rng = random.Random(int(sys.argv[2]))
    seen = Counter()
    for trial in range(int(sys.argv[3])):
        hist = [rng.randint(6, 13) for _ in range(R)]
        plan = []
        for _ in range(4):
            step = []
            for _ in range(R):
                m = rng.randint(1, K)
                step.append((m, min(m, rng.choice([0, 0, 1, 1, 2, 3, 5, 7]))))
            plan.append(step)
        # History: plain one-token decode of positions 0..hist-1. Ring block 0
        # belongs to no request here; a masked completing row reads it.
        kv0 = torch.zeros(R, PAGE, D + 4, dtype=torch.uint8)
        ring0 = torch.zeros(R + 1, 2, KPOOL, D, dtype=torch.bfloat16)
        ring0[0] = vec("another request").view(2, 1, D)
        for p in range(max(hist)):
            launch(kv0, ring0, [[(p, ("true", b, p))] if p < hist[b] else [] for b in range(R)])
        out = {}
        for mode in ("k'=m", "masked", "unsuppressed"):
            kv, ring = kv0.clone(), ring0.clone()
            out[mode] = (run(hist, plan, mode, kv, ring), kv, ring)
        ends, ref_kv, ref_ring = out["k'=m"]
        for mode in ("masked", "unsuppressed"):
            _, kv, ring = out[mode]
            seen[mode, "ring differs"] += not torch.equal(ring, ref_ring)
            for b in range(R):
                # Pools that end at or after the first anchor and are committed.
                for q in range(hist[b] // KPOOL, ends[b] // KPOOL):
                    seen[mode, "committed pools"] += 1
                    seen[mode, "pool differs"] += not torch.equal(pool(kv, b, q), pool(ref_kv, b, q))
    for key in sorted(seen):
        print("%-13s %-16s %d" % (*key, seen[key]))
    assert seen["masked", "ring differs"] == 0 and seen["masked", "pool differs"] == 0
    assert seen["unsuppressed", "pool differs"] > 0, "the check cannot see unsuppressed stashes"
    print("masked k=7 leaves the tail ring and every committed pool as k'=m does")
    '''
)


@unittest.skipUnless(
    HAVE_SRC and HAVE_TORCH and HAVE_TRITON, "needs GLM53_V11_SRC, torch, triton"
)
class KpoolTailRingInterpreterTest(unittest.TestCase):
    TRIALS = 16

    def test_masked_rows_leave_the_indexer_state_of_k_prime(self):
        with tempfile.TemporaryDirectory(prefix="v13verify-kpool-") as tmp:
            tmp = Path(tmp)
            for rel, text in STUBS.items():
                (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                (tmp / rel).write_text(text)
            (tmp / "vllm" / MODULE).write_text(patch.MODULE)
            (tmp / "glm53_tail_slots.py").write_text(patched_tail_slot_source())
            shutil.copy2(SRC / "models/glm5next/nvidia/ops/kpool_compress.py", tmp)
            res = subprocess.run(
                [sys.executable, "-c", KPOOL_SCRIPT, str(tmp), "0", str(self.TRIALS)],
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "TRITON_INTERPRET": "1"},
            )
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            print("\n  " + res.stdout.strip().replace("\n", "\n  "))


if __name__ == "__main__":
    unittest.main(verbosity=2)
