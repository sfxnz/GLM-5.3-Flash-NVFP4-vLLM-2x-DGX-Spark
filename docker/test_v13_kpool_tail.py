"""CPU tests for docker/patch_v13_kpool_tail.py (GLM53_KPOOL_TAIL_FIX).

    GLM53_V11_SRC=/path/to/v11src python3 docker/test_v13_kpool_tail.py

GLM53_V11_SRC is the directory that holds the vllm/ package of glm53-sm121-v11
(read-only; the tests patch temporary copies). Without it everything but the
Dockerfile check skips. The builder tests need torch. The simulation and
mutation tests also need triton: they run v11's own indexer kernels (the K-pool
decode update, the prefill pool compress and the tail seed) under the Triton
CPU interpreter (TRITON_INTERPRET=1), on the tail view v11 carves out of the
indexer tensor, with the tail slots built the way the serve builds them: the
generic slot kernel fills the tail group's persistent slot-mapping row, then
the kpool tail metadata builder runs on the V2 runner's metadata (no
positions; SPEC=dflash2 and SPEC=mtp) or the V1 runner's. A uniform verify
batch runs as a padded FULL graph, whose replay reads the row it captured.
They drive prompts (whole, or chunked next to a verifying request) plus
several verify steps at c=1, 2 and 4 with random acceptance and compare every
committed pool and every tail-ring slot a later step can read against
non-speculative decoding.
"""

import ast
import hashlib
import importlib.util
import json
import os
import py_compile
import re
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
KPOOL = "models/glm5next/nvidia/ops/kpool_compress.py"
ATTENTION = "models/glm5next/nvidia/attention.py"
INDEXER = "v1/attention/backends/mla/indexer.py"
INDEXER_OP = "model_executor/layers/sparse_attn_indexer_kpool.py"
ATTN_UTILS = "v1/worker/gpu/attn_utils.py"
AV_MODULE = "v1/worker/gpu/spec_decode/glm53_adaptive_verify.py"
KPOOL_MOD = "vllm.models.glm5next.nvidia.ops.kpool_compress"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


patch = load("patch_v13_kpool_tail", HERE / "patch_v13_kpool_tail.py")
verify = load("patch_v13_verify", HERE / "patch_v13_verify.py")
misc = load("patch_v13_misc", HERE / "patch_v13_misc.py")


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def top_level(src: str, names: set[str]) -> str:
    """The top-level defs, classes and assignments of `src` named in `names`."""
    keep = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.AnnAssign):
            name = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        else:
            continue
        if name in names:
            keep.append(node)
    assert len(keep) == len(names), [ast.dump(n)[:40] for n in keep]
    return "\n\n\n".join(ast.get_source_segment(src, n) for n in keep) + "\n"


def apply(src: str, edits) -> str:
    """`src` with `edits` applied (a mutation that restores an anchor counts
    as already applied)."""
    return patch.plan_edits(src, edits)[0]


def patched(rel: str, edits=None) -> str:
    """`rel` of the v11 tree as the patch (or `edits`) writes it."""
    src, applied = patch.plan_edits((SRC / rel).read_text(), (edits or patch.FILE_EDITS)[rel])
    assert edits or applied == len(patch.FILE_EDITS[rel])
    return src


def mutated(rel: str, old: str, new: str) -> dict:
    """The patch's edits with `old` replaced by `new` in the one edit of `rel` that has it."""
    edits = patch.FILE_EDITS[rel]
    assert sum(old in text for _, _, text in edits) == 1, old
    return {**patch.FILE_EDITS, rel: [(what, a, text.replace(old, new)) for what, a, text in edits]}


def ring_helper(kpool_src: str, env: dict):
    """glm53_tail_ring_slots from `kpool_src`, with GLM53_* read from `env`."""
    ns: dict = {}
    code = "import os\n\n" + top_level(kpool_src, {"GLM53_KPOOL_TAIL_FIX", "glm53_tail_ring_slots"})
    with mock.patch.dict(os.environ, env, clear=True):
        exec(compile(code, KPOOL, "exec"), ns)  # noqa: S102
    return ns


def module_stubs(**attrs) -> dict:
    """sys.modules entries for KPOOL_MOD and its parents; `attrs` on the leaf."""
    parts = KPOOL_MOD.split(".")
    stubs = {name: types.ModuleType(name) for name in (".".join(parts[:i]) for i in range(1, len(parts) + 1))}
    vars(stubs[KPOOL_MOD]).update(attrs)
    return stubs


# The kpool tail builder's module context in v1/attention/backends/mla/indexer.py.
HOST_PRELUDE = '''from types import SimpleNamespace

import torch


class AttentionMetadataBuilder:  # the tail builder only keeps the spec
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec


AttentionCGSupport = SimpleNamespace(ALWAYS="ALWAYS")
AttentionSpec = VllmConfig = CommonAttentionMetadata = object


def split_decodes_and_prefills(common_attn_metadata):
    return 0, 0, 0, 0


def DeepseekV32IndexerMetadata(**kwargs):
    return SimpleNamespace(**kwargs)


'''
HOST_NAMES = {"compute_kpool_tail_slot_mapping", "KpoolTailMetadataBuilder"}


def host_source(indexer_src: str) -> str:
    """The tail slot mapping and the tail builder of `indexer_src`, runnable
    alone (plus adaptive verify's mask attribute when `indexer_src` has it)."""
    names = HOST_NAMES | ({"GLM53_TAIL_STASH_MASK"} & set(re.findall(r"^(\w+):", indexer_src, re.M)))
    return HOST_PRELUDE + top_level(indexer_src, names)


# ---------------------------------------------------------------------------
# Patch mechanics
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SRC, "set GLM53_V11_SRC to the dir holding the v11 vllm/")
class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v13kpooltail-"))
        self.root = self.tmp / "vllm"
        for rel in ("__init__.py", KPOOL, ATTENTION, INDEXER):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / rel, self.root / rel)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_patch(self, name="patch_v13_kpool_tail.py", root=None):
        return subprocess.run(
            [sys.executable, str(HERE / name), str(root or self.root)],
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

    def test_apply_twice_is_idempotent(self):
        before = tree_digest(self.root)
        first = self.run_patch()
        self.assertEqual(first.returncode, 0, first.stderr)
        after = tree_digest(self.root)
        self.assertEqual({k for k in after if before.get(k) != after[k]}, {KPOOL, ATTENTION, INDEXER})
        second = self.run_patch()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("(0 files written)", second.stdout)
        self.assertEqual(tree_digest(self.root), after)

    def test_refuses_on_drift_and_writes_nothing(self):
        for rel, old, new in (
            (KPOOL, "phys_slot = safe_pos % POOL_SIZE", "phys_slot = 0"),
            (INDEXER, "    out = slot_mapping.clone()\n", "    out = slot_mapping.clone()  # drift\n"),
        ):
            path = self.root / rel
            text = path.read_text()
            path.write_text(text.replace(old, new))
            before = tree_digest(self.root)
            res = self.run_patch()
            self.assertNotEqual(res.returncode, 0, rel)
            self.assertIn("refusing", res.stderr)
            self.assertEqual(tree_digest(self.root), before, "no partial writes")
            path.write_text(text)

    def test_touched_files_compile(self):
        self.assertEqual(self.run_patch().returncode, 0)
        for rel in (KPOOL, ATTENTION, INDEXER):
            py_compile.compile(
                str(self.root / rel),
                cfile=str(self.tmp / "pyc" / (rel.replace("/", "_") + "c")),
                doraise=True,
            )

    def test_every_ring_access_follows_the_ring_size(self):
        """After the patch no kernel addresses a tail block by the pool size."""
        self.assertEqual(self.run_patch().returncode, 0)
        src = (self.root / KPOOL).read_text()

        def body(kernel):
            text = src[src.index(f"def {kernel}(") :]
            return text[: text.index("\ndef ")]

        seed = body("_kpool_tail_seed_kernel")
        for text in ("// KPOOL", "t % KPOOL", "2 * KPOOL", "KPOOL * HEAD_DIM"):
            self.assertNotIn(text, seed)
        decode = body("_kpool_decode_update_batched_kernel")
        self.assertNotIn("// POOL_SIZE", decode)
        # Only the pool phase is still taken mod the pool size.
        self.assertEqual(re.findall(r"(\w+) = [^\n]*% POOL_SIZE", decode), ["slot"])

    def test_dockerfile_order_applies_and_reruns_clean(self):
        """Every v13 patch in Dockerfile order on one tree (the .py files of
        v11), so adaptive verify and this patch are applied together."""
        dockerfile = (HERE / "Dockerfile.sm121-v13").read_text()
        order = re.findall(r"python3 /tmp/(patch_v13_\w+\.py) ", dockerfile)
        self.assertEqual(order[-2:], ["patch_v13_verify.py", "patch_v13_kpool_tail.py"])
        root = self.tmp / "chain" / "vllm"
        shutil.copytree(
            SRC,
            root,
            ignore=lambda d, names: [n for n in names if not n.endswith(".py") and (Path(d) / n).is_file()],
        )
        before = tree_digest(root)
        for name in order:
            res = self.run_patch(name, root)
            self.assertEqual(res.returncode, 0, (name, res.stdout, res.stderr))
        after = tree_digest(root)
        for name in order:
            res = self.run_patch(name, root)
            noop = "(0 files written)" in res.stdout or " 0 file(s) written" in res.stdout
            self.assertTrue(res.returncode == 0 and noop, (name, res.stdout, res.stderr))
        self.assertEqual(tree_digest(root), after, "rerun changed files")
        changed = sorted(k for k in after if before.get(k) != after[k])
        self.assertTrue({KPOOL, ATTENTION, INDEXER} <= set(changed), changed)
        for rel in changed:
            cfile = self.tmp / "pyc" / (rel.replace("/", "_") + "c")
            py_compile.compile(str(root / rel), cfile=str(cfile), doraise=True)
        indexer = (root / INDEXER).read_text()
        self.assertEqual(indexer.count("GLM53_TAIL_STASH_MASK[:num_actual_tokens]"), 1)
        for _, _, new in patch.FILE_EDITS[INDEXER]:
            self.assertEqual(indexer.count(new), 1)
        self.assertEqual((root / KPOOL).read_text(), patched(KPOOL))
        self.assertEqual((root / ATTENTION).read_text(), patched(ATTENTION))

    def test_indexer_edits_commute_with_misc_and_verify(self):
        """The three patches that edit indexer.py give the same file in either order."""
        src = (SRC / INDEXER).read_text()
        edits = [misc.FILE_EDITS[INDEXER], verify.FILE_EDITS[INDEXER], patch.FILE_EDITS[INDEXER]]
        results = []
        for order in (edits, edits[::-1]):
            text = src
            for e in order:
                text, applied = patch.plan_edits(text, e)
                self.assertEqual(applied, len(e))
            results.append(text)
        self.assertEqual(results[0], results[1])


class PatchStaticTests(unittest.TestCase):
    def test_dockerfile_runs_this_patch_last_before_compileall(self):
        text = (HERE / "Dockerfile.sm121-v13").read_text()
        self.assertEqual(text.count('python3 /tmp/patch_v13_kpool_tail.py "$VLLM_ROOT"'), 1)
        self.assertLess(text.index("patch_v13_verify.py"), text.index("patch_v13_kpool_tail.py"))
        self.assertLess(text.index("patch_v13_kpool_tail.py"), text.index("compileall"))

    def test_other_v13_patches_leave_these_files_alone(self):
        for name in ("misc", "fp8", "census", "verify"):
            text = (HERE / f"patch_v13_{name}.py").read_text()
            self.assertNotIn("kpool_compress.py", text, name)
            self.assertNotIn("nvidia/attention.py", text, name)

    def test_switch_is_read_once_and_only_as_1(self):
        texts = "".join(new for edits in patch.FILE_EDITS.values() for _, _, new in edits)
        self.assertEqual(texts.count('os.environ.get("GLM53_KPOOL_TAIL_FIX") == "1"'), 1)
        self.assertEqual(texts.count("os.environ"), 1)


# ---------------------------------------------------------------------------
# Ring size and the tail spec
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SRC, "set GLM53_V11_SRC to the dir holding the v11 vllm/")
class RingSizeTests(unittest.TestCase):
    def test_off_keeps_kpool(self):
        for env in ({}, {"GLM53_KPOOL_TAIL_FIX": "0"}, {"GLM53_KPOOL_TAIL_FIX": "true"}):
            ns = ring_helper(patched(KPOOL), env)
            self.assertFalse(ns["GLM53_KPOOL_TAIL_FIX"])
            self.assertEqual([ns["glm53_tail_ring_slots"](4, k) for k in range(12)], [4] * 12)

    def test_on_is_the_next_power_of_two_past_the_bound(self):
        ring = ring_helper(patched(KPOOL), {"GLM53_KPOOL_TAIL_FIX": "1"})["glm53_tail_ring_slots"]
        self.assertEqual({k: ring(4, k) for k in (0, 1, 4, 5, 7)}, {0: 4, 1: 4, 4: 8, 5: 8, 7: 16})
        for kpool in (2, 4, 8, 16):
            for k in range(40):
                r = ring(kpool, k)
                self.assertGreaterEqual(r, max(kpool, k + kpool - 1), (kpool, k))
                self.assertEqual(r & (r - 1), 0, (kpool, k))  # a power of two
                self.assertTrue(r == kpool or r // 2 < k + kpool - 1, (kpool, k))  # minimal
                # Up to kpool * 32 it divides every block size the kpool indexer
                # accepts (a multiple of kpool * 32), so the scheduler's LCM of
                # the KV group block sizes stays the same.
                if r <= 32 * kpool:
                    self.assertEqual((32 * kpool) % r, 0, (kpool, k))


def tail_spec_kwargs(attention_src: str, kpool_src: str, env: dict, num_spec, block_size=2304):
    """Run Glm5NextTailCache.get_kv_cache_spec from `attention_src` against a
    stub vllm_config; return (KpoolTailSpec kwargs, log lines)."""
    cls = next(
        n
        for n in ast.parse(attention_src).body
        if isinstance(n, ast.ClassDef) and n.name == "Glm5NextTailCache"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_kv_cache_spec")
    code = "from __future__ import annotations\n\n"
    code += textwrap.dedent(ast.get_source_segment(attention_src, fn))
    lines = []
    ns = {
        "KpoolTailSpec": lambda **kw: kw,
        "torch": types.SimpleNamespace(bfloat16="bfloat16"),
        "logger": types.SimpleNamespace(info_once=lambda msg, *a: lines.append(msg % a)),
    }
    exec(compile(code, ATTENTION, "exec"), ns)  # noqa: S102
    # The patched method imports the switch and the ring size from kpool_compress.
    stubs = module_stubs(**(ring_helper(kpool_src, env) if "glm53_tail_ring_slots" in kpool_src else {}))
    spec = None if num_spec is None else types.SimpleNamespace(num_speculative_tokens=num_spec)
    config = types.SimpleNamespace(
        speculative_config=spec, cache_config=types.SimpleNamespace(block_size=block_size)
    )
    tail = types.SimpleNamespace(_index_kpool=4, head_dim=128)
    with mock.patch.dict(sys.modules, stubs):
        return ns["get_kv_cache_spec"](tail, config), lines


@unittest.skipUnless(HAVE_SRC, "set GLM53_V11_SRC to the dir holding the v11 vllm/")
class TailSpecTests(unittest.TestCase):
    def setUp(self):
        self.attention, self.kpool = patched(ATTENTION), patched(KPOOL)
        self.v11, _ = tail_spec_kwargs((SRC / ATTENTION).read_text(), "", {}, 7)

    def spec(self, env, num_spec, **kw):
        return tail_spec_kwargs(self.attention, self.kpool, env, num_spec, **kw)

    def test_off_is_v11(self):
        for env in ({}, {"GLM53_KPOOL_TAIL_FIX": "0"}):
            for num_spec in (None, 4, 7):
                kwargs, lines = self.spec(env, num_spec)
                self.assertEqual(kwargs, self.v11)
                self.assertEqual(lines, [])
        self.assertEqual((self.v11["block_size"], self.v11["sliding_window"]), (4, 4))

    def test_on_sizes_the_ring_for_the_verify_window(self):
        on = {"GLM53_KPOOL_TAIL_FIX": "1"}
        for num_spec, ring in ((7, 16), (4, 8), (5, 8), (None, 4)):
            kwargs, lines = self.spec(on, num_spec)
            self.assertEqual(kwargs, {**self.v11, "block_size": ring}, num_spec)
            self.assertEqual(
                lines,
                [f"GLM53_KPOOL_TAIL_FIX: kpool tail ring {ring} slots (index_kpool=4, k={num_spec or 0})"],
            )

    def test_on_refuses_a_ring_that_does_not_divide_the_block(self):
        with self.assertRaises(AssertionError):
            self.spec({"GLM53_KPOOL_TAIL_FIX": "1"}, 7, block_size=2312)
        self.spec({"GLM53_KPOOL_TAIL_FIX": "1"}, 7, block_size=2048)


# ---------------------------------------------------------------------------
# The tail builder on the runners' metadata (torch only)
# ---------------------------------------------------------------------------
def tail_builder(indexer_src: str, switch: bool, ring: int):
    """KpoolTailMetadataBuilder of `indexer_src`, built as the runner builds
    it; it reads the switch from kpool_compress (stubbed)."""
    ns: dict = {}
    exec(compile(host_source(indexer_src), INDEXER, "exec"), ns)  # noqa: S102
    with mock.patch.dict(sys.modules, module_stubs(GLM53_KPOOL_TAIL_FIX=switch)):
        return ns["KpoolTailMetadataBuilder"](types.SimpleNamespace(block_size=ring), [], None, "cpu"), ns


@unittest.skipUnless(HAVE_SRC and HAVE_TORCH, "needs GLM53_V11_SRC and torch")
class TailBuilderTests(unittest.TestCase):
    """One mixed batch as each runner hands it to the tail builder: a verify
    request (8 rows from position 21), a prefill chunk that starts mid-prompt
    (6 rows from 36) and a one-token decode (position 3), padded by 8 tokens of
    a padding request. The generic slot kernel filled the persistent row with
    bt[req][pos // R] * R + pos % R and -1 past the real tokens."""

    RING, BLOCKS = 16, [5, 9, 2]
    POS = [list(range(21, 29)), list(range(36, 42)), [3]]

    def batch(self, v2: bool):
        import torch

        qsl = [0, 8, 14, 15, 15]  # the padding request has no tokens
        n, real = 23, 15
        pos = [p for r in self.POS for p in r]
        bt = torch.zeros(4, 8, dtype=torch.int32)
        bt[:3, 0] = torch.tensor(self.BLOCKS, dtype=torch.int32)
        row = torch.full((64,), -1, dtype=torch.int64)  # BlockTables.slot_mappings[tail group]
        req = [b for b, r in enumerate(self.POS) for _ in r]
        for t, (b, p) in enumerate(zip(req, pos)):
            row[t] = int(bt[b, p // self.RING]) * self.RING + p % self.RING
        want = torch.tensor([self.BLOCKS[b] * self.RING + p % self.RING for b, p in zip(req, pos)])
        cm = types.SimpleNamespace(
            slot_mapping=row[:n],
            block_table_tensor=bt,
            query_start_loc=torch.tensor(qsl, dtype=torch.int32),
            query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
            seq_lens=torch.tensor([29, 42, 4, 0], dtype=torch.int32),
            # V1 hands the model positions (padding entries are stale); V2 none.
            positions=None if v2 else torch.tensor(pos + [7] * (n - real)),
            num_actual_tokens=n,
            num_reqs=4,
            max_seq_len=42,
        )
        return cm, row, want, real

    def test_off_is_v11(self):
        v11_src, kt_src = (SRC / INDEXER).read_text(), patched(INDEXER)
        for v2 in (True, False):
            outs = []
            for src in (v11_src, kt_src):
                cm, row, _, _ = self.batch(v2)
                before = row.clone()
                builder, _ = tail_builder(src, False, self.RING)
                out = builder.build(0, cm).slot_mapping
                # v11: V2 keeps the generic slots (the row itself); V1 maps into a clone.
                self.assertEqual(out.data_ptr() == row.data_ptr(), v2)
                self.assertTrue(row.equal(before))
                outs.append(out.clone())
            self.assertTrue(outs[0].equal(outs[1]), v2)

    def test_on_maps_the_real_tokens_in_place(self):
        import torch

        for v2 in (True, False):
            cm, row, want, real = self.batch(v2)
            builder, _ = tail_builder(patched(INDEXER), True, self.RING)
            out = builder.build(0, cm).slot_mapping
            self.assertEqual(out.data_ptr(), row.data_ptr())
            self.assertTrue(out[:real].equal(want), (v2, out[:real], want))
            self.assertTrue(row[real:].equal(torch.full((64 - real,), -1)), v2)

    def test_on_with_adaptive_verify_masks_rows_on_v2(self):
        """With both patches the V2 builder runs adaptive verify's tail mask
        (v11's V2 builder never calls the slot mapping, so the mask was dead)."""
        import torch

        src = apply(apply((SRC / INDEXER).read_text(), verify.FILE_EDITS[INDEXER]), patch.FILE_EDITS[INDEXER])
        cm, row, want, real = self.batch(True)
        builder, ns = tail_builder(src, True, self.RING)
        mask = torch.zeros(64, dtype=torch.bool)
        mask[[5, 6, 7]] = True  # verify rows past the width
        ns["GLM53_TAIL_STASH_MASK"] = mask
        out = builder.build(0, cm).slot_mapping
        want[[5, 6, 7]] = -1
        self.assertTrue(out[:real].equal(want))


# ---------------------------------------------------------------------------
# Simulation on v11's own kernels (Triton CPU interpreter)
# ---------------------------------------------------------------------------
STUBS = {
    "vllm/__init__.py": "",
    "vllm/logger.py": "import logging\n\n\ndef init_logger(name):\n    return logging.getLogger(name)\n",
    "vllm/triton_utils/__init__.py": "import triton\nimport triton.language as tl\n",
    "vllm/v1/__init__.py": "",
    "vllm/v1/worker/__init__.py": "",
    "vllm/v1/worker/gpu/__init__.py": "",
    "vllm/v1/worker/gpu/spec_decode/__init__.py": "",
}

SIM_SCRIPT = textwrap.dedent(
    r'''
    import importlib
    import importlib.util
    import json
    import os
    import random
    import sys
    import zlib
    from collections import Counter
    from types import SimpleNamespace

    import torch

    root, cfg = sys.argv[1], json.loads(sys.argv[2])
    sys.path.insert(0, root)
    import host_common  # v11's tail view carving and prefill pool insert
    from vllm.v1.worker.gpu.spec_decode.glm53_adaptive_verify import AdaptiveVerify

    KPOOL_MOD = "vllm.models.glm5next.nvidia.ops.kpool_compress"


    def kernels(name, path, fix):
        os.environ["GLM53_KPOOL_TAIL_FIX"] = "1" if fix else "0"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ns = {"torch": torch, "kpool_compress_and_write_cache": mod.kpool_compress_and_write_cache}
        exec(host_common.COMPRESS_INSERT_SRC, ns)
        mod.compress_insert = ns["_kpool_compress_insert"]
        return mod


    KC = {
        "v11": kernels("kc_v11", f"{root}/kpool_v11.py", False),
        "off": kernels("kc_off", f"{root}/kpool_patched.py", False),
        "fix": kernels("kc_fix", f"{root}/kpool_patched.py", True),
    }
    del os.environ["GLM53_KPOOL_TAIL_FIX"]

    D, KPOOL, PAGE = 128, 4, 64  # index head dim, index_kpool (09b04e5), pool page
    PAGE_BYTES = PAGE * (D + 4)  # one pool page per block here (the real block holds 9)
    NB = 32  # room for a block id taken by kpool instead of the ring (a mutation)
    K, C = cfg["k"], cfg["c"]  # draft tokens, concurrent requests
    POOL_PAGE = [10, 11, 12, 13][:C]  # request b's indexer block
    TAIL_BLOCK = [5, 6, 7, 8][:C]  # request b's tail block; v11's seed lands in page 1
    STATES = [3, 1, 0, 2][:C]  # request b's req_state (adaptive verify)
    OWNED = set(POOL_PAGE) | set(TAIL_BLOCK)
    FILL = 0x3C  # every byte of the shared indexer tensor starts at this value
    APE = torch.randn(KPOOL, D, generator=torch.Generator().manual_seed(7))
    MAX_TOKENS, BT_COLS = 256, 64  # the slot-mapping row; the V2 tail block table width
    PAD, STALE_POS = 1, 6  # padding requests of a FULL batch; their position entries
    RING = cfg.get("ring") or KC["fix"].glm53_tail_ring_slots(KPOOL, K)
    AV = "_av" if cfg["av"] else ""


    def vec(key):
        """A row's indexer K and gate: a function of the inputs the row saw."""
        g = torch.Generator().manual_seed(zlib.crc32(repr(key).encode()))
        return torch.randn(2, D, generator=g).to(torch.bfloat16)


    class Layout:
        """The indexer tensor with the tail co-owning it (v11's GLM-5-Next
        layout), or v11's reference: a contiguous tail tensor of its own."""

        def __init__(self, ring, padded, stale):
            self.ring = ring
            self.raw = torch.full((NB * PAGE_BYTES,), FILL, dtype=torch.uint8)
            self.kv = self.raw.view(NB, PAGE, D + 4)
            if padded:
                spec = SimpleNamespace(
                    dtype=torch.bfloat16, page_size_padded=PAGE_BYTES, page_size_bytes=PAGE_BYTES
                )
                self.tail = host_common._reshape_attention_kv_cache(
                    self.raw, spec, (NB, 2, ring, D), (0, 1, 2, 3), NB, None
                )
            else:
                self.tail = torch.zeros(NB, 2, ring, D, dtype=torch.bfloat16)
            # Stale ring contents: a slot read before this request wrote it
            # shows up as a differing pool.
            g = torch.Generator().manual_seed(stale)
            for blk in TAIL_BLOCK:
                self.tail[blk] = torch.randn(2, ring, D, generator=g).to(torch.bfloat16)


    class Runner:
        """How a run builds the tail slots: the generic slot kernel fills the
        tail group's persistent slot-mapping row, then the kpool tail builder
        (v11's or the patched one) runs on the V2 runner's metadata (no
        positions) or the V1 runner's (positions)."""

        def __init__(self, host, kern, ring, v2, full):
            self.host = importlib.import_module(host)
            sys.modules[KPOOL_MOD] = KC[kern]  # the patched builder reads the switch there
            spec = SimpleNamespace(block_size=ring)
            self.builder = self.host.KpoolTailMetadataBuilder(spec, [], None, "cpu")
            self.ring, self.v2, self.full = ring, v2, full
            self.row = torch.full((MAX_TOKENS,), -1, dtype=torch.int64)
            self.builds = self.in_place = 0

        def slots(self, batch, blocks, mask, pad):
            """Slots for batch[i] = [(pos, key)] (tail block blocks[i]) plus
            `pad` padding requests of 1 + K tokens."""
            qsl = [0]
            for r in batch:
                qsl.append(qsl[-1] + len(r))
            real = qsl[-1]
            n = real + pad * (1 + K)
            qsl += [real] * pad
            bt = torch.zeros(len(qsl) - 1, BT_COLS, dtype=torch.int32)  # padding rows zeroed
            bt[: len(blocks), 0] = torch.tensor(blocks, dtype=torch.int32)
            pos = [p for r in batch for p, _ in r]
            self.row.fill_(-1)  # the generic slot kernel pads past the real tokens
            for t, (i, p) in enumerate((i, p) for i, r in enumerate(batch) for p, _ in r):
                self.row[t] = int(bt[i, p // self.ring]) * self.ring + p % self.ring
            cm = SimpleNamespace(
                slot_mapping=self.row[:n],
                block_table_tensor=bt,
                query_start_loc=torch.tensor(qsl, dtype=torch.int32),
                query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
                seq_lens=torch.tensor([r[-1][0] + 1 for r in batch] + [0] * pad, dtype=torch.int32),
                positions=None if self.v2 else torch.tensor(pos + [STALE_POS] * (n - real)),
                num_actual_tokens=n,
                num_reqs=len(qsl) - 1,
                max_seq_len=max(p for p in pos) + 1,
            )
            if hasattr(self.host, "GLM53_TAIL_STASH_MASK"):
                self.host.GLM53_TAIL_STASH_MASK = mask
            out = self.builder.build(0, cm).slot_mapping
            self.builds += 1
            self.in_place += out.data_ptr() == self.row.data_ptr()
            return out


    def pool_slot(b, p):
        """Pool-granular slot mapping: only a pool's last token has a slot."""
        return POOL_PAGE[b] * PAGE + p // KPOOL if p % KPOOL == KPOOL - 1 else -1


    def forward(kc, run, lay, rows, state=None, widths=None, uniform=False):
        """One target forward over rows[b] = [(pos, key)] of request b ([]: not
        scheduled), in the runner's batch order: decodes first (the indexer runs
        a request of <= 1 + K tokens on its decode path), then prefills. A
        uniform verify batch runs as a padded FULL graph: its replay reads the
        slot-mapping row it captured, not the tensor the builder returns."""
        dec = [b for b, r in enumerate(rows) if 0 < len(r) <= 1 + K]
        pre = [b for b, r in enumerate(rows) if len(r) > 1 + K]
        batch = [rows[b] for b in dec + pre]
        qsl = [0]
        for r in batch:
            qsl.append(qsl[-1] + len(r))
        full = run.full and uniform
        pad = PAD if full else 0
        mask = None
        if state is not None:
            logits, local, states = [], [], []
            for i, b in enumerate(dec + pre):
                if widths[b] is not None:
                    state.verify_len[STATES[b]] = widths[b]
                    logits += range(qsl[i], qsl[i + 1])
                    local += range(len(batch[i]))
                    states += [STATES[b]] * len(batch[i])
            state.prepare(SimpleNamespace(
                num_tokens_after_padding=MAX_TOKENS, num_draft_tokens=len(logits),
                logits_indices=torch.tensor(logits, dtype=torch.int64),
                expanded_local_pos=torch.tensor(local, dtype=torch.int64),
                expanded_idx_mapping=torch.tensor(states, dtype=torch.int64),
            ))
            mask = state.masked
        flat = run.slots(batch, [TAIL_BLOCK[b] for b in dec + pre], mask, pad)
        tslots = run.row if full else flat
        if dec:
            nd, n = len(dec) + pad, max(len(rows[b]) for b in dec)
            key = torch.zeros(nd, n, 2, D, dtype=torch.bfloat16)
            pos, slot, tslot = (torch.full((nd, n), -1, dtype=torch.int32) for _ in range(3))
            for i, b in enumerate(dec):
                for t, (p, k) in enumerate(rows[b]):
                    key[i, t], pos[i, t], slot[i, t] = vec(k), p, pool_slot(b, p)
                    tslot[i, t] = tslots[qsl[i] + t]
            for i in range(len(dec), nd):  # padding: stale positions, no pool slot
                start = qsl[-1] + (i - len(dec)) * n
                pos[i], tslot[i] = STALE_POS, tslots[start : start + n]
            kc.kpool_decode_update_and_maybe_write_cache_batched(
                lay.kv, lay.tail, tslot, key[:, :, 0].contiguous(), key[:, :, 1].contiguous(),
                APE, slot, pos, KPOOL, D, round_scale=True,
            )
        if pre:
            kg = torch.stack([vec(k) for b in pre for _, k in rows[b]])
            k, g = kg[:, 0].contiguous(), kg[:, 1].contiguous()
            slot = torch.tensor([pool_slot(b, p) for b in pre for p, _ in rows[b]])
            kc.compress_insert(k, g, APE, lay.kv, slot, KPOOL, D, round_scale=True)
            kc.kpool_seed_tail_cache(lay.tail, k, g, tslots[qsl[len(dec)] : qsl[-1]], KPOOL, D)


    def run_spec(kc, run, lay, sched, av):
        """sched[s][b]: ("prompt", n) a prompt chunk of n tokens, ("verify", m,
        a) 1 + K verify rows with the drafts past a rejected (and the rows past
        m masked, with adaptive verify), or None."""
        P = [0] * C
        state = AdaptiveVerify(0.0, K, None, 4, MAX_TOKENS, "cpu") if av else None
        for s, step in enumerate(sched):
            rows = []
            for b, op in enumerate(step):
                if op is None:
                    rows.append([])
                elif op[0] == "prompt":
                    rows.append([(P[b] + j, ("true", b, P[b] + j)) for j in range(op[1])])
                else:
                    rows.append([
                        (P[b] + j, ("true", b, P[b] + j) if j <= op[2] else
                         ("draft" if j <= op[1] else "masked", s, b, P[b] + j))
                        for j in range(K + 1)
                    ])
            widths = [op[1] if op and op[0] == "verify" else None for op in step]
            forward(kc, run, lay, rows, state, widths, uniform=None not in widths)
            for b, op in enumerate(step):
                if op:
                    P[b] += op[1] if op[0] == "prompt" else op[2] + 1
        return P


    def run_plain(kc, run, lay, sched, ends):
        """Non-speculative decoding of the committed tokens: each request's
        prompt chunks in forwards of their own, then one row per request per step."""
        P = [0] * C
        for step in sched:
            for b, op in enumerate(step):
                if op and op[0] == "prompt":
                    rows = [[] for _ in range(C)]
                    rows[b] = [(P[b] + j, ("true", b, P[b] + j)) for j in range(op[1])]
                    forward(kc, run, lay, rows)
                    P[b] += op[1]
        for p in range(min(P), max(ends)):
            rows = [[(p, ("true", b, p))] if P[b] <= p < ends[b] else [] for b in range(C)]
            if any(rows):
                forward(kc, run, lay, rows)


    def schedule(rng):
        """Every prompt whole at step 0, then cfg["steps"] verify steps. With
        cfg["chunked"], request 1's prompt comes in two chunks next to request
        0's verify steps: a prefill-path chunk ending pool-aligned (as the
        mamba-align split ends them), then a chunk of either path."""
        def verify():
            m = rng.randint(1, K) if cfg["av"] else K
            return ("verify", m, min(m, rng.choice([0, 0, 1, 2, rng.randint(0, K), K])))

        if cfg.get("chunked"):
            sched = [
                [("prompt", rng.randint(5, 13)), None],
                [verify(), ("prompt", KPOOL * rng.randint(3, 5))],
                [verify(), ("prompt", rng.randint(2, 12))],
            ]
        else:
            sched = [[("prompt", rng.randint(5, 13)) for _ in range(C)]]
        return sched + [[verify() for _ in range(C)] for _ in range(cfg["steps"])]


    def pool(lay, b, q):
        """Pool q of request b: its fp8 K and its fp32 scale bytes."""
        flat = lay.kv[POOL_PAGE[b]].reshape(-1)
        return torch.cat([flat[q * D:(q + 1) * D], flat[PAGE * D + 4 * q:PAGE * D + 4 * q + 4]])


    def foreign(lay):
        """Bytes changed in blocks no request owns."""
        pages = lay.raw.view(NB, PAGE_BYTES)
        return sum(int((pages[p] != FILL).sum()) for p in range(NB) if p not in OWNED)


    RUNS = {  # name: (kernels, indexer, ring, V2 runner, FULL graphs, padded tail view, speculative)
        "fix": ("fix", "kt", RING, True, True, True, True),  # as served: V2, SPEC=dflash2 and mtp
        "fix-v1": ("fix", "kt", RING, False, True, True, True),  # the V1 runner (positions)
        "off": ("off", "kt", KPOOL, True, True, True, True),  # patched source, switch off
        "v11": ("v11", "v11", KPOOL, True, True, True, True),  # v11 as served
        "v11-contig": ("v11", "v11", KPOOL, False, False, False, True),  # overwrite alone
        "v11-nonspec": ("v11", "v11", KPOOL, False, False, True, False),  # seed addressing alone
    }
    rng = random.Random(cfg["seed"])
    seen = Counter()
    seen["fix|ring"] = RING
    for trial in range(cfg["trials"]):
        sched = schedule(rng)
        ends, verifies = [0] * C, [0] * C
        for step in sched:
            for b, op in enumerate(step):
                verifies[b] += bool(op and op[0] == "verify")
        for step in sched:
            for b, op in enumerate(step):
                if op and op[0] == "verify":
                    verifies[b] -= 1
                    # The case that needs the whole bound, with a later step to read it.
                    seen["worst-case steps"] += verifies[b] > 0 and ends[b] % KPOOL == KPOOL - 2 and op[2] == 0
                if op:
                    ends[b] += op[1] if op[0] == "prompt" else op[2] + 1
        ref = Layout(KPOOL, padded=False, stale=1)
        run_plain(KC["v11"], Runner("host_v11", "v11", KPOOL, False, False), ref, sched, ends)
        for name in cfg["runs"]:
            kern, idx, ring, v2, full, padded, spec = RUNS[name]
            run = Runner(f"host_{idx}{AV}", kern, ring, v2, full)
            lay = Layout(ring, padded, stale=2)
            if spec:
                assert run_spec(KC[kern], run, lay, sched, cfg["av"]) == ends
            else:
                run_plain(KC[kern], run, lay, sched, ends)
            seen[name + "|builds"] += run.builds
            seen[name + "|in place"] += run.in_place
            for b in range(C):
                for q in range(ends[b] // KPOOL):  # every pool whose tokens are committed
                    seen[name + "|pools"] += 1
                    seen[name + "|pools differ"] += not torch.equal(pool(lay, b, q), pool(ref, b, q))
                # The ring slots a later step can read: the committed part of the open pool.
                for c in range(ends[b] // KPOOL * KPOOL, ends[b]):
                    seen[name + "|ring slots"] += 1
                    got = lay.tail[TAIL_BLOCK[b], :, c % ring]
                    seen[name + "|ring differs"] += not torch.equal(got, ref.tail[TAIL_BLOCK[b], :, c % KPOOL])
            if padded:
                seen[name + "|foreign bytes"] += foreign(lay)
            if name == "off":
                off_raw = lay.raw
            if name == "v11" and "off" in cfg["runs"]:
                seen["off|raw differs from v11"] += not torch.equal(off_raw, lay.raw)
    print(json.dumps(dict(sorted(seen.items()))))
    '''
)


def build_sim_root(root: Path, edits: dict) -> None:
    """Stub vllm, v11's kernels and the kernels as `edits` patch them, v11's
    tail view carving and prefill insert, the adaptive-verify module, and the
    tail builder four ways: v11 and patched by `edits`, each without and with
    adaptive verify (applied first, as the Dockerfile orders them)."""
    for rel, text in STUBS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    (root / "vllm" / AV_MODULE).write_text(verify.MODULE)
    shutil.copy2(SRC / KPOOL, root / "kpool_v11.py")
    (root / "kpool_patched.py").write_text(patched(KPOOL, edits))
    (root / "host_common.py").write_text(
        "from __future__ import annotations\n\nfrom math import prod\n\nimport torch\n\n\n"
        "def get_dtype_size(dtype):\n    return torch.tensor([], dtype=dtype).element_size()\n\n\n"
        + top_level((SRC / ATTN_UTILS).read_text(), {"_reshape_attention_kv_cache"})
        + "\n\nCOMPRESS_INSERT_SRC = "
        + repr(top_level((SRC / INDEXER_OP).read_text(), {"_kpool_compress_insert"}))
        + "\n"
    )
    v11 = (SRC / INDEXER).read_text()
    av = apply(v11, verify.FILE_EDITS[INDEXER])
    for name, src in (
        ("host_v11", v11),
        ("host_v11_av", av),
        ("host_kt", apply(v11, edits[INDEXER])),
        ("host_kt_av", apply(av, edits[INDEXER])),
    ):
        (root / f"{name}.py").write_text(host_source(src))


def simulate(edits=None, **cfg) -> dict:
    """Run SIM_SCRIPT with the patch's `edits`; return its counters (or {"crashed": stderr})."""
    with tempfile.TemporaryDirectory(prefix="v13kpooltail-sim-") as tmp:
        build_sim_root(Path(tmp), edits or patch.FILE_EDITS)
        res = subprocess.run(
            [sys.executable, "-c", SIM_SCRIPT, tmp, json.dumps(cfg)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "TRITON_INTERPRET": "1"},
        )
    if res.returncode != 0:  # a negative code is a signal (an out-of-bounds address)
        return {"crashed": f"rc={res.returncode} {res.stderr[-2000:]}"}
    return json.loads(res.stdout.strip().splitlines()[-1])


# Scenario: (name, config). k=7 is DFlash2-7, k=4 the MTP-4 rollback and k=5
# at c=4 the four-way rollback (NUM_SPECULATIVE_TOKENS=5 MAX_NUM_SEQS=4).
SCENARIOS = [
    ("DFlash2 k=7, c=1", dict(k=7, c=1, av=False, trials=10, steps=5, seed=1)),
    ("DFlash2 k=7, c=2", dict(k=7, c=2, av=False, trials=10, steps=5, seed=2)),
    ("MTP k=4, c=2", dict(k=4, c=2, av=False, trials=10, steps=5, seed=3)),
    ("k=7, c=2 + GLM53_ADAPTIVE_VERIFY", dict(k=7, c=2, av=True, trials=10, steps=5, seed=4)),
    ("k=5, c=4", dict(k=5, c=4, av=False, trials=5, steps=5, seed=5)),
    ("k=7, c=2, chunked prompt next to verify", dict(k=7, c=2, av=False, chunked=True, trials=10, steps=4, seed=6)),
]
ALL_RUNS = ["fix", "fix-v1", "off", "v11", "v11-contig", "v11-nonspec"]


def table(title: str, seen: dict) -> str:
    rows = [f"  {title}"]
    for run in ALL_RUNS:
        if f"{run}|pools" in seen:
            rows.append(
                "    %-12s pools differ %3d/%-3d  readable ring slots differ %2d/%-2d  foreign bytes %d"
                % (run, seen[f"{run}|pools differ"], seen[f"{run}|pools"], seen[f"{run}|ring differs"],
                   seen[f"{run}|ring slots"], seen.get(f"{run}|foreign bytes", 0))
            )
    return "\n".join(rows)


@unittest.skipUnless(HAVE_SRC and HAVE_TORCH and HAVE_TRITON, "needs GLM53_V11_SRC, torch, triton")
class KpoolTailSimulationTest(unittest.TestCase):
    """The fix leaves every committed pool and every readable ring slot equal
    to non-speculative decoding, on the V2 path the serve runs and on V1; v11
    does not."""

    def test_committed_state_equals_non_speculative_decoding(self):
        report = []
        for title, cfg in SCENARIOS:
            with self.subTest(title):
                seen = simulate(runs=ALL_RUNS, **cfg)
                self.assertNotIn("crashed", seen)
                report.append(table(title, seen))
                self.assertEqual(seen["fix|ring"], 16 if cfg["k"] == 7 else 8)
                for run in ("fix", "fix-v1"):
                    self.assertGreater(seen[f"{run}|pools"], 40)
                    # The fix: nothing differs, nothing outside a request's blocks moves.
                    self.assertEqual(seen[f"{run}|pools differ"], 0, run)
                    self.assertEqual(seen[f"{run}|ring differs"], 0, run)
                    self.assertEqual(seen[f"{run}|foreign bytes"], 0, run)
                    # Every build wrote the slots into the persistent row.
                    self.assertEqual(seen[f"{run}|in place"], seen[f"{run}|builds"], run)
                # Switch off: byte-identical to v11 on the same layout.
                self.assertEqual(seen["off|raw differs from v11"], 0)
                # v11 as served (V2): pools built from rejected drafts and, from
                # c=2 on, from other requests (one ring in block 0 for all).
                self.assertGreater(seen["v11|pools differ"], 0)
                # v11's bug, documented: rejected rows overwrite committed slots.
                self.assertGreater(seen["v11-contig|pools differ"], 0)
                self.assertGreater(seen["v11-contig|ring differs"], 0)
                # v11's seed addressing, documented: without speculation the
                # boundary pools still differ and bytes land in foreign blocks.
                self.assertEqual(seen["v11-nonspec|ring differs"], 0)
                self.assertGreater(seen["v11-nonspec|pools differ"], 0)
                self.assertGreater(seen["v11-nonspec|foreign bytes"], 0)
        print("\n" + "\n".join(report))


# Each mutation breaks one part of the fix; the simulation must see it.
MUTATIONS = {
    "stash still pos % kpool": (KPOOL, "phys_slot = safe_pos % RING", "phys_slot = safe_pos % POOL_SIZE"),
    "completion reads pos % kpool": (
        KPOOL,
        """            denom = tl.full((BLOCK_D,), 0.0, tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % RING""",
        """            denom = tl.full((BLOCK_D,), 0.0, tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % POOL_SIZE""",
    ),
    "decode block id by kpool": (
        KPOOL,
        "block = tl.maximum(tail_slot, 0).to(tl.int64) // RING",
        "block = tl.maximum(tail_slot, 0).to(tl.int64) // POOL_SIZE",
    ),
    "seed keeps v11 addressing": (
        KPOOL,
        "        block_elems, half_elems = tail_kv_cache.stride(0), tail_kv_cache.stride(1)",
        "        block_elems, half_elems = 2 * ring * head_dim, ring * head_dim",
    ),
    "seed block id by kpool": (KPOOL, "blk = t // RING", "blk = t // KPOOL"),
    "ring one short of the bound": (
        KPOOL,
        "return max(kpool, 1 << (num_speculative_tokens + kpool - 2).bit_length())",
        "return num_speculative_tokens + kpool - 2",
    ),
    "V2 keeps the generic slots": (
        INDEXER,
        "        if positions is not None or self.glm53_tail_fix:\n",
        "        if positions is not None:\n",
    ),
    "tail slots in a clone (FULL replay reads the row)": (
        INDEXER,
        "    out = slot_mapping if in_place else slot_mapping.clone()\n",
        "    out = slot_mapping.clone()\n",
    ),
    "V2 positions off by a request": (
        INDEXER,
        "last = seq_lens[:num_reqs] - query_start_loc[1 : num_reqs + 1]",
        "last = seq_lens[:num_reqs] - query_start_loc[:num_reqs]",
    ),
    "CUDA-graph padding mapped": (
        INDEXER,
        """            num_tokens = int(
                common_attn_metadata.query_start_loc_cpu[common_attn_metadata.num_reqs]
            )""",
        "            num_tokens = common_attn_metadata.num_actual_tokens",
    ),
}


@unittest.skipUnless(HAVE_SRC and HAVE_TORCH and HAVE_TRITON, "needs GLM53_V11_SRC, torch, triton")
class KpoolTailMutationTest(unittest.TestCase):
    CFG = dict(k=7, c=2, av=False, trials=10, steps=5, seed=7, runs=["fix"])

    def test_the_bound_is_exact(self):
        """R = k + kpool - 1 is enough (one less is a mutation below); the
        shipped power of two sits above it."""
        tight = mutated(
            KPOOL,
            "return max(kpool, 1 << (num_speculative_tokens + kpool - 2).bit_length())",
            "return num_speculative_tokens + kpool - 1",
        )
        seen = simulate(tight, **self.CFG)
        self.assertEqual((seen["fix|ring"], seen["fix|pools differ"], seen["fix|ring differs"]), (10, 0, 0))
        self.assertGreater(seen["worst-case steps"], 0)

    def test_every_mutation_is_caught(self):
        for name, (rel, old, new) in MUTATIONS.items():
            with self.subTest(name):
                seen = simulate(mutated(rel, old, new), **self.CFG)
                caught = "crashed" in seen or (
                    seen["fix|pools differ"] + seen["fix|ring differs"] + seen["fix|foreign bytes"] > 0
                )
                self.assertTrue(caught, (name, seen))
                if "crashed" not in seen:
                    print(f"\n  {name}: pools differ {seen['fix|pools differ']}/{seen['fix|pools']}, "
                          f"ring differs {seen['fix|ring differs']}, foreign bytes {seen['fix|foreign bytes']}, "
                          f"in place {seen['fix|in place']}/{seen['fix|builds']}")
                else:
                    print(f"\n  {name}: simulation crashed ({seen['crashed'].split()[0]})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
