"""CPU tests for docker/patch_v13_kpool_tail.py (GLM53_KPOOL_TAIL_FIX).

    GLM53_V11_SRC=/path/to/v11src python3 docker/test_v13_kpool_tail.py

GLM53_V11_SRC is the directory that holds the vllm/ package of glm53-sm121-v11
(read-only; the tests patch temporary copies). Without it everything but the
Dockerfile check skips. The simulation and mutation tests also need torch and
triton: they run v11's own indexer kernels (the K-pool decode update, the
prefill pool compress and the tail seed) under the Triton CPU interpreter
(TRITON_INTERPRET=1), on the tail view v11 carves out of the indexer tensor,
with v11's tail slot mapping. They drive a prompt plus several verify steps at
c=1 and c=2 with random acceptance and compare every committed pool and every
tail-ring slot a later step can read against non-speculative decoding.
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


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def top_level(src: str, names: set[str]) -> str:
    """The top-level defs and assignments of `src` named in `names`."""
    keep = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef):
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


def patched(rel: str) -> str:
    """`rel` of the v11 tree as the patch writes it."""
    src, applied = patch.plan_edits((SRC / rel).read_text(), patch.FILE_EDITS[rel])
    assert applied == len(patch.FILE_EDITS[rel])
    return src


def mutated(rel: str, old: str, new: str) -> str:
    """`rel` patched with `old` replaced by `new` in the one edit that has it."""
    edits = patch.FILE_EDITS[rel]
    assert sum(old in text for _, _, text in edits) == 1, old
    edits = [(what, anchor, text.replace(old, new)) for what, anchor, text in edits]
    return patch.plan_edits((SRC / rel).read_text(), edits)[0]


def ring_helper(kpool_src: str, env: dict):
    """glm53_tail_ring_slots from `kpool_src`, with GLM53_* read from `env`."""
    ns: dict = {}
    code = "import os\n\n" + top_level(kpool_src, {"GLM53_KPOOL_TAIL_FIX", "glm53_tail_ring_slots"})
    with mock.patch.dict(os.environ, env, clear=True):
        exec(compile(code, KPOOL, "exec"), ns)  # noqa: S102
    return ns


# ---------------------------------------------------------------------------
# Patch mechanics
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAVE_SRC, "set GLM53_V11_SRC to the dir holding the v11 vllm/")
class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v13kpooltail-"))
        self.root = self.tmp / "vllm"
        for rel in ("__init__.py", KPOOL, ATTENTION):
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
        self.assertEqual({k for k in after if before.get(k) != after[k]}, {KPOOL, ATTENTION})
        second = self.run_patch()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("(0 files written)", second.stdout)
        self.assertEqual(tree_digest(self.root), after)

    def test_refuses_on_drift_and_writes_nothing(self):
        path = self.root / KPOOL
        path.write_text(path.read_text().replace("phys_slot = safe_pos % POOL_SIZE", "phys_slot = 0"))
        before = tree_digest(self.root)
        res = self.run_patch()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("refusing", res.stderr)
        self.assertEqual(tree_digest(self.root), before, "no partial writes")

    def test_touched_files_compile(self):
        self.assertEqual(self.run_patch().returncode, 0)
        for rel in (KPOOL, ATTENTION):
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
        self.assertEqual((root / KPOOL).read_text(), patched(KPOOL))
        self.assertEqual((root / ATTENTION).read_text(), patched(ATTENTION))


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
    parts = KPOOL_MOD.split(".")
    stubs = {name: types.ModuleType(name) for name in (".".join(parts[:i]) for i in range(1, len(parts) + 1))}
    if "glm53_tail_ring_slots" in kpool_src:
        vars(stubs[KPOOL_MOD]).update(ring_helper(kpool_src, env))
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
    import host_av  # compute_kpool_tail_slot_mapping with the adaptive-verify hook
    import host_v11  # v11 compute_kpool_tail_slot_mapping, tail view, prefill insert
    from vllm.v1.worker.gpu.spec_decode.glm53_adaptive_verify import AdaptiveVerify


    def kernels(name, path, fix):
        os.environ["GLM53_KPOOL_TAIL_FIX"] = "1" if fix else "0"
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ns = {"torch": torch, "kpool_compress_and_write_cache": mod.kpool_compress_and_write_cache}
        exec(host_v11.COMPRESS_INSERT_SRC, ns)
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
    POOL_PAGE = [10, 11]  # request b's indexer block
    TAIL_BLOCK = [5, 6]  # request b's tail block; v11's seed lands in page 1
    STATES = [3, 1]  # request b's req_state (adaptive verify)
    OWNED = set(POOL_PAGE) | set(TAIL_BLOCK)
    FILL = 0x3C  # every byte of the shared indexer tensor starts at this value
    APE = torch.randn(KPOOL, D, generator=torch.Generator().manual_seed(7))
    K, R = cfg["k"], cfg["c"]


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
                self.tail = host_v11._reshape_attention_kv_cache(
                    self.raw, spec, (NB, 2, ring, D), (0, 1, 2, 3), NB, None
                )
            else:
                self.tail = torch.zeros(NB, 2, ring, D, dtype=torch.bfloat16)
            # Stale ring contents: a slot read before this request wrote it
            # shows up as a differing pool.
            g = torch.Generator().manual_seed(stale)
            for blk in TAIL_BLOCK:
                self.tail[blk] = torch.randn(2, ring, D, generator=g).to(torch.bfloat16)


    def tail_slots(host, lay, positions, qsl, mask):
        """The tail slots the kpool tail metadata builder hands the indexer."""
        host.GLM53_TAIL_STASH_MASK = mask
        n = qsl[-1]
        return host.compute_kpool_tail_slot_mapping(
            torch.full((n,), -1, dtype=torch.int64),
            torch.tensor(TAIL_BLOCK[: len(qsl) - 1], dtype=torch.int32).view(-1, 1),
            torch.tensor(qsl, dtype=torch.int32),
            torch.tensor(positions),
            n,
            len(qsl) - 1,
            lay.ring,  # the tail spec's block_size
        )


    def pool_slot(b, p):
        """Pool-granular slot mapping: only a pool's last token has a slot."""
        return POOL_PAGE[b] * PAGE + p // KPOOL if p % KPOOL == KPOOL - 1 else -1


    def decode(kc, host, lay, rows, mask=None):
        """The indexer's decode write: rows[b] = [(pos, key)] of request b,
        padded with -1 as the non-uniform decode scatter pads them."""
        qsl = [0]
        for r in rows:
            qsl.append(qsl[-1] + len(r))
        flat = tail_slots(host, lay, [p for r in rows for p, _ in r], qsl, mask)
        n = max(len(r) for r in rows)
        key = torch.zeros(len(rows), n, 2, D, dtype=torch.bfloat16)
        pos, slot, tslot = (torch.full((len(rows), n), -1, dtype=torch.int32) for _ in range(3))
        for b, r in enumerate(rows):
            for t, (p, k) in enumerate(r):
                key[b, t], pos[b, t], slot[b, t] = vec(k), p, pool_slot(b, p)
                tslot[b, t] = flat[qsl[b] + t]
        kc.kpool_decode_update_and_maybe_write_cache_batched(
            lay.kv, lay.tail, tslot, key[:, :, 0].contiguous(), key[:, :, 1].contiguous(),
            APE, slot, pos, KPOOL, D, round_scale=True,
        )


    def prefill(kc, host, lay, rows):
        """The indexer's prefill write: pool compress, then the tail seed."""
        qsl = [0]
        for r in rows:
            qsl.append(qsl[-1] + len(r))
        kg = torch.stack([vec(k) for r in rows for _, k in r])
        k, g = kg[:, 0].contiguous(), kg[:, 1].contiguous()
        slot = torch.tensor([pool_slot(b, p) for b, r in enumerate(rows) for p, _ in r])
        kc.compress_insert(k, g, APE, lay.kv, slot, KPOOL, D, round_scale=True)
        tslot = tail_slots(host, lay, [p for r in rows for p, _ in r], qsl, None)
        kc.kpool_seed_tail_cache(lay.tail, k, g, tslot, KPOOL, D)


    def prompt(kc, host, lay, lens):
        """One prefill step. The indexer runs a request of <= 1 + k tokens on
        its decode path (treat_short_extends_as_decodes)."""
        rows = [[(p, ("true", b, p)) for p in range(h)] for b, h in enumerate(lens)]
        short = [r if len(r) <= 1 + K else [] for r in rows]
        long_ = [r if len(r) > 1 + K else [] for r in rows]
        if any(short):
            decode(kc, host, lay, short)
        if any(long_):
            prefill(kc, host, lay, long_)


    def run_spec(kc, host, lay, lens, plan, av):
        """Verify steps: 1 + k rows per request, drafts past `a` rejected;
        with adaptive verify, rows past `m` masked (no stash)."""
        prompt(kc, host, lay, lens)
        P = list(lens)
        state = AdaptiveVerify(0.0, K, None, 4, R * (K + 1), "cpu") if av else None
        for s, step in enumerate(plan):
            rows = [
                [
                    (P[b] + j, ("true", b, P[b] + j) if j <= a else
                     ("draft" if j <= m else "masked", s, b, P[b] + j))
                    for j in range(K + 1)
                ]
                for b, (m, a) in enumerate(step)
            ]
            mask = None
            if av:
                for b, (m, _) in enumerate(step):
                    state.verify_len[STATES[b]] = m
                n = R * (K + 1)
                state.prepare(SimpleNamespace(
                    num_tokens_after_padding=n, num_draft_tokens=R * K,
                    logits_indices=torch.arange(n),
                    expanded_local_pos=torch.arange(K + 1).repeat(R),
                    expanded_idx_mapping=torch.tensor(STATES[:R]).repeat_interleave(K + 1),
                ))
                mask = state.masked
            decode(kc, host, lay, rows, mask)
            P = [p + a + 1 for p, (_, a) in zip(P, step)]
        return P


    def run_plain(kc, host, lay, lens, ends):
        """Non-speculative decoding of the committed tokens, one row per step."""
        prompt(kc, host, lay, lens)
        for p in range(min(lens), max(ends)):
            decode(kc, host, lay, [[(p, ("true", b, p))] if lens[b] <= p < ends[b] else []
                                   for b in range(R)])


    def pool(lay, b, q):
        """Pool q of request b: its fp8 K and its fp32 scale bytes."""
        flat = lay.kv[POOL_PAGE[b]].reshape(-1)
        return torch.cat([flat[q * D:(q + 1) * D], flat[PAGE * D + 4 * q:PAGE * D + 4 * q + 4]])


    def foreign(lay):
        """Bytes changed in blocks no request owns."""
        pages = lay.raw.view(NB, PAGE_BYTES)
        return sum(int((pages[p] != FILL).sum()) for p in range(NB) if p not in OWNED)


    RUNS = {  # name: (kernels, ring, padded tail view, speculative)
        "fix": ("fix", cfg.get("ring") or KC["fix"].glm53_tail_ring_slots(KPOOL, K), True, True),
        "off": ("off", KPOOL, True, True),  # patched source, switch off
        "v11": ("v11", KPOOL, True, True),
        "v11-contig": ("v11", KPOOL, False, True),  # the seed is right here: aliasing alone
        "v11-nonspec": ("v11", KPOOL, True, False),  # no aliasing: the seed addressing alone
    }
    rng = random.Random(cfg["seed"])
    seen = Counter()
    seen["fix|ring"] = RUNS["fix"][1]
    for trial in range(cfg["trials"]):
        lens = [rng.randint(5, 13) for _ in range(R)]
        plan = []
        for _ in range(cfg["steps"]):
            step = []
            for _ in range(R):
                m = rng.randint(1, K) if cfg["av"] else K
                step.append((m, min(m, rng.choice([0, 0, 0, 1, 1, 2, 3, K]))))
            plan.append(step)
        ends = list(lens)
        for s, step in enumerate(plan):
            for b, (m, a) in enumerate(step):
                # The case that needs the whole bound, with a later step to read it.
                seen["worst-case steps"] += s + 1 < len(plan) and ends[b] % KPOOL == KPOOL - 2 and a == 0
            ends = [p + a + 1 for p, (_, a) in zip(ends, step)]
        host = host_av if cfg["av"] else host_v11
        ref = Layout(KPOOL, padded=False, stale=1)
        run_plain(KC["v11"], host_v11, ref, lens, ends)
        for name in cfg["runs"]:
            kern, ring, padded, spec = RUNS[name]
            lay = Layout(ring, padded, stale=2)
            if spec:
                assert run_spec(KC[kern], host, lay, lens, plan, cfg["av"]) == ends
            else:
                run_plain(KC[kern], host_v11, lay, lens, ends)
            for b in range(R):
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


def build_sim_root(root: Path, kpool_src: str) -> None:
    """Stub vllm, v11's kernels and host helpers, the patched kernels, and the
    adaptive-verify module and tail-slot hook as patch_v13_verify writes them."""
    for rel, text in STUBS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    (root / "vllm" / AV_MODULE).write_text(verify.MODULE)
    shutil.copy2(SRC / KPOOL, root / "kpool_v11.py")
    (root / "kpool_patched.py").write_text(kpool_src)
    idx = (SRC / INDEXER).read_text()
    (root / "host_v11.py").write_text(
        "from __future__ import annotations\n\nfrom math import prod\n\nimport torch\n\n\n"
        "def get_dtype_size(dtype):\n    return torch.tensor([], dtype=dtype).element_size()\n\n\n"
        + top_level(idx, {"compute_kpool_tail_slot_mapping"})
        + "\n\n"
        + top_level((SRC / ATTN_UTILS).read_text(), {"_reshape_attention_kv_cache"})
        + "\n\nCOMPRESS_INSERT_SRC = "
        + repr(top_level((SRC / INDEXER_OP).read_text(), {"_kpool_compress_insert"}))
        + "\n"
    )
    av_idx, applied = verify.plan_edits(idx, verify.FILE_EDITS[INDEXER])
    assert applied == len(verify.FILE_EDITS[INDEXER])
    (root / "host_av.py").write_text(
        "import torch\n\n\n" + top_level(av_idx, {"compute_kpool_tail_slot_mapping", "GLM53_TAIL_STASH_MASK"})
    )


def simulate(kpool_src: str, **cfg) -> dict:
    """Run SIM_SCRIPT on `kpool_src`; return its counters (or {"crashed": stderr})."""
    with tempfile.TemporaryDirectory(prefix="v13kpooltail-sim-") as tmp:
        build_sim_root(Path(tmp), kpool_src)
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


# Scenario: (name, config). k=7 is DFlash2-7; k=4 is the MTP-4 rollback.
SCENARIOS = [
    ("DFlash2 k=7, c=1", dict(k=7, c=1, av=False, trials=10, steps=5, seed=1)),
    ("DFlash2 k=7, c=2", dict(k=7, c=2, av=False, trials=10, steps=5, seed=2)),
    ("MTP k=4, c=2", dict(k=4, c=2, av=False, trials=10, steps=5, seed=3)),
    ("k=7, c=2 + GLM53_ADAPTIVE_VERIFY", dict(k=7, c=2, av=True, trials=10, steps=5, seed=4)),
]
ALL_RUNS = ["fix", "off", "v11", "v11-contig", "v11-nonspec"]


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
    to non-speculative decoding; v11 does not."""

    def test_committed_state_equals_non_speculative_decoding(self):
        src = patched(KPOOL)
        report = []
        for title, cfg in SCENARIOS:
            with self.subTest(title):
                seen = simulate(src, runs=ALL_RUNS, **cfg)
                self.assertNotIn("crashed", seen)
                report.append(table(title, seen))
                self.assertEqual(seen["fix|ring"], 16 if cfg["k"] == 7 else 8)
                self.assertGreater(seen["fix|pools"], 40)
                # The fix: nothing differs, nothing outside a request's blocks moves.
                self.assertEqual(seen["fix|pools differ"], 0)
                self.assertEqual(seen["fix|ring differs"], 0)
                self.assertEqual(seen["fix|foreign bytes"], 0)
                # Switch off: byte-identical to v11 on the same layout.
                self.assertEqual(seen["off|raw differs from v11"], 0)
                # v11's bug, documented: rejected rows overwrite committed slots.
                self.assertGreater(seen["v11-contig|pools differ"], 0)
                self.assertGreater(seen["v11-contig|ring differs"], 0)
                self.assertGreater(seen["v11|pools differ"], 0)
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
}


@unittest.skipUnless(HAVE_SRC and HAVE_TORCH and HAVE_TRITON, "needs GLM53_V11_SRC, torch, triton")
class KpoolTailMutationTest(unittest.TestCase):
    CFG = dict(k=7, c=2, av=False, trials=10, steps=5, seed=5, runs=["fix"])

    def test_the_bound_is_exact(self):
        """R = k + kpool - 1 is enough (one less is the mutation below); the
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
                          f"ring differs {seen['fix|ring differs']}, foreign bytes {seen['fix|foreign bytes']}")
                else:
                    print(f"\n  {name}: simulation crashed ({seen['crashed'].split()[0]})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
