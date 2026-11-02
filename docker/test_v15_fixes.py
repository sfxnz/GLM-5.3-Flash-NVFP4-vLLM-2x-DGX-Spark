"""CPU tests for patch_v15_fixes.py.

    GLM53_V14_SRC=/path/to/v14src python3 docker/test_v15_fixes.py

GLM53_V14_SRC is the directory that holds the `vllm/` package tree copied out
of glm53-sm121-v14. Without it the source tests skip. The behaviour tests also
need torch and the image's Python packages, so they skip on a bare host; run
them in a CPU-only container (no --gpus):

    docker run --rm --network none -v "$PWD/docker:/t:ro" --entrypoint python3 \\
      -e GLM53_V14_SRC=/usr/local/lib/python3.12/dist-packages \\
      glm53-sm121-v14 /t/test_v15_fixes.py

Set GLM53_SNAPSHOT (checkpoint snapshot dir) and GLM53_VIDEO_CLIPS (a dir of
short .mp4 clips) to add the end-to-end video test with the real processor.
"""

import importlib.util
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_v15_fixes as P  # noqa: E402

SRC = os.environ.get("GLM53_V14_SRC", "")
HAVE_SRC = bool(SRC) and (Path(SRC) / "vllm").is_dir()
HAVE_TORCH = importlib.util.find_spec("torch") is not None


class PlanEditsTest(unittest.TestCase):
    def test_refuses_missing_anchor(self):
        with self.assertRaises(SystemExit):
            P.plan_edits("nothing here", [("x", "anchor", "new")])

    def test_already_applied_is_noop(self):
        self.assertEqual(P.plan_edits("a new b", [("x", "old", "new")]), ("a new b", 0))


@unittest.skipUnless(HAVE_SRC, "GLM53_V14_SRC not set")
class SourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "vllm"
        for rel in P.FILE_EDITS:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(SRC) / "vllm" / rel, self.root / rel)
        (self.root / "__init__.py").touch()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_every_anchor_once(self):
        for rel, edits in P.FILE_EDITS.items():
            src = (self.root / rel).read_text()
            for what, old, _new in edits:
                self.assertEqual(src.count(old), 1, what)

    def test_apply_twice_and_compile(self):
        self.assertEqual(len(P.main(self.root)), len(P.FILE_EDITS))
        self.assertEqual(P.main(self.root), [])
        for rel in P.FILE_EDITS:
            py_compile.compile(str(self.root / rel), doraise=True)

    def test_switches_default_off(self):
        P.main(self.root)
        text = {rel: (self.root / rel).read_text() for rel in P.FILE_EDITS}
        for rel, flag in [
            ("v1/core/sched/scheduler.py", "GLM53_PREFIX_CACHE_DFLASH"),
            ("v1/worker/gpu/model_states/mamba_hybrid.py", "GLM53_PREFIX_CACHE_DFLASH"),
            ("models/glm5next/nvidia/multimodal.py", "GLM53_VIDEO_FRAMES_FIX"),
            (
                "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py",
                "GLM53_MLA_PLAN_NOSYNC",
            ),
        ]:
            self.assertIn(f'os.environ.get("{flag}") == "1"', text[rel], rel)
        mla = text["v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py"]
        self.assertIn("_SM90_STATE.plan(num_rows, kv_lens)", mla)
        self.assertIn(
            "return super()._get_video_second_idx_glm46v(metadata, total_frames)",
            text["models/glm5next/nvidia/multimodal.py"],
        )

    def test_refuses_drift(self):
        path = self.root / "v1/worker/gpu/model_states/mamba_hybrid.py"
        path.write_text(path.read_text().replace("self.cache_config.block_size", "x"))
        with self.assertRaises(SystemExit):
            P.main(self.root)


# --------------------------------------------------------------------------
# Behaviour tests: the patched modules run in a subprocess per switch setting,
# importing from an overlay tree (the v14 tree symlinked, patched files real).
# --------------------------------------------------------------------------

PREFIX_CACHE = r'''
import json, math
from types import SimpleNamespace
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_groups_glm5_next, generate_scheduler_kv_cache_config,
    get_request_block_hasher, init_none_hash, resolve_kv_cache_block_sizes)
from vllm.v1.core.sched import scheduler as S
from vllm.v1.kv_cache_interface import (
    KVCacheConfig, KpoolTailSpec, MambaSpec, MLAAttentionSpec, SlidingWindowSpec)
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

ATTN, KPOOL, K, MNBT = 4608, 4, 7, 4624

def layout():
    # The serve's layout: attention block 4608 (fp8 KV), kpool tail, the v11
    # DFlash draft group at a quarter block, KDA groups in align mode.
    specs = {}
    for i in range(4):
        specs[f"l{i}.attn"] = MLAAttentionSpec(block_size=ATTN, num_kv_heads=1,
            head_size=576, dtype=torch.uint8, cache_dtype_str="fp8_e4m3")
        specs[f"l{i}.indexer"] = MLAAttentionSpec(block_size=ATTN, num_kv_heads=1,
            head_size=132, dtype=torch.uint8, compress_ratio=KPOOL)
        specs[f"l{i}.tail"] = KpoolTailSpec(block_size=KPOOL, num_kv_heads=1,
            head_size=64, dtype=torch.bfloat16, sliding_window=KPOOL)
    for i in range(4, 12):
        specs[f"l{i}.kda"] = MambaSpec(block_size=ATTN, shapes=((16, 16),),
            dtypes=(torch.float32,), mamba_cache_mode="align", num_speculative_blocks=K)
    for i in range(5):
        specs[f"draft.{i}.attn"] = SlidingWindowSpec(block_size=16, num_kv_heads=8,
            head_size=128, dtype=torch.bfloat16, sliding_window=2048)
    pp = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1))
    groups = _get_kv_cache_groups_glm5_next(pp, specs)
    return generate_scheduler_kv_cache_config(
        [KVCacheConfig(num_blocks=400, kv_cache_tensors=[], kv_cache_groups=groups)])

DFLASH = SimpleNamespace(use_eagle=lambda: True, use_dflash=lambda: True,
                         use_dspark=lambda: False)
MTP = SimpleNamespace(use_eagle=lambda: True, use_dflash=lambda: False,
                      use_dspark=lambda: False)

def scenario(budget, L):
    kvc = layout()
    groups = kvc.kv_cache_groups
    engine_bs = min(g.kv_cache_spec.block_size for g in groups
                    if g.kv_cache_spec.participates_in_prefix_caching)  # core.py
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=engine_bs, enable_prefix_caching=True,
                                     prefix_match_unit=None),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        kv_transfer_config=None)
    sched_bs, hash_bs = resolve_kv_cache_block_sizes(kvc, cfg)
    drop = S._glm53_eagle_block_drop(DFLASH)
    init_none_hash(sha256)
    hasher = get_request_block_hasher(hash_bs, sha256)
    mgr = KVCacheManager(kvc, max_model_len=65536, scheduler_block_size=sched_bs,
                         hash_block_size=hash_bs, max_in_flight_tokens=8192,
                         enable_caching=True, use_eagle=drop)
    s = object.__new__(S.Scheduler)
    s.cache_config = cfg.cache_config
    s.use_eagle = True
    s.use_eagle_block_drop = drop
    s._glm53_mamba_block_size = S._glm53_mamba_block_size(kvc)
    s.max_num_scheduled_tokens = MNBT
    s.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    s.hash_block_size = hash_bs
    s.mamba_partial_cache_hit = False
    mamba_gid = next(i for i, g in enumerate(groups) if isinstance(g.kv_cache_spec, MambaSpec))
    draft_gid = next(i for i, g in enumerate(groups)
                     if type(g.kv_cache_spec) is SlidingWindowSpec)
    # Worker model: the forward writes the state after `end` tokens into the
    # running column (end - 1) // mamba_block (mamba_get_block_table_tensor).
    state_at = {}
    out = {"engine_block_size": engine_bs, "stalls": 0}

    def run(rid):
        r = Request(rid, list(range(1, L + 1)), SamplingParams(max_tokens=1), None,
                    block_hasher=hasher)
        mgr.new_step_starts()
        blocks, hit, _ = mgr.get_computed_blocks(r)
        if hit:
            out["hit_state"] = state_at.get(blocks.blocks[mamba_gid][-1].block_id)
            out["hit_draft_blocks"] = len(blocks.blocks[draft_gid])
        ends, first = [], True
        while r.num_computed_tokens + (hit if first else 0) < L:
            mgr.new_step_starts()
            done = r.num_computed_tokens + (hit if first else 0)
            n = s._mamba_block_aligned_split(r, min(budget, L - done), hit if first else 0)
            if n == 0:
                out["stalls"] += 1
                n = s._mamba_block_aligned_split(r, min(MNBT, L - done), hit if first else 0)
            assert n > 0
            assert mgr.allocate_slots(r, n, hit if first else 0,
                                      blocks if first else None) is not None
            r.num_computed_tokens = done + n
            first = False
            ends.append(r.num_computed_tokens)
            row = mgr.coordinator.single_type_managers[mamba_gid].req_to_blocks[rid]
            state_at[row[(r.num_computed_tokens - 1) // ATTN].block_id] = r.num_computed_tokens
        mgr.free(r)
        return hit, ends

    _, out["a_ends"] = run("A")
    out["hit"], out["b_ends"] = run("B")
    return out

def seed(num_computed):
    st = object.__new__(MambaHybridModelState)
    st.cache_config = SimpleNamespace(block_size=1152, mamba_block_size=ATTN)
    st._align_mode = True
    st.rope_state = None
    st.num_accepted_tokens_gpu = torch.ones(2, dtype=torch.int32)
    st._mamba_state_idx_gpu = torch.zeros(2, dtype=torch.int32)
    st.add_request(1, SimpleNamespace(num_computed_tokens=num_computed))
    return int(st._mamba_state_idx_gpu[1])

print(json.dumps({
    "drop_dflash": S._glm53_eagle_block_drop(DFLASH),
    "drop_mtp": S._glm53_eagle_block_drop(MTP),
    "seed_9216": seed(9216),
    "c1_9216": scenario(MNBT, 9216),
    "c1_13000": scenario(MNBT, 13000),
    "short_budget_13000": scenario(4600, 13000),
}))
'''

VIDEO = r'''
import json, math
from types import SimpleNamespace
import numpy as np
from vllm.models.glm5next.nvidia.multimodal import Glm5NextProcessingInfo
from vllm.transformers_utils.processors.glm5next import Glm5NextVideoProcessor

vp = Glm5NextVideoProcessor()
info = object.__new__(Glm5NextProcessingInfo)
info.get_video_processor = lambda **kw: vp

def case(total, fps, duration, loaded=None):
    meta = {"fps": fps, "duration": duration, "total_num_frames": total,
            "do_sample_frames": loaded is None}
    if loaded is None:
        frames = len(vp.sample_frames(SimpleNamespace(
            total_num_frames=total, fps=fps, duration=duration)))
        delivered = total
    else:
        meta["frames_indices"] = np.linspace(0, total - 1, loaded, dtype=int).tolist()
        frames = delivered = loaded
    grid_t = math.ceil(frames / vp.temporal_patch_size)
    return {"grid_t": grid_t,
            "placeholder_t": len(info._get_video_second_idx_glm46v(meta, delivered))}

print(json.dumps({
    "2s_10fps": case(20, 10.0, 2.0),
    "1s_30fps": case(30, 30.0, 1.0),
    "10s_30fps_32_loaded": case(300, 30.0, 10.0, loaded=32),
    "600s_2fps": case(1200, 2.0, 600.0),
}))
'''

VIDEO_E2E = r'''
import json, os, sys
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.media.video import VideoMediaIO
from vllm.multimodal.media.image import ImageMediaIO

snap, clip_dir = sys.argv[1], sys.argv[2]
mc = ModelConfig(model=snap, trust_remote_code=True,
                 limit_mm_per_prompt={"image": 4, "video": 1}, mm_processor_cache_gb=0)
proc = MULTIMODAL_REGISTRY.create_processor(mc)
res = {}
for name in sorted(os.listdir(clip_dir)):
    if not name.endswith(".mp4"):
        continue
    data = open(os.path.join(clip_dir, name), "rb").read()
    frames, meta = VideoMediaIO(ImageMediaIO()).load_bytes(data).media
    items = proc.info.parse_mm_data({"video": [(frames, meta)]})
    out = proc.apply(ProcessorInputs(
        prompt="<|begin_of_video|><|video|><|end_of_video|>describe",
        mm_data_items=items), TimingContext())
    t, h, w = (int(x) for x in out["mm_kwargs"]["video"][0]["video_grid_thw"].data)
    res[name] = {"encoder": t * h * w // 4,
                 "placeholder": out["mm_placeholders"]["video"][0].get_num_embeds()}
print(json.dumps(res))
'''

MLA_PLAN = r'''
import json
from types import SimpleNamespace
import torch
import triton, triton.language as tl
import vllm.triton_utils as tu
torch.cuda.is_current_stream_capturing = lambda: False
# No CUDA driver here, so vLLM hands out placeholder triton; the clamp kernel
# runs under the real Triton CPU interpreter instead.
tu.triton, tu.tl = triton, tl
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm90 as M

TOPK, KPOOL, WIDTH, HEADS, SMS = 2048, 4, 2176, 32, 48
MAX_WORKS = 16384

def mla_plan(lens):
    # Python mirror of MLAPlan (scheduler.cuh), non-causal, one query row per
    # batch entry: split each row's kv_len into kv_len_limit chunks.
    f = lambda x: 32 if x <= 8 else 64 if x <= 16 else 128 if x <= 32 else (
        192 if x <= 64 else -(-x // 256) * 256)
    limit = f(max(-(-sum(lens) // SMS), 1))
    works = []
    for row, n in enumerate(lens):
        start = 0
        while True:
            take = min(n - start, limit)
            works.append((row, start, start + take))
            start += take
            if start >= n:
                break
    ws = torch.zeros(4 * MAX_WORKS + SMS + 1, dtype=torch.int32)
    for i, (row, s, e) in enumerate(works):
        ws[i], ws[2 * MAX_WORKS + i], ws[3 * MAX_WORKS + i] = row, s, e
    ws[4 * MAX_WORKS + SMS] = len(works)
    return ws, limit

class State:
    plan_bounded = M._SM90State.glm53_plan_bounded
    glm53_planned_rows = -1
    _glm53_plan_copied = None
    topk_width = WIDTH

    def __init__(self):
        self._lens_cpu = torch.full((64,), WIDTH, dtype=torch.int32)
        self.plans = 0

    def plan(self, num_tokens, kv_lens):
        self.plans += 1
        self._lens_cpu.fill_(WIDTH)
        self._lens_cpu[:num_tokens] = kv_lens
        self.ws, self.limit = mla_plan(kv_lens.tolist())

    def _glm53_save_plan(self):
        self._glm53_int_ws = self.ws
        self._glm53_saved_kv_end = self.ws[3 * MAX_WORKS: 4 * MAX_WORKS].clone()
        self._glm53_offsets = (4 * MAX_WORKS + SMS, 0, 2 * MAX_WORKS, 3 * MAX_WORKS)

builder = object.__new__(M.FlashInferMLASparseSM90Builder)
builder._index_topk, builder._index_kpool = TOPK, KPOOL
builder._async_scheduling = False
builder._glm53_adaptive_verification = False

def cam(seq_lens, q_lens, upper):
    qsl = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=len(seq_lens), query_start_loc_cpu=qsl, query_start_loc=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor(upper, dtype=torch.int32), positions=None)

def step(state, seq_lens, q_lens, upper):
    c = cam(seq_lens, q_lens, upper)
    _, exact = builder._kv_lens_host(cam(seq_lens, q_lens, seq_lens))
    num_rows, ctx = builder._glm53_ctx_bounds(c)
    req_id = torch.repeat_interleave(torch.arange(len(q_lens), dtype=torch.int32),
                                     torch.tensor(q_lens))
    state.plan_bounded(num_rows, ctx, c, req_id, TOPK, KPOOL)
    ws = state.ws
    n = int(ws[4 * MAX_WORKS + SMS])
    rows = ws[:n].tolist()
    starts = ws[2 * MAX_WORKS: 2 * MAX_WORKS + n].tolist()
    ends = ws[3 * MAX_WORKS: 3 * MAX_WORKS + n].tolist()
    covered = [0] * num_rows
    first_nonempty = [False] * num_rows
    for r, s, e in zip(rows, starts, ends):
        assert e >= s
        covered[r] += e - s
        if s == 0 and e > 0:
            first_nonempty[r] = True
    return {"exact": exact.tolist(), "covered": covered,
            "first_chunk_nonempty": all(first_nonempty), "plans": state.plans,
            "planned_minus_exact_max": int(max(
                state._lens_cpu[:num_rows] - exact))}

st = State()
# Two requests, 8 verify rows each: one past index_topk, one below it. The
# host bound assumes every draft was accepted (async spec decode).
s1 = step(st, [3000, 1500], [8, 8], [3007, 1507])
s2 = step(st, [3003, 1502], [8, 8], [3011, 1510])
s3 = step(st, [3009, 1540], [8, 8], [3017, 1545])
print(json.dumps({"s1": s1, "s2": s2, "s3": s3}))
'''


def _overlay(src_root: Path, dst_root: Path) -> None:
    """Symlink the v14 vllm tree into dst_root, copy and patch the files v15
    edits."""
    src_pkg = src_root / "vllm"
    for dirpath, dirnames, filenames in os.walk(src_pkg):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        rel = Path(dirpath).relative_to(src_pkg)
        (dst_root / "vllm" / rel).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            os.symlink(Path(dirpath) / name, dst_root / "vllm" / rel / name)
    for rel in P.FILE_EDITS:
        target = dst_root / "vllm" / rel
        target.unlink()
        shutil.copy(src_pkg / rel, target)
    P.main(dst_root / "vllm")


def _run(script: str, overlay: Path, flags: dict[str, str], *args: str) -> dict:
    env = dict(os.environ, PYTHONPATH=str(overlay), PYTHONDONTWRITEBYTECODE="1", **flags)
    proc = subprocess.run(
        [sys.executable, "-c", script, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr[-4000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


@unittest.skipUnless(HAVE_SRC and HAVE_TORCH, "needs GLM53_V14_SRC and torch (CPU container)")
class BehaviourTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        _overlay(Path(SRC), cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_prefix_cache_dflash(self):
        off = _run(PREFIX_CACHE, self.tmp, {})
        on = _run(PREFIX_CACHE, self.tmp, {"GLM53_PREFIX_CACHE_DFLASH": "1"})
        # The scheduler's split unit is the draft group's quarter block.
        self.assertEqual(off["c1_9216"]["engine_block_size"], 1152)
        # Off: DFlash counts as EAGLE, so a 2-block prompt never hits.
        self.assertTrue(off["drop_dflash"])
        self.assertEqual(off["c1_9216"]["hit"], 0)
        # Off, prefill budget under one mamba block (more than two decoding
        # requests): a chunk crosses 4608 without stopping, and the block
        # hashed for 4608 holds the KDA state after 3456 tokens.
        bad = off["short_budget_13000"]
        self.assertEqual(bad["hit"], 4608)
        self.assertNotEqual(bad["hit_state"], bad["hit"])
        # On: no drop for DFlash (MTP keeps it), hits at mamba boundaries with
        # the matching state and the draft window's KV.
        self.assertFalse(on["drop_dflash"])
        self.assertTrue(on["drop_mtp"])
        for case, hit in [("c1_9216", 4608), ("c1_13000", 9216), ("short_budget_13000", 9216)]:
            got = on[case]
            self.assertEqual(got["hit"], hit, case)
            self.assertEqual(got["hit_state"], hit, case)
            self.assertGreater(got["hit_draft_blocks"], 0, case)
            self.assertTrue(all(e % 4608 == 0 for e in got["a_ends"][:-1]), case)
        # The price of the right unit: a budget under 4608 waits a step.
        self.assertGreater(on["short_budget_13000"]["stalls"], 0)
        self.assertEqual(off["c1_13000"]["stalls"], 0)
        # vLLM #55601: the resumed KDA column is in mamba blocks.
        self.assertEqual(off["seed_9216"], (9216 - 1) // 1152)
        self.assertEqual(on["seed_9216"], (9216 - 1) // 4608)

    def test_video_placeholders_match_encoder(self):
        off = _run(VIDEO, self.tmp, {})
        on = _run(VIDEO, self.tmp, {"GLM53_VIDEO_FRAMES_FIX": "1"})
        self.assertEqual(off["2s_10fps"]["placeholder_t"], 3 * off["2s_10fps"]["grid_t"])
        self.assertEqual(off["1s_30fps"]["placeholder_t"], 3 * off["1s_30fps"]["grid_t"])
        self.assertNotEqual(off["600s_2fps"]["placeholder_t"], off["600s_2fps"]["grid_t"])
        for case, got in on.items():
            self.assertEqual(got["placeholder_t"], got["grid_t"], case)
        self.assertEqual(off["10s_30fps_32_loaded"], on["10s_30fps_32_loaded"])

    @unittest.skipUnless(
        os.environ.get("GLM53_SNAPSHOT") and os.environ.get("GLM53_VIDEO_CLIPS"),
        "GLM53_SNAPSHOT / GLM53_VIDEO_CLIPS not set",
    )
    def test_video_end_to_end(self):
        args = (os.environ["GLM53_SNAPSHOT"], os.environ["GLM53_VIDEO_CLIPS"])
        flags = {"HF_HUB_OFFLINE": "1"}
        off = _run(VIDEO_E2E, self.tmp, flags, *args)
        on = _run(VIDEO_E2E, self.tmp, dict(flags, GLM53_VIDEO_FRAMES_FIX="1"), *args)
        print("\nvideo e2e off:", off, "\nvideo e2e on: ", on)
        self.assertTrue(any(v["encoder"] != v["placeholder"] for v in off.values()))
        for name, got in on.items():
            self.assertEqual(got["encoder"], got["placeholder"], name)

    def test_mla_plan_bounded_clamp(self):
        res = _run(MLA_PLAN, self.tmp, {"GLM53_MLA_PLAN_NOSYNC": "1", "TRITON_INTERPRET": "1"})
        for name in ("s1", "s2", "s3"):
            got = res[name]
            # Each row's clamped works cover exactly its valid count, starting
            # with a non-empty chunk at kv_start 0.
            self.assertEqual(got["covered"], got["exact"], name)
            self.assertTrue(got["first_chunk_nonempty"], name)
            self.assertGreaterEqual(got["planned_minus_exact_max"], 0, name)
        # Growth inside the slack reuses the plan; growth past it replans.
        self.assertEqual(res["s1"]["plans"], 1)
        self.assertEqual(res["s2"]["plans"], 1)
        self.assertEqual(res["s3"]["plans"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
