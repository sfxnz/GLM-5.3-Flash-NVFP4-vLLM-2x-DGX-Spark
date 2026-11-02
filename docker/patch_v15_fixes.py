"""v15 fixes for the GLM-5.3-Flash serve image (on top of v14).

Every behaviour is opt-in through a GLM53_* environment variable and is off
by default. With none of them set, v15 runs v14's code.

  GLM53_PREFIX_CACHE_DFLASH=1   prefix-cache hits with DFlash2: no EAGLE
                                last-block drop for DFlash (vLLM #54163), KDA
                                state indexed in mamba blocks (vLLM #55601 and
                                the scheduler's align split)
  GLM53_VIDEO_FRAMES_FIX=1      video placeholder timestamps from the pixel
                                path's frame sampler (vLLM #55647)
  GLM53_MLA_PLAN_NOSYNC=1       SM90 sparse-MLA plan from host upper bounds,
                                reused across steps, kv_end clamped on device
                                (vLLM #58684)

Usage: python3 patch_v15_fixes.py [VLLM_ROOT]
VLLM_ROOT defaults to the image's site-packages vllm directory.

Every edit is an exact-substring replace. An edit whose replacement text is
already present counts as applied (so reruns are no-ops); an edit whose anchor
does not occur exactly once refuses the whole run before anything is written.
"""

import ast
import sys
from pathlib import Path

DEFAULT_ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"

# --------------------------------------------------------------------------
# GLM53_PREFIX_CACHE_DFLASH: v1/core/sched/scheduler.py
#
# 1. vLLM #54163: use_eagle() also matches dflash, so the KV cache manager
#    drops the last cached block of every group and a 2-block prompt never
#    hits. DFlash builds a position's draft KV from the target hidden state
#    at that position alone, so nothing needs dropping. use_eagle itself stays
#    (encoder shift, drafter look-ahead).
# 2. The align split clips prefill chunks at mamba-block boundaries in units
#    of cache_config.block_size. EngineCore sets that to the smallest
#    prefix-cacheable group, and our DFlash draft group is a quarter of the
#    attention block (patch_v11_dflash_kv_groups.py), so the unit is 1152
#    while KDA state is checkpointed every 4608 tokens. A chunk can then cross
#    a mamba boundary without stopping on it, and the block hashed for that
#    boundary holds the state of an earlier position.
# --------------------------------------------------------------------------
SCHED_EDITS = [
    (
        "scheduler.py: import os",
        "import itertools\nimport time\n",
        "import itertools\nimport os\nimport time\n",
    ),
    (
        "scheduler.py: prefix cache flag and helpers",
        "logger = init_logger(__name__)\n",
        '''logger = init_logger(__name__)

_GLM53_PREFIX_CACHE_DFLASH = os.environ.get("GLM53_PREFIX_CACHE_DFLASH") == "1"


def _glm53_eagle_block_drop(speculative_config) -> bool:
    """GLM53_PREFIX_CACHE_DFLASH (v15), vLLM #54163: only drafters whose KV at
    a position needs the next token (EAGLE, EAGLE3, MTP) drop the last block."""
    if speculative_config is None or not speculative_config.use_eagle():
        return False
    if not _GLM53_PREFIX_CACHE_DFLASH:
        return True
    return not (speculative_config.use_dflash() or speculative_config.use_dspark())


def _glm53_mamba_block_size(kv_cache_config) -> int | None:
    """GLM53_PREFIX_CACHE_DFLASH (v15): the align-mode mamba block size, or
    None to keep cache_config.block_size as the split unit."""
    if not _GLM53_PREFIX_CACHE_DFLASH:
        return None
    return next(
        (
            g.kv_cache_spec.block_size
            for g in kv_cache_config.kv_cache_groups
            if getattr(g.kv_cache_spec, "mamba_cache_mode", None) == "align"
        ),
        None,
    )
''',
    ),
    (
        "scheduler.py: block drop decision",
        "            self.use_eagle = speculative_config.use_eagle()\n",
        "            self.use_eagle = speculative_config.use_eagle()\n"
        "        self.use_eagle_block_drop = _glm53_eagle_block_drop(speculative_config)\n",
    ),
    (
        "scheduler.py: block drop to the KV cache manager",
        "            use_eagle=self.use_eagle,\n",
        "            use_eagle=self.use_eagle_block_drop,\n",
    ),
    (
        "scheduler.py: mamba split unit",
        """        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
""",
        """        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self._glm53_mamba_block_size = _glm53_mamba_block_size(kv_cache_config)
""",
    ),
    (
        "scheduler.py: split in mamba blocks",
        "        block_size = self.cache_config.block_size\n",
        "        block_size = self._glm53_mamba_block_size or self.cache_config.block_size\n",
    ),
    (
        "scheduler.py: split back-off follows the block drop",
        "        if self.use_eagle:\n            last_cache_position = max(last_cache_position - block_size, 0)\n",
        "        if self.use_eagle_block_drop:\n            last_cache_position = max(last_cache_position - block_size, 0)\n",
    ),
]

# --------------------------------------------------------------------------
# GLM53_PREFIX_CACHE_DFLASH: v1/worker/gpu/model_states/mamba_hybrid.py
#
# vLLM #55601: a request resumed from a prefix hit seeds its running KDA state
# column in mamba blocks. cache_config.block_size is only the mamba block in
# a worker whose config EngineCore did not shrink (our mp executor); with the
# switch the unit no longer depends on that.
# --------------------------------------------------------------------------
MAMBA_EDITS = [
    (
        "mamba_hybrid.py: import os",
        "from dataclasses import dataclass\nfrom typing import Any\n",
        "import os\nfrom dataclasses import dataclass\nfrom typing import Any\n",
    ),
    (
        "mamba_hybrid.py: flag",
        "from vllm.v1.worker.utils import AttentionGroup\n",
        "from vllm.v1.worker.utils import AttentionGroup\n\n"
        '_GLM53_PREFIX_CACHE_DFLASH = os.environ.get("GLM53_PREFIX_CACHE_DFLASH") == "1"\n',
    ),
    (
        "mamba_hybrid.py: seed in mamba blocks",
        """            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )
""",
        """            block_size = (
                self.cache_config.mamba_block_size
                if _GLM53_PREFIX_CACHE_DFLASH
                else self.cache_config.block_size
            )
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // block_size
            )
""",
    ),
]

# --------------------------------------------------------------------------
# GLM53_PREFIX_CACHE_DFLASH supersedes v13's GLM53_DFLASH_PREFIX_CACHE_FIX
# (v1/core/kv_cache_coordinator.py). With no group dropping a block, the v13
# rule that a short draft hit must not shrink the target's hit would leave the
# draft window without KV for the cached span; the fixed-point search shrinks
# the hit to a boundary every group holds instead.
# --------------------------------------------------------------------------
COORD_EDITS = [
    (
        "kv_cache_coordinator.py: v13 fix yields to v15",
        '_GLM53_PREFIX_CACHE_FIX = os.environ.get("GLM53_DFLASH_PREFIX_CACHE_FIX") == "1"\n',
        '_GLM53_PREFIX_CACHE_FIX = (\n'
        '    os.environ.get("GLM53_DFLASH_PREFIX_CACHE_FIX") == "1"\n'
        '    and os.environ.get("GLM53_PREFIX_CACHE_DFLASH") != "1"\n'
        ')\n',
    ),
]

# --------------------------------------------------------------------------
# GLM53_VIDEO_FRAMES_FIX: models/glm5next/nvidia/multimodal.py
#
# vLLM #55647. Pixels are sampled by Glm5NextVideoProcessor
# (glm_sample_frame_indices, fps_interval 2) but the placeholder timestamps
# come from GLM-4.6V's sampler (3 fps under 30 s), so a clip the loader hands
# over whole (32 frames or fewer, do_sample_frames=True) gets 3x more
# placeholder frames than the encoder produces. One timestamp per temporal
# patch of the pixel path's own sample makes the count grid_t by construction.
# --------------------------------------------------------------------------
MM_EDITS = [
    (
        "multimodal.py: import os",
        "from collections.abc import Mapping\nfrom functools import partial\n",
        "import os\nfrom collections.abc import Mapping\nfrom functools import partial\n",
    ),
    (
        "multimodal.py: flag",
        "from vllm.v1.attention.backends.registry import AttentionBackendEnum\n",
        "from vllm.v1.attention.backends.registry import AttentionBackendEnum\n\n"
        '_GLM53_VIDEO_FRAMES_FIX = os.environ.get("GLM53_VIDEO_FRAMES_FIX") == "1"\n',
    ),
    (
        "multimodal.py: placeholder timestamps from the pixel sampler",
        "    def _get_vision_info(\n",
        '''    def _get_video_second_idx_glm46v(self, metadata, total_frames: int) -> list[int]:
        if not _GLM53_VIDEO_FRAMES_FIX:
            return super()._get_video_second_idx_glm46v(metadata, total_frames)
        from vllm.transformers_utils.processors.glm5next import (
            glm_sample_frame_indices,
        )

        video_processor = self.get_video_processor()
        temporal_patch_size = int(
            getattr(video_processor, "temporal_patch_size", 1) or 1
        )
        video_fps = float(metadata["fps"])
        if not metadata.get("do_sample_frames", True):
            frame_indices = [int(idx) for idx in metadata["frames_indices"]]
            num_frames = int(total_frames)
        else:
            frame_indices = [
                int(idx)
                for idx in glm_sample_frame_indices(
                    int(metadata.get("total_num_frames", total_frames)),
                    video_fps,
                    float(metadata.get("duration") or 0),
                    target_fps=video_processor.fps_interval,
                    max_frame_count=video_processor.max_frame_count_dynamic,
                    temporal_patch_size=temporal_patch_size,
                )
            ]
            num_frames = len(frame_indices)
        if not frame_indices or num_frames <= 0:
            raise ValueError(
                "The GLM-5-Next frame sampler selected no frames for this video "
                f"(fps={video_fps}, duration={metadata.get('duration')!r}, "
                f"total_num_frames={metadata.get('total_num_frames')!r})."
            )
        num_padded_frames = num_frames + (-num_frames % temporal_patch_size)
        grid_t = max(num_padded_frames // temporal_patch_size, 1)
        last = len(frame_indices) - 1
        return [
            int(frame_indices[min(i * temporal_patch_size, last)] / video_fps)
            for i in range(grid_t)
        ]

    def _get_vision_info(
''',
    ),
]

# --------------------------------------------------------------------------
# GLM53_MLA_PLAN_NOSYNC: v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py
#
# vLLM #58684. Under async scheduling the builder copies positions to the host
# (a D2H sync) because FlashInfer bakes each row's kv_len into its schedule.
# With the switch it plans from seq_lens_cpu_upper_bound plus 32 tokens of
# slack, reuses that plan while the bounds still fit, and clamps every work
# item's kv_end to the row's exact valid count on device.
#
# FA2 on sm_121 (our v8 path) takes the same MLAPlan (scheduler.cuh) as FA3.
# Its kernel (mla.cuh) sizes the tile loop and the logits mask by kv_end,
# writes an empty work as lse = -inf, and merges split-KV partials in kv_start
# order, so a clamped-empty trailing chunk adds nothing and chunk 0 (kv_start
# 0, valid >= 1) is never empty. Loads past kv_end stay inside the row's own
# topk_width slice of kv_indices (fixed-stride kv_indptr here, unlike
# upstream's cumsum), whose -1 tail forward_mqa already clamps to slot 0.
# Our GLM53_ADAPTIVE_VERIFY keeps every shape, so the host query split equals
# the device one; vLLM's own adaptive verification falls back to the batch max.
# --------------------------------------------------------------------------
MLA_EDITS = [
    (
        "flashinfer_mla_sparse_sm90.py: import os",
        "from typing import Any, ClassVar\n\nimport torch\n",
        "import os\nfrom typing import Any, ClassVar\n\nimport torch\n",
    ),
    (
        "flashinfer_mla_sparse_sm90.py: import triton",
        "from vllm.platforms.interface import DeviceCapability\n",
        "from vllm.platforms.interface import DeviceCapability\n"
        "from vllm.triton_utils import tl, triton\n",
    ),
    (
        "flashinfer_mla_sparse_sm90.py: flag, plan layout, clamp kernel",
        "_WORKSPACE_BYTES = 128 * 1024 * 1024\n",
        '''_WORKSPACE_BYTES = 128 * 1024 * 1024

_GLM53_MLA_PLAN_NOSYNC = os.environ.get("GLM53_MLA_PLAN_NOSYNC") == "1"
# FlashInfer MLAPlanInfo field indices (attention/scheduler.cuh).
_PLAN_INFO_LEN = 18
_PI_NUM_BLKS_Y = 1
_PI_Q_INDPTR = 2
_PI_KV_INDPTR = 3
_PI_KV_START = 13
_PI_KV_END = 14
_PI_WORK_INDPTR = 15
# Planned lengths exceed the host bound by up to this many tokens so a plan
# stays valid across consecutive steps; the device clamp restores exactness.
_PLAN_SLACK = 32


@triton.jit
def _clamp_work_kv_end(
    int_ws,
    saved_kv_end,
    seq_lens,
    query_start_loc,
    req_id_per_token,
    num_works_idx,
    q_indptr_off,
    kv_start_off,
    kv_end_off,
    TOPK: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    works = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = works < tl.load(int_ws + num_works_idx)
    row = tl.load(int_ws + q_indptr_off + works, mask=mask, other=0)
    req = tl.load(req_id_per_token + row, mask=mask, other=0)
    seq_len = tl.load(seq_lens + req, mask=mask, other=0)
    query_end = tl.load(query_start_loc + req + 1, mask=mask, other=0)
    ctx = seq_len - query_end + row + 1
    valid = tl.where(ctx <= TOPK, ctx, TOPK + ctx % KPOOL)
    start = tl.load(int_ws + kv_start_off + works, mask=mask, other=0)
    end = tl.load(saved_kv_end + works, mask=mask, other=0)
    tl.store(
        int_ws + kv_end_off + works,
        tl.maximum(tl.minimum(end, valid), start),
        mask=mask,
    )
''',
    ),
    (
        "flashinfer_mla_sparse_sm90.py: bounded plan and clamp",
        "\n\n_SM90_STATE: _SM90State | None = None\n",
        '''
    def glm53_plan_bounded(
        self,
        num_tokens: int,
        ctx_bound: torch.Tensor,
        cam: CommonAttentionMetadata,
        req_id_per_token: torch.Tensor,
        topk: int,
        kpool: int,
    ) -> None:
        """GLM53_MLA_PLAN_NOSYNC (v15): plan from host upper bounds of each
        row's context, reuse the plan while they fit, clamp kv_end on device."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInferMLASparseSM90 plan() called inside CUDA graph capture"
            )
        max_valid = min(topk + kpool - 1, self.topk_width)
        bound = torch.clamp(ctx_bound, max=max_valid).to(torch.int32)
        planned = (
            self._lens_cpu[:num_tokens]
            if self.glm53_planned_rows == num_tokens
            else None
        )
        if planned is None or not bool(
            ((planned >= bound) & (planned <= bound + _PLAN_SLACK)).all()
        ):
            # MLAPlan refills the pinned int buffer on the host and copies it
            # asynchronously; with no D2H sync per step the previous copy may
            # still be queued, so wait for it before overwriting the buffer.
            if self._glm53_plan_copied is not None:
                self._glm53_plan_copied.synchronize()
            self.plan(num_tokens, torch.clamp(bound + _PLAN_SLACK, max=max_valid))
            self._glm53_save_plan()
            self.glm53_planned_rows = num_tokens
        block = 1024
        _clamp_work_kv_end[(triton.cdiv(self._glm53_saved_kv_end.numel(), block),)](
            self._glm53_int_ws,
            self._glm53_saved_kv_end,
            cam.seq_lens,
            cam.query_start_loc,
            req_id_per_token,
            *self._glm53_offsets,
            TOPK=topk,
            KPOOL=kpool,
            BLOCK_SIZE=block,
        )

    glm53_planned_rows = -1
    _glm53_plan_copied: "torch.cuda.Event | None" = None

    def _glm53_save_plan(self) -> None:
        plan_info = [int(x) for x in self.wrapper._plan_info]
        assert len(plan_info) == _PLAN_INFO_LEN, plan_info
        max_works = (plan_info[_PI_KV_INDPTR] - plan_info[_PI_Q_INDPTR]) // 4
        int_ws = self.wrapper._int_workspace_buffer.view(torch.int32)
        offsets = (
            plan_info[_PI_WORK_INDPTR] // 4 + plan_info[_PI_NUM_BLKS_Y],
            plan_info[_PI_Q_INDPTR] // 4,
            plan_info[_PI_KV_START] // 4,
            plan_info[_PI_KV_END] // 4,
        )
        saved = getattr(self, "_glm53_saved_kv_end", None)
        if saved is None or saved.numel() != max_works:
            saved = torch.empty(max_works, dtype=torch.int32, device=self.device)
        saved.copy_(int_ws[offsets[3] : offsets[3] + max_works])
        self._glm53_saved_kv_end = saved
        self._glm53_int_ws = int_ws
        self._glm53_offsets = offsets
        self._glm53_plan_copied = torch.cuda.Event()
        self._glm53_plan_copied.record()


_SM90_STATE: _SM90State | None = None
''',
    ),
    (
        "flashinfer_mla_sparse_sm90.py: builder flags",
        """        self._index_kpool = int(getattr(hf_config, "index_kpool", 1))
""",
        """        self._index_kpool = int(getattr(hf_config, "index_kpool", 1))
        spec_config = vllm_config.speculative_config
        self._glm53_adaptive_verification = bool(
            getattr(spec_config, "enable_adaptive_verification", False)
        )

    def _glm53_ctx_bounds(
        self, cam: CommonAttentionMetadata
    ) -> tuple[int, torch.Tensor] | None:
        \"\"\"GLM53_MLA_PLAN_NOSYNC (v15): sync-free host upper bounds of each
        row's context (position + 1), or None when there is no host bound.\"\"\"
        num_reqs = cam.num_reqs
        qsl = cam.query_start_loc_cpu[: num_reqs + 1]
        num_rows = int(qsl[-1])
        sl_host = getattr(cam, "seq_lens_cpu_upper_bound", None)
        if num_rows == 0 or sl_host is None:
            return None
        if self._glm53_adaptive_verification:
            # vLLM's adaptive verification can split queries differently on
            # the device; bound every row by the longest request.
            return num_rows, torch.full(
                (num_rows,), int(sl_host[:num_reqs].max()), dtype=torch.int64
            )
        q_lens = (qsl[1:] - qsl[:-1]).to(torch.int64)
        first_ctx = sl_host[:num_reqs].to(torch.int64) - q_lens + 1
        req_of_row = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int64), q_lens
        )
        rows = torch.arange(num_rows, dtype=torch.int64)
        return num_rows, first_ctx[req_of_row] + rows - qsl.to(torch.int64)[req_of_row]
""",
    ),
    (
        "flashinfer_mla_sparse_sm90.py: build plans bounded",
        """        if _SM90_STATE is not None:
            num_rows, kv_lens = self._kv_lens_host(common_attn_metadata)
            _SM90_STATE.plan(num_rows, kv_lens)
        return metadata
""",
        """        if _SM90_STATE is not None:
            bounds = (
                self._glm53_ctx_bounds(common_attn_metadata)
                if _GLM53_MLA_PLAN_NOSYNC
                else None
            )
            if bounds is not None:
                _SM90_STATE.glm53_plan_bounded(
                    bounds[0],
                    bounds[1],
                    common_attn_metadata,
                    metadata.req_id_per_token,
                    self._index_topk,
                    max(self._index_kpool, 1),
                )
                return metadata
            num_rows, kv_lens = self._kv_lens_host(common_attn_metadata)
            _SM90_STATE.plan(num_rows, kv_lens)
            if _GLM53_MLA_PLAN_NOSYNC:
                _SM90_STATE.glm53_planned_rows = -1
        return metadata
""",
    ),
]

FILE_EDITS = {
    "v1/core/sched/scheduler.py": SCHED_EDITS,
    "v1/worker/gpu/model_states/mamba_hybrid.py": MAMBA_EDITS,
    "v1/core/kv_cache_coordinator.py": COORD_EDITS,
    "models/glm5next/nvidia/multimodal.py": MM_EDITS,
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py": MLA_EDITS,
}


def plan_edits(src: str, edits: list[tuple[str, str, str]]) -> tuple[str, int]:
    """Apply ``edits`` to ``src``; return (new_src, n_applied). Raise
    SystemExit if an anchor is missing or ambiguous."""
    applied = 0
    for what, old, new in edits:
        if src.count(new) == 1:
            continue  # already applied
        if src.count(old) != 1:
            raise SystemExit(
                f"refusing: anchor for '{what}' found {src.count(old)} times "
                "(expected 1); upstream source drifted"
            )
        src = src.replace(old, new)
        applied += 1
    return src, applied


def main(root: Path) -> list[Path]:
    if not (root / "__init__.py").is_file():
        raise SystemExit(f"refusing: {root} is not a vllm package directory")
    staged: dict[Path, str] = {}
    for rel, edits in FILE_EDITS.items():
        path = root / rel
        if not path.is_file():
            raise SystemExit(f"refusing: {path} missing")
        src, applied = plan_edits(path.read_text(), edits)
        ast.parse(src, filename=str(path))
        print(f"{rel}: {applied}/{len(edits)} edits to apply")
        if applied:
            staged[path] = src
    for path, content in staged.items():
        path.write_text(content)
    print(f"v15 fixes applied ({len(staged)} files written)")
    return list(staged)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
