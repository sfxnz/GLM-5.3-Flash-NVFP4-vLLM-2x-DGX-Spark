"""v13 expert-census patch for the GLM-5.3-Flash serve image (on top of v11).

  GLM53_EXPERT_CENSUS=<dir>          record routed top-k expert ids per MoE
                                     layer for every token row of real steps
  GLM53_EXPERT_CENSUS_STEPS=<int>    steps to record (default 2000)
  GLM53_EXPERT_CENSUS_SKIP=<int>     real steps to skip first (default 0)

Off unless GLM53_EXPERT_CENSUS is set: no hook is bound, no buffer is
allocated, and the served graphs are the v11 graphs. Measurement only; the
per-step device-to-host copy makes a census boot's timings unpublishable.
tools/census_report.py reads the files; tools/README.md has the procedure.

Only the V2 model runner (v1/worker/gpu/model_runner.py) is hooked. That is
the runner DFlash2 forces (config/vllm.py _is_dflash2_draft), i.e. the
recipe default SPEC=dflash2. SPEC=mtp runs the V1 runner and records nothing
(no "GLM53_EXPERT_CENSUS: recording" log line).

Usage: python3 patch_v13_census.py [VLLM_ROOT]
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
# New module: v1/worker/gpu/glm53_expert_census.py
# --------------------------------------------------------------------------
CENSUS_MODULE = '''"""GLM53_EXPERT_CENSUS (v13): routed-expert census of real forward steps.

Off unless GLM53_EXPERT_CENSUS names a directory. It must be a bind-mounted
path inside the container, e.g. /cache/huggingface/glm53-census/<run> (run.sh
mounts the host HF cache there). Each rank then records real forward steps
SKIP .. SKIP+STEPS-1 (GLM53_EXPERT_CENSUS_SKIP, default 0;
GLM53_EXPERT_CENSUS_STEPS, default 2000). Both ranks see the same scheduler
outputs, so the step counter and the recorded window agree across ranks
without any cross-node trigger. Dummy, profile and capture runs do not count.

Per step it stores the routed top-k expert ids of every token row at every
MoE layer (logical ids, taken where vLLM's own --enable-return-routed-experts
takes them: BaseRouter._select_experts -> capture_fn, before EPLB mapping),
and one segment per request (SEG_COLUMNS):
  step       real-step index (0 = first real step after boot)
  req        census-local request id (stable across steps)
  row0       first token row of the request in the step
  nrows      token rows of the request (verify: 1 anchor + ndraft drafts)
  ndraft     draft tokens scheduled for verification
  nsampled   tokens the sampler emitted (verify: accepted drafts + 1)
  ncomputed  tokens already in the KV cache before the step
  prefill    1 while the request is still prefilling its prompt

CUDA graphs: capture() only copies topk_ids into a preallocated device buffer.
The hook is bound after model load, before torch.compile and graph capture,
so the copy is a node of every captured graph and replays with it (the MoE
forward is the opaque vllm.moe_forward* custom op, so Dynamo never traces
it). The host copies the buffer out after the step's sampler ran, before any
later forward can overwrite it. No ENFORCE_EAGER needed.

Files per rank R in the census directory:
  census-rank{R}.json         meta: layers, top_k, num_experts, window
  census-rank{R}-{NNNN}.npz   topk uint16 [rows, layers, top_k], seg int32
                              [segments, 8]; written every FLUSH_EVERY steps,
                              at the end of the window and at exit
"""

import atexit
import json
import os
from pathlib import Path

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

SEG_COLUMNS = (
    "step",
    "req",
    "row0",
    "nrows",
    "ndraft",
    "nsampled",
    "ncomputed",
    "prefill",
)
FLUSH_EVERY = 100


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "")
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        value = minimum - 1
    if value < minimum:
        raise ValueError(f"{name}={raw!r}: want an integer >= {minimum}")
    return value


def census_segments(
    step, req_uids, query_start_loc, num_draft, num_sampled, num_computed, prefilling
):
    """One int32 row per request of a step, columns SEG_COLUMNS."""
    n = len(req_uids)
    seg = np.zeros((n, len(SEG_COLUMNS)), dtype=np.int32)
    seg[:, 0] = step
    seg[:, 1] = req_uids
    seg[:, 2] = query_start_loc[:n]
    seg[:, 3] = np.diff(query_start_loc[: n + 1])
    seg[:, 4] = 0 if num_draft is None else num_draft[:n]
    seg[:, 5] = num_sampled[:n]
    seg[:, 6] = num_computed[:n]
    seg[:, 7] = prefilling[:n]
    return seg


def write_chunk(path: Path, topk: np.ndarray, seg: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, topk=topk, seg=seg)
    os.replace(tmp, path)


class ExpertCensus:
    def __init__(self, out_dir, layer_ids, top_k, max_tokens, device, rank, steps, skip):
        self.out_dir = out_dir
        self.rank = rank
        self.steps = steps
        self.skip = skip
        self.layer_ids = sorted(layer_ids)
        self._layer_index = torch.tensor(self.layer_ids, dtype=torch.long, device=device)
        self.buf = torch.zeros(
            (max_tokens, self.layer_ids[-1] + 1, top_k), dtype=torch.int32, device=device
        )
        self.step = 0
        self.chunk = 0
        self._staged = None
        self._uids: dict[str, int] = {}
        self._topk: list[np.ndarray] = []
        self._seg: list[np.ndarray] = []

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        # Runs inside the forward, so inside captured graphs: device copy only.
        self.buf[: topk_ids.shape[0], layer_id].copy_(topk_ids)

    def stage(self, input_batch) -> None:
        self._staged = input_batch

    def commit(self, num_sampled: torch.Tensor) -> None:
        batch, self._staged = self._staged, None
        if batch is None or self.step >= self.skip + self.steps:
            return
        step = self.step
        self.step += 1
        if step < self.skip:
            return
        n_req = batch.num_reqs
        qsl = batch.query_start_loc_np
        n = batch.num_tokens
        if int(qsl[n_req]) != n:
            # Adaptive verification lays the rows out on the GPU only.
            logger.warning_once(
                "GLM53_EXPERT_CENSUS: row layout not known on the host "
                "(adaptive verification?); such steps are not recorded"
            )
            return
        uids = [self._uids.setdefault(r, len(self._uids)) for r in batch.req_ids]
        rows = self.buf[:n].index_select(1, self._layer_index).cpu().numpy()
        self._topk.append(rows.astype(np.uint16))
        self._seg.append(
            census_segments(
                step,
                np.asarray(uids),
                qsl,
                batch.num_draft_tokens_per_req,
                num_sampled[:n_req].cpu().numpy(),
                batch.num_computed_tokens_np,
                batch.is_prefilling_np,
            )
        )
        done = self.step >= self.skip + self.steps
        if done or len(self._seg) >= FLUSH_EVERY:
            self.flush()
        if done:
            logger.info(
                "GLM53_EXPERT_CENSUS: rank %d recorded steps %d..%d into %s",
                self.rank,
                self.skip,
                self.step - 1,
                self.out_dir,
            )

    def flush(self) -> None:
        if not self._seg:
            return
        path = self.out_dir / f"census-rank{self.rank}-{self.chunk:04d}.npz"
        write_chunk(path, np.concatenate(self._topk), np.concatenate(self._seg))
        self.chunk += 1
        self._topk, self._seg = [], []


def maybe_create_census(model, max_num_tokens, device, vllm_config):
    """Return an ExpertCensus bound to ``model``'s MoE routers, or None."""
    out = os.environ.get("GLM53_EXPERT_CENSUS", "")
    if not out:
        return None
    steps = _env_int("GLM53_EXPERT_CENSUS_STEPS", 2000, 1)
    skip = _env_int("GLM53_EXPERT_CENSUS_SKIP", 0, 0)
    if vllm_config.parallel_config.use_sequence_parallel_moe:
        raise ValueError(
            "GLM53_EXPERT_CENSUS: sequence-parallel MoE shards topk_ids across "
            "ranks; not supported"
        )
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        bind_routed_experts_capturer,
    )

    layer_ids = [m.layer_id for m in model.modules() if isinstance(m, MoERunner)]
    if not layer_ids:
        raise ValueError("GLM53_EXPERT_CENSUS: the target model has no MoE layers")
    model_config = vllm_config.model_config
    top_k = model_config.get_num_experts_per_tok()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    census = ExpertCensus(
        out_dir, layer_ids, top_k, max_num_tokens, device, rank, steps, skip
    )
    # Raises for a monolithic MoE kernel without routing capture.
    bind_routed_experts_capturer(model, census)
    meta = {
        "rank": rank,
        "layers": census.layer_ids,
        "top_k": top_k,
        "num_experts": model_config.get_num_experts(),
        "skip": skip,
        "steps": steps,
        "max_tokens": max_num_tokens,
        "seg_columns": list(SEG_COLUMNS),
    }
    (out_dir / f"census-rank{rank}.json").write_text(json.dumps(meta) + "\\n")
    atexit.register(census.flush)
    logger.info(
        "GLM53_EXPERT_CENSUS: recording %d MoE layers x top-%d for real steps "
        "%d..%d on rank %d into %s",
        len(layer_ids),
        top_k,
        skip,
        skip + steps - 1,
        rank,
        out_dir,
    )
    return census
'''

# --------------------------------------------------------------------------
# Hooks: v1/worker/gpu/model_runner.py (V2 model runner)
# --------------------------------------------------------------------------
RUNNER_EDITS = [
    (
        "model_runner.py: census attribute",
        """        self.routed_experts_capturer: RoutedExpertsCapturer | None = None
""",
        """        self.routed_experts_capturer: RoutedExpertsCapturer | None = None
        self._glm53_census = None  # GLM53_EXPERT_CENSUS (v13), set in load_model
""",
    ),
    (
        "model_runner.py: census bind after load",
        """        get_offloader().post_init()

    def get_model(self) -> nn.Module:
""",
        """        get_offloader().post_init()

        # GLM53_EXPERT_CENSUS (v13): None unless the env var names a directory.
        from vllm.v1.worker.gpu.glm53_expert_census import maybe_create_census

        self._glm53_census = maybe_create_census(
            self.model, self.max_num_tokens, self.device, self.vllm_config
        )

    def get_model(self) -> nn.Module:
""",
    ),
    (
        "model_runner.py: census stage after forward",
        """            routed_experts = capturer.get_routed_experts(slot_mappings, num_toks)
""",
        """            routed_experts = capturer.get_routed_experts(slot_mappings, num_toks)
        if not dummy_run and self._glm53_census is not None:
            self._glm53_census.stage(input_batch)
""",
    ),
    (
        "model_runner.py: census commit after sampling",
        """        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )
""",
        """        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )
        if self._glm53_census is not None:
            self._glm53_census.commit(num_sampled)
""",
    ),
]

FILE_EDITS = {
    "v1/worker/gpu/model_runner.py": RUNNER_EDITS,
}

NEW_FILES = {
    "v1/worker/gpu/glm53_expert_census.py": CENSUS_MODULE,
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
    for rel, content in NEW_FILES.items():
        path = root / rel
        if path.is_file():
            if path.read_text() != content:
                raise SystemExit(f"refusing: {path} exists with other content")
            print(f"{rel}: present")
            continue
        ast.parse(content, filename=str(path))
        print(f"{rel}: new file")
        staged[path] = content
    # All anchors checked and every result parses: now write.
    for path, content in staged.items():
        path.write_text(content)
    print(f"v13 census patch applied ({len(staged)} files written)")
    return list(staged)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
