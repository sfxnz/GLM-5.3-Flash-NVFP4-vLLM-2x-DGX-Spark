"""Opt-in FP8 weight-only (Marlin W8A16) for the BF16 non-MoE linears.

Every verify step streams ~8.3 GB/rank of BF16 projection weights that the
nvidia pack (and the DFlash2 drafter) keep unquantized: KDA in/out projections,
MLA q_b/o, shared experts, the lm_head (read twice: DFlash2 shares it) and the
drafter itself. This patch lets the recipe store selected groups as FP8 e4m3
with one scale per output channel and run them through vLLM's Marlin FP8 GEMM
(BF16 activations, in-register dequant), halving those bytes.

GLM53_FP8_W8A16 is a comma list of groups: draft, shared, mla, kda_o, kda_in,
lm_head. Unset or empty skips the new code entirely (v11 behaviour). The swap
runs at the end of model_loader.utils.process_weights_after_loading, once per
loaded model (target, then drafter), and frees each BF16 weight it replaces.

Stays BF16 by construction: indexer, router gate, mHC, embeddings, kv_b
(absorbed into the MLA BMMs), fused_qkv_a, KDA f_b/g_b/conv, vision tower.

Usage: python3 patch_v13_fp8.py [VLLM_ROOT]
(default /usr/local/lib/python3.12/dist-packages/vllm). Exact-substring
anchors; refuses on drift before writing anything; re-running is a no-op.
"""

import py_compile
import sys
from pathlib import Path

ROOT = Path(
    sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm"
)
MODULE = "model_executor/layers/quantization/glm53_fp8_w8a16.py"

MODULE_SRC = '''\
# SPDX-License-Identifier: Apache-2.0
"""GLM53_FP8_W8A16: opt-in FP8 weight-only (Marlin W8A16) for BF16 linears.

Installed by the GLM-5.3-Flash 2x DGX Spark recipe (docker/patch_v13_fp8.py).
model_loader.utils.process_weights_after_loading calls
apply_glm53_fp8_w8a16() only when GLM53_FP8_W8A16 is set.

Groups (comma list):
  kda_in   KDA in_proj_qkvbfg_a (merged q|k|v|b|f_a|g_a)
  kda_o    KDA o_proj
  mla      MLA q_b_proj and o_proj
  shared   MoE shared_experts gate_up_proj and down_proj
  lm_head  target ParallelLMHead (the DFlash2 drafter shares this module)
  draft    every BF16 linear of the DFlash/DFlash2 drafter

Quantization: symmetric e4m3, one scale per output channel of the per-rank
shard. The scale is amax/448 rounded to BF16, which is exactly the scale Marlin
applies, and W/scale is rounded once to e4m3.
"""

import os

import torch

ENV = "GLM53_FP8_W8A16"
GROUPS = ("draft", "shared", "mla", "kda_o", "kda_in", "lm_head")
FP8_MAX = 448.0

_TARGET_MODELS = frozenset({"Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"})
_DRAFT_MODEL = "DFlashQwen3ForCausalLM"
# owner class name -> group -> attribute paths of the linears to swap
_TARGET_SITES = {
    "Glm5NextLinearAttention": {
        "kda_in": ("in_proj_qkvbfg_a",),
        "kda_o": ("o_proj",),
    },
    "Glm5NextMLAAttention": {"mla": ("q_b_proj", "o_proj")},
    "Glm5NextMoE": {
        "shared": ("shared_experts.gate_up_proj", "shared_experts.down_proj"),
    },
}


def parse_groups(value: str) -> list[str]:
    groups = list(dict.fromkeys(g.strip() for g in value.split(",") if g.strip()))
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise ValueError(
            f"{ENV}: unknown group(s) {unknown}; valid: {','.join(GROUPS)}"
        )
    return groups


def quantize_per_channel(
    weight: torch.Tensor, chunk_elems: int = 1 << 24
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric e4m3 quantization of an (N, K) weight.

    Returns (q, scale): q is float8_e4m3fn (N, K), scale is (N,) in
    weight.dtype, and weight ~= q.float() * scale.float()[:, None]. Rows go in
    chunks so the fp32 transient stays at chunk_elems * 4 bytes.
    """
    n, k = weight.shape
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty(n, dtype=weight.dtype, device=weight.device)
    rows = max(1, chunk_elems // k)
    for i in range(0, n, rows):
        w = weight[i : i + rows].float()
        s = (w.abs().amax(dim=1) / FP8_MAX).clamp_min(1e-12).to(weight.dtype)
        scale[i : i + rows] = s
        q[i : i + rows] = (
            (w / s.float().unsqueeze(1))
            .clamp_(-FP8_MAX, FP8_MAX)
            .to(torch.float8_e4m3fn)
        )
    return q, scale


def dequantize_per_channel(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float().unsqueeze(1)


class Fp8MarlinW8A16Method:
    """Replacement quant_method: Marlin FP8 weight-only GEMM, BF16 activations.

    Deliberately not a QuantizeMethodBase: the loader's post-load loops must
    skip it, because the weight is already in its final Marlin layout. apply()
    makes no host sync and allocates only through the caching allocator (the
    same ops as vLLM's MarlinFP8 linear kernel), so it is CUDA-graph safe.
    """

    def __init__(self, gemm) -> None:
        self._gemm = gemm

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        out = self._gemm(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=None,
        )
        return out if bias is None else out + bias


def quantize_layer_to_marlin_fp8(layer: torch.nn.Module) -> None:
    """Swap one bias-free BF16 (N, K) linear or LM head to Marlin FP8 in place."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )
    from vllm.model_executor.utils import replace_parameter

    n, k = layer.weight.shape
    sizes = (("output_size_per_partition", n), ("input_size_per_partition", k))
    for attr, size in sizes:
        if getattr(layer, attr, size) != size:
            raise RuntimeError(f"{ENV}: {attr}={getattr(layer, attr)} != {size}")
        setattr(layer, attr, size)
    q, scale = quantize_per_channel(layer.weight.data)
    replace_parameter(layer, "weight", q)  # drops the BF16 weight
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    layer.orig_dtype = scale.dtype
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    layer.quant_method = Fp8MarlinW8A16Method(apply_fp8_marlin_linear)


def _collect_sites(model, groups, linear_base):
    """Return (kind, [(group, name, layer)]); kind is target, draft or None."""
    mro = {c.__name__ for c in type(model).__mro__}
    sites = []
    if mro & _TARGET_MODELS:
        for name, module in model.named_modules():
            cls = type(module).__name__
            for group, paths in _TARGET_SITES.get(cls, {}).items():
                if group in groups:
                    for path in paths:
                        layer = module.get_submodule(path)
                        sites.append((group, f"{name}.{path}", layer))
            if cls == "ParallelLMHead" and "lm_head" in groups:
                sites.append(("lm_head", name, module))
        return "target", sites
    if _DRAFT_MODEL in mro:
        if "draft" in groups:
            for name, module in model.named_modules():
                if isinstance(module, linear_base):
                    sites.append(("draft", name, module))
        return "draft", sites
    return None, sites


def apply_glm53_fp8_w8a16(model: torch.nn.Module, target_device: torch.device) -> None:
    from vllm.logger import init_logger
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    logger = init_logger(__name__)
    groups = parse_groups(os.environ.get(ENV, ""))
    model_name = type(model).__name__
    kind, sites = _collect_sites(model, groups, LinearBase)
    if kind is None:
        logger.info("%s: %s is not a GLM-5.3 target or DFlash drafter; unchanged",
                    ENV, model_name)
        return
    wanted = {g for g in groups if (g == "draft") == (kind == "draft")}
    missing = wanted - {g for g, _, _ in sites}
    if missing:
        raise RuntimeError(f"{ENV}: no {sorted(missing)} layers found in {model_name}")
    if not sites:
        return
    inner = getattr(model, "model", None)
    if kind == "draft" and not hasattr(inner, "_num_attn_layers"):
        # The fused context-KV buffers would be rebuilt from packed weights.
        logger.warning("%s: drafter weights not loaded; draft left BF16", ENV)
        return
    if target_device.type != "cuda":
        raise RuntimeError(f"{ENV} needs a CUDA target device, got {target_device}")

    reserved_before = torch.cuda.memory_reserved(target_device)
    plain = (UnquantizedLinearMethod, UnquantizedEmbeddingMethod)
    stats: dict[str, list[int]] = {}
    skipped = []
    for group, name, layer in sites:
        w = layer.weight
        if (
            not isinstance(layer.quant_method, plain)
            or w.dtype != torch.bfloat16
            or w.dim() != 2
            or getattr(layer, "bias", None) is not None
        ):
            skipped.append(name)
            continue
        bf16_bytes = w.numel() * w.element_size()
        del w  # the swap below must drop the last reference to the BF16 weight
        quantize_layer_to_marlin_fp8(layer)
        st = stats.setdefault(group, [0, 0, 0])
        st[0] += 1
        st[1] += bf16_bytes
        st[2] += layer.weight.numel() * layer.weight.element_size()
        st[2] += layer.weight_scale.numel() * layer.weight_scale.element_size()
    torch.cuda.empty_cache()
    logger.info(
        "%s: %s %s; torch reserved %.2f -> %.2f GiB",
        ENV,
        model_name,
        ", ".join(
            f"{g}={n} layers {b / 2**20:.1f}->{f / 2**20:.1f} MiB"
            for g, (n, b, f) in stats.items()
        ),
        reserved_before / 2**30,
        torch.cuda.memory_reserved(target_device) / 2**30,
    )
    if skipped:
        logger.warning("%s: left %d selected layers as they were (not plain "
                       "bias-free BF16): %s", ENV, len(skipped), skipped[:8])
'''

HOOK_OLD = """    if model_config.quantization == "torchao":
        set_torchao_reload_attrs(model, model_config)
"""
HOOK_NEW = HOOK_OLD + """
    # GLM53_FP8_W8A16 (recipe patch_v13_fp8): opt-in FP8 weight-only swap of
    # selected BF16 linears. Unset or empty keeps this function as it was.
    import os as _os

    if _os.environ.get("GLM53_FP8_W8A16", "").strip():
        from vllm.model_executor.layers.quantization.glm53_fp8_w8a16 import (
            apply_glm53_fp8_w8a16,
        )

        apply_glm53_fp8_w8a16(model, target_device)
"""

ENV_OLD = """    for var in ray_noset_env_vars:
        factors[var] = normalize_value(os.getenv(var))
"""
ENV_NEW = ENV_OLD + """
    # GLM53_FP8_W8A16 (recipe patch_v13_fp8) changes the linears inside the
    # torch.compile'd drafter, so it must key the compile cache. Hashed only
    # when set, so the unset key matches v11.
    if os.getenv("GLM53_FP8_W8A16", "").strip():
        factors["GLM53_FP8_W8A16"] = normalize_value(os.getenv("GLM53_FP8_W8A16"))
"""

EDITS = [
    ("model_executor/model_loader/utils.py", HOOK_OLD, HOOK_NEW, "post-load hook"),
    ("envs.py", ENV_OLD, ENV_NEW, "compile-cache factor"),
]


def main() -> None:
    writes: dict[Path, str] = {}
    for rel, old, new, what in EDITS:
        path = ROOT / rel
        text = path.read_text()
        if text.count(new) == 1:
            print(f"{rel}: {what} already applied")
            continue
        if text.count(old) != 1:
            raise SystemExit(f"unexpected source for {what} in {rel}; refusing")
        writes[path] = text.replace(old, new)

    module = ROOT / MODULE
    if module.exists():
        if module.read_text() != MODULE_SRC:
            raise SystemExit(f"{MODULE} exists with other content; refusing")
        print(f"{MODULE}: already installed")
    else:
        writes[module] = MODULE_SRC

    for path, text in writes.items():
        path.write_text(text)
    for rel in [MODULE] + [rel for rel, *_ in EDITS]:
        py_compile.compile(str(ROOT / rel), doraise=True)
    print(f"patch_v13_fp8: {len(writes)} file(s) written under {ROOT}")


if __name__ == "__main__":
    main()
