"""DFlash aux-hidden-state capture for Glm5Next (fork-only model).

The DFlash/DFlash2 drafter needs target hidden states from the layers named in
the draft's dflash_config.target_layer_ids. vLLM routes that through the
EAGLE3 interface (set_eagle3_aux_hidden_state_layers), which Glm5Next does not
implement -- TP ranks die with "Model does not support EAGLE3 interface".

Capture semantics follow sgl-project/sglang#36708 (the reference DFlash2
integration for GLM-5.3-Flash): the completed output of layer k, taken at the
top of iteration k+1, contracted from the n hyper-connection streams to
nominal width by averaging. In this fork a layer's hc_post is deferred into
the next layer's fused kernel, so the tap materializes it explicitly;
MHCPostOp returns a fresh tensor, so the deferred (x, residual, post, comb)
flow into the next layer is untouched.
"""

from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py")
s = p.read_text()

old = """from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)"""
new = """from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsPP,
)"""
if s.count(old) != 1:
    raise SystemExit("unexpected interfaces import; refusing to patch")
s = s.replace(old, new)

old = "class Glm5NextModel(nn.Module):"
new = "class Glm5NextModel(EagleModelMixin, nn.Module):"
if s.count(old) != 1:
    raise SystemExit("unexpected Glm5NextModel class def; refusing to patch")
s = s.replace(old, new)

old = """        for layer in self._active_layers:
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
"""
new = """        aux_hidden_states: list[torch.Tensor] = []
        prev_layer = None
        for layer_idx, layer in enumerate(
            self._active_layers, start=self.start_layer
        ):
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(
                    self._dflash_aux_hidden_state(
                        hidden_states, residual, post, comb, prev_layer,
                        full_num_tokens,
                    )
                )
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
            prev_layer = layer
"""
if s.count(old) != 1:
    raise SystemExit("unexpected Glm5NextModel layer loop; refusing to patch")
s = s.replace(old, new)

old = """        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:"""
new = """        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def _dflash_aux_hidden_state(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
        prev_layer: nn.Module | None,
        full_num_tokens: int,
    ) -> torch.Tensor:
        # Completed output of the previous layer at nominal width. Mid-stack,
        # a layer's hc_post is deferred into the next layer's fused kernel, so
        # materialize it here (MHCPostOp returns a fresh tensor) and average
        # the hyper-connection streams -- the DFlash capture semantics from
        # sgl-project/sglang#36708.
        if post is None:
            aux = hidden_states
        else:
            assert prev_layer is not None
            aux = hc_contract(
                prev_layer.hc_post(hidden_states, residual, post, comb),
                prev_layer.n,
            )
        if self.is_sequence_parallel:
            aux = sp_all_gather(aux)[:full_num_tokens]
        return aux

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:"""
if s.count(old) != 1:
    raise SystemExit("unexpected Glm5NextModel forward tail; refusing to patch")
s = s.replace(old, new)

old = """class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):"""
new = """class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid, SupportsEagle3
):"""
if s.count(old) != 1:
    raise SystemExit("unexpected Glm5NextForCausalLM class def; refusing to patch")
s = s.replace(old, new)

old = """class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid
):"""
new = """class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, SupportsEagle3
):"""
if s.count(old) != 1:
    raise SystemExit(
        "unexpected Glm5NextForConditionalGeneration class def; refusing to patch"
    )
s = s.replace(old, new)

p.write_text(s)
print("glm5next DFlash aux-hidden-state capture applied")
