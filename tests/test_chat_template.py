"""CPU checks for chat_template.jinja. Run: python3 -m unittest discover -s tests

tests/data/nvidia_hub_chat_template.jinja is the nvidia/GLM-5.3-Flash-NVFP4 Hub
template at 09b04e5e74bca08ca8549fc736d4cdd8624bfde3 (git blob fb94d40d).
"""
import itertools
import json
import unittest
from pathlib import Path

try:
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment
except ImportError:  # pragma: no cover - CI installs jinja2
    jinja2 = None

REPO = Path(__file__).resolve().parent.parent
LOCAL = REPO / "chat_template.jinja"
HUB = Path(__file__).resolve().parent / "data" / "nvidia_hub_chat_template.jinja"
GEN_START = "{%- if add_generation_prompt -%}\n"

# The two documented differences from the Hub template, both keyed on the parser's thinking rule: the
# generation prompt closes an empty think block when thinking is off, and the system effort line then says
# Low instead of Max unless low or high was requested (Max beside a closed think block corrupted tables).
HUB_EFFORT_LINE = "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}\n"
LOCAL_EFFORT_LINES = (
    "{#- Thinking off with no low/high effort renders Low, not Max: Max beside a closed think block corrupted dense numeric output (evidence/r3/b0 utf8 A/B). -#}\n"
    "{%- set _head_thinking = thinking if thinking is defined else none -%}\n"
    "{%- set _head_enable_thinking = enable_thinking if enable_thinking is defined else none -%}\n"
    "{%- set _head_thinking_on = (_head_thinking is none and _head_enable_thinking is none) or _head_thinking or _head_enable_thinking -%}\n"
    "{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else ('max' if _head_thinking_on else 'low') -%}\n"
)
LOCAL_GEN_BLOCK = """{%- if add_generation_prompt -%}
    <|assistant|>
    {#- vLLM's glm45/glm47 parser: on when both kwargs are unset, else thinking or enable_thinking. -#}
    {%- set _thinking = thinking if thinking is defined else none -%}
    {%- set _enable_thinking = enable_thinking if enable_thinking is defined else none -%}
    {%- if (_thinking is none and _enable_thinking is none) or _thinking or _enable_thinking -%}
        {{- '<think>' -}}
    {%- else -%}
        {{- '<think></think>' -}}
    {%- endif -%}
{%- endif -%}
"""
HUB_GEN_BLOCK = """{%- if add_generation_prompt -%}
    <|assistant|>{{- '<think>' -}}
{%- endif -%}
"""

UNSET = object()


def parser_thinking(kwargs):
    """vllm/parser/glm47_moe.py Glm47MoeParser.__init__ (v11 image), used by glm45 and glm47."""
    thinking = kwargs.get("thinking", None)
    enable_thinking = kwargs.get("enable_thinking", None)
    if thinking is None and enable_thinking is None:
        return True
    return bool(thinking) or bool(enable_thinking)


def render(path, **kwargs):
    """Render like transformers' apply_chat_template (the path vLLM uses)."""
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env.filters["tojson"] = tojson
    template = env.from_string(path.read_text())
    return template.render(
        messages=[{"role": "user", "content": "hi"}], add_generation_prompt=True, **kwargs
    )


@unittest.skipIf(jinja2 is None, "jinja2 not importable")
class KwargMatrix(unittest.TestCase):
    def test_generation_prompt_follows_parser(self):
        values = (True, False, UNSET)
        efforts = (UNSET, "low", "high", "medium", "minimal", "max", "none")
        for enable, think, effort in itertools.product(values, values, efforts):
            kwargs = {}
            if enable is not UNSET:
                kwargs["enable_thinking"] = enable
            if think is not UNSET:
                kwargs["thinking"] = think
            if effort is not UNSET:
                kwargs["reasoning_effort"] = effort
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                out = render(LOCAL, **kwargs)
                want = "<|assistant|><think>" if parser_thinking(kwargs) else "<|assistant|><think></think>"
                self.assertTrue(out.endswith(want), out[-60:])
                label = {"low": "Low", "high": "High"}.get(effort, "Max" if parser_thinking(kwargs) else "Low")
                self.assertTrue(out.startswith(f"[gMASK]<sop><|system|>Reasoning Effort: {label}"), out[:60])

    def test_thinking_off_default_matches_clean_ab_arm(self):
        # The r3 utf8 A/B's clean thinking-off arm sent enable_thinking false with reasoning_effort low.
        self.assertEqual(render(LOCAL, enable_thinking=False), render(LOCAL, enable_thinking=False, reasoning_effort="low"))
        self.assertEqual(render(LOCAL, enable_thinking=True), render(HUB, enable_thinking=True))

    def test_thinking_true_overrides_server_default(self):
        # Server default {"enable_thinking": false} merged with a request {"thinking": true}.
        out = render(LOCAL, enable_thinking=False, thinking=True)
        self.assertTrue(out.endswith("<|assistant|><think>"))

    def test_vllm_passes_both_kwargs_through(self):
        # vllm/renderers/hf.py drops chat_template_kwargs the template never references.
        import jinja2.meta

        env = ImmutableSandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
        names = jinja2.meta.find_undeclared_variables(env.parse(LOCAL.read_text()))
        self.assertLessEqual({"thinking", "enable_thinking", "reasoning_effort"}, names)

    def test_hub_template_always_opens_think(self):
        self.assertTrue(render(HUB, enable_thinking=False).endswith("<|assistant|><think>"))


class HubParity(unittest.TestCase):
    def test_only_documented_difference(self):
        local, hub = LOCAL.read_text(), HUB.read_text()
        self.assertEqual(local.count(GEN_START), 1)
        self.assertEqual(hub.count(GEN_START), 1)
        local_head, local_tail = local.split(GEN_START)
        hub_head, hub_tail = hub.split(GEN_START)
        self.assertEqual(hub_head.count(HUB_EFFORT_LINE), 1)
        self.assertEqual(local_head, hub_head.replace(HUB_EFFORT_LINE, LOCAL_EFFORT_LINES),
                         "chat_template.jinja drifted from the Hub template outside the two documented differences")
        self.assertEqual(GEN_START + local_tail, LOCAL_GEN_BLOCK)
        self.assertEqual(GEN_START + hub_tail, HUB_GEN_BLOCK)


if __name__ == "__main__":
    unittest.main()
