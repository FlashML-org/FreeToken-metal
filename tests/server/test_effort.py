"""Reasoning effort: quantization onto a checkpoint's vocabulary, the chat-template probes
that learn the vocabulary and the thinking toggles, and the routes that carry an effort.

With FTMETAL_TEST_MODEL_PATH set, the real checkpoint's chat template is probed too (its
tokenizer and template only, no weights).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue

from ftmetal.effort import KNOWN_REASONING_EFFORTS, EffortProfile, quantize_effort
from ftmetal.server.api_server import create_app
from ftmetal.server.model_meta import derive_think_gears
from ftmetal.tokenizer import TokenizeManager
from scripted_server import ScriptedScheduler, tiny_server_args

MODEL_PATH = os.environ.get("FTMETAL_TEST_MODEL_PATH")
ALL_EFFORTS = frozenset(KNOWN_REASONING_EFFORTS)

# Heads put in front of the tiny checkpoint's Qwen3.6-shaped template, which reads only
# enable_thinking. This one grades effort and rejects what it does not know (gpt-oss style).
GRADES_EFFORT_HEAD = """\
{%- set effort = reasoning_effort if reasoning_effort is defined else 'medium' %}
{%- if effort not in ['low', 'medium', 'high'] %}
{{- raise_exception('reasoning_effort must be low, medium or high') }}
{%- endif %}
{{- '<|im_start|>system\\nReasoning: ' + effort + '<|im_end|>\\n' }}
"""
# Interpolates any reasoning_strength (muse-glimmer style).
STRENGTH_LADDER_HEAD = """\
{%- if reasoning_strength is defined %}
{{- '<|im_start|>system\\nReasoning strength: ' + reasoning_strength + '<|im_end|>\\n' }}
{%- endif %}
"""

IGNORES = EffortProfile(supported=ALL_EFFORTS, default=None, consumes_effort=False)
OPENAI_TRIPLE = EffortProfile(
    supported=frozenset({"low", "medium", "high"}),
    default="medium",
    consumes_effort=True,
    validates=True,
)
# DSV4's encoder: two gears, the top one opt-in only.
HIGH_MAX = EffortProfile(
    supported=frozenset({"high", "max"}), default="high", consumes_effort=True, validates=True
)
# Grades effort without validating it.
GRADER = EffortProfile(supported=ALL_EFFORTS, default=None, consumes_effort=True)
STRENGTH = EffortProfile(
    supported=ALL_EFFORTS, default=None, consumes_effort=True, strength_dialect=True
)

# (profile, requested effort, the gear it is served as; None sends no effort)
QUANTIZE_CASES: list[tuple[EffortProfile, JsonValue, str | None]] = [
    (IGNORES, "high", None),
    (OPENAI_TRIPLE, "low", "low"),
    (OPENAI_TRIPLE, "high", "high"),
    (OPENAI_TRIPLE, "xhigh", "high"),
    (OPENAI_TRIPLE, "minimal", "low"),
    # 0.2 from low on the scale: too far to round.
    (OPENAI_TRIPLE, "none", None),
    (OPENAI_TRIPLE, "max", "high"),
    (OPENAI_TRIPLE, "turbo", None),
    # Only named levels sit on the scale.
    (OPENAI_TRIPLE, 0.9, None),
    (HIGH_MAX, "medium", None),
    # max is reachable by its own name only, never by rounding.
    (HIGH_MAX, "xhigh", "high"),
    (HIGH_MAX, "max", "max"),
    # A grader that validates nothing is served the OpenAI triple.
    (GRADER, "xhigh", "high"),
    (GRADER, "max", "high"),
    # The strength dialect's ladder tops out at xhigh.
    (STRENGTH, "xhigh", "xhigh"),
    (STRENGTH, "max", "xhigh"),
    (STRENGTH, "minimal", "low"),
]


@pytest.mark.parametrize(("profile", "requested", "served"), QUANTIZE_CASES)
def test_quantize_effort(profile: EffortProfile, requested: JsonValue, served: str | None) -> None:
    assert quantize_effort(requested, profile) == served


# (head of the chat template, None for the tiny checkpoint's own, and its probed profile)
EFFORT_PROBE_CASES: list[tuple[str | None, EffortProfile]] = [
    (None, IGNORES),
    (GRADES_EFFORT_HEAD, OPENAI_TRIPLE),
    (STRENGTH_LADDER_HEAD, STRENGTH),
]


def use_template_head(model_dir: Path, head: str | None) -> None:
    if head is not None:
        template_path = model_dir / "chat_template.jinja"
        template_path.write_text(head + template_path.read_text())


@pytest.mark.parametrize(("head", "profile"), EFFORT_PROBE_CASES)
def test_probe_learns_the_effort_vocabulary(
    tiny_model_dir: Path, head: str | None, profile: EffortProfile
) -> None:
    use_template_head(tiny_model_dir, head)
    assert TokenizeManager(str(tiny_model_dir)).effort_profile() == profile


# (chat template, toggleable, has_adaptive, default state, thinking gears, default gear)
THINKING_PROBE_CASES: list[tuple[str, bool, bool, str, tuple[str, ...], str]] = [
    (
        "{{ messages[0].content }}{% if enable_thinking is not defined or enable_thinking %}"
        "<think>{% endif %}",
        True,
        False,
        "on",
        ("off", "on"),
        "on",
    ),
    (
        "{{ messages[0].content }}{% if enable_thinking is defined and enable_thinking %}"
        "<think>{% endif %}",
        True,
        False,
        "off",
        ("off", "on"),
        "off",
    ),
    # No toggle spelling reaches thinking; the reasoning parser still labels the state.
    (
        "{{ messages[0].content }}{% if thinking is defined and thinking %}<think>{% endif %}",
        False,
        False,
        "on",
        ("on",),
        "on",
    ),
    (
        "{{ messages[0].content }}"
        "{% set mode = thinking_mode if thinking_mode is defined else 'adaptive' %}"
        "{% if mode == 'enabled' %}<think>{% elif mode == 'adaptive' %}<maybe>{% endif %}",
        True,
        True,
        "adaptive",
        ("off", "adaptive", "on"),
        "adaptive",
    ),
]


@pytest.mark.parametrize(
    ("template", "toggleable", "has_adaptive", "default_state", "gears", "default_gear"),
    THINKING_PROBE_CASES,
)
def test_probe_learns_the_thinking_toggles(
    tiny_model_dir: Path,
    template: str,
    toggleable: bool,
    has_adaptive: bool,
    default_state: str,
    gears: tuple[str, ...],
    default_gear: str,
) -> None:
    (tiny_model_dir / "chat_template.jinja").write_text(template)
    profile = TokenizeManager(str(tiny_model_dir)).thinking_profile()
    assert (profile.toggleable, profile.has_adaptive, profile.default_state) == (
        toggleable,
        has_adaptive,
        default_state,
    )
    derived = derive_think_gears(profile, parser_configured=True)
    assert derived is not None and derived[:2] == (gears, default_gear)


# (route, head of the chat template, the requested effort, how the prompt starts, how it ends)
EFFORT_ROUTE_CASES: list[tuple[str, str | None, str, str, str]] = [
    ("/v1/chat/completions", None, "high", "<|im_start|>user\n", "assistant\n<think>\n"),
    ("/v1/chat/completions", None, "none", "<|im_start|>user\n", "<think>\n\n</think>\n\n"),
    ("/v1/chat/completions", None, " Off", "<|im_start|>user\n", "<think>\n\n</think>\n\n"),
    (
        "/v1/chat/completions",
        GRADES_EFFORT_HEAD,
        "xhigh",
        "<|im_start|>system\nReasoning: high<|im_end|>\n",
        "assistant\n<think>\n",
    ),
    (
        "/v1/chat/completions",
        GRADES_EFFORT_HEAD,
        "minimal",
        "<|im_start|>system\nReasoning: low<|im_end|>\n",
        "assistant\n<think>\n",
    ),
    (
        "/v1/chat/completions",
        GRADES_EFFORT_HEAD,
        "none",
        "<|im_start|>system\nReasoning: medium<|im_end|>\n",
        "<think>\n\n</think>\n\n",
    ),
    (
        "/v1/chat/completions",
        STRENGTH_LADDER_HEAD,
        "max",
        "<|im_start|>system\nReasoning strength: xhigh<|im_end|>\n",
        "assistant\n<think>\n",
    ),
    ("/v1/responses", None, "none", "<|im_start|>user\n", "<think>\n\n</think>\n\n"),
    ("/v1/responses", None, "high", "<|im_start|>user\n", "assistant\n<think>\n"),
    (
        "/v1/responses",
        GRADES_EFFORT_HEAD,
        "low",
        "<|im_start|>system\nReasoning: low<|im_end|>\n",
        "assistant\n<think>\n",
    ),
]


@pytest.mark.parametrize(("route", "head", "effort", "start", "end"), EFFORT_ROUTE_CASES)
def test_requested_effort_reaches_the_chat_template(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    route: str,
    head: str | None,
    effort: str,
    start: str,
    end: str,
) -> None:
    use_template_head(tiny_model_dir, head)
    if route == "/v1/chat/completions":
        body: dict[str, JsonValue] = {
            "model": "tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": effort,
        }
    else:
        body = {"model": "tiny", "input": "hi", "reasoning": {"effort": effort}}
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as client:
        assert client.post(route, json=body).status_code == 200
    prompt = scripted.instances[0].prompt_text(0)
    assert prompt.startswith(start) and prompt.endswith(end), prompt


# (head of the chat template, supported_reasoning_efforts, default_reasoning_effort)
MODEL_EFFORT_CASES: list[tuple[str | None, list[str] | None, str | None]] = [
    (None, None, None),
    (GRADES_EFFORT_HEAD, ["high", "medium", "low"], "medium"),
    (STRENGTH_LADDER_HEAD, ["xhigh", "high", "medium", "low"], None),
]


@pytest.mark.parametrize(("head", "supported", "default"), MODEL_EFFORT_CASES)
def test_models_list_the_served_efforts(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    head: str | None,
    supported: list[str] | None,
    default: str | None,
) -> None:
    use_template_head(tiny_model_dir, head)
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as client:
        [card] = client.get("/v1/models").json()["data"]
    assert (card["supported_reasoning_efforts"], card["default_reasoning_effort"]) == (
        supported,
        default,
    )


# (chat template, thinking_toggleable, default_thinking)
MODEL_THINKING_SWITCH_CASES: list[tuple[str, bool, bool | None]] = [
    (THINKING_PROBE_CASES[0][0], True, True),
    (THINKING_PROBE_CASES[1][0], True, False),
    (THINKING_PROBE_CASES[2][0], False, None),
    (THINKING_PROBE_CASES[3][0], True, True),
]


@pytest.mark.parametrize(
    ("template", "toggleable", "default_thinking"), MODEL_THINKING_SWITCH_CASES
)
def test_models_report_the_thinking_switch(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    template: str,
    toggleable: bool,
    default_thinking: bool | None,
) -> None:
    (tiny_model_dir / "chat_template.jinja").write_text(template)
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as client:
        [card] = client.get("/v1/models").json()["data"]
    assert (card["thinking_toggleable"], card["default_thinking"]) == (toggleable, default_thinking)


@pytest.mark.skipif(MODEL_PATH is None, reason="FTMETAL_TEST_MODEL_PATH is not set")
def test_real_template_takes_effort_as_the_thinking_switch() -> None:
    """Qwen3.6 and Gemma4 read enable_thinking and no effort."""
    assert MODEL_PATH is not None
    profile = TokenizeManager(MODEL_PATH).thinking_profile()
    assert not profile.efforts.consumes_effort and profile.toggleable
