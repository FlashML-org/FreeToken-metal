"""The reasoning and tool-call parsers on hand-picked completions, streamed in chunks and
parsed whole."""

from __future__ import annotations

import json

import pytest
from pydantic import JsonValue

from ftmetal.server.api_models import Tool
from ftmetal.server.function_call_parser import FunctionCallParser
from ftmetal.server.reasoning_parser import ReasoningParser

# (parser, force_reasoning, streamed chunks, reasoning, content streamed, content one-shot):
# chunk edges split the markers so the hold-back has to reassemble them.
REASONING_CASES: list[tuple[str, bool, list[str], str, str, str]] = [
    (
        "qwen3",
        True,
        ["Let me", " th", "ink.</th", "ink>\n\nAns", "wer <th", "e end"],
        "Let me think.",
        "\n\nAnswer <the end",
        "Answer <the end",
    ),
    ("qwen3", False, ["<thi", "nk>abc</think>", "def"], "abc", "def", "def"),
    ("qwen3", False, ["no ", "block <", "th"], "", "no block <th", "no block <th"),
    ("qwen3", True, ["still ", "thinking </thi"], "still thinking </thi", "", ""),
    # A tool call right after an unclosed think goes to the tool parser.
    (
        "qwen3",
        True,
        ["I need a shell.<tool_", "call><function=run></function></tool_call>"],
        "I need a shell.",
        "<tool_call><function=run></function></tool_call>",
        "<tool_call><function=run></function></tool_call>",
    ),
    # Once the think is closed, a tool marker inside it is only text.
    (
        "qwen3",
        True,
        ["Mention <tool_call> only.</th", "ink>Visible"],
        "Mention <tool_call> only.",
        "Visible",
        "Visible",
    ),
    (
        "gemma4",
        False,
        ["<|channel>thou", "ght\nLet me think.<chan", "nel|>The answer."],
        "Let me think.",
        "The answer.",
        "The answer.",
    ),
    ("gemma4", False, ["Just ", "the answer."], "", "Just the answer.", "Just the answer."),
    ("gemma4", True, ["Thinking<channel|>", "Done"], "Thinking", "Done", "Done"),
    ("flint", False, ["<thi", "nk>Let me th", "ink.</think>Done"], "Let me think.", "Done", "Done"),
    ("flint", False, ["Just ", "the answer."], "", "Just the answer.", "Just the answer."),
    (
        "deepseekv32",
        True,
        ["reason ", "<｜DSML｜function_calls>", "<｜DSML｜invoke"],
        "reason ",
        "<｜DSML｜function_calls><｜DSML｜invoke",
        "<｜DSML｜function_calls><｜DSML｜invoke",
    ),
    (
        "gpt_oss",
        False,
        [
            "<|channel|>analysis<|message|>think",
            "ing<|end|><|start|>assistant<|channel|>final<|message|>hi",
            " there<|return|>",
        ],
        "thinking",
        "hi there",
        "hi there",
    ),
    ("minimax_m3", False, ["\n</mm:th", "ink>Hello"], "", "Hello", "Hello"),
]


@pytest.mark.parametrize(
    ("name", "force_reasoning", "chunks", "reasoning", "content", "content_one_shot"),
    REASONING_CASES,
)
def test_reasoning_parser_splits_reasoning_from_content(
    name: str,
    force_reasoning: bool,
    chunks: list[str],
    reasoning: str,
    content: str,
    content_one_shot: str,
) -> None:
    streamed = ReasoningParser(name, force_reasoning=force_reasoning)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in chunks:
        reasoning_delta, content_delta = streamed.parse_stream_chunk(chunk)
        reasoning_parts.append(reasoning_delta)
        content_parts.append(content_delta)
    reasoning_delta, content_delta = streamed.flush()
    assert "".join([*reasoning_parts, reasoning_delta]) == reasoning
    assert "".join([*content_parts, content_delta]) == content
    one_shot = ReasoningParser(name, force_reasoning=force_reasoning)
    assert one_shot.parse_non_stream("".join(chunks)) == (reasoning.strip(), content_one_shot)


TOOL_DICTS: list[dict[str, JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}},
        },
    },
]
TOOLS = [Tool.model_validate(tool) for tool in TOOL_DICTS]
QWEN3_CODER_PARIS = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "</function>\n</tool_call>"
)
QWEN3_CODER_CET = (
    "<tool_call>\n<function=get_time>\n<parameter=zone>\nCET\n</parameter>\n</function>\n"
    "</tool_call>"
)
GEMMA4_PARIS = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
GEMMA4_CET = '<|tool_call>call:get_time{zone:<|"|>CET<|"|>}<tool_call|>'

# (parser, streamed chunks, content around the calls, the calls as (name, arguments))
TOOL_CALL_CASES: list[tuple[str, list[str], str, list[tuple[str, dict[str, JsonValue]]]]] = [
    (
        "qwen3_coder",
        [
            "Let me check.\n<tool_",
            "call>\n<function=get_weather>\n<parameter=city>\nPar",
            "is\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
        ],
        "Let me check.",
        [("get_weather", {"city": "Paris", "days": 3})],
    ),
    (
        "qwen3_coder",
        [QWEN3_CODER_PARIS, "\n", QWEN3_CODER_CET],
        "",
        [("get_weather", {"city": "Paris"}), ("get_time", {"zone": "CET"})],
    ),
    ("qwen3_coder", ["No tools ", "needed."], "No tools needed.", []),
    (
        "qwen25",
        [
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}\n',
            "</tool_call>",
        ],
        "",
        [("get_weather", {"city": "Paris", "days": 3})],
    ),
    (
        "gemma4",
        ['Checking.<|tool_call>call:get_weather{city:<|"|>Par', 'is<|"|>,days:3}<tool_call|>'],
        "Checking.",
        [("get_weather", {"city": "Paris", "days": 3})],
    ),
    (
        "gemma4",
        [GEMMA4_PARIS, GEMMA4_CET],
        "",
        [("get_weather", {"city": "Paris"}), ("get_time", {"zone": "CET"})],
    ),
]


@pytest.mark.parametrize(("name", "chunks", "content", "calls"), TOOL_CALL_CASES)
def test_tool_call_parser_extracts_calls(
    name: str, chunks: list[str], content: str, calls: list[tuple[str, dict[str, JsonValue]]]
) -> None:
    streamed = FunctionCallParser(TOOLS, name)
    assert streamed.supports_streaming() and streamed.args_fragments_prefix_stable()
    text: list[str] = []
    names: dict[int, str | None] = {}
    arguments: dict[int, str] = {}
    for chunk in chunks:
        for kind, payload in streamed.parse_stream_events(chunk):
            if kind == "text":
                text.append(payload)
            else:
                for fragment in payload:
                    names.setdefault(fragment.tool_index, fragment.name)
                    arguments[fragment.tool_index] = (
                        arguments.get(fragment.tool_index, "") + fragment.parameters
                    )
    text.append(streamed.finish_stream())
    assert "".join(text).strip() == content
    assert [(names[i], json.loads(arguments[i])) for i in sorted(names)] == calls
    one_shot = FunctionCallParser(TOOLS, name).parse_non_stream("".join(chunks))
    assert one_shot.normal_text.strip() == content
    assert [(call.name, json.loads(call.parameters)) for call in one_shot.calls] == calls
