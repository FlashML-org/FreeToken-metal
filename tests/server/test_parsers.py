"""The reasoning parsers on hand-picked completions, streamed in chunks and parsed whole."""

from __future__ import annotations

import pytest

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
