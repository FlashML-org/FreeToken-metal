"""The Anthropic Messages API end to end over the real tokenizer worker and a scripted
scheduler: thinking, text and tool_use blocks, whole and streamed, and token counting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue

from ftmetal.server.api_server import create_app
from scripted_server import SCRIPT, ScriptedScheduler, named_sse_events, tiny_server_args

TOOLS: list[JsonValue] = [
    {
        "name": "get_weather",
        "description": "The weather in a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
    {"name": "get_time", "input_schema": {"properties": {"zone": {"type": "string"}}}},
]
QWEN3_CODER_PARIS = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"
    "</tool_call>"
)
GEMMA4_PARIS = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
# A content block as (type, text) or ("tool_use", (name, input)).
Block = tuple[str, str | tuple[str | None, JsonValue]]
PARIS_USE: Block = ("tool_use", ("get_weather", {"city": "Paris"}))


def messages_body(content: str = "What is 6 x 7?") -> dict[str, JsonValue]:
    return {"model": "tiny", "max_tokens": 512, "messages": [{"role": "user", "content": content}]}


def blocks_of(content: JsonValue) -> list[Block]:
    """The content blocks of a whole message; empty text blocks dropped, text stripped."""
    assert isinstance(content, list)
    blocks: list[Block] = []
    for block in content:
        assert isinstance(block, dict)
        if block["type"] == "thinking":
            thinking = block["thinking"]
            assert isinstance(thinking, str) and block["signature"] == ""
            blocks.append(("thinking", thinking))
        elif block["type"] == "text":
            text = block["text"]
            assert isinstance(text, str)
            if text.strip():
                blocks.append(("text", text.strip()))
        else:
            assert block["type"] == "tool_use"
            name = block["name"]
            assert isinstance(name, str) and isinstance(block["id"], str)
            assert block["id"].startswith("toolu_")
            blocks.append(("tool_use", (name, block["input"])))
    return blocks


def streamed_blocks(body: str) -> tuple[list[Block], str, JsonValue]:
    """(content blocks, stop_reason, stop_sequence) reassembled from the stream's events."""
    events = named_sse_events(body)
    assert events[0][0] == "message_start" and events[-1][0] == "message_stop"
    assert all(name == payload["type"] for name, payload in events)
    blocks: list[tuple[str, list[str], str | None]] = []
    stop_reason: str = ""
    stop_sequence: JsonValue = None
    for name, payload in events[1:-1]:
        if name == "content_block_start":
            block = payload["content_block"]
            assert isinstance(block, dict) and payload["index"] == len(blocks)
            if block["type"] == "thinking":
                assert block == {"type": "thinking", "thinking": "", "signature": ""}, block
            tool_name = block.get("name")
            blocks.append(
                (str(block["type"]), [], tool_name if isinstance(tool_name, str) else None)
            )
        elif name == "content_block_delta":
            delta = payload["delta"]
            assert isinstance(delta, dict) and payload["index"] == len(blocks) - 1
            for key in ("thinking", "text", "partial_json"):
                value = delta.get(key)
                if isinstance(value, str):
                    blocks[-1][1].append(value)
        elif name == "message_delta":
            delta = payload["delta"]
            assert isinstance(delta, dict) and isinstance(delta["stop_reason"], str)
            stop_reason, stop_sequence = delta["stop_reason"], delta.get("stop_sequence")
        else:
            assert name == "content_block_stop", name
    assembled: list[Block] = []
    for block_type, parts, tool_name in blocks:
        text = "".join(parts)
        if block_type == "tool_use":
            assembled.append(("tool_use", (tool_name, json.loads(text))))
        elif block_type == "thinking":
            assembled.append(("thinking", text))
        elif text.strip():
            assembled.append(("text", text.strip()))
    return assembled, stop_reason, stop_sequence


# (tool-call parser, reasoning parser, scripted completion, request fields, content blocks,
# stop_reason, stop_sequence)
MESSAGES_CASES: list[
    tuple[str, str | None, str, dict[str, JsonValue], list[Block], str, JsonValue]
] = [
    (
        "qwen3_coder",
        "qwen3",
        SCRIPT,
        {"thinking": {"type": "enabled", "budget_tokens": 1024}},
        [("thinking", "Let me think."), ("text", "The answer is 42.")],
        "end_turn",
        None,
    ),
    (
        "qwen3_coder",
        "qwen3",
        f"I should look it up.</think>\n\n{QWEN3_CODER_PARIS}",
        {"tools": TOOLS},
        [("thinking", "I should look it up."), PARIS_USE],
        "tool_use",
        None,
    ),
    (
        "qwen3_coder",
        "qwen3",
        "Forty-two.",
        {"thinking": {"type": "disabled"}},
        [("text", "Forty-two.")],
        "end_turn",
        None,
    ),
    (
        "qwen3_coder",
        "qwen3",
        "one two three four",
        {"thinking": {"type": "disabled"}, "stop_sequences": ["three"]},
        [("text", "one two")],
        "stop_sequence",
        "three",
    ),
    (
        "qwen3_coder",
        "qwen3",
        "one two three four",
        {"thinking": {"type": "disabled"}, "max_tokens": 3},
        [("text", "one")],
        "max_tokens",
        None,
    ),
    (
        "gemma4",
        "gemma4",
        f"<|channel>thought\nChecking the weather.<channel|>On it.{GEMMA4_PARIS}",
        {"tools": TOOLS, "tool_choice": {"type": "any"}},
        [("thinking", "Checking the weather."), ("text", "On it."), PARIS_USE],
        "tool_use",
        None,
    ),
]


@pytest.mark.parametrize(
    (
        "tool_call_parser",
        "reasoning_parser",
        "script",
        "fields",
        "blocks",
        "stop_reason",
        "stop_sequence",
    ),
    MESSAGES_CASES,
)
def test_messages_return_thinking_text_and_tool_use_blocks(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    monkeypatch: pytest.MonkeyPatch,
    tool_call_parser: str,
    reasoning_parser: str | None,
    script: str,
    fields: dict[str, JsonValue],
    blocks: list[Block],
    stop_reason: str,
    stop_sequence: JsonValue,
) -> None:
    monkeypatch.setattr(scripted, "script", script)
    args = tiny_server_args(tiny_model_dir, tool_call_parser, reasoning_parser)
    with TestClient(create_app(args)) as client:
        body = {**messages_body(), **fields}
        message = client.post("/v1/messages", json=body).json()
        streamed = client.post("/v1/messages", json={**body, "stream": True})
    assert (message["type"], message["role"], message["model"]) == ("message", "assistant", "tiny")
    assert message["id"].startswith("msg_")
    assert blocks_of(message["content"]) == blocks
    assert (message["stop_reason"], message.get("stop_sequence")) == (stop_reason, stop_sequence)
    assert message["usage"]["input_tokens"] > 0 and message["usage"]["output_tokens"] > 0
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert streamed_blocks(streamed.text) == (blocks, stop_reason, stop_sequence)


def test_count_tokens_counts_what_a_generation_reports(client: TestClient) -> None:
    body = {**messages_body(), "system": "Be brief.", "tools": TOOLS}
    usage = client.post("/v1/messages", json=body).json()["usage"]
    counted = client.post("/v1/messages/count_tokens", json=body)
    assert counted.status_code == 200
    assert counted.json() == {"input_tokens": usage["input_tokens"]}


# (--enable-cache-report, usage): the scripted prompts hit 5 cached tokens, which input_tokens
# then excludes, as Anthropic bills.
CACHE_REPORT_CASES: list[tuple[bool, int, JsonValue]] = [(False, 0, None), (True, 5, 5)]


@pytest.mark.parametrize(("enable_cache_report", "excluded", "cache_read"), CACHE_REPORT_CASES)
def test_usage_reports_cache_reads_when_asked(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    monkeypatch: pytest.MonkeyPatch,
    enable_cache_report: bool,
    excluded: int,
    cache_read: JsonValue,
) -> None:
    monkeypatch.setattr(scripted, "cached_tokens", 5)
    args = tiny_server_args(tiny_model_dir, enable_cache_report=enable_cache_report)
    with TestClient(create_app(args)) as client:
        counted = client.post("/v1/messages/count_tokens", json=messages_body()).json()
        usage = client.post("/v1/messages", json=messages_body()).json()["usage"]
    assert usage["input_tokens"] == counted["input_tokens"] - excluded
    assert usage.get("cache_read_input_tokens") == cache_read


# (route, request body, status code, error message fragment): Anthropic clients get every
# error as {"type": "error", "error": {...}}.
REFUSED_CASES: list[tuple[str, dict[str, JsonValue], int, str]] = [
    ("/v1/messages", {"model": "tiny", "messages": []}, 400, "max_tokens: Field required"),
    ("/v1/messages", {**messages_body(), "max_tokens": 0}, 400, "max_tokens must be positive"),
    ("/v1/messages/count_tokens", {"model": "tiny", "messages": []}, 400, "at least one message"),
    (
        "/v1/messages",
        {
            **messages_body(),
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "base64", "data": "AA=="}}],
                }
            ],
        },
        400,
        "no vision support",
    ),
    (
        "/v1/messages/count_tokens",
        {
            "model": "tiny",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "base64", "data": "AA=="}}],
                }
            ],
        },
        400,
        "no vision support",
    ),
]


@pytest.mark.parametrize(("route", "body", "status_code", "error"), REFUSED_CASES)
def test_invalid_requests_get_anthropic_errors(
    client: TestClient, route: str, body: dict[str, JsonValue], status_code: int, error: str
) -> None:
    response = client.post(route, json=body)
    assert response.status_code == status_code
    payload = response.json()
    assert payload["type"] == "error" and payload["error"]["type"] == "invalid_request_error"
    assert error in payload["error"]["message"]


def test_conversation_turns_reach_the_chat_template(
    client: TestClient, scripted: type[ScriptedScheduler]
) -> None:
    body = {
        **messages_body(),
        "system": [{"type": "text", "text": "Be brief."}],
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Look it up.", "signature": ""},
                    {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny"}],
            },
            # A system turn inside the array stays where it is, after the cached prefix; this
            # template renders none there, so it joins the tool result before it.
            {"role": "system", "content": "Use metric units."},
        ],
    }
    assert client.post("/v1/messages", json=body).status_code == 200
    prompt = scripted.instances[0].prompt_text(0)
    assert prompt.startswith("<|im_start|>system\nBe brief.<|im_end|>\n")
    turns = ["user\nWeather in Paris?", "assistant\nChecking.", "tool\nSunny\n\nUse metric units."]
    assert all(f"<|im_start|>{turn}<|im_end|>" in prompt for turn in turns)
