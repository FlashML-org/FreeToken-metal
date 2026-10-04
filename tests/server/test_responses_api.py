"""The OpenAI Responses API end to end over the real tokenizer worker and a scripted
scheduler: reasoning, message and function_call items, whole and streamed, and the
stateless routes."""

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
        "type": "function",
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
    {"type": "web_search"},
]
QWEN3_CODER_PARIS = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"
    "</tool_call>"
)
GEMMA4_PARIS = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'

# An output item as (type, text) or ("function_call", (name, arguments)).
Item = tuple[str, str | tuple[str, JsonValue]]
PARIS_CALL: Item = ("function_call", ("get_weather", {"city": "Paris"}))


def item_text(item: dict[str, JsonValue], part_type: str) -> str:
    """The text of a reasoning or message item's single content part."""
    content = item["content"]
    assert isinstance(content, list) and len(content) == 1
    [part] = content
    assert isinstance(part, dict) and part["type"] == part_type
    text = part["text"]
    assert isinstance(text, str)
    return text


def items_of(output: JsonValue) -> list[Item]:
    """The items of a response's output, each checked for its frame; message text is stripped
    and a whitespace-only message dropped, as a stream keeps the whitespace before a call."""
    assert isinstance(output, list)
    items: list[Item] = []
    for item in output:
        assert isinstance(item, dict) and isinstance(item["id"], str)
        if item["type"] == "reasoning":
            assert item["id"].startswith("rs_")
            items.append(("reasoning", item_text(item, "reasoning_text")))
        elif item["type"] == "message":
            assert item["id"].startswith("msg_") and item["role"] == "assistant"
            text = item_text(item, "output_text").strip()
            if text:
                items.append(("message", text))
        else:
            assert item["type"] == "function_call" and item["id"].startswith("fc_")
            assert isinstance(item["call_id"], str) and item["call_id"].startswith("call_")
            assert isinstance(item["name"], str) and isinstance(item["arguments"], str)
            items.append(("function_call", (item["name"], json.loads(item["arguments"]))))
    return items


def streamed_items(body: str) -> tuple[list[Item], dict[str, JsonValue]]:
    """(items, the terminal response) of a stream, after checking that the deltas add up to
    the items the stream closes."""
    events = named_sse_events(body)
    assert [name for name, _ in events[:2]] == ["response.created", "response.in_progress"]
    assert [payload["sequence_number"] for _, payload in events] == list(range(len(events)))
    deltas: dict[str, str] = {}
    done: list[JsonValue] = []
    for name, payload in events:
        assert payload["type"] == name
        if name.endswith(".delta"):
            item_id = payload["item_id"]
            assert isinstance(item_id, str) and isinstance(payload["delta"], str)
            deltas[item_id] = deltas.get(item_id, "") + payload["delta"]
        elif name == "response.output_item.done":
            done.append(payload["item"])
    terminal_name, terminal = events[-1]
    assert terminal_name in ("response.completed", "response.incomplete")
    response = terminal["response"]
    assert isinstance(response, dict) and response["output"] == done
    for item in done:
        assert isinstance(item, dict)
        if item["type"] == "function_call":
            assert deltas[str(item["id"])] == item["arguments"]
        elif item["type"] == "reasoning":
            assert deltas[str(item["id"])] == item_text(item, "reasoning_text")
        else:
            assert deltas[str(item["id"])] == item_text(item, "output_text")
    return items_of(done), response


# (tool-call parser, reasoning parser, scripted completion, request fields, output items,
# status)
RESPONSES_CASES: list[tuple[str, str | None, str, dict[str, JsonValue], list[Item], str]] = [
    (
        "qwen3_coder",
        "qwen3",
        SCRIPT,
        {"instructions": "Be brief."},
        [("reasoning", "Let me think."), ("message", "The answer is 42.")],
        "completed",
    ),
    (
        "qwen3_coder",
        "qwen3",
        f"I should look it up.</think>\n\n{QWEN3_CODER_PARIS}",
        {"tools": TOOLS, "reasoning": {"summary": "auto"}},
        [("reasoning", "I should look it up."), PARIS_CALL],
        "completed",
    ),
    (
        "gemma4",
        "gemma4",
        f"<|channel>thought\nChecking the weather.<channel|>On it.{GEMMA4_PARIS}",
        {"tools": TOOLS, "tool_choice": {"type": "function", "name": "get_weather"}},
        [("reasoning", "Checking the weather."), ("message", "On it."), PARIS_CALL],
        "completed",
    ),
    (
        "qwen3_coder",
        "qwen3",
        SCRIPT,
        {"max_output_tokens": 3},
        [("reasoning", "Let")],
        "incomplete",
    ),
]


@pytest.mark.parametrize(
    ("tool_call_parser", "reasoning_parser", "script", "fields", "items", "status"),
    RESPONSES_CASES,
)
def test_responses_return_reasoning_message_and_function_call_items(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    monkeypatch: pytest.MonkeyPatch,
    tool_call_parser: str,
    reasoning_parser: str | None,
    script: str,
    fields: dict[str, JsonValue],
    items: list[Item],
    status: str,
) -> None:
    monkeypatch.setattr(scripted, "script", script)
    args = tiny_server_args(tiny_model_dir, tool_call_parser, reasoning_parser)
    with TestClient(create_app(args)) as client:
        body = {"model": "tiny", "input": "What is 6 x 7?", **fields}
        response = client.post("/v1/responses", json=body).json()
        streamed = client.post("/v1/responses", json={**body, "stream": True})
    assert (response["object"], response["model"], response["status"]) == (
        "response",
        "tiny",
        status,
    )
    assert response["id"].startswith("resp_")
    assert items_of(response["output"]) == items
    usage = response["usage"]
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["input_tokens_details"]["cached_tokens"] == 0
    if status == "incomplete":
        assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert streamed.headers["content-type"].startswith("text/event-stream")
    streamed_output, streamed_response = streamed_items(streamed.text)
    assert streamed_output == items and streamed_response["status"] == status


def test_input_items_reach_the_chat_template(
    client: TestClient, scripted: type[ScriptedScheduler]
) -> None:
    body = {
        "model": "tiny",
        "instructions": "Be brief.",
        "input": [
            {"type": "message", "role": "developer", "content": "Use metric units."},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "Look it up."}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "Sunny"},
        ],
    }
    assert client.post("/v1/responses", json=body).status_code == 200
    prompt = scripted.instances[0].prompt_text(0)
    assert prompt.startswith("<|im_start|>system\nBe brief.\n\nUse metric units.<|im_end|>\n")
    assert "<|im_start|>user\nHi<|im_end|>" in prompt
    assert "<|im_start|>tool\nSunny<|im_end|>" in prompt


def test_cached_tokens_are_reported_when_asked(
    tiny_model_dir: Path, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scripted, "cached_tokens", 5)
    with TestClient(
        create_app(tiny_server_args(tiny_model_dir, enable_cache_report=True))
    ) as client:
        response = client.post("/v1/responses", json={"model": "tiny", "input": "hi"}).json()
    assert response["usage"]["input_tokens_details"]["cached_tokens"] == 5


# (method, route, request body, status code, error message fragment)
REFUSED_CASES: list[tuple[str, str, dict[str, JsonValue] | None, int, str]] = [
    ("GET", "/v1/responses/resp_1", None, 404, "not found (stateless server)"),
    ("POST", "/v1/responses/resp_1/cancel", None, 404, "not found (stateless server)"),
    (
        "POST",
        "/v1/responses",
        {"model": "tiny", "input": "hi", "previous_response_id": "resp_1"},
        400,
        "previous_response_id is not supported",
    ),
    (
        "POST",
        "/v1/responses",
        {"model": "tiny", "input": "hi", "background": True},
        400,
        "background",
    ),
    (
        "POST",
        "/v1/responses",
        {"model": "tiny", "input": "hi", "max_output_tokens": 0},
        400,
        "max_output_tokens must be a positive integer",
    ),
]


@pytest.mark.parametrize(("method", "route", "body", "status_code", "error"), REFUSED_CASES)
def test_unsupported_requests_are_refused(
    client: TestClient,
    method: str,
    route: str,
    body: dict[str, JsonValue] | None,
    status_code: int,
    error: str,
) -> None:
    response = client.request(method, route, json=body)
    assert response.status_code == status_code
    assert error in response.json()["error"]["message"]
