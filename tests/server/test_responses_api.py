"""The OpenAI Responses API end to end over the real tokenizer worker and a scripted
scheduler: reasoning, message and function_call items, whole and streamed, image input, and
the stateless routes."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import JsonValue
from tokenizers import Tokenizer

from ftmetal.mm import MM_PAD_SHIFT_VALUE
from ftmetal.mm.config import ENCODER_KINDS, EncoderKind, MultimodalConfig
from ftmetal.server.api_server import create_app
from scripted_server import SCRIPT, ScriptedScheduler, named_sse_events, tiny_server_args
from tiny_vl_model import write_qwen_vl_model

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


def png_data_url(width: int, height: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 30, 200)).save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def image_input(image_url: str) -> list[JsonValue]:
    """A user message asking about one image."""
    content: list[JsonValue] = [
        {"type": "input_image", "image_url": image_url},
        {"type": "input_text", "text": "What is it?"},
    ]
    return [{"type": "message", "role": "user", "content": content}]


def image_span(input_ids: list[int]) -> tuple[int, int]:
    """The half-open span of the one image's pad ids."""
    rows = [i for i, token in enumerate(input_ids) if token >= MM_PAD_SHIFT_VALUE]
    assert rows == list(range(rows[0], rows[-1] + 1))
    return rows[0], rows[-1] + 1


def test_input_image_reaches_the_scheduler_as_the_chat_route_sends_it(
    tiny_model_dir: Path, scripted: type[ScriptedScheduler]
) -> None:
    write_qwen_vl_model(tiny_model_dir)
    image_url = png_data_url(64, 64)
    chat_content: list[JsonValue] = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "What is it?"},
    ]
    chat_body = {"model": "tiny", "messages": [{"role": "user", "content": chat_content}]}
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as client:
        response = client.post(
            "/v1/responses", json={"model": "tiny", "input": image_input(image_url)}
        )
        streamed = client.post(
            "/v1/responses", json={"model": "tiny", "input": image_input(image_url), "stream": True}
        )
        chat = client.post("/v1/chat/completions", json=chat_body)
    assert response.status_code == streamed.status_code == chat.status_code == 200
    admitted = [scripted.instances[0].admitted[uid][0] for uid in range(3)]
    assert admitted[0] == admitted[1] == admitted[2]
    # 64x64 is scaled up to the 64-token minimum.
    start, end = image_span(admitted[0])
    assert end - start == 64
    assert response.json()["usage"]["input_tokens"] == len(admitted[0])


def test_tool_output_image_rides_the_following_user_turn(
    tiny_model_dir: Path, scripted: type[ScriptedScheduler]
) -> None:
    write_qwen_vl_model(tiny_model_dir)
    output: list[JsonValue] = [
        {"type": "input_text", "text": "the screenshot"},
        {"type": "input_image", "image_url": png_data_url(64, 64)},
    ]
    body = {
        "model": "tiny",
        "input": [
            {"type": "message", "role": "user", "content": "Take a screenshot."},
            {"type": "function_call", "call_id": "call_1", "name": "screenshot", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": output},
        ],
    }
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as client:
        assert client.post("/v1/responses", json=body).status_code == 200
    input_ids = scripted.instances[0].admitted[0][0]
    start, end = image_span(input_ids)
    tokenizer = Tokenizer.from_file(str(tiny_model_dir / "tokenizer.json"))
    before = tokenizer.decode(input_ids[:start], skip_special_tokens=False)
    after = tokenizer.decode(input_ids[end:], skip_special_tokens=False)
    assert before.endswith(
        "<|im_start|>tool\nthe screenshot<|im_end|>\n<|im_start|>user\n<|vision_start|>"
    )
    assert after.startswith("<|vision_end|><|im_end|>\n<|im_start|>assistant\n")


# (whether the checkpoint has a vision tower, the encoder kinds left unbuilt, the refusal)
IMAGE_REFUSAL_CASES: list[tuple[bool, frozenset[EncoderKind], str]] = [
    (False, frozenset(), "the served model has no vision support"),
    (True, frozenset(ENCODER_KINDS), "image input is disabled on this server (--text-model-only)"),
    (True, frozenset({"vision"}), "image input is disabled on this server (--mm-disable)"),
]


@pytest.mark.parametrize(("has_vision", "disabled_encoders", "reason"), IMAGE_REFUSAL_CASES)
def test_text_only_server_refuses_images_before_any_stream(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    has_vision: bool,
    disabled_encoders: frozenset[EncoderKind],
    reason: str,
) -> None:
    write_qwen_vl_model(tiny_model_dir, has_vision)
    args = dataclasses.replace(
        tiny_server_args(tiny_model_dir), mm=MultimodalConfig(disabled_encoders=disabled_encoders)
    )
    with TestClient(create_app(args)) as client:
        for stream in (False, True):
            body = {"model": "tiny", "input": image_input(png_data_url(8, 8)), "stream": stream}
            response = client.post("/v1/responses", json=body)
            assert response.status_code == 400 and reason in response.json()["error"]["message"]
    assert scripted.instances[0].admitted == {}


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
