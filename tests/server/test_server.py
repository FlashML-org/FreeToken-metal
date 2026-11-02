"""The OpenAI API and /generate end to end over the real tokenizer worker and a scripted
scheduler, image input on a tiny multimodal checkpoint, plus the server's argument parsing.

With FTMETAL_TEST_MODEL_PATH set, one short request also runs on the real model.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import http.client
import io
import json
import logging
import os
import queue
import threading
import time
from contextlib import aclosing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import JsonValue
from tokenizers import Tokenizer

from ftmetal.core import SamplingParams
from ftmetal.env import ENV_PREFIXES
from ftmetal.message.frontend import BatchFrontendMsg, UserReply
from ftmetal.message.tokenizer import AbortMsg, BaseTokenizerMsg, FinishReason
from ftmetal.mm import MM_PAD_SHIFT_VALUE
from ftmetal.mm.config import ENCODER_KINDS, EncoderKind, Modality, MultimodalConfig
from ftmetal.server.api_server import FrontendManager, create_app
from ftmetal.server.args import (
    ServerArgs,
    infer_reasoning_parser,
    infer_tool_call_parser,
    parse_args,
)
from scripted_server import (
    SCRIPT,
    SCRIPT_TOKENS,
    WAIT_SECONDS,
    ScriptedScheduler,
    running_server,
    sse_events,
    tiny_server_args,
    wait_for_abort,
)
from tiny_vl_model import write_qwen_vl_model

MODEL_PATH = os.environ.get("FTMETAL_TEST_MODEL_PATH")
TOOLS: list[JsonValue] = [
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
QWEN3_CODER_PARIS = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
)
QWEN3_CODER_CET = (
    "<tool_call>\n<function=get_time>\n<parameter=zone>\nCET\n</parameter>\n</function>\n"
    "</tool_call>"
)
GEMMA4_PARIS = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|>'
GEMMA4_CET = '<|tool_call>call:get_time{zone:<|"|>CET<|"|>}<tool_call|>'
PARIS_CALL: tuple[str, JsonValue] = ("get_weather", {"city": "Paris", "days": 3})
CET_CALL: tuple[str, JsonValue] = ("get_time", {"zone": "CET"})


def chat_body(content: str = "What is 6 x 7?") -> dict[str, JsonValue]:
    return {"model": "tiny", "messages": [{"role": "user", "content": content}]}


def png_base64(width: int, height: int) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 30, 200)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def image_chat_body(image_base64: str, stream: bool = False) -> dict[str, JsonValue]:
    """A chat completion asking about one image, sent as a data: URL."""
    image_part: dict[str, JsonValue] = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
    }
    content: list[JsonValue] = [image_part, {"type": "text", "text": "What is it?"}]
    return {"model": "tiny", "messages": [{"role": "user", "content": content}], "stream": stream}


def uid_of(completion: dict[str, JsonValue]) -> int:
    completion_id = completion["id"]
    assert isinstance(completion_id, str)
    return int(completion_id.removeprefix("chatcmpl-"))


def tool_calls_of(calls: JsonValue) -> list[tuple[str, JsonValue]]:
    """(name, arguments) of an OpenAI tool_calls list, after checking each call's frame."""
    assert isinstance(calls, list)
    named: list[tuple[str, JsonValue]] = []
    for index, call in enumerate(calls):
        assert isinstance(call, dict) and isinstance(call["function"], dict)
        assert call["type"] == "function" and call["index"] == index
        assert isinstance(call["id"], str) and call["id"].startswith("call_")
        name, arguments = call["function"]["name"], call["function"]["arguments"]
        assert isinstance(name, str) and isinstance(arguments, str)
        named.append((name, json.loads(arguments)))
    return named


def streamed_chat(events: list[str]) -> tuple[str, str, list[tuple[str, JsonValue]], str]:
    """(reasoning, content, tool calls, finish reason) reassembled from chunk deltas."""
    assert events[-1] == "[DONE]"
    reasoning: list[str] = []
    content: list[str] = []
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    finish_reasons: list[str] = []
    for event in events[:-1]:
        chunk = json.loads(event)
        assert chunk["object"] == "chat.completion.chunk"
        for choice in chunk["choices"]:
            delta = choice["delta"]
            reasoning.append(delta.get("reasoning_content", ""))
            content.append(delta.get("content", ""))
            for call in delta.get("tool_calls", []):
                if "id" in call:
                    assert call["id"].startswith("call_") and call["type"] == "function"
                    names[call["index"]] = call["function"]["name"]
                fragment = call["function"]["arguments"]
                arguments[call["index"]] = arguments.get(call["index"], "") + fragment
            if choice["finish_reason"] is not None:
                finish_reasons.append(choice["finish_reason"])
    assert len(finish_reasons) == 1
    calls = [(names[index], json.loads(arguments[index])) for index in sorted(names)]
    return "".join(reasoning), "".join(content), calls, finish_reasons[0]


# (tool-call parser, reasoning parser, scripted completion, request fields, reasoning, content,
# tool calls, finish reason): the content compares stripped, as a stream keeps the whitespace
# around the reasoning block and the calls that a whole completion trims.
CHAT_CASES: list[
    tuple[str, str | None, str, dict[str, JsonValue], str, str, list[tuple[str, JsonValue]], str]
] = [
    ("qwen3_coder", "qwen3", SCRIPT, {}, "Let me think.", "The answer is 42.", [], "stop"),
    (
        "qwen3_coder",
        "qwen3",
        f"I should look it up.</think>\n\n{QWEN3_CODER_PARIS}",
        {"tools": TOOLS},
        "I should look it up.",
        "",
        [PARIS_CALL],
        "tool_calls",
    ),
    (
        "qwen3_coder",
        "qwen3",
        f"Two lookups.</think>\n\nOn it.\n{QWEN3_CODER_PARIS}\n{QWEN3_CODER_CET}",
        {"tools": TOOLS, "tool_choice": "required"},
        "Two lookups.",
        "On it.",
        [PARIS_CALL, CET_CALL],
        "tool_calls",
    ),
    # tool_choice none: the model sees no tools and its markup stays content.
    (
        "qwen3_coder",
        "qwen3",
        f"Hm.</think>\n\n{QWEN3_CODER_CET}",
        {"tools": TOOLS, "tool_choice": "none"},
        "Hm.",
        QWEN3_CODER_CET,
        [],
        "stop",
    ),
    (
        "gemma4",
        "gemma4",
        f"<|channel>thought\nChecking the weather.<channel|>{GEMMA4_PARIS}",
        {"tools": TOOLS},
        "Checking the weather.",
        "",
        [PARIS_CALL],
        "tool_calls",
    ),
    (
        "gemma4",
        "gemma4",
        f"Sure.{GEMMA4_PARIS}{GEMMA4_CET}",
        {"tools": TOOLS},
        "",
        "Sure.",
        [PARIS_CALL, CET_CALL],
        "tool_calls",
    ),
    ("gemma4", "gemma4", "Forty-two.", {}, "", "Forty-two.", [], "stop"),
]


@pytest.mark.parametrize(
    (
        "tool_call_parser",
        "reasoning_parser",
        "script",
        "fields",
        "reasoning",
        "content",
        "calls",
        "finish_reason",
    ),
    CHAT_CASES,
)
def test_chat_completion_splits_reasoning_content_and_tool_calls(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    monkeypatch: pytest.MonkeyPatch,
    tool_call_parser: str,
    reasoning_parser: str | None,
    script: str,
    fields: dict[str, JsonValue],
    reasoning: str,
    content: str,
    calls: list[tuple[str, JsonValue]],
    finish_reason: str,
) -> None:
    monkeypatch.setattr(scripted, "script", script)
    args = tiny_server_args(tiny_model_dir, tool_call_parser, reasoning_parser)
    with TestClient(create_app(args)) as client:
        body = {**chat_body(), **fields}
        completion = client.post("/v1/chat/completions", json=body).json()
        events = sse_events(client.post("/v1/chat/completions", json={**body, "stream": True}).text)
    choice = completion["choices"][0]
    message = choice["message"]
    assert message["role"] == "assistant"
    assert message.get("reasoning_content", "") == reasoning
    assert message["content"] == content
    assert tool_calls_of(message.get("tool_calls", [])) == calls
    assert choice["finish_reason"] == finish_reason
    streamed_reasoning, streamed_content, streamed_calls, streamed_finish = streamed_chat(events)
    assert (streamed_reasoning, streamed_content.strip()) == (reasoning, content)
    assert (streamed_calls, streamed_finish) == (calls, finish_reason)


def test_chat_completion_counts_usage(client: TestClient, tiny_model_dir: Path) -> None:
    response = client.post("/v1/chat/completions", json=chat_body())
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion" and body["model"] == "tiny"
    tokenizer = Tokenizer.from_file(str(tiny_model_dir / "tokenizer.json"))
    usage = body["usage"]
    assert usage["completion_tokens"] == len(tokenizer.encode(SCRIPT).ids) + 1
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert "prompt_tokens_details" not in usage


def test_streamed_chunks_follow_the_openai_format(client: TestClient) -> None:
    body = {**chat_body(), "stream": True, "stream_options": {"include_usage": True}}
    response = client.post("/v1/chat/completions", json=body)
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert {chunk["id"] for chunk in chunks} == {chunks[0]["id"]}
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert all(isinstance(chunk["created"], int) for chunk in chunks)
    assert chunks[0]["choices"] == [
        {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
    ]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks[1:-2]]
    assert all(len(delta) == 1 for delta in deltas)
    assert chunks[-2]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert "usage" not in chunks[-2]
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0


# (--enable-cache-report, prompt_tokens_details): the scripted prompts hit 5 cached tokens.
CACHE_REPORT_CASES: list[tuple[bool, JsonValue]] = [(False, None), (True, {"cached_tokens": 5})]


@pytest.mark.parametrize(("enable_cache_report", "details"), CACHE_REPORT_CASES)
def test_usage_reports_the_prefix_cache_hit_when_asked(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    monkeypatch: pytest.MonkeyPatch,
    enable_cache_report: bool,
    details: JsonValue,
) -> None:
    monkeypatch.setattr(scripted, "cached_tokens", 5)
    args = tiny_server_args(tiny_model_dir, enable_cache_report=enable_cache_report)
    with TestClient(create_app(args)) as client:
        usage = client.post("/v1/chat/completions", json=chat_body()).json()["usage"]
        body = {**chat_body(), "stream": True, "stream_options": {"include_usage": True}}
        events = sse_events(client.post("/v1/chat/completions", json=body).text)
        completion = client.post("/v1/completions", json={"model": "tiny", "prompt": "hi"}).json()
    streamed_usage = json.loads(events[-2])["usage"]
    for reported in (usage, streamed_usage, completion["usage"]):
        assert reported.get("prompt_tokens_details") == details


def test_stop_string_finishes_with_stop_and_aborts(
    client: TestClient, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scripted, "script", "one two three four five")
    body = {
        **chat_body(),
        "chat_template_kwargs": {"enable_thinking": False},
        "stop": ["three", "nine"],
    }
    completion = client.post("/v1/chat/completions", json=body).json()
    choice = completion["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "one two "}
    assert choice["finish_reason"] == "stop"
    assert completion["usage"]["completion_tokens"] == len("one two three")
    wait_for_abort(scripted.instances[0], uid_of(completion))


def test_max_tokens_finishes_with_length(client: TestClient) -> None:
    body = {**chat_body(), "max_completion_tokens": 3}
    completion = client.post("/v1/chat/completions", json=body).json()
    assert completion["choices"][0]["finish_reason"] == "length"
    assert completion["choices"][0]["message"]["reasoning_content"] == "Let"
    assert completion["usage"]["completion_tokens"] == 3


def test_models_lists_the_served_model(client: TestClient, tiny_model_dir: Path) -> None:
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    [card] = models["data"]
    assert (card["id"], card["root"], card["owned_by"]) == (
        "tiny-model",
        str(tiny_model_dir),
        "ftmetal",
    )
    # The tiny checkpoint has no config.json, so its context length is unknown.
    assert card["max_model_len"] is None and card["context_length"] is None
    assert client.get("/v1").json() == {"status": "ok"}


# (request fields, the end of the rendered prompt): how a request turns thinking on or off.
THINKING_CASES: list[tuple[dict[str, JsonValue], str]] = [
    ({}, "<|im_start|>assistant\n<think>\n"),
    ({"chat_template_kwargs": {"enable_thinking": False}}, "<think>\n\n</think>\n\n"),
    ({"thinking": {"type": "disabled"}}, "<think>\n\n</think>\n\n"),
    ({"thinking": {"type": "enabled"}}, "<|im_start|>assistant\n<think>\n"),
]


@pytest.mark.parametrize(("fields", "prompt_end"), THINKING_CASES)
def test_thinking_toggles_reach_the_chat_template(
    client: TestClient,
    scripted: type[ScriptedScheduler],
    fields: dict[str, JsonValue],
    prompt_end: str,
) -> None:
    completion = client.post("/v1/chat/completions", json={**chat_body(), **fields}).json()
    assert scripted.instances[0].prompt_text(uid_of(completion)).endswith(prompt_end)


# (tool_choice, the tools the chat template renders): tool_choice none hides them all, a named
# function shows only itself while every tool stays parseable.
TEMPLATE_TOOL_CASES: list[tuple[JsonValue, list[str]]] = [
    (None, ["get_weather", "get_time"]),
    ("auto", ["get_weather", "get_time"]),
    ({"type": "function", "function": {"name": "get_time"}}, ["get_time"]),
    ("none", []),
]


@pytest.mark.parametrize(("tool_choice", "rendered"), TEMPLATE_TOOL_CASES)
def test_tools_reach_the_chat_template(
    client: TestClient,
    scripted: type[ScriptedScheduler],
    tool_choice: JsonValue,
    rendered: list[str],
) -> None:
    body = {**chat_body(), "tools": TOOLS, "tool_choice": tool_choice}
    completion = client.post("/v1/chat/completions", json=body).json()
    prompt = scripted.instances[0].prompt_text(uid_of(completion))
    if rendered:
        tools = json.loads(prompt[prompt.index("<tools>") + 7 : prompt.index("</tools>")])
        assert [tool["function"]["name"] for tool in tools] == rendered
    else:
        assert "<tools>" not in prompt


def test_request_failures_surface_as_errors(
    client: TestClient, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    out_of_order = {
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": "late"}],
    }
    for stream in (False, True):
        # A stream's prompt is rendered before the headers go out, so it fails as a 400 too.
        rejected = client.post("/v1/chat/completions", json={**out_of_order, "stream": stream})
        assert rejected.status_code == 400
        assert "System message must be at the beginning" in rejected.json()["error"]["message"]

    monkeypatch.setattr(scripted, "max_prompt_tokens", 4)
    too_long = client.post("/v1/chat/completions", json=chat_body())
    assert too_long.status_code == 400
    assert too_long.json()["error"]["code"] == "context_length_exceeded"
    events = sse_events(
        client.post("/v1/chat/completions", json={**chat_body(), "stream": True}).text
    )
    assert json.loads(events[-2])["error"]["code"] == "context_length_exceeded"
    assert events[-1] == "[DONE]"


# (request body, choice texts, finish reasons, prompt tokens, completion tokens): a raw prompt
# gets no chat template and spends one token per byte or special token.
COMPLETION_CASES: list[tuple[dict[str, JsonValue], list[str], list[FinishReason], int, int]] = [
    ({"prompt": "6 x 7 ="}, [SCRIPT], ["stop"], 7, SCRIPT_TOKENS),
    ({"prompt": ["hi", "<|im_start|>"]}, [SCRIPT, SCRIPT], ["stop", "stop"], 3, 2 * SCRIPT_TOKENS),
    ({"prompt": "hi", "max_tokens": 3}, ["Let"], ["length"], 2, 3),
    (
        {"prompt": "hi", "max_tokens": SCRIPT_TOKENS + 2, "ignore_eos": True},
        [f"{SCRIPT}<|im_end|>Le"],
        ["length"],
        2,
        SCRIPT_TOKENS + 2,
    ),
    # n is ignored: one choice per prompt.
    ({"prompt": "hi", "n": 2}, [SCRIPT], ["stop"], 2, SCRIPT_TOKENS),
]


@pytest.mark.parametrize(
    ("body", "texts", "finish_reasons", "prompt_tokens", "completion_tokens"), COMPLETION_CASES
)
def test_completion_returns_the_raw_text(
    client: TestClient,
    body: dict[str, JsonValue],
    texts: list[str],
    finish_reasons: list[FinishReason],
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    response = client.post("/v1/completions", json={"model": "tiny", **body})
    assert response.status_code == 200
    completion = response.json()
    assert completion["object"] == "text_completion" and completion["id"].startswith("cmpl-")
    assert completion["choices"] == [
        {"index": index, "text": text, "finish_reason": finish_reason, "logprobs": None}
        for index, (text, finish_reason) in enumerate(zip(texts, finish_reasons))
    ]
    assert completion["usage"] == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def test_streamed_completion_chunks_carry_the_raw_text(client: TestClient) -> None:
    body = {
        "model": "tiny",
        "prompt": "hi",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    response = client.post("/v1/completions", json=body)
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert {chunk["id"] for chunk in chunks} == {chunks[0]["id"]}
    assert all(chunk["object"] == "text_completion.chunk" for chunk in chunks)
    choices = [chunk["choices"][0] for chunk in chunks[:-2]]
    assert all(choice["text"] and choice["finish_reason"] is None for choice in choices)
    assert "".join(choice["text"] for choice in choices) == SCRIPT
    assert chunks[-2]["choices"] == [
        {"index": 0, "text": "", "finish_reason": "stop", "logprobs": None}
    ]
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": SCRIPT_TOKENS,
        "total_tokens": 2 + SCRIPT_TOKENS,
    }


def test_generate_streams_the_raw_text_sampled_greedily(
    client: TestClient, scripted: type[ScriptedScheduler]
) -> None:
    response = client.post("/generate", json={"prompt": "hi", "max_tokens": 64})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert events[-1] == "[DONE]"
    texts = [json.loads(event)["text"] for event in events[:-1]]
    assert all(texts) and "".join(texts) == SCRIPT
    # The framework's greedy defaults, not the checkpoint's recommended sampling.
    assert scripted.instances[0].admitted[0] == ([71, 72], SamplingParams(max_tokens=64))


# (route, request body, the end of the prompt a next turn does not share)
REUSE_CASES: list[tuple[str, dict[str, JsonValue], str]] = [
    ("/v1/chat/completions", chat_body("hi"), "<|im_start|>assistant\n<think>\n"),
    ("/v1/completions", {"model": "tiny", "prompt": "hi", "max_tokens": 4}, ""),
    ("/generate", {"prompt": "hi", "max_tokens": 4}, ""),
]


def test_routes_admit_the_prompt_with_its_reusable_prefix_end(
    client: TestClient, scripted: type[ScriptedScheduler], tiny_model_dir: Path
) -> None:
    tokenizer = Tokenizer.from_file(str(tiny_model_dir / "tokenizer.json"))
    for uid, (route, body, unshared) in enumerate(REUSE_CASES):
        assert client.post(route, json=body).status_code == 200, route
        scheduler = scripted.instances[0]
        input_ids = scheduler.admitted[uid][0]
        reuse_len = scheduler.reuse_lens[uid]
        assert reuse_len is not None, route
        assert tokenizer.decode(input_ids[reuse_len:], skip_special_tokens=False) == unshared


# (route, request body, status code, a fragment of the error body)
REFUSED_CASES: list[tuple[str, dict[str, JsonValue], int, str]] = [
    ("/v1/chat/completions", {**chat_body(), "n": 2}, 400, '"param":"n"'),
    (
        "/v1/chat/completions",
        {**chat_body(), "reasoning_effort": "turbo"},
        400,
        "reasoning_effort must be one of",
    ),
    (
        "/v1/chat/completions",
        {**chat_body(), "function_call": "auto"},
        400,
        "use tools/tool_choice",
    ),
    ("/v1/chat/completions", {**chat_body(), "logit_bias": {"1": 1.0}}, 400, "logit_bias is not"),
    (
        "/v1/chat/completions",
        {**chat_body(), "response_format": {"type": "json_object"}},
        400,
        '"param":"response_format"',
    ),
    ("/v1/chat/completions", {**chat_body(), "thinking": {"type": "on"}}, 400, "thinking.type"),
    ("/v1/chat/completions", {**chat_body(), "max_tokens": 0}, 400, "max_tokens must be at least"),
    ("/v1/chat/completions", {"model": "tiny", "messages": []}, 400, "No messages provided"),
    ("/v1/chat/completions", {"model": "tiny"}, 422, "messages"),
    ("/v1/completions", {"model": "tiny", "prompt": [1, 2, 3]}, 400, "token-id prompt"),
    ("/v1/completions", {"model": "tiny", "prompt": [[1, 2], [3]]}, 400, "token-id prompt"),
    ("/v1/completions", {"model": "tiny", "prompt": "hi", "echo": True}, 400, "echo is not"),
    ("/v1/completions", {"model": "tiny", "prompt": "hi", "logprobs": 1}, 400, "logprobs is not"),
    (
        "/v1/completions",
        {"model": "tiny", "prompt": ["a", "b"], "stream": True},
        400,
        "only support a single text prompt",
    ),
    ("/v1/completions", {"model": "tiny", "prompt": "hi", "max_tokens": 0}, 400, "max_tokens"),
    ("/generate", {"prompt": "hi", "max_tokens": 0}, 400, "max_tokens must be at least 1, got 0"),
    ("/generate", {"prompt": [1, 2, 3], "max_tokens": 4}, 422, "prompt"),
]


@pytest.mark.parametrize(("route", "body", "status_code", "error"), REFUSED_CASES)
def test_invalid_requests_are_refused(
    client: TestClient, route: str, body: dict[str, JsonValue], status_code: int, error: str
) -> None:
    response = client.post(route, json=body)
    assert response.status_code == status_code and error in response.text


def test_image_reaches_the_scheduler_as_its_expanded_span(
    tiny_model_dir: Path, scripted: type[ScriptedScheduler]
) -> None:
    image_token_id = write_qwen_vl_model(tiny_model_dir)
    image_base64 = png_base64(64, 64)
    image_block: dict[str, JsonValue] = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": image_base64},
    }
    content: list[JsonValue] = [image_block, {"type": "text", "text": "What is it?"}]
    messages_body: dict[str, JsonValue] = {
        "model": "tiny",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": content}],
    }
    not_an_image = base64.b64encode(b"plain text").decode()
    with TestClient(create_app(tiny_server_args(tiny_model_dir))) as client:
        counted = client.post("/v1/messages/count_tokens", json=messages_body).json()
        usage = client.post("/v1/messages", json=messages_body).json()["usage"]
        chat = client.post("/v1/chat/completions", json=image_chat_body(image_base64))
        refused = client.post("/v1/chat/completions", json=image_chat_body(not_an_image, True))
    input_ids, _ = scripted.instances[0].admitted[0]
    # 64x64 is scaled up to the 64-token minimum; the placeholder is gone.
    pad_ids = [token for token in input_ids if token >= MM_PAD_SHIFT_VALUE]
    assert len(pad_ids) == 64 and len(set(pad_ids)) == 1 and image_token_id not in input_ids
    assert counted["input_tokens"] == usage["input_tokens"] == len(input_ids)
    # The OpenAI route renders the same prompt, the same image to the same pad id.
    assert chat.status_code == 200 and scripted.instances[0].admitted[1][0] == input_ids
    # The bytes are probed before the stream starts.
    assert refused.status_code == 400 and "could not load image" in refused.text


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
            body = image_chat_body(png_base64(8, 8), stream)
            response = client.post("/v1/chat/completions", json=body)
            assert response.status_code == 400 and reason in response.text


def test_raw_prompt_failures_surface_as_errors(
    client: TestClient, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scripted, "max_prompt_tokens", 4)
    body = {"model": "tiny", "prompt": "a long prompt"}
    rejected = client.post("/v1/completions", json=body)
    assert rejected.status_code == 400
    assert rejected.json()["error"]["message"] == "prompt is too long: 13 tokens"
    events = sse_events(client.post("/v1/completions", json={**body, "stream": True}).text)
    assert json.loads(events[-2])["error"]["message"] == "prompt is too long: 13 tokens"
    assert events[-1] == "[DONE]"
    generate_body = {"prompt": "a long prompt", "max_tokens": 4}
    events = sse_events(client.post("/generate", json=generate_body).text)
    assert events == [json.dumps({"error": "prompt is too long: 13 tokens"}), "[DONE]"]


def test_a_dead_engine_fails_its_requests_and_refuses_new_ones(
    client: TestClient, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scripted, "fails", True)
    failed = client.post("/v1/chat/completions", json=chat_body())
    assert failed.status_code == 400 and failed.json()["error"]["message"] == "the engine failed"
    deadline = time.monotonic() + WAIT_SECONDS
    while any(thread.name == "ftmetal-scheduler" for thread in threading.enumerate()):
        assert time.monotonic() < deadline, "the scheduler thread outlived its failure"
        time.sleep(0.01)
    refused = client.post("/v1/chat/completions", json=chat_body())
    assert refused.status_code == 503
    assert refused.json() == {"error": "server unavailable: maintenance failed (restart required)"}


# (route, streamed request body): every protocol aborts a stream its client drops.
DISCONNECT_CASES: list[tuple[str, dict[str, JsonValue]]] = [
    ("/v1/chat/completions", {**chat_body(), "stream": True}),
    ("/v1/completions", {"model": "tiny", "prompt": "hi", "stream": True}),
    ("/v1/messages", {**chat_body(), "max_tokens": 10000, "stream": True}),
    ("/v1/responses", {"model": "tiny", "input": "hi", "stream": True}),
    ("/generate", {"prompt": "hi", "max_tokens": 10000}),
]


@pytest.mark.parametrize(("route", "body"), DISCONNECT_CASES)
def test_client_disconnect_aborts_the_request(
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    body: dict[str, JsonValue],
) -> None:
    monkeypatch.setattr(scripted, "script", "tick " * 2000)
    monkeypatch.setattr(scripted, "step_seconds", 0.005)
    with running_server(create_app(tiny_server_args(tiny_model_dir))) as port:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT_SECONDS)
        headers = {"Content-Type": "application/json"}
        connection.request("POST", route, body=json.dumps(body), headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        lines = [response.readline() for _ in range(12)]
        assert all(line.startswith((b"data: ", b"event: ", b"\n")) for line in lines)
        connection.close()
        wait_for_abort(scripted.instances[0], 0)


def test_frontend_keeps_early_replies_and_aborts_on_early_close(tiny_model_dir: Path) -> None:
    frontend = FrontendManager(
        config=tiny_server_args(tiny_model_dir),
        send_tokenizer=queue.Queue(),
        backend_threads=[threading.current_thread()],
    )

    async def read_all(uid: int) -> list[str]:
        return [reply.incremental_output async for reply in frontend.wait_for_ack(uid)]

    async def read_one(uid: int) -> None:
        async with aclosing(frontend.wait_for_ack(uid)) as replies:
            async for _ in replies:
                break

    async def scenario() -> tuple[list[str], BaseTokenizerMsg | None]:
        finished = frontend.new_user()
        # The whole reply lands before anyone reads it, as with a fast engine and a stream.
        frontend.deliver(
            BatchFrontendMsg(
                data=[
                    UserReply(uid=finished, incremental_output="hi", finished=False),
                    UserReply(uid=finished, incremental_output="", finished=True),
                ]
            )
        )
        texts = await read_all(finished)
        abandoned = frontend.new_user()
        frontend.deliver(UserReply(uid=abandoned, incremental_output="a", finished=False))
        await read_one(abandoned)
        # A later abort of the same request (stream_with_cancellation's) sends nothing more.
        await frontend.abort_user(abandoned)
        return texts, frontend.send_tokenizer.get_nowait()

    texts, sent = asyncio.run(scenario())
    assert texts == ["hi", ""]
    assert sent == AbortMsg(uid=1)
    assert frontend.send_tokenizer.empty()
    assert frontend.ack_map == {}


@pytest.mark.parametrize(
    ("flags", "env_value", "command_graph"),
    [
        ([], None, True),
        ([], "0", False),
        ([], "1", True),
        (["--disable-command-graph"], None, False),
        (["--disable-command-graph"], "1", False),
    ],
)
def test_disable_command_graph_flag_overrides_the_environment(
    tiny_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    env_value: str | None,
    command_graph: bool,
) -> None:
    for prefix in ENV_PREFIXES:
        monkeypatch.delenv(f"{prefix}COMMAND_GRAPH", raising=False)
    if env_value is not None:
        monkeypatch.setenv("FTMETAL_COMMAND_GRAPH", env_value)
    args = parse_args(["--model", str(tiny_model_dir), *flags], prog="ftm serve")
    assert args.command_graph is command_graph


QWEN3_6_CONFIG: dict[str, JsonValue] = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {"model_type": "qwen3_5_moe_text"},
}
GEMMA4_CONFIG: dict[str, JsonValue] = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "text_config": {"model_type": "gemma4_text"},
}

# (config.json, flags, tool-call parser, reasoning parser)
PARSER_SELECTION_CASES: list[tuple[dict[str, JsonValue] | None, list[str], str, str | None]] = [
    (QWEN3_6_CONFIG, [], "qwen3_coder", "qwen3"),
    (GEMMA4_CONFIG, [], "gemma4", "gemma4"),
    (None, [], "llama3", None),
    (QWEN3_6_CONFIG, ["--reasoning-parser", "off"], "qwen3_coder", None),
    (
        GEMMA4_CONFIG,
        ["--tool-call-parser", "qwen25", "--reasoning-parser", "qwen3"],
        "qwen25",
        "qwen3",
    ),
]


@pytest.mark.parametrize(
    ("config", "flags", "tool_call_parser", "reasoning_parser"), PARSER_SELECTION_CASES
)
def test_parsers_follow_the_model_family(
    tiny_model_dir: Path,
    config: dict[str, JsonValue] | None,
    flags: list[str],
    tool_call_parser: str,
    reasoning_parser: str | None,
) -> None:
    if config is not None:
        (tiny_model_dir / "config.json").write_text(json.dumps(config))
    args = parse_args(["--model", str(tiny_model_dir), *flags], prog="ftm serve")
    assert (args.tool_call_parser, args.reasoning_parser) == (tool_call_parser, reasoning_parser)
    assert args.served_model_name == "tiny-model"


def test_cache_type_flag_defaults_to_radix(tiny_model_dir: Path) -> None:
    assert parse_args(["--model", str(tiny_model_dir)], prog="ftm serve").cache_type == "radix"
    args = parse_args(["--model", str(tiny_model_dir), "--cache-type", "naive"], prog="ftm serve")
    assert args.cache_type == "naive"


def test_swa_windows_flag_takes_a_window_count(tiny_model_dir: Path) -> None:
    flags = ["--model", str(tiny_model_dir)]
    assert parse_args(flags, prog="ftm serve").swa_windows is None
    assert parse_args([*flags, "--swa-windows", "12"], prog="ftm serve").swa_windows == 12
    for value in ("0", "1.5"):
        with pytest.raises(SystemExit):
            parse_args([*flags, "--swa-windows", value], prog="ftm serve")


def test_gdn_snapshots_flag_takes_a_snapshot_count(tiny_model_dir: Path) -> None:
    flags = ["--model", str(tiny_model_dir)]
    assert parse_args(flags, prog="ftm serve").gdn_snapshots is None
    assert parse_args([*flags, "--gdn-snapshots", "12"], prog="ftm serve").gdn_snapshots == 12
    for value in ("0", "1.5"):
        with pytest.raises(SystemExit):
            parse_args([*flags, "--gdn-snapshots", value], prog="ftm serve")


def test_multimodal_flags_reach_the_server_args(tiny_model_dir: Path) -> None:
    flags = ["--model", str(tiny_model_dir)]
    defaults = parse_args(flags, prog="ftm serve")
    assert defaults.mm == MultimodalConfig() and defaults.allowed_local_media_path == ""
    args = parse_args(
        [
            *flags,
            "--text-model-only",
            "--image-min-tokens",
            "64",
            "--image-max-tokens",
            "1024",
            "--allowed-media-domains",
            "images.example.com",
            "--allowed-local-media-path",
            str(tiny_model_dir),
        ],
        prog="ftm serve",
    )
    assert args.mm == MultimodalConfig(
        disabled_encoders=frozenset(ENCODER_KINDS), image_min_tokens=64, image_max_tokens=1024
    )
    assert args.allowed_media_domains == "images.example.com"
    assert args.allowed_local_media_path == os.path.realpath(tiny_model_dir)


# (encoder flags, the encoder kinds left unbuilt, whether the server is text-only, the
# modalities served from a checkpoint with a vision tower)
ENCODER_FLAG_CASES: list[tuple[list[str], frozenset[EncoderKind], bool, frozenset[Modality]]] = [
    (["--mm-disable", "vision"], frozenset({"vision"}), False, frozenset()),
    # No family registers an audio tower, so this leaves the vision tower serving.
    (["--mm-disable", "audio"], frozenset({"audio"}), False, frozenset({"image"})),
    (["--mm-disable", "vision", "audio"], frozenset(ENCODER_KINDS), True, frozenset()),
    (["--text-model-only", "--mm-disable", "audio"], frozenset(ENCODER_KINDS), True, frozenset()),
]


@pytest.mark.parametrize(
    ("flags", "disabled_encoders", "text_model_only", "modalities"), ENCODER_FLAG_CASES
)
def test_encoder_flags_leave_their_towers_unbuilt(
    tiny_model_dir: Path,
    flags: list[str],
    disabled_encoders: frozenset[EncoderKind],
    text_model_only: bool,
    modalities: frozenset[Modality],
) -> None:
    write_qwen_vl_model(tiny_model_dir)
    args = parse_args(["--model", str(tiny_model_dir), *flags], prog="ftm serve")
    assert args.mm.disabled_encoders == disabled_encoders
    assert args.mm.text_model_only is text_model_only
    assert args.served_modalities == modalities


# (flags, with {model} for the checkpoint directory; the startup refusal)
REFUSED_MULTIMODAL_FLAG_CASES: list[tuple[list[str], str]] = [
    (["--image-min-tokens", "2048", "--image-max-tokens", "1024"], "2048 exceeds"),
    (["--image-max-tokens", "0"], "expected a positive integer"),
    (["--allowed-local-media-path", "{model}/missing"], "is not a directory"),
    (["--mm-disable", "video"], "invalid choice: 'video'"),
]


@pytest.mark.parametrize(("flags", "error"), REFUSED_MULTIMODAL_FLAG_CASES)
def test_multimodal_flags_are_validated_at_startup(
    tiny_model_dir: Path, capsys: pytest.CaptureFixture[str], flags: list[str], error: str
) -> None:
    argv = ["--model", str(tiny_model_dir), *[flag.format(model=tiny_model_dir) for flag in flags]]
    with pytest.raises(SystemExit):
        parse_args(argv, prog="ftm serve")
    assert error in capsys.readouterr().err


# (flags, (--moe-strategy, --moe-cache-size, --moe-cache-rate, --moe-cache-auto,
# --kv-reserve-tokens, prefill overlap, --memory-ratio) as the server args hold them, the
# warning logged)
MOE_FLAG_CASES: list[
    tuple[list[str], tuple[str, int, float | None, bool, int, bool, float | None], str]
] = [
    ([], ("auto", 0, None, False, 8192, True, None), ""),
    # An offload without a size flag sizes its cache automatically.
    (["--moe-strategy", "offload"], ("offload", 0, None, True, 8192, True, None), ""),
    (
        ["--moe-strategy", "offload", "--moe-cache-rate", "0.7"],
        ("offload", 0, 0.7, False, 8192, True, None),
        "",
    ),
    # A rate of 0 sets nothing.
    (
        ["--moe-strategy", "offload", "--moe-cache-rate", "0"],
        ("offload", 0, None, True, 8192, True, None),
        "",
    ),
    (
        ["--moe-backend", "offload", "--moe-cache-size", "600"],
        ("offload", 600, None, False, 8192, True, None),
        "--moe-backend is deprecated; use --moe-strategy",
    ),
    (
        ["--moe-cache-auto", "--kv-reserve-tokens", "4096", "--memory-ratio", "0.43"],
        ("auto", 0, None, True, 4096, True, 0.43),
        "",
    ),
    (
        ["--disable-moe-prefill-overlap", "--moe-cache-policy", "lru"],
        ("auto", 0, None, False, 8192, False, None),
        "",
    ),
    (
        ["--expert-load", "serial"],
        ("auto", 0, None, False, 8192, True, None),
        "--expert-load is ignored",
    ),
]


@pytest.mark.parametrize(("flags", "fields", "warning"), MOE_FLAG_CASES)
def test_moe_flags_reach_the_server_args(
    tiny_model_dir: Path,
    caplog: pytest.LogCaptureFixture,
    flags: list[str],
    fields: tuple[str, int, float | None, bool, int, bool, float | None],
    warning: str,
) -> None:
    with caplog.at_level(logging.WARNING):
        args = parse_args(["--model", str(tiny_model_dir), *flags], prog="ftm serve")
    assert (
        args.moe_strategy,
        args.moe_cache_size,
        args.moe_cache_rate,
        args.moe_cache_auto,
        args.kv_reserve_tokens,
        args.moe_prefill_overlap,
        args.memory_ratio,
    ) == fields
    assert args.moe_cache_policy == "lru"
    assert warning in caplog.text


# (flags, the startup refusal)
REFUSED_MOE_FLAG_CASES: list[tuple[list[str], str]] = [
    (["--moe-strategy", "cpu"], "--moe-strategy cpu computes experts on the CPU"),
    (["--moe-backend", "hybrid"], "--moe-strategy hybrid computes experts on the CPU"),
    (["--moe-cache-size", "600", "--moe-cache-rate", "0.5"], "not allowed with argument"),
    (["--moe-cache-rate", "0.5", "--moe-cache-auto"], "not allowed with argument"),
    (["--moe-cache-rate", "1.5"], "expected a number in [0, 1]"),
    (["--moe-cache-size", "-1"], "expected a non-negative integer"),
    (["--memory-ratio", "0"], "expected a number in (0, 1]"),
    (["--moe-cache-policy", "lfu"], "invalid choice: 'lfu'"),
    (["--moe-victim-cache-gib", "-1"], "expected auto or a non-negative number"),
    (["--moe-victim-cache-gib", "inf"], "expected auto or a non-negative number"),
]


# (flags, --moe-victim-cache-gib as the server args hold it)
VICTIM_CACHE_FLAG_CASES: list[tuple[list[str], float | str]] = [
    ([], 0),
    (["--moe-victim-cache-gib", "auto"], "auto"),
    (["--moe-victim-cache-gib", "4.5"], 4.5),
]


@pytest.mark.parametrize(("flags", "expected"), VICTIM_CACHE_FLAG_CASES)
def test_victim_cache_flag_reaches_the_server_args(
    tiny_model_dir: Path, flags: list[str], expected: float | str
) -> None:
    args = parse_args(["--model", str(tiny_model_dir), *flags], prog="ftm serve")
    assert args.moe_victim_cache_gib == expected


@pytest.mark.parametrize(("flags", "error"), REFUSED_MOE_FLAG_CASES)
def test_moe_flags_are_validated_at_startup(
    tiny_model_dir: Path, capsys: pytest.CaptureFixture[str], flags: list[str], error: str
) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model", str(tiny_model_dir), *flags], prog="ftm serve")
    assert error in capsys.readouterr().err


@pytest.mark.skipif(MODEL_PATH is None, reason="FTMETAL_TEST_MODEL_PATH is not set")
def test_real_model_answers_over_http() -> None:
    assert MODEL_PATH is not None
    config = ServerArgs(
        model_path=MODEL_PATH,
        max_running_req=1,
        max_seq_len_override=4096,
        num_page_override=256,
        tool_call_parser=infer_tool_call_parser(MODEL_PATH),
        reasoning_parser=infer_reasoning_parser(MODEL_PATH),
    )
    with TestClient(create_app(config)) as real_client:
        body = {
            **chat_body("What is 2 + 3? Reply with the number only."),
            "max_tokens": 32,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        choice = real_client.post("/v1/chat/completions", json=body).json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert "5" in choice["message"]["content"]
        streamed = {**chat_body("Why is the sky blue?"), "stream": True, "max_tokens": 16}
        events = sse_events(real_client.post("/v1/chat/completions", json=streamed).text)
    chunks = [json.loads(event) for event in events[:-1]]
    assert any("reasoning_content" in chunk["choices"][0]["delta"] for chunk in chunks[1:])
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
