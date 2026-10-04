"""The OpenAI API and /generate end to end over the real tokenizer worker and a scripted
scheduler.

The scheduler thread runs ScriptedScheduler, which samples a fixed text for every
request, so the routes, the message flow and the detokenizer are tested without a
model. With FTMETAL_TEST_MODEL_PATH set, one short request also runs on the real model.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import queue
import threading
import time
from collections.abc import Iterator
from contextlib import aclosing
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from pydantic import JsonValue
from tokenizers import Tokenizer

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.env import ENV_PREFIXES
from ftmetal.message.frontend import BatchFrontendMsg, UserReply
from ftmetal.message.tokenizer import AbortMsg, BaseTokenizerMsg, FinishReason
from ftmetal.scheduler.scheduler import ForwardData, Scheduler, TokenEvent
from ftmetal.scheduler.status import SchedulerStatusReporter
from ftmetal.server import api_server
from ftmetal.server.api_server import create_app
from ftmetal.server.args import ServerArgs, parse_args
from ftmetal.server.generation import FrontendManager
from ftmetal.server.reasoning_parser import ThinkReasoningParser

MODEL_PATH = os.environ.get("FTMETAL_TEST_MODEL_PATH")
SCRIPT = "Let me think.</think>\n\nThe answer is 42."
# One token per byte except </think>, plus EOS.
SCRIPT_TOKENS = len(SCRIPT.encode()) - len("</think>") + 2
WAIT_SECONDS = 5.0


class ScriptedScheduler(Scheduler):
    """Samples script's tokens then EOS for every request, one token per step."""

    script: ClassVar[str] = SCRIPT
    step_seconds: ClassVar[float] = 0.0
    max_prompt_tokens: ClassVar[int] = 4096
    instances: ClassVar[list[ScriptedScheduler]] = []

    def __init__(self, config: EngineConfig, eos_token_ids: set[int]) -> None:
        self.tokenizer = Tokenizer.from_file(str(Path(config.model_path) / "tokenizer.json"))
        self.eos_token_id = min(eos_token_ids)
        self.results: list[TokenEvent] = []
        self.live_uids: set[int] = set()
        self.status_reporter = SchedulerStatusReporter(lambda line: None)
        # uid -> (tokens left, whether the last one is EOS)
        self.running: dict[int, tuple[list[int], bool]] = {}
        self.admitted: dict[int, tuple[list[int], SamplingParams]] = {}
        self.aborted: list[int] = []
        ScriptedScheduler.instances.append(self)

    def add_request(self, uid: int, input_ids: np.ndarray, sampling_params: SamplingParams) -> None:
        if len(input_ids) > self.max_prompt_tokens:
            raise ValueError(f"prompt is too long: {len(input_ids)} tokens")
        self.admitted[uid] = (input_ids.tolist(), sampling_params)
        script_tokens = [*self.tokenizer.encode(self.script).ids, self.eos_token_id]
        # Sampling past EOS repeats the script until max_tokens.
        repeats = sampling_params.max_tokens // len(script_tokens) + 1
        tokens = script_tokens * repeats if sampling_params.ignore_eos else script_tokens
        self.running[uid] = (
            tokens[: sampling_params.max_tokens],
            len(tokens) <= sampling_params.max_tokens,
        )

    def abort_request(self, uid: int) -> None:
        self.running.pop(uid, None)
        self.aborted.append(uid)

    @property
    def runnable(self) -> bool:
        return len(self.running) > 0

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        time.sleep(self.step_seconds)
        for uid, (tokens, ends_with_eos) in list(self.running.items()):
            token = tokens.pop(0)
            finish_reason: Literal["stop", "length"] | None = None
            if not tokens:
                finish_reason = "stop" if ends_with_eos else "length"
                self.running.pop(uid)
            self.results.append(TokenEvent(uid, token, not tokens, finish_reason))
        return None


def wait_for_abort(scheduler: ScriptedScheduler, uid: int) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    while uid not in scheduler.aborted:
        assert time.monotonic() < deadline, f"request {uid} was never aborted"
        time.sleep(0.01)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> type[ScriptedScheduler]:
    ScriptedScheduler.instances.clear()
    monkeypatch.setattr(api_server, "Scheduler", ScriptedScheduler)
    return ScriptedScheduler


@pytest.fixture
def client(tiny_model_dir: Path, scripted: type[ScriptedScheduler]) -> Iterator[TestClient]:
    with TestClient(create_app(ServerArgs(model_path=str(tiny_model_dir)))) as test_client:
        yield test_client


def chat_body(content: str = "What is 6 x 7?") -> dict[str, JsonValue]:
    return {"model": "tiny", "messages": [{"role": "user", "content": content}]}


def sse_events(body: str) -> list[str]:
    return [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]


def uid_of(completion: dict[str, JsonValue]) -> int:
    completion_id = completion["id"]
    assert isinstance(completion_id, str)
    return int(completion_id.removeprefix("chatcmpl-"))


def test_chat_completion_splits_reasoning_and_counts_usage(
    client: TestClient, tiny_model_dir: Path
) -> None:
    response = client.post("/v1/chat/completions", json=chat_body())
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion" and body["model"] == "tiny"
    choice = body["choices"][0]
    assert choice["message"] == {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning_content": "Let me think.",
    }
    assert choice["finish_reason"] == "stop"
    tokenizer = Tokenizer.from_file(str(tiny_model_dir / "tokenizer.json"))
    usage = body["usage"]
    assert usage["completion_tokens"] == len(tokenizer.encode(SCRIPT).ids) + 1
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_streamed_chunks_follow_the_openai_format(client: TestClient) -> None:
    body = {**chat_body(), "stream": True, "stream_options": {"include_usage": True}}
    response = client.post("/v1/chat/completions", json=body)
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert {chunk["id"] for chunk in chunks} == {chunks[0]["id"]}
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0]["choices"] == [
        {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
    ]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks[1:-2]]
    assert all(len(delta) == 1 for delta in deltas)
    reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
    content = "".join(delta.get("content", "") for delta in deltas)
    assert (reasoning, content) == ("Let me think.", "\n\nThe answer is 42.")
    assert chunks[-2]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert chunks[-2]["usage"] is None
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0


def test_disabled_thinking_leaves_no_reasoning(
    client: TestClient, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scripted, "script", "Forty-two.")
    body = {**chat_body(), "chat_template_kwargs": {"enable_thinking": False}}
    message = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "Forty-two.", "reasoning_content": None}


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
    assert choice["message"]["content"] == "one two"
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
    assert [(model["id"], model["root"]) for model in models["data"]] == [
        ("tiny-model", str(tiny_model_dir))
    ]


def test_unsupported_and_invalid_requests_are_refused(client: TestClient) -> None:
    refused = client.post("/v1/chat/completions", json={**chat_body(), "n": 2})
    assert refused.status_code == 400 and refused.json()["error"]["param"] == "n"
    tools = [{"type": "function", "function": {"name": "f"}}]
    refused = client.post("/v1/chat/completions", json={**chat_body(), "tools": tools})
    assert refused.status_code == 400 and refused.json()["error"]["param"] == "tools"
    empty = client.post("/v1/chat/completions", json={"model": "tiny", "messages": []})
    assert empty.status_code == 422


def test_request_failures_surface_as_errors(
    client: TestClient, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    out_of_order = {
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": "late"}],
    }
    rejected = client.post("/v1/chat/completions", json=out_of_order)
    assert rejected.status_code == 400
    assert "System message must be at the beginning" in rejected.json()["error"]["message"]
    streamed = client.post("/v1/chat/completions", json={**out_of_order, "stream": True})
    events = sse_events(streamed.text)
    assert "System message" in json.loads(events[-2])["error"]["message"]
    assert events[-1] == "[DONE]"

    monkeypatch.setattr(scripted, "max_prompt_tokens", 4)
    too_long = client.post("/v1/chat/completions", json=chat_body())
    assert too_long.status_code == 400
    assert too_long.json()["error"]["code"] == "context_length_exceeded"
    assert client.post("/v1/chat/completions", json=chat_body("x")).status_code == 400


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


# (route, request body, status code, a fragment of the error body)
REFUSED_CASES: list[tuple[str, dict[str, JsonValue], int, str]] = [
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
    ("/v1/completions", {"model": "tiny", "prompt": "hi", "max_tokens": 0}, 422, "max_tokens"),
    ("/generate", {"prompt": "hi", "max_tokens": 0}, 400, "max_tokens must be at least 1, got 0"),
    ("/generate", {"prompt": [1, 2, 3], "max_tokens": 4}, 422, "prompt"),
]


@pytest.mark.parametrize(("route", "body", "status_code", "error"), REFUSED_CASES)
def test_raw_prompt_requests_are_refused(
    client: TestClient, route: str, body: dict[str, JsonValue], status_code: int, error: str
) -> None:
    response = client.post(route, json=body)
    assert response.status_code == status_code and error in response.text


def test_raw_prompt_failures_surface_as_errors(
    client: TestClient, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scripted, "max_prompt_tokens", 4)
    body = {"model": "tiny", "prompt": "a long prompt"}
    rejected = client.post("/v1/completions", json=body)
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "context_length_exceeded"
    events = sse_events(client.post("/v1/completions", json={**body, "stream": True}).text)
    assert json.loads(events[-2])["error"]["code"] == "context_length_exceeded"
    assert events[-1] == "[DONE]"
    generate_body = {"prompt": "a long prompt", "max_tokens": 4}
    events = sse_events(client.post("/generate", json=generate_body).text)
    assert events == [json.dumps({"error": "prompt is too long: 13 tokens"}), "[DONE]"]


def test_client_disconnect_aborts_the_request(
    tiny_model_dir: Path, scripted: type[ScriptedScheduler], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scripted, "script", "tick " * 2000)
    monkeypatch.setattr(scripted, "step_seconds", 0.005)
    app = create_app(ServerArgs(model_path=str(tiny_model_dir)))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + WAIT_SECONDS
    while not server.started:
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=WAIT_SECONDS)
        body = json.dumps({**chat_body(), "stream": True})
        headers = {"Content-Type": "application/json"}
        connection.request("POST", "/v1/chat/completions", body=body, headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        lines = [response.readline() for _ in range(10)]
        assert all(line.startswith(b"data: ") for line in lines[::2])
        connection.close()
        wait_for_abort(scripted.instances[0], 0)
    finally:
        server.should_exit = True
        thread.join()


def test_frontend_keeps_early_replies_and_aborts_on_early_close() -> None:
    frontend = FrontendManager(
        send_tokenizer=queue.Queue(), backend_threads=[threading.current_thread()]
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
        return texts, frontend.send_tokenizer.get_nowait()

    texts, sent = asyncio.run(scenario())
    assert texts == ["hi", ""]
    assert sent == AbortMsg(uid=1)
    assert frontend.ack_map == {}


def parse_in_chunks(parser: ThinkReasoningParser, chunks: list[str]) -> tuple[str, str]:
    reasoning: list[str] = []
    content: list[str] = []
    for chunk in chunks:
        reasoning_delta, content_delta = parser.parse_stream_chunk(chunk)
        reasoning.append(reasoning_delta)
        content.append(content_delta)
    reasoning_delta, content_delta = parser.flush()
    return "".join([*reasoning, reasoning_delta]), "".join([*content, content_delta])


def test_reasoning_parser_reassembles_split_markers() -> None:
    chunks = ["Let me", " th", "ink.</th", "ink>\n\nAns", "wer <th", "e end"]
    assert parse_in_chunks(ThinkReasoningParser(force_reasoning=True), chunks) == (
        "Let me think.",
        "\n\nAnswer <the end",
    )
    explicit = ["<thi", "nk>abc</think>", "def"]
    assert parse_in_chunks(ThinkReasoningParser(force_reasoning=False), explicit) == ("abc", "def")
    plain = ["no ", "block <", "th"]
    assert parse_in_chunks(ThinkReasoningParser(force_reasoning=False), plain) == (
        "",
        "no block <th",
    )
    truncated = ["still ", "thinking </thi"]
    assert parse_in_chunks(ThinkReasoningParser(force_reasoning=True), truncated) == (
        "still thinking </thi",
        "",
    )


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


@pytest.mark.skipif(MODEL_PATH is None, reason="FTMETAL_TEST_MODEL_PATH is not set")
def test_real_model_answers_over_http() -> None:
    assert MODEL_PATH is not None
    config = ServerArgs(
        model_path=MODEL_PATH, max_running_req=1, max_seq_len_override=4096, num_page_override=256
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
