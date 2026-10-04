"""API-agnostic generation core: submit a conversation, stream its semantic events.

Port of FreeToken server/generation.py (GenSpec, the events, resolve_sampling,
submit_generation, generate_events, generate_full) and of api_server.py's
FrontendManager, kept here so every protocol adapter imports it without a cycle.
Divergences: the tokenizer worker is a thread of this process, so replies reach the
event loop through call_soon_threadsafe and wait per request on an asyncio.Queue;
closing a request's event stream before its terminal reply aborts it; generate_full
folds the streamed events; stream_generate reports a failed request as an
{"error": ...} event before [DONE] instead of ending the stream silently; no tool
calls, images, keepalives or request log.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field

from pydantic import JsonValue

from ftmetal.core import SamplingParams
from ftmetal.message.frontend import BaseFrontendMsg, BatchFrontendMsg, UserReply
from ftmetal.message.tokenizer import (
    AbortMsg,
    BaseTokenizerMsg,
    ChatMessage,
    FinishReason,
    TokenizeMsg,
)
from ftmetal.server.args import ServerArgs
from ftmetal.server.reasoning_parser import ReasoningParser, build_reasoning_parser
from ftmetal.tokenizer import resolve_thinking_mode


class EngineUnavailableError(RuntimeError):
    """The scheduler or tokenizer thread is not running, so no request can be served."""


class GenerationError(ValueError):
    """A request failed before producing output, e.g. a prompt longer than the context."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


@dataclass(kw_only=True)
class FrontendManager:
    config: ServerArgs
    # The tokenizer worker's inbox; None stops it.
    send_tokenizer: queue.Queue[BaseTokenizerMsg | None]
    # Set by the lifespan once the scheduler thread is up.
    loop: asyncio.AbstractEventLoop | None = None
    backend_threads: list[threading.Thread] = field(default_factory=list)
    uid_counter: int = 0
    ack_map: dict[int, asyncio.Queue[UserReply]] = field(default_factory=dict)

    def new_user(self) -> int:
        if not self.backend_threads or not all(
            thread.is_alive() for thread in self.backend_threads
        ):
            raise EngineUnavailableError("the engine is not running")
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = asyncio.Queue()
        return uid

    def send_one(self, msg: BaseTokenizerMsg) -> None:
        self.send_tokenizer.put(msg)

    def post(self, msg: BaseFrontendMsg) -> None:
        """Hand replies to the event loop; called from the tokenizer thread."""
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.deliver, msg)

    def deliver(self, msg: BaseFrontendMsg) -> None:
        replies = msg.data if isinstance(msg, BatchFrontendMsg) else [msg]
        for reply in replies:
            assert isinstance(reply, UserReply)
            # A request whose stream closed has left the map; its late replies drop.
            pending = self.ack_map.get(reply.uid)
            if pending is not None:
                pending.put_nowait(reply)

    async def wait_for_ack(self, uid: int) -> AsyncGenerator[UserReply]:
        """uid's replies up to the terminal one; closing the stream earlier aborts uid."""
        pending = self.ack_map[uid]
        finished = False
        try:
            while not finished:
                reply = await pending.get()
                finished = reply.finished
                yield reply
        finally:
            self.ack_map.pop(uid, None)
            if not finished:
                self.abort_user(uid)

    async def stream_generate(self, uid: int) -> AsyncGenerator[bytes]:
        """uid's raw text as {"text": ...} server-sent events, then [DONE]."""
        async with aclosing(self.wait_for_ack(uid)) as replies:
            async for reply in replies:
                if reply.error is not None:
                    event = json.dumps({"error": reply.error}, ensure_ascii=False)
                    yield f"data: {event}\n\n".encode()
                elif reply.incremental_output:
                    event = json.dumps({"text": reply.incremental_output}, ensure_ascii=False)
                    yield f"data: {event}\n\n".encode()
        yield b"data: [DONE]\n\n"

    def abort_user(self, uid: int) -> None:
        self.ack_map.pop(uid, None)
        self.send_one(AbortMsg(uid=uid))


@dataclass(kw_only=True)
class GenSpec:
    """What to generate, whatever protocol asked for it."""

    messages: list[ChatMessage]
    sampling_params: SamplingParams
    chat_template_kwargs: dict[str, JsonValue] = field(default_factory=dict)


@dataclass
class ReasoningDelta:
    text: str


@dataclass
class ContentDelta:
    text: str


@dataclass
class GenDone:
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int
    matched_stop: str | None = None


GenEvent = ReasoningDelta | ContentDelta | GenDone


@dataclass
class GenResult:
    reasoning: str
    content: str
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int
    matched_stop: str | None = None


def resolve_sampling(
    *,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    max_tokens: int | None,
    ignore_eos: bool,
    stop: str | list[str] | None,
    model_sampling: SamplingParams,
    default_max_tokens: int,
) -> SamplingParams:
    """A request's sampling fields; the ones it omits come from the checkpoint's defaults."""
    stop_strs = [stop] if isinstance(stop, str) else list(stop or [])
    return SamplingParams(
        temperature=model_sampling.temperature if temperature is None else temperature,
        top_k=model_sampling.top_k if top_k is None else top_k,
        top_p=model_sampling.top_p if top_p is None else top_p,
        ignore_eos=ignore_eos,
        max_tokens=default_max_tokens if max_tokens is None else max_tokens,
        # An empty stop string would match at once.
        stop_strs=[stop_str for stop_str in stop_strs if stop_str],
    )


def submit_generation(spec: GenSpec, frontend: FrontendManager) -> int:
    uid = frontend.new_user()
    frontend.send_one(
        TokenizeMsg(
            uid=uid,
            text=spec.messages,
            sampling_params=spec.sampling_params,
            chat_template_kwargs=spec.chat_template_kwargs,
        )
    )
    return uid


def _make_reasoning_parser(spec: GenSpec, state: FrontendManager) -> ReasoningParser | None:
    """Build a reasoning parser for this generation, or None if the server has no
    reasoning parser configured. ``force_reasoning`` matches the encode-side
    thinking mode so chat-mode content is never mislabeled as reasoning."""
    parser_name = state.config.reasoning_parser
    if parser_name == "qwen3":
        # The qwen3 chat template opens an implicit <think> (thinking on) unless
        # enable_thinking is explicitly false, so the model emits only the closing
        # </think>. Mirror that default here, else the chain-of-thought leaks into content.
        force_reasoning = (spec.chat_template_kwargs or {}).get("enable_thinking") is not False
    elif parser_name == "glm":
        # GLM's template honors enable_thinking (default on) even with tools; the
        # generic fallback would force thinking and mislabel disabled output as reasoning.
        force_reasoning = (spec.chat_template_kwargs or {}).get("enable_thinking") is not False
    elif parser_name == "gemma4":
        # Gemma4 defaults thinking off even when tools are present: its template injects an
        # empty thought channel before generation. Do not let Codex tool definitions make all
        # visible text look like hidden reasoning.
        ctk = spec.chat_template_kwargs or {}
        force_reasoning = (
            ctk.get("thinking_mode") == "thinking"
            or bool(ctk.get("enable_thinking"))
            or bool(ctk.get("thinking"))
        )
    elif parser_name == "minimax_m3":
        # M3's template pre-opens <mm:think> only in thinking_mode "enabled" (the
        # model then emits just the closing tag); "adaptive" (default) leaves the
        # model to open the tag itself and "disabled" pre-closes it.
        force_reasoning = (spec.chat_template_kwargs or {}).get("thinking_mode") == "enabled"
    else:
        force_reasoning = resolve_thinking_mode(spec.chat_template_kwargs, None) == "thinking"
    return build_reasoning_parser(state.config, force_reasoning)


async def generate_events(
    uid: int, spec: GenSpec, frontend: FrontendManager
) -> AsyncGenerator[GenEvent]:
    """Reasoning and content deltas of one request, then exactly one GenDone."""
    parser = _make_reasoning_parser(spec, frontend)
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason: FinishReason | None = None
    matched_stop: str | None = None
    async with aclosing(frontend.wait_for_ack(uid)) as replies:
        async for reply in replies:
            if reply.error is not None:
                raise GenerationError(reply.error, reply.error_code)
            prompt_tokens += reply.prompt_tokens_delta
            completion_tokens += reply.completion_tokens_delta
            finish_reason = reply.finish_reason
            matched_stop = reply.matched_stop
            if not reply.incremental_output:
                continue
            elif parser is None:
                yield ContentDelta(reply.incremental_output)
            else:
                reasoning, content = parser.parse_stream_chunk(reply.incremental_output)
                if reasoning:
                    yield ReasoningDelta(reasoning)
                if content:
                    yield ContentDelta(content)
    if parser is not None:
        reasoning, content = parser.flush()
        if reasoning:
            yield ReasoningDelta(reasoning)
        if content:
            yield ContentDelta(content)
    assert finish_reason is not None
    yield GenDone(finish_reason, prompt_tokens, completion_tokens, matched_stop)


async def generate_full(uid: int, spec: GenSpec, frontend: FrontendManager) -> GenResult:
    """The whole completion; whitespace around the reasoning block is trimmed."""
    reasoning: list[str] = []
    content: list[str] = []
    done: GenDone | None = None
    async with aclosing(generate_events(uid, spec, frontend)) as events:
        async for event in events:
            if isinstance(event, ReasoningDelta):
                reasoning.append(event.text)
            elif isinstance(event, ContentDelta):
                content.append(event.text)
            else:
                done = event
    assert done is not None
    return GenResult(
        reasoning="".join(reasoning).strip(),
        content="".join(content).strip(),
        finish_reason=done.finish_reason,
        prompt_tokens=done.prompt_tokens,
        completion_tokens=done.completion_tokens,
        matched_stop=done.matched_stop,
    )


__all__ = [
    "ContentDelta",
    "EngineUnavailableError",
    "FrontendManager",
    "GenDone",
    "GenEvent",
    "GenResult",
    "GenSpec",
    "GenerationError",
    "ReasoningDelta",
    "generate_events",
    "generate_full",
    "resolve_sampling",
    "submit_generation",
]
