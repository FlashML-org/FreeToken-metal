"""Messages into the tokenizer worker: prompts to encode, sampled tokens to decode,
admissions, aborts and scheduler errors.

Port of FreeToken message/tokenizer.py. Divergences: plain dataclasses passed over
queue.Queue (no ZMQ encoder/decoder); stop strings are matched by the detokenizer, so
DetokenizeMsg carries neither matched_stop nor stop_strs; no pool-usage snapshot,
multimodal input or cache-rebuild messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Required, TypedDict

from pydantic import JsonValue

from ftmetal.core import SamplingParams

FinishReason = Literal["stop", "length"]


class ChatMessage(TypedDict, total=False):
    """One conversation turn as the chat template reads it."""

    role: Required[Literal["system", "user", "assistant"]]
    content: Required[str]
    reasoning_content: str


@dataclass(kw_only=True)
class BaseTokenizerMsg:
    pass


@dataclass(kw_only=True)
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: list[BaseTokenizerMsg]


@dataclass(kw_only=True)
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool
    finish_reason: FinishReason | None = None


@dataclass(kw_only=True)
class PromptAdmittedMsg(BaseTokenizerMsg):
    """Sent once per request, after its first prefill batch is prepared."""

    uid: int
    prompt_tokens: int
    # The prefix-cache hit: prompt tokens served from the cache instead of recomputed.
    cached_tokens: int = 0


@dataclass(kw_only=True)
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    # A raw prompt, or a conversation for the chat template.
    text: str | list[ChatMessage]
    sampling_params: SamplingParams
    chat_template_kwargs: dict[str, JsonValue] = field(default_factory=dict)
    tools: list[dict[str, JsonValue]] | None = None


@dataclass(kw_only=True)
class AbortMsg(BaseTokenizerMsg):
    uid: int


@dataclass(kw_only=True)
class ErrorReplyMsg(BaseTokenizerMsg):
    """A request the scheduler cannot serve, or the acknowledgement of an abort."""

    uid: int
    error: str
    # Machine-readable class surfaced as the OpenAI error code, e.g. context_length_exceeded.
    code: str | None = None


__all__ = [
    "AbortMsg",
    "BaseTokenizerMsg",
    "BatchTokenizerMsg",
    "ChatMessage",
    "DetokenizeMsg",
    "ErrorReplyMsg",
    "FinishReason",
    "PromptAdmittedMsg",
    "TokenizeMsg",
]
