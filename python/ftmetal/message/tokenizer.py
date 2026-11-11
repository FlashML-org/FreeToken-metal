"""Messages into the tokenizer worker: prompts to encode, sampled tokens to decode,
admissions, aborts and scheduler errors.

Port of FreeToken message/tokenizer.py. Divergences: plain dataclasses passed over
queue.Queue (no ZMQ encoder/decoder); stop strings are matched by the detokenizer, so
DetokenizeMsg carries neither matched_stop nor stop_strs, and carries the timings snapshot
whenever the request has stop strings (the detokenizer can end it on any token); no GPU-memory
figure or cache-rebuild messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import JsonValue

from ftmetal.core import SamplingParams

FinishReason = Literal["stop", "length"]


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
    # Pool occupancy after this step (not-evictable used, total), passed through to the
    # frontend's /v1/stats; 0/0 for a pool the model does not have.
    kv_used_pages: int = 0
    kv_total_pages: int = 0
    mamba_used_slots: int = 0
    mamba_total_slots: int = 0
    swa_used_tokens: int = 0
    swa_total_tokens: int = 0
    # The scheduler's ReqStats.finish snapshot: on the terminal reply, and on every reply of
    # a timings_per_token request or one with stop strings; None otherwise.
    timings: dict[str, float | int] | None = None


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
    # A raw prompt, or a conversation for the chat template: OpenAI-shaped message dicts.
    text: str | list[dict[str, JsonValue]]
    sampling_params: SamplingParams
    chat_template_kwargs: dict[str, JsonValue] = field(default_factory=dict)
    tools: list[dict[str, JsonValue]] | None = None
    # The raw bytes of the request's images, in prompt order.
    images: list[bytes] | None = None
    # How system messages after the conversation's first turn are rendered (auto, preserve
    # or fold); None leaves them to the chat template.
    inline_system_policy: str | None = None


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
    "DetokenizeMsg",
    "ErrorReplyMsg",
    "FinishReason",
    "PromptAdmittedMsg",
    "TokenizeMsg",
]
