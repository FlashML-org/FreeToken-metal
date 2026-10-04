"""Replies to the API server: incremental text, usage deltas, the pool occupancy and the
terminal state.

Port of FreeToken message/frontend.py. Divergences: plain dataclasses handed to the
event loop (no ZMQ encoder/decoder); no GPU-memory figure or cache-rebuild reply.
"""

from __future__ import annotations

from dataclasses import dataclass

from ftmetal.message.tokenizer import FinishReason


@dataclass(kw_only=True)
class BaseFrontendMsg:
    pass


@dataclass(kw_only=True)
class BatchFrontendMsg(BaseFrontendMsg):
    data: list[BaseFrontendMsg]


@dataclass(kw_only=True)
class UserReply(BaseFrontendMsg):
    uid: int
    incremental_output: str
    finished: bool
    prompt_tokens_delta: int = 0
    completion_tokens_delta: int = 0
    # The prefix-cache hit, reported with prompt_tokens_delta.
    cached_tokens: int = 0
    # Pool occupancy after the step that sampled this token (not-evictable used, total);
    # 0/0 when not reported or the model has no such pool.
    kv_used_pages: int = 0
    kv_total_pages: int = 0
    mamba_used_slots: int = 0
    mamba_total_slots: int = 0
    swa_used_tokens: int = 0
    swa_total_tokens: int = 0
    # Set, with finished, when the request failed before producing output.
    error: str | None = None
    error_code: str | None = None
    finish_reason: FinishReason | None = None
    # The stop string that ended generation; it and everything after it are not emitted.
    matched_stop: str | None = None


__all__ = ["BaseFrontendMsg", "BatchFrontendMsg", "UserReply"]
