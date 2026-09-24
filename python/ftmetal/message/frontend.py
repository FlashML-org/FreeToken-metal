"""Replies to the API server: incremental text, usage deltas and the terminal state.

Port of FreeToken message/frontend.py. Divergences: plain dataclasses handed to the
event loop (no ZMQ encoder/decoder); no cached-token count, pool-usage snapshot or
cache-rebuild reply.
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
    # Set, with finished, when the request failed before producing output.
    error: str | None = None
    error_code: str | None = None
    finish_reason: FinishReason | None = None
    # The stop string that ended generation; it and everything after it are not emitted.
    matched_stop: str | None = None


__all__ = ["BaseFrontendMsg", "BatchFrontendMsg", "UserReply"]
