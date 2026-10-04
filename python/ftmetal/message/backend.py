"""Messages into the scheduler: new requests, aborts and exit.

Port of FreeToken message/backend.py. Divergences: plain dataclasses passed over
queue.Queue (no ZMQ encoder/decoder); input ids are a host numpy int32 array; UserMsg
carries reuse_len, where the prompt's reusable prefix ends (FreeToken has none); no
multimodal items and no cache-rebuild message.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftmetal.core import SamplingParams


@dataclass(kw_only=True)
class BaseBackendMsg:
    pass


@dataclass(kw_only=True)
class BatchBackendMsg(BaseBackendMsg):
    data: list[BaseBackendMsg]


@dataclass(kw_only=True)
class ExitMsg(BaseBackendMsg):
    pass


@dataclass(kw_only=True)
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: np.ndarray
    sampling_params: SamplingParams
    # The prompt's tokens a next turn can share: a conversation without the chat template's
    # generation suffix, all of a raw prompt.
    reuse_len: int


@dataclass(kw_only=True)
class AbortBackendMsg(BaseBackendMsg):
    uid: int


__all__ = ["AbortBackendMsg", "BaseBackendMsg", "BatchBackendMsg", "ExitMsg", "UserMsg"]
