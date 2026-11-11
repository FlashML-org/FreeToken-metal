"""Messages into the scheduler: new requests, aborts and exit.

Port of FreeToken message/backend.py. Divergences: plain dataclasses passed over
queue.Queue (no ZMQ encoder/decoder); input ids are a host numpy int32 array; UserMsg
carries reuse_len, where the prompt's reusable prefix ends (FreeToken has none), in the
coordinates of the expanded image spans; MMItem lives in mm/item.py, which core.py imports
too; mrope_positions is an int32 numpy array; no cache-rebuild message.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftmetal.core import SamplingParams
from ftmetal.mm.item import MMItem


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
    # Per-image processor outputs, in prompt order.
    mm_items: list[MMItem] | None = None
    # Precomputed int32 [3, len(input_ids)] mrope positions and the decode delta; None for
    # text-only requests and 1-D rope models.
    mrope_positions: np.ndarray | None = None
    mrope_delta: int = 0


@dataclass(kw_only=True)
class AbortBackendMsg(BaseBackendMsg):
    uid: int


@dataclass(kw_only=True)
class FlushCacheBackendMsg(BaseBackendMsg):
    """Drop every unlocked prefix-cache entry once the scheduler is idle, waiting up to
    timeout_s for it."""

    request_id: str
    timeout_s: float = 0.0


__all__ = [
    "AbortBackendMsg",
    "BaseBackendMsg",
    "BatchBackendMsg",
    "ExitMsg",
    "FlushCacheBackendMsg",
    "UserMsg",
]
