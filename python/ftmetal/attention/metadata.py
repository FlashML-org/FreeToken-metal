"""Per-step attention metadata records the batch carries.

Split out of FreeToken attention/base.py and attention/linear.py (AttnType,
BaseAttnMetadata, FLAMetadata) so core.py can type the batch without importing the
backends. The batch keys its metadata by AttnType, one record per backend. Divergence:
a record names its step_inputs, the arrays a CommandGraph replay rebinds (FreeToken
copies them into a backend's static capture buffers instead).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import mlx.core as mx


class AttnType(str, Enum):
    FULL = "full"
    LINEAR = "linear"


class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> mx.array:
        """Row of each request's last query token in the flattened batch."""

    @abstractmethod
    def step_inputs(self) -> list[mx.array]:
        """The device arrays a replay of a captured decode step rebinds, in capture order."""


@dataclass
class LinearAttnMetadata(BaseAttnMetadata):
    # Query index pointer over the flattened batch, int32 [size + 1].
    cu_seqlens: mx.array
    # Live GDN state slot per request, int32 [size].
    cache_indices: mx.array
    # Prefill: the slot taking each request's snapshot (its next ping-pong slot), or -1 when
    # this chunk tracks none, int32 [size].
    track_slots: mx.array
    # Host copies for loops that pick per-request token ranges.
    seq_lens_host: list[int]
    slots_host: list[int]
    track_slots_host: list[int]
    # The snapshot's token count per request (its mamba_last_track_seqlen); 0 when untracked.
    track_seqlens_host: list[int]

    def get_last_indices(self, bs: int) -> mx.array:
        return self.cu_seqlens[1 : 1 + bs] - 1

    def step_inputs(self) -> list[mx.array]:
        return [self.cu_seqlens, self.cache_indices, self.track_slots]


__all__ = ["AttnType", "BaseAttnMetadata", "LinearAttnMetadata"]
