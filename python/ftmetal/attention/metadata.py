"""Per-step attention metadata records the batch carries.

Split out of FreeToken attention/base.py and attention/linear.py (BaseAttnMetadata,
FLAMetadata) so core.py can type the batch without importing the backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import mlx.core as mx


class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> mx.array:
        """Row of each request's last query token in the flattened batch."""


@dataclass
class LinearAttnMetadata:
    # Query index pointer over the flattened batch, int32 [size + 1].
    cu_seqlens: mx.array
    # Live GDN state slot per request, int32 [size].
    cache_indices: mx.array
    # Host copies for loops that pick per-request token ranges.
    seq_lens_host: list[int]
    slots_host: list[int]


__all__ = ["BaseAttnMetadata", "LinearAttnMetadata"]
