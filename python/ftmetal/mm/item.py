"""One image of a request (MMItem) and a chunk's gather of its encoder rows (MMGatherRow).

Port of the MMItem of FreeToken message/backend.py. Divergences: its own module, which the
request records (core.py) and the messages import alike; the feature is a numpy array, a
bf16 one held as uint16 bits (numpy has no bf16); the family extras are typed fields instead of
model_specific_data; offsets are tuples; no precomputed_embeddings; a gather plan row is an
MMGatherRow, not a tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ftmetal.mm.config import Modality


@dataclass(kw_only=True)
class MMItem:
    """One image of a request: its preprocessed pixels and the input_ids spans its
    embeddings take, half-open [start, end) (an item may occupy several)."""

    modality: Modality
    hash: int
    pad_value: int
    offsets: list[tuple[int, int]]
    # The processor's [patches, patch_dim] pixels, Qwen VL's bf16 as uint16 bits and Gemma4's
    # float32 (FreeToken's dtypes); None once this request will not encode them.
    feature: np.ndarray | None
    # Qwen VL: the (t, h, w) patch grid before the spatial merge.
    grid_thw: tuple[int, int, int] | None = None
    # Gemma4: each patch's int32 (x, y), [patches, 2], and the soft tokens they pool into.
    position_ids: np.ndarray | None = None
    num_soft_tokens: int | None = None

    @property
    def num_tokens(self) -> int:
        return sum(end - start for start, end in self.offsets)

    def validate(self) -> None:
        if not self.offsets or any(end <= start for start, end in self.offsets):
            raise ValueError(f"MMItem offsets must be non-empty half-open spans: {self.offsets}")


class MMGatherRow(NamedTuple):
    """Encoder rows [row_lo, row_hi) of an image that land at a chunk position."""

    uid: int
    item_hash: int
    row_lo: int
    row_hi: int
    num_tokens: int
    position: int


__all__ = ["MMGatherRow", "MMItem"]
