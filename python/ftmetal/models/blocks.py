"""Model base class.

Port of FreeToken models/blocks.py (BaseLLMModel). Divergence: forward takes the
Batch instead of reading it from the global Context.
"""

from __future__ import annotations

from abc import abstractmethod

import mlx.core as mx

from ftmetal.core import Batch
from ftmetal.layers.base import BaseOP


class BaseLLMModel(BaseOP):
    @abstractmethod
    def forward(self, batch: Batch) -> mx.array:
        """Logits [batch.size, vocab] for the last token of every request."""


def select_last_rows(x: mx.array, batch: Batch) -> mx.array:
    """The hidden row of each request's last token; decode rows are already one per request."""
    if batch.is_decode:
        return x
    return x[batch.attn_metadata.get_last_indices(batch.size)]


__all__ = ["BaseLLMModel", "select_last_rows"]
