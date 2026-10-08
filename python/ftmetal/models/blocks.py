"""Model base class, the multimodal model protocol and the embedding merge.

Port of FreeToken models/blocks.py (BaseLLMModel, SupportsMultimodal, embed_input_ids).
Divergences: forward takes the Batch instead of reading it from the global Context;
SupportsMultimodal has no place_encoder_weights (the tower stays resident).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import mlx.core as mx

from ftmetal.core import Batch
from ftmetal.layers.base import BaseOP
from ftmetal.layers.embedding import Embedding
from ftmetal.mm.item import MMItem


class BaseLLMModel(BaseOP):
    @abstractmethod
    def forward(self, batch: Batch) -> mx.array:
        """Logits [batch.padded_size, vocab] at the last token of each of batch.padded_reqs."""


def select_last_rows(x: mx.array, batch: Batch) -> mx.array:
    """The hidden row of each request's last token; decode rows are already one per request."""
    if batch.is_decode:
        return x
    metadata = next(iter(batch.attn_metadata.values()))
    return x[metadata.get_last_indices(batch.size)]


@runtime_checkable
class SupportsMultimodal(Protocol):
    """What the engine asks of a model that serves images; its text forward takes its token
    embeddings from embed_input_ids."""

    def encode(self, item: MMItem) -> mx.array:
        """[item.num_tokens, hidden] soft tokens of one image."""
        ...


def embed_input_ids(embed_tokens: Embedding, batch: Batch) -> mx.array:
    """Token embeddings of the batch; on a chunk with image rows, batch.mm_embeds' leading
    columns replace the rows batch.mm_rows."""
    if batch.mm_embeds is None:
        return embed_tokens.forward(batch.input_ids)
    else:
        assert batch.mm_rows is not None
        # Image rows carry content pad ids above the vocab: clamped for the lookup, the scatter
        # overwrites those rows.
        x = embed_tokens.forward(mx.minimum(batch.input_ids, embed_tokens.num_embeddings - 1))
        x[batch.mm_rows] = batch.mm_embeds[:, : x.shape[1]].astype(x.dtype)
        return x


__all__ = ["BaseLLMModel", "SupportsMultimodal", "embed_input_ids", "select_last_rows"]
