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
class FinishesLoading(Protocol):
    """A model with work once its weights load: opening tables it reads from the checkpoint
    files as it runs instead of loading them, building projections of the loaded ones."""

    def finish_loading(self, model_path: str) -> int:
        """Returns the host bytes it holds outside MLX's allocator (a table's row cache)."""
        ...


@runtime_checkable
class ServesStepLookups(Protocol):
    """A model whose decode steps look rows up through a host service, as the MoE offload
    cache does: armed before a step is submitted, checked once its tokens land (a failed
    lookup computed with stand-in rows). A decode batch of more rows than one step's lookup
    takes looks them up on the host, so it is not captured."""

    @property
    def max_lookup_rows(self) -> int: ...

    def lookup_buffers(self) -> list[mx.array]:
        """The long-lived buffers a step's lookups bind (a CommandGraph's externals)."""
        ...

    def arm_lookups(self) -> None: ...

    def check_lookups(self) -> None: ...


@runtime_checkable
class SupportsMultimodal(Protocol):
    """What the engine asks of a model that serves images; its text forward takes its token
    embeddings from embed_input_ids."""

    def encode(self, item: MMItem) -> mx.array:
        """[item.num_tokens, hidden] soft tokens of one image."""
        ...


def embed_input_ids(embed_tokens: Embedding, batch: Batch, scale: float | None = None) -> mx.array:
    """Token embeddings of the batch, times scale when given; on a chunk with image rows,
    batch.mm_embeds' leading columns replace the rows batch.mm_rows, unscaled."""
    if batch.mm_embeds is None:
        x = embed_tokens.forward(batch.input_ids)
        return x if scale is None else x * scale
    else:
        assert batch.mm_rows is not None
        # Image rows carry content pad ids above the vocab: clamped for the lookup, the scatter
        # overwrites those rows.
        x = embed_tokens.forward(mx.minimum(batch.input_ids, embed_tokens.num_embeddings - 1))
        x = x if scale is None else x * scale
        x[batch.mm_rows] = batch.mm_embeds[:, : x.shape[1]].astype(x.dtype)
        return x


__all__ = [
    "BaseLLMModel",
    "FinishesLoading",
    "ServesStepLookups",
    "SupportsMultimodal",
    "embed_input_ids",
    "select_last_rows",
]
