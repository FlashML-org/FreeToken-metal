"""QSA (Qwen Sparse Attention): full attention over the keys a block indexer selects, on
the paged K/V pool and the indexer's keys at the same token slots.

Port of FreeToken attention/qsa_sparse.py (QSASparseAttnBackend, QSAIndexerInputs). Each
query attends its index_topk_blocks best complete blocks of index_block_size keys (scored
by sum over the index heads of relu(q . k) / sqrt(index_head_dim) against each block's
pooled, normed and roped key) and its own incomplete block; while a query's complete blocks
fit index_topk_blocks it attends every key, which is plain causal attention. Divergences:
the backend is the paged full-attention backend plus the indexer (one subclass per kernel
set, as attention/full), on QSAKVCache's per-slot index keys (kvcache/qsa_pool.py), whose
block keys are written as each block's last token is stored (a block's earlier keys are
read back from their slots, so no per-request ring); the metadata carries each query's
block window of token slots; ties in the top-k go to the lower block; a prefill batch whose
requests all fit the budget runs the dense kernels.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx
import numpy as np

from ftmetal.attention.base import AttentionSpec
from ftmetal.attention.full.common import FullAttnBackend, FullAttnMetadata
from ftmetal.attention.metadata import AttnType
from ftmetal.core import Batch
from ftmetal.kvcache.base import BasePool
from ftmetal.kvcache.qsa_pool import QSAKVCache
from ftmetal.models.config import KVCacheGroupSpec, ModelConfig


class BlockKeyNorm(Protocol):
    def __call__(self, pooled: mx.array, positions: mx.array) -> mx.array:
        """Block keys [N, D] of the pooled raw keys [N, D] of blocks starting at positions [N]
        (int32 [3, N] under the 3-axis rope)."""
        ...


@dataclass(frozen=True, kw_only=True)
class QSAIndexInputs:
    # [T, index heads, D], normed and roped at the query positions.
    queries: mx.array
    # [T, D], the raw keys (no norm, no rope) the block keys pool.
    keys: mx.array
    block_key_norm: BlockKeyNorm


def select_blocks(scores: mx.array, topk: int) -> mx.array:
    """bool [rows, blocks]: each row's topk highest finite scores, ties to the lower block."""
    num_blocks = scores.shape[-1]
    if num_blocks <= topk:
        return scores > -mx.inf
    kth = mx.sort(scores, axis=-1)[:, num_blocks - topk : num_blocks - topk + 1]
    above = scores > kth
    tied = scores == kth
    needed = topk - mx.sum(above, axis=-1, keepdims=True)
    chosen = above | (tied & (mx.cumsum(tied.astype(mx.int32), axis=-1) <= needed))
    return chosen & (scores > -mx.inf)


def block_scores(queries: mx.array, block_keys: mx.array) -> mx.array:
    """fp32 [Q, N]: sum over the heads of relu(q . k) / sqrt(D) of queries [Q, heads, D] against
    block keys [N, D], head by head (no [Q, heads, N] product)."""
    keys = block_keys.astype(mx.float32).T
    scores = mx.zeros((queries.shape[0], block_keys.shape[0]), dtype=mx.float32)
    for head in range(queries.shape[1]):
        scores = scores + mx.maximum(queries[:, head].astype(mx.float32) @ keys, 0)
    return scores / math.sqrt(queries.shape[-1])


def visible_blocks(
    scores: mx.array, query_ends: mx.array, block_size: int, topk: int, num_blocks: int
) -> mx.array:
    """bool [Q, num_blocks]: the blocks each query ending at query_ends [Q] (its position + 1)
    attends, its own (incomplete) block included; scores [Q, N] cover the complete blocks."""
    complete = query_ends // block_size
    block_ids = mx.arange(num_blocks, dtype=mx.int32)[None]
    width = scores.shape[1]
    scores = mx.where(block_ids[:, :width] < complete[:, None], scores, -mx.inf)
    chosen = select_blocks(scores, topk)
    if width < num_blocks:
        chosen = mx.concatenate(
            [chosen, mx.zeros((chosen.shape[0], num_blocks - width), dtype=mx.bool_)], axis=1
        )
    own = block_ids <= ((query_ends - 1) // block_size)[:, None]
    tail = block_ids >= complete[:, None]
    is_sparse = (complete > topk)[:, None]
    return mx.where(is_sparse, chosen[:, :num_blocks] | tail, True) & own


class QSAAttnBackend(FullAttnBackend):
    def __init__(
        self,
        config: ModelConfig,
        spec: KVCacheGroupSpec,
        kvcache: BasePool,
        page_table: np.ndarray,
        page_size: int,
    ) -> None:
        super().__init__(config, spec, kvcache, page_table, page_size)
        assert isinstance(kvcache, QSAKVCache), "QSA reads the paged K/V and index-key pool"
        assert spec.attn_type is AttnType.QSA and spec.index_block_size > 0
        self.qsa_pool = kvcache
        self.block_size = spec.index_block_size
        self.topk_blocks = spec.index_topk_blocks

    def prepare_metadata(self, batch: Batch) -> None:
        super().prepare_metadata(batch)
        metadata = batch.attn_metadata[AttnType.QSA]
        assert isinstance(metadata, FullAttnMetadata)
        offsets = np.arange(1 - self.block_size, 1, dtype=np.int32)
        rows = []
        for req in batch.padded_reqs:
            positions = np.arange(req.cached_len, req.device_len, dtype=np.int32)
            window = np.maximum(positions[:, None] + offsets[None], 0)
            rows.append(self.page_table_host[req.table_idx][window])
        metadata.window_slots = mx.array(np.concatenate(rows).astype(np.int32))

    def forward(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> mx.array:
        raise NotImplementedError("QSA layers call qsa_forward with their index inputs")

    def qsa_forward(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        index: QSAIndexInputs,
        layer_id: int,
        batch: Batch,
    ) -> mx.array:
        """q [T, num_heads, D], k / v [T, num_kv_heads, D] -> [T, num_heads, D]: stores this
        step's keys and index keys, then attends each query's selected keys."""
        metadata = batch.attn_metadata[AttnType.QSA]
        assert isinstance(metadata, FullAttnMetadata) and metadata.window_slots is not None
        pool = self.qsa_pool
        pool.store_kv(k, v, batch.out_loc, layer_id)
        # Raw keys first: a block closing at a query pools the keys of its window's slots.
        pool.store_index_keys(index.keys, batch.out_loc, layer_id)
        pooled = self.window_mean(pool.index_cache(layer_id), metadata.window_slots)
        positions = batch.positions
        rope_positions = batch.get_attn_positions()
        if rope_positions.ndim == 1:
            starts = mx.maximum(positions - (self.block_size - 1), 0)
        elif metadata.block_starts is None:
            # A block's first token is its window's first slot, and an earlier forward may
            # have written it: the forward's tokens keep their 3-axis positions at their slots.
            pool.store_rope_positions(rope_positions, batch.out_loc)
            assert pool.rope_positions is not None
            starts = pool.rope_positions[metadata.window_slots[:, 0]].T
            metadata.block_starts = starts
        else:
            starts = metadata.block_starts
        closing = (positions % self.block_size == self.block_size - 1)[:, None]
        stored = mx.where(closing, index.block_key_norm(pooled, starts), index.keys)
        pool.store_index_keys(stored, batch.out_loc, layer_id)
        if metadata.max_seqlen_q == 1:
            return self.sparse_decode(q, index.queries, layer_id, metadata)
        elif all(
            seq_len // self.block_size <= self.topk_blocks for seq_len in metadata.seq_lens_host
        ):
            return self.attend(q, k, v, layer_id, batch, metadata)
        else:
            return self.sparse_prefill(q, index.queries, layer_id, metadata)

    def window_mean(self, index_cache: mx.array, window_slots: mx.array) -> mx.array:
        """[T, D]: the fp32 mean of the keys at each window's slots, in the keys' dtype."""
        windows = index_cache[window_slots].astype(mx.float32)
        return mx.mean(windows, axis=1).astype(index_cache.dtype)

    @abstractmethod
    def sparse_decode(
        self, q: mx.array, index_queries: mx.array, layer_id: int, metadata: FullAttnMetadata
    ) -> mx.array:
        """One query per request at its last key: attention over its selected keys."""

    @abstractmethod
    def sparse_prefill(
        self, q: mx.array, index_queries: mx.array, layer_id: int, metadata: FullAttnMetadata
    ) -> mx.array:
        """A prefill's queries, some past the budget: attention over each one's selected keys."""


__all__ = [
    "BlockKeyNorm",
    "QSAAttnBackend",
    "QSAIndexInputs",
    "block_scores",
    "select_blocks",
    "visible_blocks",
]
