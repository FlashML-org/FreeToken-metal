"""Paged K/V pool for QSA layers, with the block indexer's keys at the same token slots.

Port of FreeToken kvcache/qsa_pool.py (QSAKVCache). Divergences: the indexer's state is
one key per token slot per layer [num_tokens, index_head_dim] instead of a compressed
slab of one row per block (a 1/block_size shadow of the pages, which needs pages of whole
blocks) and a per-request ring of the open block's raw keys: the slot of a block's last
token holds the block's pooled, normed and roped key, every other slot its token's raw
index key, so a block closing in a later forward reads its earlier keys from their slots,
any page size works and a prefix's pages carry its index keys with them. Under the 3-axis
rope, every token's rope position is kept at its slot too (FreeToken's rope_positions).
"""

from __future__ import annotations

from typing import Self

import mlx.core as mx

from ftmetal.kernel.kvcache.pool_rows import scatter_rows
from ftmetal.kvcache.base import PoolConfig
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.models.config import KVCacheGroupSpec


class QSAKVCache(MHAKVCache):
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        index_head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: mx.Dtype,
        layer_ids: tuple[int, ...] | None = None,
        has_rope_positions: bool = False,
    ) -> None:
        super().__init__(num_kv_heads, num_layers, head_dim, num_pages, page_size, dtype, layer_ids)
        self.index_keys = [
            mx.zeros((self.num_tokens, index_head_dim), dtype=dtype) for _ in self.layer_ids
        ]
        # [num_tokens, 3] int32, the (t, h, w) rope position at each token slot.
        self.rope_positions = (
            mx.zeros((self.num_tokens, 3), dtype=mx.int32) if has_rope_positions else None
        )
        mx.eval(*self.index_keys, *self.rope_arrays())

    @classmethod
    def from_spec(cls, spec: KVCacheGroupSpec, config: PoolConfig, num_pages: int) -> Self:
        return cls(
            num_kv_heads=spec.num_kv_heads,
            num_layers=spec.num_layers,
            head_dim=spec.head_dim,
            index_head_dim=spec.index_head_dim,
            # + 1: the dummy page padded page-table entries point at.
            num_pages=num_pages + 1,
            page_size=config.page_size,
            dtype=config.dtype,
            layer_ids=spec.layer_ids,
            has_rope_positions=spec.has_rope_positions,
        )

    def index_cache(self, layer_id: int) -> mx.array:
        return self.index_keys[self.dense(layer_id)]

    def store_index_keys(self, keys: mx.array, out_loc: mx.array, layer_id: int) -> None:
        dense = self.dense(layer_id)
        self.index_keys[dense] = scatter_rows(self.index_keys[dense], keys, out_loc)

    def store_rope_positions(self, positions: mx.array, out_loc: mx.array) -> None:
        """positions int32 [3, T] of the tokens at out_loc."""
        assert self.rope_positions is not None
        self.rope_positions = scatter_rows(self.rope_positions, positions.T, out_loc)

    def rope_arrays(self) -> list[mx.array]:
        return [] if self.rope_positions is None else [self.rope_positions]

    def arrays(self) -> list[mx.array]:
        return [*self.k, *self.v, *self.index_keys, *self.rope_arrays()]

    def set_arrays(self, arrays: list[mx.array]) -> None:
        count = len(self.k)
        assert len(arrays) == 3 * count + len(self.rope_arrays())
        self.k, self.v, self.index_keys = (
            arrays[:count],
            arrays[count : 2 * count],
            arrays[2 * count : 3 * count],
        )
        self.rope_positions = None if self.rope_positions is None else arrays[3 * count]


__all__ = ["QSAKVCache"]
