"""Paged K/V pool for full-attention layers.

Port of FreeToken kvcache/mha_pool.py (MHAKVCache). Divergences: one [num_tokens,
num_kv_heads, head_dim] array per storage layer for K and for V (the page size is
folded into the token axis, one buffer per layer); store_kv is the in-place
kv_write primitive and the pool is rebound to its result; kv_cost and from_spec take
the FULL group's spec.
"""

from __future__ import annotations

from typing import Self

import mlx.core as mx

from ftmetal.kernel.kvcache.pool_rows import kv_write
from ftmetal.kvcache.base import BaseKVCachePool, PoolConfig, PoolCost
from ftmetal.models.config import KVCacheGroupSpec


class MHAKVCache(BaseKVCachePool):
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: mx.Dtype,
        layer_ids: tuple[int, ...] | None = None,
    ) -> None:
        self._num_layers = num_layers
        self._dtype = dtype
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_tokens = num_pages * page_size
        if layer_ids is None:
            layer_ids = tuple(range(num_layers))
        self.layer_ids = layer_ids
        self.layer_map = {layer_id: dense for dense, layer_id in enumerate(layer_ids)}
        shape = (self.num_tokens, num_kv_heads, head_dim)
        self.k = [mx.zeros(shape, dtype=dtype) for _ in layer_ids]
        self.v = [mx.zeros(shape, dtype=dtype) for _ in layer_ids]
        mx.eval(*self.k, *self.v)

    @classmethod
    def kv_cost(cls, spec: KVCacheGroupSpec, config: PoolConfig) -> PoolCost:
        return PoolCost(spec.token_bytes(config.dtype) * config.page_size, 0)

    @classmethod
    def from_spec(cls, spec: KVCacheGroupSpec, config: PoolConfig, num_pages: int) -> Self:
        return cls(
            num_kv_heads=spec.num_kv_heads,
            num_layers=spec.num_layers,
            head_dim=spec.head_dim,
            # + 1: the dummy page padded page-table entries point at.
            num_pages=num_pages + 1,
            page_size=config.page_size,
            dtype=config.dtype,
            layer_ids=spec.layer_ids,
        )

    def dense(self, layer_id: int) -> int:
        return self.layer_map[layer_id]

    def k_cache(self, layer_id: int) -> mx.array:
        return self.k[self.dense(layer_id)]

    def v_cache(self, layer_id: int) -> mx.array:
        return self.v[self.dense(layer_id)]

    def store_kv(self, k: mx.array, v: mx.array, out_loc: mx.array, layer_id: int) -> None:
        dense = self.dense(layer_id)
        self.k[dense], self.v[dense] = kv_write(self.k[dense], self.v[dense], k, v, out_loc)

    def arrays(self) -> list[mx.array]:
        return [*self.k, *self.v]

    def set_arrays(self, arrays: list[mx.array]) -> None:
        count = len(self.k)
        assert len(arrays) == 2 * count
        self.k, self.v = arrays[:count], arrays[count:]

    @property
    def dtype(self) -> mx.Dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers


__all__ = ["MHAKVCache"]
