"""KV pool interface.

Port of FreeToken kvcache/base.py (BaseKVCachePool). Divergences: pools hold one
mx.array per storage layer and expose arrays() so the engine can eval them at
startup and at every step; the prefix-cache interfaces arrive with the radix cache
in round 2; runtime rebuild is dropped.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import mlx.core as mx

from ftmetal.models.config import ModelConfig

logger = logging.getLogger(__name__)


class BaseKVCachePool(ABC):
    @classmethod
    @abstractmethod
    def kv_cost(cls, model_config: ModelConfig, page_size: int, dtype: mx.Dtype) -> tuple[int, int]:
        """(bytes per page, fixed bytes) of this pool family's own buffers."""

    @classmethod
    def solve_num_pages(
        cls,
        model_config: ModelConfig,
        page_size: int,
        dtype: mx.Dtype,
        available_memory: int,
        num_page_override: int | None = None,
    ) -> int:
        """Largest usable page count fitting available_memory bytes."""
        cache_per_page, fixed_cache_size = cls.kv_cost(model_config, page_size, dtype)
        num_pages = num_page_override
        if num_pages is None:
            num_pages = (available_memory - fixed_cache_size) // cache_per_page
        if num_pages <= 1:
            raise RuntimeError("not enough memory for the KV cache; lower --max-seq-len")
        logger.info(
            f"KV cache: {num_pages} pages x {page_size} tokens, "
            f"{(num_pages * cache_per_page + fixed_cache_size) / 2**30:.2f} GiB"
        )
        return num_pages

    @abstractmethod
    def k_cache(self, layer_id: int) -> mx.array: ...

    @abstractmethod
    def v_cache(self, layer_id: int) -> mx.array: ...

    @abstractmethod
    def store_kv(self, k: mx.array, v: mx.array, out_loc: mx.array, layer_id: int) -> None: ...

    @abstractmethod
    def arrays(self) -> list[mx.array]:
        """Every pool array, for eval."""

    @property
    @abstractmethod
    def dtype(self) -> mx.Dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


__all__ = ["BaseKVCachePool"]
