"""Pool interfaces: one pool per attention type, sized from its KVCacheGroupSpec.

Port of FreeToken kvcache/base.py (BaseKVCachePool). Divergences: BasePool is the base
of every pool family, the GDN state pool included; kv_cost prices one spec as (bytes per
page, fixed bytes) and from_spec builds the pool from it; solve_num_pages is the engine
budget's (engine/cache_budget.py); pools hold one mx.array per storage layer and expose
arrays() so the engine can eval them at startup and at every step, and set_arrays() so a
graph replay can rebind them; the prefix-cache
types live with the radix caches and the CacheManager; runtime rebuild is dropped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, NamedTuple, Protocol, Self

import mlx.core as mx

from ftmetal.models.config import KVCacheGroupSpec


class PoolConfig(Protocol):
    """The EngineConfig fields pools size themselves from; a Protocol because engine.config
    imports the model registry, which reaches core, which imports kvcache."""

    @property
    def page_size(self) -> int: ...

    @property
    def dtype(self) -> mx.Dtype: ...

    @property
    def max_running_req(self) -> int: ...

    @property
    def max_extend_tokens(self) -> int: ...

    @property
    def linear_state_checkpoints(self) -> int | None: ...

    @property
    def cache_type(self) -> Literal["naive", "radix"]: ...

    @property
    def swa_windows(self) -> int | None: ...


class PoolCost(NamedTuple):
    bytes_per_page: int
    fixed_bytes: int


class BasePool(ABC):
    @classmethod
    @abstractmethod
    def kv_cost(cls, spec: KVCacheGroupSpec, config: PoolConfig) -> PoolCost:
        """What the pool for spec costs, before it exists."""

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: KVCacheGroupSpec, config: PoolConfig, num_pages: int) -> Self:
        """The pool for spec when the paged pools hold num_pages usable pages."""

    @abstractmethod
    def arrays(self) -> list[mx.array]:
        """Every pool array, for eval."""

    @abstractmethod
    def set_arrays(self, arrays: list[mx.array]) -> None:
        """Rebind the pool to arrays, in arrays() order (a replayed step's updated pools)."""


class BaseKVCachePool(BasePool):
    @abstractmethod
    def k_cache(self, layer_id: int) -> mx.array: ...

    @abstractmethod
    def v_cache(self, layer_id: int) -> mx.array: ...

    @abstractmethod
    def store_kv(self, k: mx.array, v: mx.array, out_loc: mx.array, layer_id: int) -> None: ...

    @property
    @abstractmethod
    def dtype(self) -> mx.Dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


__all__ = ["BaseKVCachePool", "BasePool", "PoolConfig", "PoolCost"]
