"""Pool families keyed by attention type, the pools of a model's group specs, and their
arrays in the one order a CommandGraph capture and its replays take them.

Port of FreeToken kvcache/__init__.py (resolve_pool_class, create_kv_pool). Divergences:
every KVCacheGroupSpec gets its own pool, looked up by AttnType in POOL_FAMILIES, so the
GDN state pool is one of them instead of one KV family per model plus a LinearStatePool
the engine builds, and FreeToken's HybridSWAKVCache is the FULL pool plus the SWA one;
the "+1 dummy page" is the paged pool's own from_spec rule; the prefix caches are built
by the scheduler, not registered here.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.attention.metadata import AttnType
from ftmetal.kvcache.base import BasePool, PoolConfig, PoolCost
from ftmetal.kvcache.linear_state_pool import LinearStatePool
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.kvcache.qsa_pool import QSAKVCache
from ftmetal.kvcache.swa_pool import SWAKVCache
from ftmetal.models.config import KVCacheGroupSpec

POOL_FAMILIES: dict[AttnType, type[BasePool]] = {
    AttnType.FULL: MHAKVCache,
    AttnType.SWA: SWAKVCache,
    AttnType.LINEAR: LinearStatePool,
    AttnType.QSA: QSAKVCache,
}


def pool_costs(specs: tuple[KVCacheGroupSpec, ...], config: PoolConfig) -> dict[AttnType, PoolCost]:
    return {spec.attn_type: POOL_FAMILIES[spec.attn_type].kv_cost(spec, config) for spec in specs}


def create_pools(
    specs: tuple[KVCacheGroupSpec, ...], config: PoolConfig, num_pages: int
) -> dict[AttnType, BasePool]:
    """One pool per spec; num_pages counts the usable pages the budget solved for."""
    return {
        spec.attn_type: POOL_FAMILIES[spec.attn_type].from_spec(spec, config, num_pages)
        for spec in specs
    }


def pool_arrays(pools: dict[AttnType, BasePool]) -> list[mx.array]:
    return [array for pool in pools.values() for array in pool.arrays()]


def set_pool_arrays(pools: dict[AttnType, BasePool], arrays: list[mx.array]) -> None:
    """Rebind every pool to its part of arrays, in pool_arrays order."""
    offset = 0
    for pool in pools.values():
        count = len(pool.arrays())
        pool.set_arrays(arrays[offset : offset + count])
        offset += count
    assert offset == len(arrays)


__all__ = ["POOL_FAMILIES", "create_pools", "pool_arrays", "pool_costs", "set_pool_arrays"]
