"""Pool families keyed by attention type, and the pools of a model's group specs.

Port of FreeToken kvcache/__init__.py (resolve_pool_class, create_kv_pool). Divergences:
every KVCacheGroupSpec gets its own pool, looked up by AttnType in POOL_FAMILIES, so the
GDN state pool is one of them instead of one KV family per model plus a LinearStatePool
the engine builds; the "+1 dummy page" is the paged pool's own from_spec rule; the
prefix caches are built by the scheduler, not registered here.
"""

from __future__ import annotations

from ftmetal.attention.metadata import AttnType
from ftmetal.kvcache.base import BasePool, PoolConfig, PoolCost
from ftmetal.kvcache.linear_state_pool import LinearStatePool
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.models.config import KVCacheGroupSpec

POOL_FAMILIES: dict[AttnType, type[BasePool]] = {
    AttnType.FULL: MHAKVCache,
    AttnType.LINEAR: LinearStatePool,
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


__all__ = ["POOL_FAMILIES", "create_pools", "pool_costs"]
