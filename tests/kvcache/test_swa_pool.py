"""Window pool contracts: full slots map to window slots of their own, frees return them
once, the sentinel backs everything else, and kv_cost prices what from_spec allocates."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.attention.metadata import AttnType
from ftmetal.engine.config import EngineConfig
from ftmetal.kvcache import create_pools, pool_costs
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.kvcache.swa_pool import SENTINEL_SLOT, SWAKVCache, naive_swa_num_tokens
from ftmetal.models.config import KVCacheGroupSpec

SPEC = KVCacheGroupSpec(
    name="swa",
    attn_type=AttnType.SWA,
    layer_ids=(0, 2),
    num_kv_heads=2,
    head_dim=8,
    sliding_window=16,
)
FULL_SPEC = KVCacheGroupSpec(
    name="full", attn_type=AttnType.FULL, layer_ids=(1,), num_kv_heads=1, head_dim=16
)


def test_alloc_translate_free_round_trip() -> None:
    pool = SWAKVCache(SPEC, num_swa_tokens=8, num_full_tokens=32, dtype=mx.float16)
    assert pool.swa_available_size == 7
    full_slots = np.array([5, 9, 30], dtype=np.int32)
    pool.alloc_swa(full_slots)
    swa_slots = pool.translate(full_slots)
    assert len(set(swa_slots.tolist())) == 3 and SENTINEL_SLOT not in swa_slots
    assert pool.translate(np.array([0, 31], dtype=np.int32)).tolist() == [SENTINEL_SLOT] * 2
    assert pool.swa_available_size == 4

    # A second free of the same slots (and a never-allocated one) returns nothing more.
    pool.free_swa(np.array([5, 9], dtype=np.int32))
    pool.free_swa(np.array([5, 9, 12], dtype=np.int32))
    assert pool.swa_available_size == 6
    assert pool.translate(full_slots).tolist() == [SENTINEL_SLOT, SENTINEL_SLOT, swa_slots[2]]
    pool.free_swa(full_slots)
    assert pool.swa_available_size == 7
    assert SENTINEL_SLOT not in pool.free_swa_slots

    with pytest.raises(RuntimeError, match="SWA pool exhausted"):
        pool.alloc_swa(np.arange(8, dtype=np.int32))


def test_store_kv_writes_the_translated_window_slots() -> None:
    pool = SWAKVCache(SPEC, num_swa_tokens=8, num_full_tokens=32, dtype=mx.float16)
    full_slots = np.array([3, 4], dtype=np.int32)
    pool.alloc_swa(full_slots)
    swa_loc = mx.array(pool.translate(full_slots))
    k = mx.random.normal((2, 2, 8)).astype(mx.float16)
    v = mx.random.normal((2, 2, 8)).astype(mx.float16)
    pool.store_kv(k, v, swa_loc, layer_id=2)
    assert mx.array_equal(pool.k_cache(2)[swa_loc], k)
    assert mx.array_equal(pool.v_cache(2)[swa_loc], v)
    assert pool.k_cache(0).abs().sum().item() == 0
    with pytest.raises(KeyError):
        pool.k_cache(1)


@pytest.mark.parametrize("page_size", [1, 4])
def test_kv_cost_prices_what_from_spec_allocates(page_size: int) -> None:
    config = EngineConfig(
        model_path="unused",
        dtype=mx.float16,
        page_size=page_size,
        max_running_req=3,
        max_extend_tokens=64,
    )
    num_pages = 10
    costs = pool_costs((FULL_SPEC, SPEC), config)
    pools = create_pools((FULL_SPEC, SPEC), config, num_pages)
    swa_pool, full_pool = pools[AttnType.SWA], pools[AttnType.FULL]
    assert isinstance(swa_pool, SWAKVCache) and isinstance(full_pool, MHAKVCache)
    num_swa_tokens = naive_swa_num_tokens(16, 64, 3)
    assert num_swa_tokens == 4 * 96
    assert swa_pool.swa_num_tokens == num_swa_tokens
    # The window pool is all fixed cost; its mapping covers the FULL pool's slots, dummy page
    # included.
    assert costs[AttnType.SWA].bytes_per_page == 0
    assert sum(array.nbytes for array in swa_pool.arrays()) == costs[AttnType.SWA].fixed_bytes
    assert len(swa_pool.full_to_swa) == full_pool.num_tokens == (num_pages + 1) * page_size
