"""Pool contracts: in-place stores land where the page table says, slots recycle cleanly,
and every pool family's kv_cost prices exactly what its from_spec allocates."""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.attention.metadata import AttnType
from ftmetal.engine.config import EngineConfig
from ftmetal.kvcache import create_pools, pool_costs
from ftmetal.kvcache.linear_state_pool import LinearStatePool
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.models.config import KVCacheGroupSpec
from ftmetal.models.qwen3_5_moe.config import parse_config
from tiny_model import TINY_TEXT_CONFIG


def test_store_kv_writes_only_the_addressed_rows() -> None:
    pool = MHAKVCache(
        num_kv_heads=2,
        num_layers=4,
        head_dim=8,
        num_pages=4,
        page_size=4,
        dtype=mx.float16,
        layer_ids=(1, 3),
    )
    k = mx.random.normal((3, 2, 8)).astype(mx.float16)
    v = mx.random.normal((3, 2, 8)).astype(mx.float16)
    out_loc = mx.array([5, 0, 15], dtype=mx.int32)
    pool.store_kv(k, v, out_loc, layer_id=3)
    assert mx.array_equal(pool.k_cache(3)[out_loc], k)
    assert mx.array_equal(pool.v_cache(3)[out_loc], v)
    assert pool.k_cache(3).sum().item() == k.sum().item()
    assert pool.k_cache(1).sum().item() == 0
    with pytest.raises(KeyError):
        pool.k_cache(0)


def test_linear_state_pool_alloc_clear_copy_free() -> None:
    spec = KVCacheGroupSpec(
        name="linear",
        attn_type=AttnType.LINEAR,
        layer_ids=(0, 2),
        conv_shape=(4, 256),
        state_shape=(4, 32, 32),
    )
    pool = LinearStatePool(spec, num_slots=4, dtype=mx.float16)
    assert pool.num_free_slots == 3
    assert pool.bytes_per_slot() == spec.slot_bytes(mx.float16)
    a, b = pool.alloc(2)
    pool.recurrent_states[0] = pool.recurrent_states[0] + 1.0
    pool.copy_from(a, b)
    pool.clear_slots([a])
    assert pool.recurrent_states[0][a].sum().item() == 0
    assert pool.recurrent_states[0][b].sum().item() == pool.recurrent_states[0][b].size
    pool.free([a, b])
    assert pool.num_free_slots == 3
    with pytest.raises(RuntimeError):
        pool.alloc(4)


def test_kv_cost_prices_what_from_spec_allocates() -> None:
    specs = parse_config(TINY_TEXT_CONFIG).kv_cache_group_specs()
    config = EngineConfig(model_path="unused", dtype=mx.float16, page_size=4, max_running_req=2)
    num_pages = 10
    costs = pool_costs(specs, config)
    allocated = {
        attn_type: sum(array.nbytes for array in pool.arrays())
        for attn_type, pool in create_pools(specs, config, num_pages).items()
    }
    # The paged pool also holds the dummy page; the state pool is all fixed cost.
    assert allocated == {
        AttnType.FULL: (num_pages + 1) * costs[AttnType.FULL].bytes_per_page,
        AttnType.LINEAR: costs[AttnType.LINEAR].fixed_bytes,
    }
    assert costs[AttnType.FULL].fixed_bytes == costs[AttnType.LINEAR].bytes_per_page == 0
