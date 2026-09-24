"""Pool contracts: in-place stores land where the page table says, slots recycle cleanly."""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.kvcache.linear_state_pool import LinearStatePool, linear_state_bytes_per_req
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.models.config import LinearGatedDeltaGroupConfig


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
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0, 2),
        num_key_heads=2,
        num_value_heads=4,
        key_head_dim=32,
        value_head_dim=32,
        conv_kernel_dim=4,
    )
    pool = LinearStatePool(group, num_slots=4, dtype=mx.float16)
    assert pool.num_free_slots == 3
    assert pool.bytes_per_slot() == linear_state_bytes_per_req(group, mx.float16)
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
