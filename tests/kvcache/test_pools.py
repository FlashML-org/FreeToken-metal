"""Pool contracts: in-place stores land where the page table says, slots recycle cleanly,
the GDN state pool holds the slots its cache type needs, and every pool family's kv_cost
prices exactly what its from_spec allocates."""

from __future__ import annotations

from typing import Literal, NamedTuple

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


class SlotCase(NamedTuple):
    label: str
    cache_type: Literal["naive", "radix"]
    max_running_req: int
    gdn_snapshots: int | None
    num_slots: int


# Radix (the hybrid radix cache): 4 slots per running request (live, ping-pong pair, locked
# snapshot), the snapshot cache of gdn_snapshots slots, 8 by default, and the padding slot.
# Naive: one live slot per running request and the padding slot.
SLOT_CASES = [
    SlotCase("radix, the default snapshot cache", "radix", 2, None, 4 * 2 + 8 + 1),
    SlotCase("radix, one conversation's snapshots", "radix", 4, 2, 4 * 4 + 2 + 1),
    SlotCase("radix, a snapshot cache past the default", "radix", 8, 32, 4 * 8 + 32 + 1),
    SlotCase("naive", "naive", 2, None, 2 + 1),
    SlotCase("naive ignores the snapshot count", "naive", 8, 1, 8 + 1),
]


def slot_config(
    cache_type: Literal["naive", "radix"], max_running_req: int, gdn_snapshots: int | None
) -> EngineConfig:
    return EngineConfig(
        model_path="unused",
        dtype=mx.float16,
        page_size=4,
        max_running_req=max_running_req,
        gdn_snapshots=gdn_snapshots,
        cache_type=cache_type,
    )


def test_kv_cost_prices_what_from_spec_allocates() -> None:
    specs = parse_config(TINY_TEXT_CONFIG).kv_cache_group_specs()
    num_pages = 10
    for case in SLOT_CASES:
        config = slot_config(case.cache_type, case.max_running_req, case.gdn_snapshots)
        costs = pool_costs(specs, config)
        pools = create_pools(specs, config, num_pages)
        linear_pool = pools[AttnType.LINEAR]
        assert isinstance(linear_pool, LinearStatePool)
        assert linear_pool.num_slots == case.num_slots, case.label
        allocated = {
            attn_type: sum(array.nbytes for array in pool.arrays())
            for attn_type, pool in pools.items()
        }
        # The paged pool also holds the dummy page; the state pool is all fixed cost.
        assert allocated == {
            AttnType.FULL: (num_pages + 1) * costs[AttnType.FULL].bytes_per_page,
            AttnType.LINEAR: costs[AttnType.LINEAR].fixed_bytes,
        }, case.label
        assert costs[AttnType.FULL].fixed_bytes == costs[AttnType.LINEAR].bytes_per_page == 0


def test_fewer_snapshots_than_one_conversation_are_refused() -> None:
    specs = parse_config(TINY_TEXT_CONFIG).kv_cache_group_specs()
    with pytest.raises(ValueError, match="--gdn-snapshots 2 or more"):
        pool_costs(specs, slot_config("radix", 2, 1))
