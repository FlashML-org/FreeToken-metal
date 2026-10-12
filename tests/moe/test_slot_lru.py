"""SlotLru against offload_harness's reference of its per-pool LRUs.

SlotLru must return the same slots and loads on every call and leave the same slot_of and
last_use, no load may take a slot one of the same call's hits uses, and a miss takes a slot
of its own layer's pool only, which layers of one pool share. A call that would evict a slot
its pool used in the same step must raise, as must one naming more experts than the pool
holds, and a fill beyond a pool. The prefill buffers: prefill_slot_map keeps resident
experts and sends the others, in expert order, to the next slots of buffer layer_id % 2, and
refuses more absent experts than a buffer holds; buffers over the pools must be invalidated
first, which drops the experts their slots hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel._ext import _C
from offload_harness import ReferenceLru

# One step: the (layer, distinct expert ids) calls of resolve, in order.
Step = tuple[tuple[int, tuple[int, ...]], ...]


def random_script(
    seed: int, num_steps: int, num_layers: int, num_experts: int, top_k: int
) -> tuple[Step, ...]:
    """Every layer once per step, each with top_k distinct experts."""
    rng = np.random.default_rng(seed)
    return tuple(
        tuple(
            (layer_id, tuple(int(e) for e in rng.choice(num_experts, top_k, replace=False)))
            for layer_id in range(num_layers)
        )
        for _ in range(num_steps)
    )


def first_experts(
    num_experts: int, layer_pools: tuple[int, ...], pool_slots: tuple[int, ...]
) -> tuple[int, ...]:
    """Flat ids of each pool's lowest-numbered experts, layer by layer, filling it."""
    flat_ids: list[int] = []
    for pool, num_slots in enumerate(pool_slots):
        layers = [layer_id for layer_id, owner in enumerate(layer_pools) if owner == pool]
        pairs = [layer_id * num_experts + e for e in range(num_experts) for layer_id in layers]
        flat_ids += pairs[:num_slots]
    return tuple(flat_ids)


@dataclass(frozen=True)
class Case:
    num_experts: int
    buffer_size: int
    # Each layer's pool, and each pool's slots; the pools follow the buffers.
    layer_pools: tuple[int, ...]
    pool_slots: tuple[int, ...]
    # Flat ids filled, each into the next slot of its layer's pool.
    fill: tuple[int, ...]
    script: tuple[Step, ...]
    name: str

    def __str__(self) -> str:
        return self.name


PER_LAYER = (0, 1, 2, 3)
CASES = [
    # Misses evict the layer's never-used ids first.
    Case(
        4,
        0,
        (0, 1),
        (3, 2),
        (0, 4, 1, 5),
        (((0, (0, 3)), (1, (1, 2))), ((0, (3, 1)), (1, (0, 3)))),
        "basic",
    ),
    # Slot 0 holds the least recently used id, but this call hits it: the miss takes slot 1.
    Case(
        4,
        0,
        (0,),
        (3,),
        (0, 1, 2),
        (((0, (0,)),), ((0, (1,)),), ((0, (2,)),), ((0, (0, 3)),)),
        "hit-masked",
    ),
    # Ties on last use break by the lowest flat id, wherever its slot is in the pool.
    Case(
        3, 0, (0, 1), (2, 2), (1, 0, 5, 3), (((1, (1,)),), ((0, (2,)), (1, (2,)))), "lowest-id-tie"
    ),
    # A free slot goes before every resident expert, even one never used.
    Case(4, 0, (0, 1), (3, 3), (0, 4), (((0, (1, 2)), (1, (1,))), ((1, (2, 3)),)), "free-first"),
    # Layer 0's slots are the least recently used, yet layer 1's misses take its own.
    Case(
        4,
        0,
        (0, 1),
        (2, 2),
        (0, 1, 4, 5),
        (((0, (0, 1)),), ((1, (0, 1)),), ((1, (2, 3)),)),
        "own-pool",
    ),
    # One pool for both layers: layer 1's misses take layer 0's least recently used slots,
    # but none layer 0 used at this step.
    Case(
        4,
        0,
        (0, 0),
        (4,),
        (0, 4, 1, 5),
        (((0, (0, 2)), (1, (2,))), ((1, (0, 1)), (0, (3,)))),
        "shared-pool",
    ),
    # Every slot changes hands many times.
    Case(
        8,
        0,
        PER_LAYER[:3],
        (4, 4, 4),
        first_experts(8, PER_LAYER[:3], (4, 4, 4)),
        random_script(0, 200, 3, 8, 3),
        "random-200",
    ),
    Case(
        16,
        6,
        PER_LAYER,
        (10, 12, 8, 10),
        first_experts(16, PER_LAYER, (10, 12, 8, 10)),
        random_script(1, 100, 4, 16, 4),
        "random-buffers",
    ),
    # Layers 0 and 2 share a pool, 1 and 3 have their own.
    Case(
        16,
        0,
        (0, 1, 0, 2),
        (12, 7, 5),
        first_experts(16, (0, 1, 0, 2), (12, 7, 5)),
        random_script(2, 100, 4, 16, 4),
        "random-mixed-pools",
    ),
    Case(
        16,
        0,
        (0, 0, 0, 0),
        (20,),
        first_experts(16, (0, 0, 0, 0), (20,)),
        random_script(3, 100, 4, 16, 4),
        "random-unified",
    ),
]


def make_lru(
    num_experts: int,
    buffer_size: int,
    layer_pools: tuple[int, ...],
    pool_bounds: list[int],
) -> tuple[_C.SlotLru, mx.array, mx.array]:
    num_layers = len(layer_pools)
    slot_of = mx.full((num_layers * num_experts,), -1, dtype=mx.int32)
    last_use = mx.zeros((pool_bounds[-1],), dtype=mx.uint32)
    mx.eval(slot_of, last_use)
    lru = _C.SlotLru(
        num_layers, num_experts, buffer_size, list(layer_pools), pool_bounds, slot_of, last_use
    )
    assert lru.num_slots == pool_bounds[-1]
    return lru, slot_of, last_use


def bounds(first_slot: int, pool_slots: tuple[int, ...]) -> list[int]:
    return [first_slot + sum(pool_slots[:pool]) for pool in range(len(pool_slots) + 1)]


def check(case: Case) -> None:
    pool_bounds = bounds(2 * case.buffer_size, case.pool_slots)
    lru, slot_of, last_use = make_lru(
        case.num_experts, case.buffer_size, case.layer_pools, pool_bounds
    )
    reference = ReferenceLru(case.num_experts, list(case.layer_pools), pool_bounds, case.fill)
    filled = lru.fill(list(case.fill))
    assert filled == [reference.id_of_slot.index(flat_id) for flat_id in case.fill]
    for step, calls in enumerate(case.script, start=1):
        for layer_id, expert_ids in calls:
            slots, loads = lru.resolve(layer_id, list(expert_ids), step)
            assert (slots, loads) == reference.resolve(layer_id, list(expert_ids), step)
            loaded = {expert: slot for expert, slot in loads}
            hit_slots = {slot for expert, slot in zip(expert_ids, slots) if expert not in loaded}
            assert not hit_slots & set(loaded.values()), (step, layer_id)
            assert all(slot in reference.pool_slots(layer_id) for slot in slots), (step, layer_id)
    expected_slot_of = [-1] * (len(case.layer_pools) * case.num_experts)
    for slot, flat_id in enumerate(reference.id_of_slot):
        if flat_id >= 0:
            expected_slot_of[flat_id] = slot
    assert np.array(slot_of).tolist() == expected_slot_of
    assert np.array(last_use).tolist() == reference.last_use


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_reference_lru(case: Case) -> None:
    check(case)


def test_capacity_errors() -> None:
    lru, _, _ = make_lru(4, 0, (0, 1), [0, 2, 4])
    lru.resolve(0, [0, 1], 1)
    with pytest.raises(RuntimeError, match="would evict a slot its pool used at step 1"):
        lru.resolve(0, [2], 1)
    with pytest.raises(RuntimeError, match="pool 1's 2 slots cannot hold layer 1's 3 experts"):
        lru.resolve(1, [0, 1, 2], 2)
    assert lru.resolve(1, [0, 1], 2)[1] == [(0, 2), (1, 3)]
    with pytest.raises(ValueError, match="pool 0 has more experts than its 2 slots"):
        lru.fill([0, 1, 2])
    # Layers 0 and 1 share three slots: layer 0's two of this step leave layer 1 one.
    shared, _, _ = make_lru(4, 0, (0, 0), [0, 3])
    shared.resolve(0, [0, 1], 1)
    with pytest.raises(RuntimeError, match="would evict a slot its pool used at step 1"):
        shared.resolve(1, [0, 1], 1)


def test_prefill_slot_map() -> None:
    # Buffers [0, 3) and [3, 6) before the pools: layer 0's [6, 8), layer 1's [8, 9).
    lru, _, _ = make_lru(4, 3, (0, 1), [6, 8, 9])
    lru.fill([0, 1, 4])
    assert lru.prefill_slot_map(0) == ([6, 7, 0, 1], [(2, 0), (3, 1)])
    assert lru.prefill_slot_map(1) == ([8, 3, 4, 5], [(1, 3), (2, 4), (3, 5)])
    too_small, _, _ = make_lru(4, 1, (0,), [2, 4])
    too_small.fill([0, 1])
    with pytest.raises(RuntimeError, match="prefill buffers of 1 slots cannot hold layer 0's 2"):
        too_small.prefill_slot_map(0)


def test_overlaid_prefill_buffers() -> None:
    # Buffers [0, 4) and [4, 8) over layer 1's pool [0, 3), layer 2's [3, 6) and the start of
    # layer 0's [6, 10).
    lru, slot_of, last_use = make_lru(4, 4, (2, 0, 1), [0, 3, 6, 10])
    lru.fill([4, 5, 6, 8, 9, 10, 0, 1, 2, 3])
    lru.resolve(0, [3], 1)
    with pytest.raises(RuntimeError, match="sits in prefill buffer slot 6; invalidate"):
        lru.prefill_slot_map(0)
    lru.invalidate_prefill_buffers()
    assert np.array(slot_of).tolist() == [-1, -1, 8, 9] + [-1] * 8
    assert np.array(last_use).tolist() == [0] * 9 + [1]
    assert lru.prefill_slot_map(0) == ([0, 1, 8, 9], [(0, 0), (1, 1)])
    assert lru.prefill_slot_map(1) == ([4, 5, 6, 7], [(0, 4), (1, 5), (2, 6), (3, 7)])
    # The dropped slots are free again: a miss of layer 1 takes its pool's first.
    assert lru.resolve(1, [2], 2) == ([0], [(2, 0)])
