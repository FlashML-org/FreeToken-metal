"""SlotLru against offload_harness's reference of its per-layer LRUs.

SlotLru must return the same slots and loads on every call and leave the same slot_of and
last_use, no load may take a slot one of the same call's hits uses, and a miss takes a slot
of its own layer's run only. A call that would evict a slot its layer used in the same step
must raise, as must one naming more experts than the layer's run holds, and a fill beyond a
run. The windows: prefill_slot_map keeps resident experts and sends the others, in expert
order, to the next slots of window layer_id % 2, and refuses more absent experts than a
window holds.
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


def first_experts(num_experts: int, layer_slots: tuple[int, ...]) -> tuple[int, ...]:
    """Flat ids of each layer's lowest-numbered experts filling its run."""
    return tuple(
        layer_id * num_experts + expert
        for layer_id, num_slots in enumerate(layer_slots)
        for expert in range(num_slots)
    )


@dataclass(frozen=True)
class Case:
    num_layers: int
    num_experts: int
    window_size: int
    layer_slots: tuple[int, ...]
    # Flat ids filled, each into the next slot of its layer's run.
    fill: tuple[int, ...]
    script: tuple[Step, ...]
    name: str

    def __str__(self) -> str:
        return self.name


CASES = [
    # Misses evict the layer's never-used ids first.
    Case(
        2,
        4,
        0,
        (3, 2),
        (0, 4, 1, 5),
        (((0, (0, 3)), (1, (1, 2))), ((0, (3, 1)), (1, (0, 3)))),
        "basic",
    ),
    # Slot 0 holds the least recently used id, but this call hits it: the miss takes slot 1.
    Case(
        1,
        4,
        0,
        (3,),
        (0, 1, 2),
        (((0, (0,)),), ((0, (1,)),), ((0, (2,)),), ((0, (0, 3)),)),
        "hit-masked",
    ),
    # Ties on last use break by the lowest flat id, wherever its slot is in the run.
    Case(2, 3, 0, (2, 2), (1, 0, 5, 3), (((1, (1,)),), ((0, (2,)), (1, (2,)))), "lowest-id-tie"),
    # A free slot goes before every resident expert, even one never used.
    Case(2, 4, 0, (3, 3), (0, 4), (((0, (1, 2)), (1, (1,))), ((1, (2, 3)),)), "free-first"),
    # Layer 0's slots are the least recently used, yet layer 1's misses take its own.
    Case(
        2, 4, 0, (2, 2), (0, 1, 4, 5), (((0, (0, 1)),), ((1, (0, 1)),), ((1, (2, 3)),)), "own-run"
    ),
    # Every slot changes hands many times.
    Case(
        3,
        8,
        0,
        (4, 4, 4),
        first_experts(8, (4, 4, 4)),
        random_script(0, 200, 3, 8, 3),
        "random-200",
    ),
    Case(
        4,
        16,
        6,
        (10, 12, 8, 10),
        first_experts(16, (10, 12, 8, 10)),
        random_script(1, 100, 4, 16, 4),
        "random-windows",
    ),
]


def make_lru(
    num_layers: int, num_experts: int, window_size: int, layer_slots: tuple[int, ...]
) -> tuple[_C.SlotLru, mx.array, mx.array]:
    num_slots = 2 * window_size + sum(layer_slots)
    slot_of = mx.full((num_layers * num_experts,), -1, dtype=mx.int32)
    last_use = mx.zeros((num_slots,), dtype=mx.uint32)
    mx.eval(slot_of, last_use)
    lru = _C.SlotLru(num_layers, num_experts, window_size, list(layer_slots), slot_of, last_use)
    assert lru.num_slots == num_slots
    return lru, slot_of, last_use


def check(case: Case) -> None:
    lru, slot_of, last_use = make_lru(
        case.num_layers, case.num_experts, case.window_size, case.layer_slots
    )
    reference = ReferenceLru(case.num_experts, case.window_size, list(case.layer_slots), case.fill)
    filled = lru.fill(list(case.fill))
    assert filled == [reference.id_of_slot.index(flat_id) for flat_id in case.fill]
    for step, calls in enumerate(case.script, start=1):
        for layer_id, expert_ids in calls:
            slots, loads = lru.resolve(layer_id, list(expert_ids), step)
            assert (slots, loads) == reference.resolve(layer_id, list(expert_ids), step)
            loaded = {expert: slot for expert, slot in loads}
            hit_slots = {slot for expert, slot in zip(expert_ids, slots) if expert not in loaded}
            assert not hit_slots & set(loaded.values()), (step, layer_id)
            run = range(reference.bases[layer_id], reference.bases[layer_id + 1])
            assert all(slot in run for slot in slots), (step, layer_id)
    expected_slot_of = [-1] * (case.num_layers * case.num_experts)
    for slot, flat_id in enumerate(reference.id_of_slot):
        if flat_id >= 0:
            expected_slot_of[flat_id] = slot
    assert np.array(slot_of).tolist() == expected_slot_of
    assert np.array(last_use).tolist() == reference.last_use


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_reference_lru(case: Case) -> None:
    check(case)


def test_capacity_errors() -> None:
    lru, _, _ = make_lru(2, 4, 0, (2, 2))
    lru.resolve(0, [0, 1], 1)
    with pytest.raises(RuntimeError, match="would evict a slot it used at step 1"):
        lru.resolve(0, [2], 1)
    with pytest.raises(RuntimeError, match="2 slots cannot hold its 3 experts"):
        lru.resolve(1, [0, 1, 2], 2)
    assert lru.resolve(1, [0, 1], 2)[1] == [(0, 2), (1, 3)]
    with pytest.raises(ValueError, match="more experts than its 2 slots"):
        lru.fill([0, 1, 2])


def test_prefill_slot_map() -> None:
    # Windows [0, 3) and [3, 6); layer 0's run [6, 8), layer 1's [8, 9).
    lru, _, _ = make_lru(2, 4, 3, (2, 1))
    lru.fill([0, 1, 4])
    assert lru.prefill_slot_map(0) == ([6, 7, 0, 1], [(2, 0), (3, 1)])
    assert lru.prefill_slot_map(1) == ([8, 3, 4, 5], [(1, 3), (2, 4), (3, 5)])
    too_small, _, _ = make_lru(1, 4, 1, (2,))
    too_small.fill([0, 1])
    with pytest.raises(RuntimeError, match="windows of 1 slots cannot hold layer 0's 2 absent"):
        too_small.prefill_slot_map(0)
