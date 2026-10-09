"""SlotLru against a Python reference of the offload simulation's token-granular LRU.

The reference keeps, per slot, the expert it holds and the step it was last used. A call
first marks its hits used at this step, then gives each miss the slot least recently used
before this step: a free slot first, then the lowest flat expert id. SlotLru must return
the same slots and loads on every call and leave the same slot_of and last_use, and no
load may take a slot one of the same call's hits uses. A call that would evict a slot used
in its own step must raise, as must one naming more experts than the cache has slots. The
windows: free window slots go first, invalidate_windows drops what they hold, and
prefill_slot_map keeps resident experts and sends the others to the layer's window.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel._ext import _C

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


@dataclass(frozen=True)
class Case:
    num_layers: int
    num_experts: int
    num_slots: int
    window_slots: int
    # Flat ids filled from slot window_slots on.
    fill: tuple[int, ...]
    script: tuple[Step, ...]
    name: str

    def __str__(self) -> str:
        return self.name


CASES = [
    # Layer-interleaved fill as the simulation's; misses evict the never-used ids first.
    Case(
        2,
        4,
        5,
        0,
        (0, 4, 1, 5, 2),
        (((0, (0, 3)), (1, (1, 2))), ((0, (3, 1)), (1, (0, 3)))),
        "basic",
    ),
    # Slot 0 holds the least recently used id, but this call hits it: the miss takes slot 1.
    Case(
        1,
        4,
        3,
        0,
        (0, 1, 2),
        (((0, (0,)),), ((0, (1,)),), ((0, (2,)),), ((0, (0, 3)),)),
        "hit-masked",
    ),
    # Ties on last use break by the lowest flat id, wherever its slot is.
    Case(2, 3, 4, 0, (5, 1, 3, 0), (((1, (1,)),), ((0, (2,)), (1, (2,)))), "lowest-id-tie"),
    # A free slot goes before every resident expert, even one never used.
    Case(2, 4, 6, 0, (0, 4), (((0, (1, 2)), (1, (1,))), ((1, (2, 3)),)), "free-first"),
    # Free window slots are taken first, in slot order.
    Case(2, 2, 6, 4, (0, 2), (((0, (0, 1)), (1, (1,))), ((0, (1,)), (1, (0, 1)))), "windows"),
    # Every slot changes hands many times.
    Case(3, 8, 12, 0, tuple(range(12)), random_script(0, 200, 3, 8, 3), "random-200"),
    Case(4, 16, 72, 32, tuple(range(40)), random_script(1, 100, 4, 16, 4), "random-windows"),
]


class ReferenceLru:
    def __init__(self, case: Case) -> None:
        self.num_experts = case.num_experts
        self.id_of_slot = [-1] * case.num_slots
        self.last_use = [0] * case.num_slots
        for j, flat_id in enumerate(case.fill):
            self.id_of_slot[case.window_slots + j] = flat_id

    def resolve(
        self, layer_id: int, expert_ids: list[int], step: int
    ) -> tuple[list[int], list[tuple[int, int]]]:
        flat_ids = [layer_id * self.num_experts + expert for expert in expert_ids]
        slot_of = {flat_id: slot for slot, flat_id in enumerate(self.id_of_slot) if flat_id >= 0}
        for flat_id in flat_ids:
            if flat_id in slot_of:
                self.last_use[slot_of[flat_id]] = step
        loads: list[tuple[int, int]] = []
        for expert, flat_id in zip(expert_ids, flat_ids):
            if flat_id not in slot_of:
                victim = min(
                    (slot for slot, last in enumerate(self.last_use) if last < step),
                    key=lambda slot: (self.last_use[slot], self.id_of_slot[slot], slot),
                )
                slot_of.pop(self.id_of_slot[victim], None)
                self.id_of_slot[victim] = flat_id
                self.last_use[victim] = step
                slot_of[flat_id] = victim
                loads.append((expert, victim))
        return [slot_of[flat_id] for flat_id in flat_ids], loads


def make_lru(
    num_layers: int, num_experts: int, num_slots: int, window_slots: int
) -> tuple[_C.SlotLru, mx.array, mx.array]:
    slot_of = mx.full((num_layers * num_experts,), -1, dtype=mx.int32)
    last_use = mx.zeros((num_slots,), dtype=mx.uint32)
    mx.eval(slot_of, last_use)
    lru = _C.SlotLru(num_layers, num_experts, num_slots, window_slots, slot_of, last_use)
    return lru, slot_of, last_use


def check(case: Case) -> None:
    lru, slot_of, last_use = make_lru(
        case.num_layers, case.num_experts, case.num_slots, case.window_slots
    )
    reference = ReferenceLru(case)
    filled = lru.fill(list(case.fill))
    assert filled == [case.window_slots + j for j in range(len(case.fill))]
    for step, calls in enumerate(case.script, start=1):
        for layer_id, expert_ids in calls:
            slots, loads = lru.resolve(layer_id, list(expert_ids), step)
            assert (slots, loads) == reference.resolve(layer_id, list(expert_ids), step)
            loaded = {expert: slot for expert, slot in loads}
            hit_slots = {slot for expert, slot in zip(expert_ids, slots) if expert not in loaded}
            assert not hit_slots & set(loaded.values()), (step, layer_id)
    expected_slot_of = [-1] * (case.num_layers * case.num_experts)
    for slot, flat_id in enumerate(reference.id_of_slot):
        if flat_id >= 0:
            expected_slot_of[flat_id] = slot
    assert np.array(slot_of).tolist() == expected_slot_of
    assert np.array(last_use).tolist() == reference.last_use


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_reference_lru(case: Case) -> None:
    check(case)


def test_too_small_cache_raises() -> None:
    lru, _, _ = make_lru(2, 4, 2, 0)
    lru.resolve(0, [0, 1], 1)
    with pytest.raises(RuntimeError, match="used at step 1"):
        lru.resolve(1, [0], 1)
    with pytest.raises(RuntimeError, match="cannot hold"):
        lru.resolve(1, [0, 1, 2], 2)
    assert lru.resolve(1, [0, 1], 2)[1] == [(0, 0), (1, 1)]


def test_windows() -> None:
    lru, slot_of, last_use = make_lru(2, 2, 6, 4)
    lru.fill([0, 2])
    assert lru.resolve(0, [0, 1], 1) == ([4, 0], [(1, 0)])
    assert lru.resolve(1, [1], 1) == ([1], [(1, 1)])
    with pytest.raises(RuntimeError, match="window slot"):
        lru.prefill_slot_map(0)
    lru.invalidate_windows()
    assert np.array(slot_of).tolist() == [4, -1, 5, -1]
    assert np.array(last_use).tolist() == [0, 0, 0, 0, 1, 0]
    assert lru.prefill_slot_map(0) == ([4, 1], [(1, 1)])
    assert lru.prefill_slot_map(1) == ([5, 3], [(1, 3)])
    without_windows, _, _ = make_lru(2, 2, 6, 0)
    with pytest.raises(RuntimeError, match="no prefill windows"):
        without_windows.prefill_slot_map(0)
