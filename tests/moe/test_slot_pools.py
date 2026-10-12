"""The offload cache's slot layout and its hot-expert tables.

plan_slot_layout must give every layer one token's experts, the rest evenly or to the most
routed (layer, expert) pairs, every expert when the slots allow, sum a pool's layers, put the
pools smallest first, and lay the prefill buffers over the first pools (a whole layer each)
or before them (the most experts a layer can miss); too few slots must be refused, from
min_cache_size down. The caps: a decode's rows fit every pool's layers at once, a lookup
prefill's tokens one layer, and step_starts cuts a lookup forward where a pool would
overflow. initial_experts fills each pool with its layers' most routed experts.
find_expert_counts prefers the model directory's table to the shipped one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from ftmetal.moe.slot_pools import (
    HOT_EXPERTS_FILE,
    MoePoolGrouping,
    PrefillBufferPlacement,
    SlotLayout,
    SlotPool,
    find_expert_counts,
    initial_experts,
    min_cache_size,
    plan_slot_layout,
    read_expert_counts,
)

COUNTS = np.array([[10, 1, 5, 0], [3, 3, 9, 2]], dtype=np.int64)


@dataclass(frozen=True)
class Case:
    name: str
    cache_size: int
    num_layers: int
    num_experts: int
    top_k: int
    counts: np.ndarray | None
    grouping: MoePoolGrouping
    prefill_overlap: bool
    buffer_placement: PrefillBufferPlacement
    # (layers, first slot, slots) of each pool in slot order, and the buffers' size.
    pools: list[tuple[list[int], int, int]]
    buffer_size: int


def per_layer(runs: list[int], first_slot: int) -> list[tuple[list[int], int, int]]:
    """Pools of one layer each with these runs, smallest first from first_slot."""
    pools = []
    for layer_id in sorted(range(len(runs)), key=lambda layer_id: (runs[layer_id], layer_id)):
        pools.append(([layer_id], first_slot, runs[layer_id]))
        first_slot += runs[layer_id]
    return pools


CASES = [
    Case("even", 12, 3, 8, 2, None, "per-layer", False, "overlay", per_layer([4, 4, 4], 0), 0),
    Case(
        "even-remainder",
        14,
        3,
        8,
        2,
        None,
        "per-layer",
        False,
        "overlay",
        per_layer([5, 5, 4], 0),
        0,
    ),
    # Every expert fits: 24 slots, and the overlaid buffers hold a whole layer.
    Case(
        "all-resident",
        30,
        3,
        8,
        2,
        None,
        "per-layer",
        True,
        "overlay",
        per_layer([8, 8, 8], 0),
        8,
    ),
    # 14 slots: runs of 4, 4, 3, 3 need dedicated buffers of 1, which leave runs of 3.
    Case(
        "dedicated-iterate",
        14,
        4,
        4,
        2,
        None,
        "per-layer",
        True,
        "dedicated",
        per_layer([3, 3, 3, 3], 2),
        1,
    ),
    # Each layer's hottest expert first (layer 0's 0, layer 1's 2), then the three most
    # routed pairs left: layer 0's 2, layer 1's 0 and 1.
    Case("hot", 5, 2, 4, 1, COUNTS, "per-layer", False, "overlay", per_layer([2, 3], 0), 0),
    Case("unified-hot", 5, 2, 4, 1, COUNTS, "unified", False, "overlay", [([0, 1], 0, 5)], 0),
    # Qwen3.6's floors: one token's experts per layer, and two dedicated buffers of the rest.
    Case(
        "qwen-dedicated-floor",
        816,
        40,
        256,
        8,
        None,
        "per-layer",
        True,
        "dedicated",
        per_layer([8] * 40, 496),
        248,
    ),
    Case(
        "qwen-overlay-floor",
        512,
        40,
        256,
        8,
        None,
        "per-layer",
        True,
        "overlay",
        per_layer([13] * 32 + [12] * 8, 0),
        256,
    ),
    Case(
        "qwen-unified",
        7168,
        40,
        256,
        8,
        None,
        "unified",
        True,
        "overlay",
        [(list(range(40)), 0, 7168)],
        256,
    ),
    # A unified pool guarantees no layer any slot: dedicated buffers hold whole layers.
    Case(
        "qwen-unified-dedicated",
        1000,
        40,
        256,
        8,
        None,
        "unified",
        True,
        "dedicated",
        [(list(range(40)), 512, 488)],
        256,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_plan_slot_layout(case: Case) -> None:
    layout = plan_slot_layout(
        case.cache_size,
        case.num_layers,
        case.num_experts,
        case.top_k,
        grouping=case.grouping,
        counts=case.counts,
        prefill_overlap=case.prefill_overlap,
        buffer_placement=case.buffer_placement,
    )
    pools = [(pool.layers, pool.first_slot, pool.num_slots) for pool in layout.pools]
    assert (pools, layout.buffer_size) == (case.pools, case.buffer_size)
    assert layout.num_slots <= case.cache_size
    assert layout.pool_bounds == [first for _, first, _ in case.pools] + [layout.num_slots]
    assert sorted(layout.layer_pools) == sorted(
        pool for pool, (layers, _, _) in enumerate(case.pools) for _ in layers
    )


def test_too_few_slots_are_refused() -> None:
    floors = [
        ("per-layer", True, "dedicated", 816),
        ("per-layer", True, "overlay", 512),
        ("unified", True, "dedicated", 832),
        ("unified", True, "overlay", 512),
        ("unified", False, "overlay", 320),
    ]
    for grouping, prefill_overlap, buffer_placement, floor in floors:
        options = {
            "grouping": grouping,
            "prefill_overlap": prefill_overlap,
            "buffer_placement": buffer_placement,
        }
        assert min_cache_size(40, 256, 8, **options) == floor
        plan_slot_layout(floor, 40, 256, 8, counts=None, **options)
        with pytest.raises(ValueError, match="cannot"):
            plan_slot_layout(floor - 1, 40, 256, 8, counts=None, **options)
    with pytest.raises(ValueError, match="cannot hold the two prefill buffers of 256"):
        plan_slot_layout(
            511,
            40,
            256,
            8,
            grouping="per-layer",
            counts=None,
            prefill_overlap=True,
            buffer_placement="overlay",
        )
    with pytest.raises(ValueError, match="cannot give each of 3 MoE layers the 2 experts"):
        plan_slot_layout(
            5,
            3,
            8,
            2,
            grouping="unified",
            counts=None,
            prefill_overlap=False,
            buffer_placement="overlay",
        )


def qwen_layout(cache_size: int, grouping: MoePoolGrouping) -> SlotLayout:
    return plan_slot_layout(
        cache_size,
        40,
        256,
        8,
        grouping=grouping,
        counts=None,
        prefill_overlap=True,
        buffer_placement="overlay",
    )


def test_caps_and_steps() -> None:
    # Unified: 40 layers x 8 x 22 rows = 7040 slots fit 7168, 23 rows do not.
    unified = qwen_layout(7168, "unified")
    assert (unified.max_decode_rows, unified.max_lookup_tokens) == (22, None)
    assert unified.step_starts(22) == []
    assert unified.step_starts(23) == [38]
    # From 32 rows on every layer may route all 256 experts: 28 layers per step.
    assert unified.step_starts(100) == [28]
    assert unified.step_starts(0) == []
    every_expert = qwen_layout(40 * 256, "unified")
    assert (every_expert.max_decode_rows, every_expert.max_lookup_tokens) == (None, None)
    # Per-layer runs of 179 and 180: one layer's 22 rows, the same for decode and prefill.
    per_layer_pools = qwen_layout(7168, "per-layer")
    assert (per_layer_pools.max_decode_rows, per_layer_pools.max_lookup_tokens) == (22, 22)
    assert per_layer_pools.step_starts(22) == []
    # Layers 0 and 2 share 6 slots, layer 1 has 4: two rows' 4 experts overflow at layer 2.
    shared = SlotLayout(
        pools=[
            SlotPool(layers=[0, 2], first_slot=0, num_slots=6),
            SlotPool(layers=[1], first_slot=6, num_slots=4),
        ],
        buffer_size=0,
        num_experts=8,
        top_k=2,
    )
    assert (shared.max_decode_rows, shared.max_lookup_tokens) == (1, 2)
    assert shared.step_starts(1) == []
    assert shared.step_starts(2) == [2]


def test_initial_experts() -> None:
    layout = plan_slot_layout(
        5,
        2,
        4,
        1,
        grouping="per-layer",
        counts=COUNTS,
        prefill_overlap=False,
        buffer_placement="overlay",
    )
    assert initial_experts(layout, COUNTS) == [0, 2, 6, 4, 5]
    assert initial_experts(layout, None) == [0, 1, 4, 5, 6]
    unified = plan_slot_layout(
        5,
        2,
        4,
        1,
        grouping="unified",
        counts=COUNTS,
        prefill_overlap=False,
        buffer_placement="overlay",
    )
    assert initial_experts(unified, COUNTS) == [0, 6, 2, 4, 5]
    assert initial_experts(unified, None) == [0, 4, 1, 5, 2]


def test_model_table_before_shipped(tmp_path: Path) -> None:
    shipped = find_expert_counts(str(tmp_path), "qwen3_5_moe", 40, 256, 2048)
    assert shipped is not None and shipped.name == "qwen3_5_moe-40x256-h2048.json"
    assert read_expert_counts(shipped, 40, 256).shape == (40, 256)
    assert find_expert_counts(str(tmp_path), "qwen3_5_moe", 40, 256, 4096) is None
    table = {"num_layers": 2, "num_experts": 4, "counts": COUNTS.tolist()}
    (tmp_path / HOT_EXPERTS_FILE).write_text(json.dumps(table))
    local = find_expert_counts(str(tmp_path), "qwen3_5_moe", 2, 4, 64)
    assert local == tmp_path / HOT_EXPERTS_FILE
    assert (read_expert_counts(local, 2, 4) == COUNTS).all()
    with pytest.raises(ValueError, match="the model has 3 MoE layers of 4 experts"):
        read_expert_counts(local, 3, 4)
