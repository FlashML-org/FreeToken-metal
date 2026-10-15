"""The offload cache's slot layout and its hot-expert tables.

plan_slot_layout must give every layer one token's experts, the rest evenly or to the most
routed (layer, expert) pairs, every expert when the slots allow, sum a pool's layers, put the
pools smallest first (unified: the layers of one quant type), and lay two prefill buffers of a
whole layer each over the first pools or before them, none when every expert fits; too few
slots must be refused, from min_cache_size down. The caps: a decode's rows fit every pool's
layers at once, a lookup prefill's tokens one layer, and step_starts cuts a lookup forward
where a pool would overflow. initial_experts fills each pool with its layers' most routed
experts, coldest first. find_expert_counts prefers the model directory's table to the shipped
one.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.layers.moe import ExpertSchemes, moe_layout
from ftmetal.layers.quantization.config import AffineScheme, BankSpec
from ftmetal.moe.slot_pools import (
    HOT_EXPERTS_FILE,
    MoePoolGrouping,
    PrefillBufferPlacement,
    SlotLayout,
    find_expert_counts,
    initial_experts,
    min_cache_size,
    plan_slot_layout,
    read_expert_counts,
)

COUNTS = np.array([[10, 1, 5, 0], [3, 3, 9, 2]], dtype=np.int64)


def tiny_layout(bits: int) -> dict[str, BankSpec]:
    """Experts of hidden and intermediate 64, group 64: a slot of 6912 bytes at 4 bits (gate_up
    weight rows of 4096), 13056 at 8 (8192)."""
    return moe_layout(64, 64, ExpertSchemes.uniform(AffineScheme(bits, 64)), mx.bfloat16)


def uniform(num_layers: int) -> list[dict[str, BankSpec]]:
    return [tiny_layout(4)] * num_layers


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
    # Every expert fits in 24 slots: no buffers.
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
        0,
    ),
    # Dedicated buffers of a whole layer each, then runs of 3.
    Case(
        "dedicated",
        28,
        4,
        8,
        2,
        None,
        "per-layer",
        True,
        "dedicated",
        per_layer([3, 3, 3, 3], 0),
        8,
    ),
    # Each layer's hottest expert first (layer 0's 0, layer 1's 2), then the three most
    # routed pairs left: layer 0's 2, layer 1's 0 and 1.
    Case("hot", 5, 2, 4, 1, COUNTS, "per-layer", False, "overlay", per_layer([2, 3], 0), 0),
    # Qwen3.6's floors: one token's experts per layer, beside or under two whole layers.
    Case(
        "qwen-dedicated-floor",
        832,
        40,
        256,
        8,
        None,
        "per-layer",
        True,
        "dedicated",
        per_layer([8] * 40, 0),
        256,
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
        [(list(range(40)), 0, 488)],
        256,
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_plan_slot_layout(case: Case) -> None:
    layout = plan_slot_layout(
        case.cache_size,
        uniform(case.num_layers),
        case.num_experts,
        case.top_k,
        grouping=case.grouping,
        counts=case.counts,
        prefill_overlap=case.prefill_overlap,
        buffer_placement=case.buffer_placement,
    )
    pools = [(pool.layers, pool.first_slot, pool.num_slots) for pool in layout.pools]
    assert (pools, layout.buffer_size) == (case.pools, case.buffer_size)


def test_too_few_slots_are_refused() -> None:
    floors = [(True, "dedicated", 832), (True, "overlay", 512), (False, "overlay", 320)]
    for prefill_overlap, buffer_placement, floor in floors:
        options = {"prefill_overlap": prefill_overlap, "buffer_placement": buffer_placement}
        assert min_cache_size(40, 256, 8, **options) == floor
        for grouping in ("unified", "per-layer"):
            plan_slot_layout(floor, uniform(40), 256, 8, grouping=grouping, counts=None, **options)
            with pytest.raises(ValueError, match=f"cannot hold .*: give at least {floor}$"):
                plan_slot_layout(
                    floor - 1, uniform(40), 256, 8, grouping=grouping, counts=None, **options
                )


def qwen_layout(cache_size: int, grouping: MoePoolGrouping) -> SlotLayout:
    return plan_slot_layout(
        cache_size,
        uniform(40),
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


def test_initial_experts() -> None:
    layout = plan_slot_layout(
        5,
        uniform(2),
        4,
        1,
        grouping="per-layer",
        counts=COUNTS,
        prefill_overlap=False,
        buffer_placement="overlay",
    )
    assert initial_experts(layout, COUNTS) == [2, 0, 5, 4, 6]
    assert initial_experts(layout, None) == [1, 0, 6, 5, 4]
    unified = plan_slot_layout(
        5,
        uniform(2),
        4,
        1,
        grouping="unified",
        counts=COUNTS,
        prefill_overlap=False,
        buffer_placement="overlay",
    )
    assert initial_experts(unified, COUNTS) == [5, 4, 2, 6, 0]
    assert initial_experts(unified, None) == [2, 5, 1, 4, 0]


def test_mixed_layouts() -> None:
    # Layer 0 in 8 bits, layers 1-5 in 4: runs of 4, 4, 3, 3, 3, 3 over 20 slots.
    layouts = [tiny_layout(8)] + uniform(5)
    options = {"counts": None, "prefill_overlap": True, "buffer_placement": "overlay"}
    per_layer = plan_slot_layout(20, layouts, 4, 1, grouping="per-layer", **options)
    # The widest slots first, then the smallest pools.
    assert [pool.layers for pool in per_layer.pools] == [[0], [2], [3], [4], [5], [1]]
    weight = "gate_up_proj.weight"
    assert [pool.offsets[weight] for pool in per_layer.pools] == [
        0,
        32768,
        45056,
        57344,
        69632,
        81920,
    ]
    # Each buffer holds 4 rows of 8192 bytes: the first 65536 bytes of the weight bank lie
    # under them, layer 0's 4 rows and 3 + 3 + 2 rows of 4096 bytes after; no other bank
    # covers more of a pool.
    assert per_layer.buffer_bytes[weight] == 32768
    assert [pool.volatile_slots for pool in per_layer.pools] == [4, 3, 3, 2, 0, 0]
    assert per_layer.bank_bytes[weight] == 81920 + 4 * 4096
    # A bank is rows of the greatest common divisor of its layouts' rows.
    assert per_layer.bank_widths[weight] == 4096
    unified = plan_slot_layout(20, layouts, 4, 1, grouping="unified", **options)
    assert [(pool.layers, pool.num_slots) for pool in unified.pools] == [
        ([0], 4),
        ([1, 2, 3, 4, 5], 16),
    ]
    dedicated = plan_slot_layout(
        23,
        layouts,
        4,
        1,
        grouping="per-layer",
        counts=None,
        prefill_overlap=True,
        buffer_placement="dedicated",
    )
    assert dedicated.pools[0].offsets[weight] == 2 * 32768
    assert all(pool.volatile_slots == 0 for pool in dedicated.pools)


def test_unified_groups_by_quant_type() -> None:
    # One row format in two quant types (as GGUF's Q4_0 and Q4_K) makes two pools.
    other = {
        role: dataclasses.replace(spec, quant=AffineScheme(4, 32))
        for role, spec in tiny_layout(4).items()
    }
    layout = plan_slot_layout(
        8,
        uniform(2) + [other],
        4,
        1,
        grouping="unified",
        counts=None,
        prefill_overlap=False,
        buffer_placement="overlay",
    )
    assert [pool.layers for pool in layout.pools] == [[2], [0, 1]]


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
