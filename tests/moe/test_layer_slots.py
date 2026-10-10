"""The offload cache's per-layer slot split and its hot-expert tables.

split_layer_slots must give every layer one token's experts, the rest evenly or to the most
routed (layer, expert) pairs, every expert when the slots allow, and windows holding the
most experts a run leaves out; too few slots must be refused, from min_cache_size down.
hottest_experts fills each run with its most routed experts. load_expert_counts prefers the
model directory's table to the shipped one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from ftmetal.moe.layer_slots import (
    HOT_EXPERTS_FILE,
    hottest_experts,
    load_expert_counts,
    min_cache_size,
    split_layer_slots,
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
    prefill_overlap: bool
    layer_slots: list[int]
    window_size: int


CASES = [
    Case("even", 12, 3, 8, 2, None, False, [4, 4, 4], 0),
    Case("even-remainder", 14, 3, 8, 2, None, False, [5, 5, 4], 0),
    Case("all-resident", 24, 3, 8, 2, None, True, [8, 8, 8], 0),
    # 14 slots: runs of 4, 4, 3, 3 need windows of 1, which leave runs of 3.
    Case("windows-iterate", 14, 4, 4, 2, None, True, [3, 3, 3, 3], 1),
    # Each layer's hottest expert first (layer 0's 0, layer 1's 2), then the three most
    # routed pairs left: layer 0's 2, layer 1's 0 and 1.
    Case("hot", 5, 2, 4, 1, COUNTS, False, [2, 3], 0),
    # Qwen3.6's floor: one token's experts per layer and two windows of the rest.
    Case("qwen-floor", 816, 40, 256, 8, None, True, [8] * 40, 248),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_split_layer_slots(case: Case) -> None:
    split = split_layer_slots(
        case.cache_size,
        case.num_layers,
        case.num_experts,
        case.top_k,
        case.counts,
        case.prefill_overlap,
    )
    assert (split.layer_slots, split.window_size) == (case.layer_slots, case.window_size)
    assert split.num_slots <= case.cache_size


def test_too_few_slots_are_refused() -> None:
    assert min_cache_size(40, 256, 8, True) == 816
    with pytest.raises(ValueError, match="cannot give each of 40 MoE layers the 8 experts"):
        split_layer_slots(815, 40, 256, 8, None, True)
    with pytest.raises(ValueError, match="cannot give each of 3 MoE layers the 2 experts"):
        split_layer_slots(5, 3, 8, 2, None, False)


def test_hottest_experts() -> None:
    assert hottest_experts([2, 3], 4, COUNTS) == [0, 2, 6, 4, 5]
    assert hottest_experts([2, 1], 4, None) == [0, 1, 4]


def test_model_table_before_shipped(tmp_path: Path) -> None:
    shipped = load_expert_counts(str(tmp_path), "qwen3_5_moe", 40, 256, 2048)
    assert shipped is not None and shipped[0].shape == (40, 256)
    assert shipped[1].name == "qwen3_5_moe-40x256-h2048.json"
    assert load_expert_counts(str(tmp_path), "qwen3_5_moe", 40, 256, 4096) is None
    table = {"num_layers": 2, "num_experts": 4, "counts": COUNTS.tolist()}
    (tmp_path / HOT_EXPERTS_FILE).write_text(json.dumps(table))
    local = load_expert_counts(str(tmp_path), "qwen3_5_moe", 2, 4, 64)
    assert local is not None and local[1] == tmp_path / HOT_EXPERTS_FILE
    assert (local[0] == COUNTS).all()
    with pytest.raises(ValueError, match="the model has 3 MoE layers of 4 experts"):
        load_expert_counts(str(tmp_path), "qwen3_5_moe", 3, 4, 64)
