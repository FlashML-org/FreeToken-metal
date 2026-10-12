"""The slot layout of the routed-expert offload cache: pools of slots and prefill buffers.

The MoE layers are grouped into pools (unified: every layer of an expert layout, of which the
cache holds one; per-layer: one layer each), each a run of slots with an LRU of its own
(SlotLru), sized evenly per layer or by a hot-expert table (how often each layer's experts
are routed: the model directory's own, profiled locally, else the one shipped for the model)
summed over its layers. The two prefill buffers hold a whole layer each, over the first
slots or before the pools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# A hot-expert table in a model's directory: {"num_layers", "num_experts", "counts"}, counts
# [num_layers][num_experts] the times each expert was routed.
HOT_EXPERTS_FILE = "expert_hot_rank.json"
BUILTIN_HOT_EXPERTS = Path(__file__).parent / "hot_experts"

# Which layers share a pool: every layer of an expert layout, or each layer alone.
MoePoolGrouping = Literal["unified", "per-layer"]
# How the slots are split among the pools: evenly per layer, or by a hot-expert table.
MoePoolSlots = Literal["equal", "profile"]
# Where the prefill buffers go: over the first pools' slots, or before the pools.
PrefillBufferPlacement = Literal["overlay", "dedicated"]


@dataclass(frozen=True, kw_only=True)
class SlotPool:
    layers: list[int]
    first_slot: int
    num_slots: int


@dataclass(frozen=True, kw_only=True)
class SlotLayout:
    """The pools in slot order and the slots of each prefill buffer (0: none); the buffers
    take slots [0, 2 * buffer_size), over the first pools or before them."""

    pools: list[SlotPool]
    buffer_size: int
    num_experts: int
    top_k: int

    @property
    def num_slots(self) -> int:
        return self.pools[-1].first_slot + self.pools[-1].num_slots

    @property
    def layer_pools(self) -> list[int]:
        """Each layer's pool, an index into pools."""
        num_layers = sum(len(pool.layers) for pool in self.pools)
        pools = [0] * num_layers
        for pool_id, pool in enumerate(self.pools):
            for layer_id in pool.layers:
                pools[layer_id] = pool_id
        return pools

    @property
    def pool_bounds(self) -> list[int]:
        """The pools' first slots, then num_slots."""
        return [pool.first_slot for pool in self.pools] + [self.num_slots]

    @property
    def max_decode_rows(self) -> int | None:
        """The most rows a decode may take: then even all-distinct experts fit every pool's
        layers at once, and the decode runs as one step, as a graph replay must. None: every
        pool holds all its layers' experts."""
        caps = [
            pool.num_slots // (len(pool.layers) * self.top_k)
            for pool in self.pools
            if pool.num_slots < len(pool.layers) * self.num_experts
        ]
        return min(caps) if caps else None

    @property
    def max_lookup_tokens(self) -> int | None:
        """The most tokens a prefill that looks its experts up may take: then even
        all-distinct experts of one layer fit its pool (step_starts splits the layers). None:
        every pool holds a layer's experts."""
        caps = [
            pool.num_slots // self.top_k for pool in self.pools if pool.num_slots < self.num_experts
        ]
        return min(caps) if caps else None

    def step_starts(self, num_rows: int) -> list[int]:
        """The layers that start a new step of a lookup forward of num_rows rows: each step's
        layers hold their worst case, all-distinct experts, in their pools at once."""
        experts_per_layer = min(self.num_experts, num_rows * self.top_k)
        used = [0] * len(self.pools)
        starts: list[int] = []
        for layer_id, pool_id in enumerate(self.layer_pools):
            if used[pool_id] + experts_per_layer > self.pools[pool_id].num_slots:
                starts.append(layer_id)
                used = [0] * len(self.pools)
            used[pool_id] += experts_per_layer
        return starts


def find_expert_counts(
    model_path: str, model_type: str, num_layers: int, num_experts: int, hidden_size: int
) -> Path | None:
    """The model directory's hot-expert table, else the one shipped for this model."""
    for path in (
        Path(model_path) / HOT_EXPERTS_FILE,
        BUILTIN_HOT_EXPERTS / f"{model_type}-{num_layers}x{num_experts}-h{hidden_size}.json",
    ):
        if path.exists():
            return path
    return None


def read_expert_counts(path: Path, num_layers: int, num_experts: int) -> np.ndarray:
    """The routing counts int64 [num_layers, num_experts] of a hot-expert table."""
    table = json.loads(path.read_text())
    counts = np.array(table["counts"], dtype=np.int64)
    if counts.shape != (num_layers, num_experts):
        raise ValueError(
            f"{path}: counts are {counts.shape}, the model has {num_layers} MoE layers of "
            f"{num_experts} experts"
        )
    return counts


def split_runs(
    num_slots: int, num_layers: int, num_experts: int, top_k: int, counts: np.ndarray | None
) -> list[int]:
    """num_slots over the layers: one token's experts each, then the rest to the most routed
    (layer, expert) pairs of counts beyond those, or evenly without counts."""
    if num_slots >= num_layers * num_experts:
        return [num_experts] * num_layers
    if counts is None:
        base, extra = divmod(num_slots, num_layers)
        return [base + (layer_id < extra) for layer_id in range(num_layers)]
    floor = min(top_k, num_experts)
    rank_in_layer = np.argsort(np.argsort(-counts, axis=1, kind="stable"), axis=1, kind="stable")
    rest = np.where(rank_in_layer < floor, -1, counts).ravel()
    hottest = np.argsort(-rest, kind="stable")[: num_slots - num_layers * floor]
    return (floor + np.bincount(hottest // num_experts, minlength=num_layers)).tolist()


def min_cache_size(
    num_layers: int,
    num_experts: int,
    top_k: int,
    *,
    prefill_overlap: bool,
    buffer_placement: PrefillBufferPlacement,
) -> int:
    """The fewest slots a layout takes: one token's experts per layer, and the two prefill
    buffers of a whole layer each over them or beside them."""
    floor = num_layers * min(top_k, num_experts)
    if not prefill_overlap:
        return floor
    elif buffer_placement == "overlay":
        return max(floor, 2 * num_experts)
    else:
        return floor + 2 * num_experts


def plan_slot_layout(
    cache_size: int,
    num_layers: int,
    num_experts: int,
    top_k: int,
    *,
    grouping: MoePoolGrouping,
    counts: np.ndarray | None,
    prefill_overlap: bool,
    buffer_placement: PrefillBufferPlacement,
) -> SlotLayout:
    """cache_size slots (fewer once every expert fits) as pools and, with prefill overlap
    unless every expert fits, two prefill buffers of a whole layer each."""
    floor = min_cache_size(
        num_layers,
        num_experts,
        top_k,
        prefill_overlap=prefill_overlap,
        buffer_placement=buffer_placement,
    )
    if cache_size < floor:
        buffers = " and the two prefill buffers" if prefill_overlap else ""
        raise ValueError(
            f"{cache_size} expert slots cannot hold one token's experts per layer{buffers}: give "
            f"at least {floor}"
        )
    if prefill_overlap and cache_size < num_layers * num_experts:
        buffer_size = num_experts
    else:
        buffer_size = 0
    is_dedicated = buffer_placement == "dedicated"
    runs = split_runs(
        cache_size - (2 * buffer_size if is_dedicated else 0),
        num_layers,
        num_experts,
        top_k,
        counts,
    )
    if grouping == "unified":
        groups = [list(range(num_layers))]
    else:
        groups = [[layer_id] for layer_id in range(num_layers)]
    pool_slots = [sum(runs[layer_id] for layer_id in group) for group in groups]
    # Smallest first: overlaid buffers cover the runs whose experts their layers miss least.
    order = sorted(range(len(groups)), key=lambda pool_id: (pool_slots[pool_id], groups[pool_id]))
    first_slot = 2 * buffer_size if is_dedicated else 0
    pools: list[SlotPool] = []
    for pool_id in order:
        pools.append(
            SlotPool(layers=groups[pool_id], first_slot=first_slot, num_slots=pool_slots[pool_id])
        )
        first_slot += pool_slots[pool_id]
    return SlotLayout(pools=pools, buffer_size=buffer_size, num_experts=num_experts, top_k=top_k)


def initial_experts(layout: SlotLayout, counts: np.ndarray | None) -> list[int]:
    """Flat ids filling every pool with its layers' most routed experts (the lowest-numbered,
    layer by layer, without counts), coldest first: an overlaid buffer covers a pool's first
    slots."""
    num_experts = layout.num_experts
    flat_ids: list[int] = []
    for pool in layout.pools:
        pairs = [(layer_id, expert) for expert in range(num_experts) for layer_id in pool.layers]
        if counts is not None:
            pairs.sort(key=lambda pair: (-counts[pair], pair))
        flat_ids += [
            layer_id * num_experts + expert
            for layer_id, expert in reversed(pairs[: pool.num_slots])
        ]
    return flat_ids


__all__ = [
    "HOT_EXPERTS_FILE",
    "MoePoolGrouping",
    "MoePoolSlots",
    "PrefillBufferPlacement",
    "SlotLayout",
    "SlotPool",
    "find_expert_counts",
    "initial_experts",
    "min_cache_size",
    "plan_slot_layout",
    "read_expert_counts",
]
