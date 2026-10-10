"""Per-layer slot counts of the routed-expert offload cache.

No FreeToken donor: FreeToken's offload cache is one LRU over every layer's experts. Here
each MoE layer owns a run of slots (SlotLru), so a miss only ever evicts a slot of its own
layer, which no other layer reads within a step. The runs are sized from a hot-expert table
(how often each layer's experts are routed: the model directory's own, profiled locally,
else the one shipped for the model) or evenly; the two prefill windows hold the most experts
a layer leaves out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# A hot-expert table in a model's directory: {"num_layers", "num_experts", "counts"}, counts
# [num_layers][num_experts] the times each expert was routed.
HOT_EXPERTS_FILE = "expert_hot_rank.json"
BUILTIN_HOT_EXPERTS = Path(__file__).parent / "hot_experts"


@dataclass(frozen=True, kw_only=True)
class SlotSplit:
    """The slots of each layer's run and of each prefill window (0: no windows)."""

    layer_slots: list[int]
    window_size: int

    @property
    def num_slots(self) -> int:
        return 2 * self.window_size + sum(self.layer_slots)


def builtin_hot_experts(
    model_type: str, num_layers: int, num_experts: int, hidden_size: int
) -> Path:
    return BUILTIN_HOT_EXPERTS / f"{model_type}-{num_layers}x{num_experts}-h{hidden_size}.json"


def load_expert_counts(
    model_path: str, model_type: str, num_layers: int, num_experts: int, hidden_size: int
) -> tuple[np.ndarray, Path] | None:
    """The routing counts int64 [num_layers, num_experts] and their file: the model
    directory's table, else the one shipped for this model; None when neither exists."""
    for path in (
        Path(model_path) / HOT_EXPERTS_FILE,
        builtin_hot_experts(model_type, num_layers, num_experts, hidden_size),
    ):
        if path.exists():
            table = json.loads(path.read_text())
            counts = np.array(table["counts"], dtype=np.int64)
            if counts.shape != (num_layers, num_experts):
                raise ValueError(
                    f"{path}: counts are {counts.shape}, the model has {num_layers} MoE layers "
                    f"of {num_experts} experts"
                )
            return counts, path
    return None


def split_runs(
    num_slots: int, num_layers: int, num_experts: int, top_k: int, counts: np.ndarray | None
) -> list[int]:
    """num_slots over the layers: one token's experts each, then the rest to the most routed
    (layer, expert) pairs of counts beyond those, or evenly without counts."""
    floor = min(top_k, num_experts)
    if num_slots < num_layers * floor:
        raise ValueError(
            f"{num_slots} expert slots cannot give each of {num_layers} MoE layers the "
            f"{floor} experts of one token"
        )
    if num_slots >= num_layers * num_experts:
        return [num_experts] * num_layers
    if counts is None:
        base, extra = divmod(num_slots, num_layers)
        return [base + (layer_id < extra) for layer_id in range(num_layers)]
    rank_in_layer = np.argsort(np.argsort(-counts, axis=1, kind="stable"), axis=1, kind="stable")
    rest = np.where(rank_in_layer < floor, -1, counts).ravel()
    hottest = np.argsort(-rest, kind="stable")[: num_slots - num_layers * floor]
    return (floor + np.bincount(hottest // num_experts, minlength=num_layers)).tolist()


def split_layer_slots(
    cache_size: int,
    num_layers: int,
    num_experts: int,
    top_k: int,
    counts: np.ndarray | None,
    prefill_overlap: bool,
) -> SlotSplit:
    """cache_size slots as layer runs (split_runs) and, with prefill overlap, two windows that
    hold the most experts a run leaves out; more windows mean smaller runs, so iterate."""
    window_size = 0
    while True:
        layer_slots = split_runs(
            cache_size - 2 * window_size, num_layers, num_experts, top_k, counts
        )
        needed = num_experts - min(layer_slots) if prefill_overlap else 0
        if needed <= window_size:
            return SlotSplit(layer_slots=layer_slots, window_size=window_size)
        window_size = needed


def min_cache_size(num_layers: int, num_experts: int, top_k: int, prefill_overlap: bool) -> int:
    """The fewest slots split_layer_slots accepts: one token's experts per layer, plus the
    windows those runs need."""
    floor = min(top_k, num_experts)
    return num_layers * floor + (2 * (num_experts - floor) if prefill_overlap else 0)


def hottest_experts(
    layer_slots: list[int], num_experts: int, counts: np.ndarray | None
) -> list[int]:
    """Flat ids of each layer's most routed experts filling its run (the lowest-numbered
    without counts), layer by layer."""
    flat_ids: list[int] = []
    for layer_id, num_slots in enumerate(layer_slots):
        if counts is None:
            experts = range(num_slots)
        else:
            experts = np.argsort(-counts[layer_id], kind="stable")[:num_slots].tolist()
        flat_ids += [layer_id * num_experts + expert for expert in experts]
    return flat_ids


__all__ = [
    "BUILTIN_HOT_EXPERTS",
    "HOT_EXPERTS_FILE",
    "SlotSplit",
    "builtin_hot_experts",
    "hottest_experts",
    "load_expert_counts",
    "min_cache_size",
    "split_layer_slots",
    "split_runs",
]
