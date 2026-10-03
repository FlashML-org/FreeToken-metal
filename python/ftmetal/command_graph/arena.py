"""GraphArena: the memory the captured graphs share, as CUDA graphs share a memory pool.

plan_arena packs a capture's blocks (intermediate arrays and MLX temporaries) into an
arena by their barrier-segment lifetimes, and place_blocks rebinds them there and
releases them. Every graph binds one arena, as large as the largest plan. A temporary
a replay reads before it writes it (a host-initialized constant, found by the poison
test) stays pinned instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import mlx.core as mx
import numpy as np

from ftmetal.command_graph import _C

logger = logging.getLogger(__name__)

MIB = 1 << 20

# Arena offsets and sizes are multiples of this: Metal buffer offsets and the vector
# loads of MLX's kernels are aligned at it.
ARENA_ALIGNMENT = 256


class Placement(NamedTuple):
    offset: int
    # The block's size rounded up to ARENA_ALIGNMENT.
    size: int


@dataclass(frozen=True, kw_only=True)
class ArenaPlan:
    placements: dict[int, Placement]
    arena_bytes: int
    # The most bytes live in one barrier segment: no valid plan is smaller.
    lower_bound: int


def plan_arena(
    block_sizes: dict[int, int], dispatch_blocks: list[list[int]], separated: list[bool]
) -> ArenaPlan:
    """Pack blocks into one arena, largest first, each at its lowest free offset.

    dispatch_blocks lists the blocks each dispatch binds in replay order (ids missing
    from block_sizes are not placed); separated[d] is a barrier or an encoder boundary
    before dispatch d. Dispatches between two barriers may run concurrently, so a block
    lives from the barrier segment of its first binding to that of its last, and two
    blocks share bytes only when their segment ranges are disjoint.
    """
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    segment = 0
    for d, blocks in enumerate(dispatch_blocks):
        segment += d > 0 and separated[d]
        for block in blocks:
            if block in block_sizes:
                first.setdefault(block, segment)
                last[block] = segment
    order = sorted(
        first, key=lambda block: (-block_sizes[block], first[block] - last[block], first[block])
    )
    sizes = np.array(
        [-(-block_sizes[block] // ARENA_ALIGNMENT) * ARENA_ALIGNMENT for block in order],
        dtype=np.int64,
    )
    firsts = np.array([first[block] for block in order], dtype=np.int64)
    lasts = np.array([last[block] for block in order], dtype=np.int64)
    starts = np.zeros(len(order), dtype=np.int64)
    ends = np.zeros(len(order), dtype=np.int64)
    placements: dict[int, Placement] = {}
    for i, block in enumerate(order):
        size = int(sizes[i])
        live = (firsts[:i] <= lasts[i]) & (lasts[:i] >= firsts[i])
        offset = 0
        for start, end in sorted(zip(starts[:i][live].tolist(), ends[:i][live].tolist())):
            if start >= offset + size:
                break
            offset = max(offset, end)
        starts[i], ends[i] = offset, offset + size
        placements[block] = Placement(offset, size)
    load = np.zeros(segment + 2, dtype=np.int64)
    np.add.at(load, firsts, sizes)
    np.subtract.at(load, lasts + 1, sizes)
    return ArenaPlan(
        placements=placements,
        arena_bytes=int(ends.max(initial=0)),
        lower_bound=int(np.cumsum(load).max(initial=0)),
    )


def plan_graph_arena(graph: _C.Graph, pinned: list[int]) -> ArenaPlan:
    """plan_arena over a capture's blocks but the pinned ones."""
    sizes = graph.block_sizes
    encoders = graph.dispatch_encoders
    separated = [
        barrier or (d > 0 and encoders[d] != encoders[d - 1])
        for d, barrier in enumerate(graph.dispatch_barriers)
    ]
    block_sizes = {block: size for block, size in enumerate(sizes) if block not in pinned}
    return plan_arena(block_sizes, graph.dispatch_blocks, separated)


def place_blocks(graph: _C.Graph, plan: ArenaPlan) -> None:
    """Rebind the planned blocks into the arena to come (bind_arena) and release them; a
    block the plan leaves out must be a temporary, and stays pinned."""
    placements = plan.placements
    offsets = [
        placements[block].offset if block in placements else -1
        for block in range(len(graph.block_sizes))
    ]
    _C.place_blocks(graph, offsets)


def bit_equal(first: mx.array, second: mx.array) -> bool:
    return (
        first.shape == second.shape
        and first.dtype == second.dtype
        and mx.array_equal(first.view(mx.uint8), second.view(mx.uint8)).item() is True
    )


class CaptureBatchReplay(Protocol):
    """One replay of a graph's own capture batch from a reset state, evaluated; returns
    its logits."""

    def __call__(self) -> mx.array: ...


@dataclass(frozen=True, kw_only=True)
class ReplayCheck:
    """A graph's capture batch replay, and the logits of a clean replay of it."""

    replay: CaptureBatchReplay
    logits: mx.array


class GraphArena:
    """The memory the captured graphs share: add each graph after its capture, then bind."""

    def __init__(self) -> None:
        self.plans: dict[_C.Graph, ArenaPlan] = {}
        # What bind checks each graph with.
        self.checks: dict[_C.Graph, ReplayCheck] = {}
        # The uint8 arena every graph binds, allocated by bind.
        self.buffer: mx.array | None = None

    def add(self, graph: _C.Graph, replay: CaptureBatchReplay) -> None:
        """Plan graph's blocks into the arena and release them but the temporaries the poison
        test pins: filled with 0xFF, each changes replay's logits. One replay poisons them
        all; only when that one differs is each tested alone."""
        logits = replay()
        temporaries = graph.temporary_blocks

        def logits_change(blocks: list[int]) -> bool:
            _C.poison_blocks(graph, blocks)
            changed = not bit_equal(replay(), logits)
            _C.restore_blocks(graph)
            return changed

        if temporaries and logits_change(temporaries):
            pinned = [block for block in temporaries if logits_change([block])]
            labels = [graph.block_labels[block] for block in pinned]
            logger.warning(f"CommandGraph: {len(pinned)} temporaries stay pinned: {labels}")
        else:
            pinned = []
        plan = plan_graph_arena(graph, pinned)
        place_blocks(graph, plan)
        self.plans[graph] = plan
        self.checks[graph] = ReplayCheck(replay=replay, logits=logits)
        stats = graph.stats
        logger.info(
            f"CommandGraph batch size {logits.shape[0]}: {stats.num_dispatches} dispatches, "
            f"{len(stats.temporaries)} temporaries ({len(pinned)} fail the poison test), "
            f"arena {plan.arena_bytes / MIB:.2f} MiB (lower bound "
            f"{plan.lower_bound / MIB:.2f} MiB), pinned "
            f"{stats.capture_pinned_bytes / MIB:.1f} -> {stats.pinned_bytes / MIB:.3f} MiB"
        )

    def bind(self) -> None:
        """Allocate the arena once and bind every graph to it; each graph's replay on the
        arena filled with 0xFF must give its clean replay's logits."""
        buffer = mx.zeros((max(plan.arena_bytes for plan in self.plans.values()),), mx.uint8)
        mx.eval(buffer)
        self.buffer = buffer
        for graph, check in self.checks.items():
            _C.bind_arena(graph, buffer)
            _C.poison_arena(graph)
            if not bit_equal(check.replay(), check.logits):
                raise RuntimeError(
                    f"CommandGraph batch size {check.logits.shape[0]}: a replay on the "
                    f"poisoned arena differs from its clean replay"
                )
        # The replays hold their capture batches.
        self.checks.clear()

    @property
    def nbytes(self) -> int:
        """What the graphs keep once bound: the arena they share and their pinned buffers."""
        arena_bytes = max((plan.arena_bytes for plan in self.plans.values()), default=0)
        return arena_bytes + sum(graph.stats.pinned_bytes for graph in self.plans)


__all__ = [
    "ARENA_ALIGNMENT",
    "ArenaPlan",
    "CaptureBatchReplay",
    "GraphArena",
    "Placement",
    "place_blocks",
    "plan_arena",
    "plan_graph_arena",
]
