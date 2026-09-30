"""CommandGraph: capture one step's Metal dispatches once and replay them as one primitive.

A capture records every dispatch of one mx.eval of a step's lazy logits and updated
pools (csrc/command_graph.mm). A replay encodes the recorded dispatches into MLX's
encoder as a single GraphReplay primitive, rebinding the step inputs, the pools and a
fresh logits buffer. The constant leaves the step binds are pinned for the graph's
lifetime; its blocks (intermediate arrays and MLX temporaries) are pinned until
place_blocks moves them into an arena at the offsets plan_arena packs them to. Shapes,
grids and setBytes payloads are frozen at capture, so a value that changes per step
must reach the step as a declared step input: check_same_graph is the check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx
import numpy as np

from ftmetal.kernel._ext import _C

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


def begin_capture(
    outputs: list[mx.array],
    step_inputs: list[mx.array],
    pools: list[mx.array],
    externals: list[mx.array],
) -> None:
    """Open a capture on this thread; the caller evals outputs next, then calls end_capture.

    outputs are the unevaluated logits followed by the updated pools, in the order of
    pools (each must be updated in place). step_inputs, pools and externals (weights and
    any other long-lived buffer) are evaluated; step inputs are row contiguous.
    """
    _C.begin_capture(outputs, step_inputs, pools, externals)


def end_capture() -> _C.Graph:
    """Close the capture; raises when a recorded call or binding cannot be replayed."""
    return _C.end_capture()


def capture(
    outputs: list[mx.array],
    step_inputs: list[mx.array],
    pools: list[mx.array],
    externals: list[mx.array],
) -> _C.Graph:
    """begin_capture, eval of outputs and end_capture; a failed eval closes the capture."""
    begin_capture(outputs, step_inputs, pools, externals)
    try:
        mx.eval(*outputs)
    except BaseException:
        _C.abort_capture()
        raise
    return end_capture()


def replay(
    graph: _C.Graph, step_inputs: list[mx.array], pools: list[mx.array]
) -> tuple[mx.array, list[mx.array]]:
    """One replay on this step's inputs; returns the logits and the pools, which alias the
    pools passed in (callers rebind them)."""
    logits, *updated = _C.graph_replay(graph, step_inputs, pools)
    return logits, updated


def check_same_graph(first: _C.Graph, second: _C.Graph) -> None:
    """The double-capture check: two captures of one shape with different step inputs must
    record the same dispatches, bindings and setBytes payloads, and every leaf whose
    contents differ must be a declared step input."""
    differences = _C.compare_graphs(first, second)
    if differences:
        raise RuntimeError("the two captures differ:\n  " + "\n  ".join(differences))


def begin_count() -> None:
    """Count encoder and command-buffer calls on this thread without recording them."""
    _C.begin_count()


def end_count() -> _C.CallCounts:
    return _C.end_count()


def wait_until_available(arrays: list[mx.array]) -> None:
    """Busy-wait for evaluated arrays: no sleep, so timing loops keep CPU and GPU clocks up."""
    _C.wait_until_available(arrays)


__all__ = [
    "ARENA_ALIGNMENT",
    "ArenaPlan",
    "Placement",
    "begin_capture",
    "begin_count",
    "capture",
    "check_same_graph",
    "end_capture",
    "end_count",
    "place_blocks",
    "plan_arena",
    "plan_graph_arena",
    "replay",
    "wait_until_available",
]
