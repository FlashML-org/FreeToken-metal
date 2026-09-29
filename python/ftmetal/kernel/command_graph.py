"""CommandGraph: capture one step's Metal dispatches once and replay them as one primitive.

A capture records every dispatch of one mx.eval of a step's lazy logits and updated
pools (csrc/command_graph.mm). A replay encodes the recorded dispatches into MLX's
encoder as a single GraphReplay primitive, rebinding the step inputs, the pools and a
fresh logits buffer; every other buffer the step binds (intermediate arrays, constant
leaves, MLX temporaries) is pinned for the graph's lifetime. Shapes, grids and setBytes
payloads are frozen at capture, so a value that changes per step must reach the step as
a declared step input: check_same_graph is the check.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel._ext import _C


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
    "begin_capture",
    "begin_count",
    "check_same_graph",
    "end_capture",
    "end_count",
    "replay",
    "wait_until_available",
]
