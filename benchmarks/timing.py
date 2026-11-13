"""Shared timing of the kernel benchmarks: calls chained through their output, best of runs."""

from __future__ import annotations

import time
from typing import Protocol

import mlx.core as mx


class ChainedCall(Protocol):
    def __call__(self, previous: mx.array, /) -> mx.array: ...


def best_seconds_per_call(call: ChainedCall, first: mx.array, calls: int, repeats: int) -> float:
    """Best mean seconds per call over repeats runs of calls, each fed the previous output.

    Chaining serializes the calls on the GPU, as a layer waits for the one before it; a run is
    evaluated once. One call is evaluated before timing, which compiles its kernels.
    """
    mx.eval(call(first))
    best = float("inf")
    for _ in range(repeats):
        output = first
        started = time.perf_counter()
        for _ in range(calls):
            output = call(output)
        mx.eval(output)
        best = min(best, (time.perf_counter() - started) / calls)
    return best
