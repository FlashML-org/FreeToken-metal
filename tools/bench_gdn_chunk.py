"""Per-layer-call time of the chunked gated delta rule at the Qwen3.6 GDN shapes.

    uv run python tools/bench_gdn_chunk.py [--lengths 64,512,2048,8192] [--repeats 5]

One call is one GDN layer of one request: bf16 q, k [1, T, 16, 128], v [1, T, 32, 128],
fp32 state [1, 32, 128, 128]. Each call waits for the previous one (its gates depend on
the previous state), as a layer waits for the one before it, so no two calls overlap on
the GPU. Each length reports the best mean time per call over the repeats, for the
chunked kernel and for the per-token recurrence.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Protocol

import mlx.core as mx

from ftmetal.kernel.gdn import gated_delta_recurrence
from ftmetal.kernel.gdn_chunk import gated_delta_chunked

NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_DIM = 128
# Tokens per timed run, so short calls are averaged over many launches.
TOKENS_PER_RUN = 32768
MIN_CALLS_PER_RUN = 4


class GatedDeltaRule(Protocol):
    def __call__(
        self, q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array
    ) -> tuple[mx.array, mx.array]: ...


def make_inputs(seq_len: int) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Normalized keys and a head_dim**-0.5 query scale, as the GDN layer feeds the kernel."""
    mx.random.seed(seq_len)
    shape = (1, seq_len)
    q = mx.random.normal((*shape, NUM_K_HEADS, HEAD_DIM)) * HEAD_DIM**-0.5
    k = mx.random.normal((*shape, NUM_K_HEADS, HEAD_DIM))
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    v = mx.random.normal((*shape, NUM_V_HEADS, HEAD_DIM))
    g = mx.exp(-mx.random.uniform(0.0, 0.5, (*shape, NUM_V_HEADS)))
    beta = mx.sigmoid(mx.random.normal((*shape, NUM_V_HEADS)))
    q, k, v = (x.astype(mx.bfloat16) for x in (q, k, v))
    mx.eval(q, k, v, g, beta)
    return q, k, v, g, beta


def time_per_call(
    rule: GatedDeltaRule,
    inputs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array],
    calls: int,
    repeats: int,
) -> float:
    """Best mean seconds per call over repeats of calls chained through the state."""
    q, k, v, g, beta = inputs
    state = mx.zeros((1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM), dtype=mx.float32)
    mx.eval(rule(q, k, v, g, beta, state))
    best = float("inf")
    for _ in range(repeats):
        outputs = []
        started = time.perf_counter()
        for _ in range(calls):
            y, state = rule(q, k, v, g + 0.0 * state[:, None, :, 0, 0], beta, state)
            outputs.append(y)
        mx.eval(outputs, state)
        best = min(best, (time.perf_counter() - started) / calls)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="64,512,2048,8192")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    for seq_len in (int(length) for length in args.lengths.split(",")):
        inputs = make_inputs(seq_len)
        calls = max(MIN_CALLS_PER_RUN, TOKENS_PER_RUN // seq_len)
        chunked = time_per_call(gated_delta_chunked, inputs, calls, args.repeats)
        per_token = time_per_call(gated_delta_recurrence, inputs, calls, args.repeats)
        print(
            f"T {seq_len}: chunked {chunked * 1e6:.0f} us, per-token {per_token * 1e6:.0f} us "
            f"({per_token / chunked:.2f}x)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
