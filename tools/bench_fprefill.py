"""Per-call time of paged flash-attention prefill at the Qwen3.6 attention shapes.

    uv run python tools/bench_fprefill.py [--shapes 0:2048,30720:64] [--repeats 5]

One call is one full-attention layer of a one-request prefill chunk: bf16 q [extend, 16,
256] over a [slots, 2, 256] K/V pool at page size 1, the request's keys on shuffled
slots. Each (cached, extend) shape times flash_prefill against the path it replaces:
gather the cached prefix out of the pool, concatenate the chunk's own keys and run one
mx.fast.scaled_dot_product_attention. Calls chain through the output, as a layer waits
for the one before it, and each shape reports the best mean time per call over the
repeats, with the achieved K/V read bandwidth (each key read once) and the TFLOP/s of
the causal QK^T and PV products.
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx
import numpy as np

from ftmetal.kernel.fprefill import flash_prefill, prefill_tiles

NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
PAGE_SIZE = 1
CALLS_PER_RUN = 20


class PrefillCall:
    """One layer's prefill attention by either path, over the same pool and keys."""

    def __init__(self, cached_len: int, extend_len: int, rng: np.random.Generator) -> None:
        seq_len = cached_len + extend_len
        self.cached_len = cached_len
        self.extend_len = extend_len
        self.slots = rng.permutation(seq_len).astype(np.int32)
        self.page_table = mx.array(self.slots[None])
        self.k_cache = mx.random.normal((seq_len, NUM_KV_HEADS, HEAD_DIM)).astype(mx.bfloat16)
        self.v_cache = mx.random.normal((seq_len, NUM_KV_HEADS, HEAD_DIM)).astype(mx.bfloat16)
        fresh = mx.array(self.slots[cached_len:])
        self.prefix = mx.array(self.slots[:cached_len])
        self.k = self.k_cache[fresh]
        self.v = self.v_cache[fresh]
        # Planned once per batch, as the engine does for all of a step's layers.
        self.tiles = prefill_tiles([extend_len], [cached_len], NUM_HEADS, NUM_KV_HEADS)
        mx.eval(self.page_table, self.k_cache, self.v_cache, self.prefix, self.k, self.v)

    def kernel(self, q: mx.array) -> mx.array:
        return flash_prefill(
            q,
            self.k_cache,
            self.v_cache,
            self.page_table,
            self.tiles,
            PAGE_SIZE,
            HEAD_DIM**-0.5,
        )

    def sdpa(self, q: mx.array) -> mx.array:
        if self.cached_len > 0:
            keys = mx.concatenate([self.k_cache[self.prefix], self.k])
            values = mx.concatenate([self.v_cache[self.prefix], self.v])
        else:
            keys, values = self.k, self.v
        out = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            keys.transpose(1, 0, 2)[None],
            values.transpose(1, 0, 2)[None],
            scale=HEAD_DIM**-0.5,
            mask="causal",
        )
        return out[0].transpose(1, 0, 2)


def time_per_call(call: PrefillCall, use_kernel: bool, q: mx.array, repeats: int) -> float:
    """Best mean seconds per call over repeats of CALLS_PER_RUN calls chained through q."""
    attend = call.kernel if use_kernel else call.sdpa
    mx.eval(attend(q))
    best = float("inf")
    for _ in range(repeats):
        out = q
        started = time.perf_counter()
        for _ in range(CALLS_PER_RUN):
            out = attend(out)
        mx.eval(out)
        best = min(best, (time.perf_counter() - started) / CALLS_PER_RUN)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="0:2048,2048:2048,6144:2048,30720:2048,30720:64")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    mx.random.seed(0)
    print("cached:extend  kernel us (GB/s, TFLOP/s)  sdpa+gather us (GB/s, TFLOP/s)  speedup")
    for shape in args.shapes.split(","):
        cached_len, extend_len = (int(part) for part in shape.split(":"))
        call = PrefillCall(cached_len, extend_len, rng)
        q = mx.random.normal((extend_len, NUM_HEADS, HEAD_DIM)).astype(mx.bfloat16)
        mx.eval(q)
        kv_bytes = 2 * (cached_len + extend_len) * NUM_KV_HEADS * HEAD_DIM * 2
        visible_pairs = extend_len * cached_len + extend_len * (extend_len + 1) // 2
        flops = 4 * NUM_HEADS * HEAD_DIM * visible_pairs
        timings = [time_per_call(call, use_kernel, q, args.repeats) for use_kernel in (True, False)]
        cells = [
            f"{seconds * 1e6:9.0f} ({kv_bytes / seconds / 1e9:5.1f}, {flops / seconds / 1e12:4.2f})"
            for seconds in timings
        ]
        print(
            f"{cached_len:>6}:{extend_len:<6}  {cells[0]}  {cells[1]}  "
            f"{timings[1] / timings[0]:.2f}x",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
