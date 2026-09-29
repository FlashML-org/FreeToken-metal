"""Per-call time of paged flash-decode at the Qwen3.6 attention shapes.

    uv run python tools/bench_fdecode.py [--lengths 1024,2048,8192,32768] [--batch-size 1]

One call is one full-attention layer of a decode batch: bf16 q [B, 16, 256] over a
[slots, 2, 256] K/V pool at page size 1, each request on its own contiguous slots. Each
length runs the same batch twice: with max_seqlen_k = the length (the partition count
eager decode picks) and with max_seqlen_k = --fixed-max-seq-len and a page table that
wide (the fixed shapes a replayed graph carries). Calls chain through the output, as a
layer waits for the one before it, and each length reports the best mean time per call
over the repeats.
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx
import numpy as np

from ftmetal.kernel.fdecode import flash_decode

NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
PAGE_SIZE = 1
CALLS_PER_RUN = 200


def time_per_call(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    seq_lens: mx.array,
    max_seqlen_k: int,
    repeats: int,
) -> float:
    """Best mean seconds per call over repeats of CALLS_PER_RUN calls chained through q."""
    scale = HEAD_DIM**-0.5
    mx.eval(flash_decode(q, k_cache, v_cache, page_table, seq_lens, max_seqlen_k, PAGE_SIZE, scale))
    best = float("inf")
    for _ in range(repeats):
        out = q
        started = time.perf_counter()
        for _ in range(CALLS_PER_RUN):
            out = flash_decode(
                out, k_cache, v_cache, page_table, seq_lens, max_seqlen_k, PAGE_SIZE, scale
            )
        mx.eval(out)
        best = min(best, (time.perf_counter() - started) / CALLS_PER_RUN)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="1024,2048,8192,32768")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fixed-max-seq-len", type=int, default=262144)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    lengths = [int(length) for length in args.lengths.split(",")]
    num_slots = args.batch_size * max(lengths)
    mx.random.seed(0)
    k_cache = mx.random.normal((num_slots, NUM_KV_HEADS, HEAD_DIM)).astype(mx.bfloat16)
    v_cache = mx.random.normal((num_slots, NUM_KV_HEADS, HEAD_DIM)).astype(mx.bfloat16)
    q = mx.random.normal((args.batch_size, NUM_HEADS, HEAD_DIM)).astype(mx.bfloat16)
    mx.eval(k_cache, v_cache, q)
    starts = np.arange(args.batch_size, dtype=np.int32)[:, None] * max(lengths)
    print(f"batch size {args.batch_size}; microseconds per call", flush=True)
    for seq_len in lengths:
        seq_lens = mx.full((args.batch_size,), seq_len, dtype=mx.int32)
        timings: list[float] = []
        for max_seqlen_k in (seq_len, args.fixed_max_seq_len):
            page_table = np.zeros((args.batch_size, max_seqlen_k), dtype=np.int32)
            page_table[:, :seq_len] = starts + np.arange(seq_len, dtype=np.int32)
            seconds = time_per_call(
                q, k_cache, v_cache, mx.array(page_table), seq_lens, max_seqlen_k, args.repeats
            )
            timings.append(seconds)
        per_length, fixed = timings
        print(
            f"seq_len {seq_len}: per-length {per_length * 1e6:.1f} us, "
            f"fixed {args.fixed_max_seq_len} {fixed * 1e6:.1f} us ({fixed / per_length:.2f}x)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
