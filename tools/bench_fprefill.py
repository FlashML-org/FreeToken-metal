"""Per-call time of paged flash-attention prefill at the Qwen3.6 (default) or a given shape.

    uv run python tools/bench_fprefill.py [--shapes 0:2048,30720:64] [--repeats 5]
        [--num-heads 16 --num-kv-heads 2 --head-dim 256] [--sliding-window 1024]
        [--mma simdgroup|mpp]

One call is one full-attention layer of a one-request prefill chunk: bf16 q [extend, Hq,
D] over a [slots, Hkv, D] K/V pool at page size 1, the request's keys on shuffled slots.
Each (cached, extend) shape times flash_prefill against the path it replaces: gather the
cached prefix out of the pool, concatenate the chunk's own keys and run one
mx.fast.scaled_dot_product_attention (with a window, under the banded mask). Calls chain
through the output, as a layer waits for the one before it, and each shape reports the
best mean time per call over the repeats, with the achieved K/V read bandwidth (each key
a query can see read once) and the TFLOP/s of the visible QK^T and PV products. With a
sliding window the page-table row starts at the chunk's oldest visible key, as the SWA
backend builds it. Gemma4-26B-A4B: global layers --head-dim 512, window layers
--num-kv-heads 8 --head-dim 256 --sliding-window 1024. --mma picks the kernel's GEMMs
(default: the engine's choice for this GPU).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ftmetal.kernel.fprefill import FprefillMma, default_mma, flash_prefill, prefill_tiles

PAGE_SIZE = 1
CALLS_PER_RUN = 20


@dataclass(frozen=True)
class AttentionShape:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window: int | None
    mma: FprefillMma


class PrefillCall:
    """One layer's prefill attention by either path, over the same pool and keys."""

    def __init__(
        self, shape: AttentionShape, cached_len: int, extend_len: int, rng: np.random.Generator
    ) -> None:
        seq_len = cached_len + extend_len
        window = shape.sliding_window
        first_key = 0 if window is None else max(0, cached_len - window + 1)
        self.shape = shape
        self.cached_len = cached_len - first_key
        self.extend_len = extend_len
        self.slots = rng.permutation(seq_len).astype(np.int32)[first_key:]
        self.page_table = mx.array(self.slots[None])
        kv_shape = (seq_len, shape.num_kv_heads, shape.head_dim)
        self.k_cache = mx.random.normal(kv_shape).astype(mx.bfloat16)
        self.v_cache = mx.random.normal(kv_shape).astype(mx.bfloat16)
        fresh = mx.array(self.slots[self.cached_len :])
        self.prefix = mx.array(self.slots[: self.cached_len])
        self.k = self.k_cache[fresh]
        self.v = self.v_cache[fresh]
        self.mask: mx.array | str = "causal"
        if window is not None:
            positions = mx.arange(self.cached_len, len(self.slots))[:, None]
            key_positions = mx.arange(len(self.slots))[None, :]
            self.mask = (key_positions <= positions) & (key_positions > positions - window)
        # Planned once per batch, as the engine does for all of a step's layers.
        self.tiles = prefill_tiles(
            [extend_len], [self.cached_len], shape.num_heads, shape.num_kv_heads, window
        )
        mx.eval(self.page_table, self.k_cache, self.v_cache, self.prefix, self.k, self.v)

    def kernel(self, q: mx.array) -> mx.array:
        return flash_prefill(
            q,
            self.k_cache,
            self.v_cache,
            self.page_table,
            self.tiles,
            PAGE_SIZE,
            self.shape.head_dim**-0.5,
            self.shape.sliding_window,
            self.shape.mma,
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
            scale=self.shape.head_dim**-0.5,
            mask=self.mask,
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
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--mma", choices=["simdgroup", "mpp"], default=None)
    args = parser.parse_args()
    mma = default_mma() if args.mma is None else args.mma
    shape = AttentionShape(
        args.num_heads, args.num_kv_heads, args.head_dim, args.sliding_window, mma
    )
    window = shape.sliding_window
    rng = np.random.default_rng(0)
    mx.random.seed(0)
    print(f"kernel GEMMs: {mma}")
    print("cached:extend  kernel us (GB/s, TFLOP/s)  sdpa+gather us (GB/s, TFLOP/s)  speedup")
    for shape_text in args.shapes.split(","):
        cached_len, extend_len = (int(part) for part in shape_text.split(":"))
        call = PrefillCall(shape, cached_len, extend_len, rng)
        q = mx.random.normal((extend_len, shape.num_heads, shape.head_dim)).astype(mx.bfloat16)
        mx.eval(q)
        kv_bytes = 2 * len(call.slots) * shape.num_kv_heads * shape.head_dim * 2
        visible = np.arange(cached_len + 1, cached_len + extend_len + 1)
        visible_pairs = int((visible if window is None else np.minimum(visible, window)).sum())
        flops = 4 * shape.num_heads * shape.head_dim * visible_pairs
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
