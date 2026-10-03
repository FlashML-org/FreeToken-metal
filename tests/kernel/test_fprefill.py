"""Paged flash-prefill against the fp32 mx.fast.scaled_dot_product_attention reference.

Random K/V pools on shuffled pages; every pool row past a request's keys (page tails,
spare pages, and the spare page that pads each page-table row by a key block) holds NaN,
so a key the kernel should neither stage nor zero-fill shows up in the output. The
reference (flash_prefill_ref) gathers each request's keys and runs SDPA in fp32 with the
causal mask aligned to the request's cached prefix. Each case hits one boundary: a tile
across the causal diagonal, a cached prefix ending inside a key block, a partial last
tile, a group whose heads leave padding rows, each staging plan of the head dim and
dtype, the split-KV rule split and not, forced partitions with nothing to read. Under a
sliding window, the keys before the batch's oldest window hold huge keys and values too,
so a key the window should hide shows up in the output.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel.fprefill import MAX_BLOCK_K, flash_prefill, prefill_tiles
from ftmetal.kernel.ref import flash_prefill_ref

NUM_KV_HEADS = 2
SPARE_PAGES = 3
UNREAD_VALUE = 1e3
# Worst |error| over a row relative to the row's largest |value|. Past fp32 the kernel
# rounds only its bf16 output, half an ulp at the row's top binade (2^-8); one ulp leaves
# room for the fp32 summation order. fp32 inputs differ by summation order alone.
TOLERANCE = {mx.bfloat16: 2**-7, mx.float32: 1e-5}


@dataclass(frozen=True)
class Case:
    """One flash_prefill call: request b has cached_lens[b] keys before extend_lens[b] queries."""

    cached_lens: tuple[int, ...]
    extend_lens: tuple[int, ...]
    head_dim: int = 256
    group: int = 8
    page_size: int = 1
    dtype: mx.Dtype = mx.bfloat16
    # None: prefill_tiles' own split-KV rule.
    num_partitions: int | None = None
    window: int | None = None

    def __str__(self) -> str:
        lens = "+".join(f"{c}:{e}" for c, e in zip(self.cached_lens, self.extend_lens))
        label = f"{lens}-D{self.head_dim}-G{self.group}-p{self.page_size}"
        label += "-bf16" if self.dtype == mx.bfloat16 else "-fp32"
        if self.num_partitions is not None:
            label += f"-parts{self.num_partitions}"
        if self.window is not None:
            label += f"-W{self.window}"
        return label


# Unequal requests in one batch: a prefix ending inside a key block, an empty prefix and a
# long one; a one-key prefix, a block-aligned long prefix and an empty one.
UNEQUAL = ((37, 0, 5000), (100, 2048, 33))
SHORT_AND_LONG = ((1, 2048, 0), (2, 31, 1))
# Windowed batches: a chunk that crosses its first window start, a short turn far past the
# window, requests starting inside, at and before a window and running across window starts.
FROM_EMPTY = ((0,), (2048,))
FAR_TURN = ((5000,), (64,))
ACROSS = ((1023, 0, 3000), (2, 1100, 2048))

CASES = [
    # Qwen3.6's shape (D 256, G 8, page 1, bf16): 4 queries per tile.
    Case((0,), (1,)),
    Case((0,), (37,)),
    Case((1,), (2,)),
    Case((37,), (3,)),  # one partial tile
    Case((37,), (4,)),  # one full tile
    Case((37,), (5,)),  # a full tile and a one-query tile
    Case((2048,), (1,)),  # split by the rule
    Case((5000,), (100,)),
    Case((0,), (2048,)),
    # Tile boundaries per group: 32 // G queries per tile.
    Case((1,), (31,), head_dim=128, group=1, page_size=16),
    Case((1,), (32,), head_dim=128, group=1, page_size=16),
    Case((1,), (33,), head_dim=128, group=1, page_size=16),
    Case((37,), (10,), head_dim=128, group=3, page_size=16),  # two padding rows
    Case((37,), (11,), head_dim=128, group=3, page_size=16),
    Case((37,), (3,), head_dim=128, group=12, page_size=16),  # a padding-only row group
    # Every staging plan, one batch each: D 128 (one simdgroup per row group), D 256 (head
    # split; fp32 stages K and V in one buffer), D 512 (one buffer; fp32 8-key blocks).
    # Page 4 puts a key block across pages, page 64 a page across key blocks.
    Case(*UNEQUAL, head_dim=128, page_size=4),
    Case(*SHORT_AND_LONG, head_dim=128, group=1, page_size=4, dtype=mx.float32),
    Case(*UNEQUAL, group=3, page_size=64),
    Case(*SHORT_AND_LONG, page_size=64, dtype=mx.float32),
    Case(*UNEQUAL, head_dim=512),
    Case(*SHORT_AND_LONG, head_dim=512, page_size=16, dtype=mx.float32),
    Case(*UNEQUAL, head_dim=512, group=1, page_size=16, dtype=mx.float32),
    Case((0,), (37,), head_dim=512, page_size=64),
    Case((2048,), (1,), head_dim=512, dtype=mx.float32),
    # Forced partitions on unequal requests: short requests get partitions with no keys.
    Case(*UNEQUAL, page_size=16, num_partitions=3),
    Case(*SHORT_AND_LONG, page_size=16, num_partitions=3),
    Case(*SHORT_AND_LONG, head_dim=128, group=3, dtype=mx.float32, num_partitions=7),
    Case(*UNEQUAL, head_dim=512, num_partitions=3),
    Case(*SHORT_AND_LONG, head_dim=512, group=12, page_size=4, num_partitions=3),
    # Sliding windows: one key, inside a key block, Gemma4's (window layers D 256, G 2).
    Case(*ACROSS, group=2, window=1),
    Case(*FAR_TURN, head_dim=128, page_size=16, num_partitions=3, window=1),
    Case(*FROM_EMPTY, page_size=16, window=13),
    Case(*ACROSS, head_dim=128, group=1, page_size=16, dtype=mx.float32, window=13),
    Case(*FAR_TURN, head_dim=512, dtype=mx.float32, window=13),
    Case(*FROM_EMPTY, group=2, window=1024),
    Case(*FAR_TURN, group=2, page_size=16, window=1024),  # split by the rule
    Case(*ACROSS, group=2, page_size=16, dtype=mx.float32, num_partitions=3, window=1024),
    Case(*ACROSS, head_dim=512, window=1024),
]


def filled_pool(
    seq_lens: list[int],
    page_size: int,
    head_dim: int,
    dtype: mx.Dtype,
    seed: int,
    first_keys: list[int] | None = None,
) -> tuple[mx.array, mx.array, np.ndarray]:
    """K/V pools with random rows for keys first_keys[b]..seq_lens[b] on each request's
    shuffled pages, UNREAD_VALUE for the keys before first_keys[b] and NaN past the keys."""
    if first_keys is None:
        first_keys = [0] * len(seq_lens)
    rng = np.random.default_rng(seed)
    pages_per_request = [-(-seq_len // page_size) for seq_len in seq_lens]
    num_pages = sum(pages_per_request) + SPARE_PAGES
    pages = rng.permutation(num_pages)
    # The rows outlast the longest request by a key block, on a spare page.
    width = max(pages_per_request) + -(-MAX_BLOCK_K // page_size)
    page_table = np.full((len(seq_lens), width), pages[-1], dtype=np.int32)
    is_key = np.zeros((num_pages * page_size, 1, 1), dtype=bool)
    is_hidden = np.zeros((num_pages * page_size, 1, 1), dtype=bool)
    start = 0
    for i, (seq_len, num_request_pages) in enumerate(zip(seq_lens, pages_per_request)):
        page_table[i, :num_request_pages] = pages[start : start + num_request_pages]
        start += num_request_pages
        slots = (page_table[i, :, None] * page_size + np.arange(page_size)).reshape(-1)
        is_hidden[slots[: first_keys[i]]] = True
        is_key[slots[first_keys[i] : seq_len]] = True
    shape = (num_pages * page_size, NUM_KV_HEADS, head_dim)
    mx.random.seed(seed)
    fill = mx.where(mx.array(is_hidden), UNREAD_VALUE, mx.nan)
    k_cache = mx.where(mx.array(is_key), mx.random.normal(shape), fill).astype(dtype)
    v_cache = mx.where(mx.array(is_key), mx.random.normal(shape), fill).astype(dtype)
    return k_cache, v_cache, page_table


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_sdpa(case: Case) -> None:
    cached_lens, extend_lens = list(case.cached_lens), list(case.extend_lens)
    window = case.window
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    first_keys = [0 if window is None else max(0, c - window + 1) for c in cached_lens]
    k_cache, v_cache, page_table = filled_pool(
        seq_lens, case.page_size, case.head_dim, case.dtype, sum(seq_lens), first_keys
    )
    num_heads = NUM_KV_HEADS * case.group
    q = mx.random.normal((sum(extend_lens), num_heads, case.head_dim)).astype(case.dtype)
    tiles = prefill_tiles(extend_lens, cached_lens, num_heads, NUM_KV_HEADS, window)
    if case.num_partitions is not None:
        tiles = replace(tiles, num_partitions=case.num_partitions)
    table = mx.array(page_table)
    scale = case.head_dim**-0.5
    out = flash_prefill(q, k_cache, v_cache, table, tiles, case.page_size, scale, window)
    expected = flash_prefill_ref(
        q, k_cache, v_cache, table, extend_lens, cached_lens, case.page_size, scale, window
    )
    assert out.shape == q.shape and out.dtype == case.dtype
    error = mx.abs(out.astype(mx.float32) - expected).max(axis=2)
    assert float((error / mx.abs(expected).max(axis=2)).max()) < TOLERANCE[case.dtype]


@pytest.mark.parametrize("num_partitions", [None, 3])
def test_donated_query_buffer_keeps_the_output(num_partitions: int | None) -> None:
    """The output may take q's buffer when nothing else holds q; it must not change."""
    cached_lens, extend_lens = list(UNEQUAL[0]), list(UNEQUAL[1])
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    head_dim, group = 256, 8
    k_cache, v_cache, page_table = filled_pool(seq_lens, 1, head_dim, mx.bfloat16, seed=0)
    q = mx.random.normal((sum(extend_lens), NUM_KV_HEADS * group, head_dim)).astype(mx.bfloat16)
    tiles = prefill_tiles(extend_lens, cached_lens, NUM_KV_HEADS * group, NUM_KV_HEADS)
    if num_partitions is not None:
        tiles = replace(tiles, num_partitions=num_partitions)
    args = (k_cache, v_cache, mx.array(page_table), tiles, 1, head_dim**-0.5)
    mx.eval(q)
    kept = flash_prefill(q, *args)
    # q * 1 is q bit for bit, held by nothing but the call.
    donated = flash_prefill(q * 1, *args)
    assert mx.array_equal(kept, donated)


def test_window_partitions_follow_the_window_not_the_context() -> None:
    """A short turn far into a long context splits by the keys its window reaches."""
    full = prefill_tiles([64], [30720], num_heads=16, num_kv_heads=2)
    windowed = prefill_tiles([64], [30720], num_heads=16, num_kv_heads=2, sliding_window=1024)
    assert (full.num_partitions, windowed.num_partitions) == (8, 2)


@pytest.mark.parametrize(
    ("cached_len", "extend_len", "num_partitions"), [(30720, 2048, 1), (30720, 64, 8)]
)
def test_partition_rule_at_the_qwen3_6_shape(
    cached_len: int, extend_len: int, num_partitions: int
) -> None:
    """A long chunk fills the GPU with its tiles; a short turn over a long prefix splits."""
    tiles = prefill_tiles([extend_len], [cached_len], num_heads=16, num_kv_heads=2)
    assert tiles.num_partitions == num_partitions
