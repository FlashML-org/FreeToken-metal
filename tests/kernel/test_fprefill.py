"""Paged flash-prefill against the fp32 mx.fast.scaled_dot_product_attention reference.

Random K/V pools on shuffled pages; every pool row outside a request's keys (page tails,
spare pages) holds huge keys and values, so reading one shows up in the output. The
reference (flash_prefill_ref) gathers each request's keys and runs SDPA in fp32 with the
causal mask aligned to the request's cached prefix. The lengths put tiles across the
causal diagonal (every tile of more than one query, and a cached prefix that ends inside
a key block), leave partial last tiles, and make the split-KV rule both split and not;
num_partitions forces the split on batches of unequal requests. Groups that do not divide
the threadgroup's 32 rows (3, 12) leave padding rows in every tile. Under a sliding
window, the keys before the batch's oldest window hold huge keys and values too, so a key
the window should hide shows up in the output.
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel.fprefill import flash_prefill, prefill_tiles
from ftmetal.kernel.ref import flash_prefill_ref

PAGE_SIZES = [1, 4, 16, 64]
HEAD_DIMS = [128, 256]
# Gemma4's global layers; K and V share one staging buffer there.
WIDE_HEAD_DIM = 512
GROUPS = [1, 3, 4, 8, 12]
CACHED_LENS = [0, 1, 37, 2048, 5000]
EXTEND_LENS = [1, 2, 31, 32, 33, 100, 2048]
# Unequal requests in one batch: (cached_lens, extend_lens).
BATCHES = [([37, 0, 5000], [100, 2048, 33]), ([1, 2048, 0], [2, 31, 1])]
# Sliding windows: one key, inside a key block, Gemma4's; batches that start inside, at and
# past a window and run across window starts.
WINDOWS = [1, 13, 1024]
WINDOW_BATCHES = [([0], [2048]), ([5000], [64]), ([1023, 0, 3000], [2, 1100, 2048])]
NUM_KV_HEADS = 2
SPARE_PAGES = 3
UNREAD_VALUE = 1e3
# Worst |error| over a row relative to the row's largest |value|. Past fp32 the kernel
# rounds only its 16-bit output, half an ulp at the row's top binade (bf16 2^-8, half
# 2^-11); one ulp leaves room for the fp32 summation order. fp32 inputs differ by
# summation order alone.
TOLERANCE = {mx.bfloat16: 2**-7, mx.float16: 2**-10, mx.float32: 1e-5}


def filled_pool(
    seq_lens: list[int],
    page_size: int,
    head_dim: int,
    dtype: mx.Dtype,
    seed: int,
    first_keys: list[int] | None = None,
) -> tuple[mx.array, mx.array, np.ndarray]:
    """K/V pools with random rows for keys first_keys[b]..seq_lens[b] on each request's
    shuffled pages, UNREAD_VALUE elsewhere."""
    if first_keys is None:
        first_keys = [0] * len(seq_lens)
    rng = np.random.default_rng(seed)
    pages_per_request = [-(-seq_len // page_size) for seq_len in seq_lens]
    num_pages = sum(pages_per_request) + SPARE_PAGES
    pages = rng.permutation(num_pages)
    page_table = np.full((len(seq_lens), max(pages_per_request)), num_pages - 1, dtype=np.int32)
    is_key = np.zeros((num_pages * page_size, 1, 1), dtype=bool)
    start = 0
    for i, (seq_len, num_request_pages) in enumerate(zip(seq_lens, pages_per_request)):
        page_table[i, :num_request_pages] = pages[start : start + num_request_pages]
        start += num_request_pages
        slots = page_table[i, :, None] * page_size + np.arange(page_size)
        is_key[slots.reshape(-1)[first_keys[i] : seq_len]] = True
    shape = (num_pages * page_size, NUM_KV_HEADS, head_dim)
    mx.random.seed(seed)
    k_cache = mx.where(mx.array(is_key), mx.random.normal(shape), UNREAD_VALUE).astype(dtype)
    v_cache = mx.where(mx.array(is_key), mx.random.normal(shape), UNREAD_VALUE).astype(dtype)
    return k_cache, v_cache, page_table


def max_row_error(out: mx.array, expected: mx.array) -> float:
    error = mx.abs(out.astype(mx.float32) - expected).max(axis=2)
    return float((error / mx.abs(expected).max(axis=2)).max())


def check_prefill(
    cached_lens: list[int],
    extend_lens: list[int],
    page_size: int,
    head_dim: int,
    group: int,
    dtype: mx.Dtype,
    num_partitions: int | None,
    seed: int,
    sliding_window: int | None = None,
) -> None:
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    first_keys = [
        0 if sliding_window is None else max(0, c - sliding_window + 1) for c in cached_lens
    ]
    k_cache, v_cache, page_table = filled_pool(
        seq_lens, page_size, head_dim, dtype, seed, first_keys
    )
    q = mx.random.normal((sum(extend_lens), NUM_KV_HEADS * group, head_dim)).astype(dtype)
    planned = prefill_tiles(
        extend_lens, cached_lens, NUM_KV_HEADS * group, NUM_KV_HEADS, sliding_window
    )
    tiles = planned if num_partitions is None else replace(planned, num_partitions=num_partitions)
    table = mx.array(page_table)
    scale = head_dim**-0.5
    out = flash_prefill(q, k_cache, v_cache, table, tiles, page_size, scale, sliding_window)
    expected = flash_prefill_ref(
        q, k_cache, v_cache, table, extend_lens, cached_lens, page_size, scale, sliding_window
    )
    assert out.shape == q.shape and out.dtype == dtype
    error = max_row_error(out, expected)
    assert error < TOLERANCE[dtype], (cached_lens, extend_lens, num_partitions, error)


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("page_size", PAGE_SIZES)
def test_one_request_matches_sdpa(page_size: int, head_dim: int, group: int) -> None:
    for i, cached_len in enumerate(CACHED_LENS):
        for j, extend_len in enumerate(EXTEND_LENS):
            check_prefill(
                [cached_len],
                [extend_len],
                page_size,
                head_dim,
                group,
                mx.bfloat16,
                None,
                seed=i * len(EXTEND_LENS) + j,
            )


@pytest.mark.parametrize("num_partitions", [None, 1, 3, 7])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("page_size", PAGE_SIZES)
def test_unequal_requests_match_sdpa(
    page_size: int, head_dim: int, group: int, dtype: mx.Dtype, num_partitions: int | None
) -> None:
    for i, (cached_lens, extend_lens) in enumerate(BATCHES):
        check_prefill(
            cached_lens, extend_lens, page_size, head_dim, group, dtype, num_partitions, seed=i
        )


@pytest.mark.parametrize("num_partitions", [None, 3])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("group", [1, 3, 8])
@pytest.mark.parametrize("page_size", [1, 64])
def test_wide_head_matches_sdpa(
    page_size: int, group: int, dtype: mx.Dtype, num_partitions: int | None
) -> None:
    for i, (cached_lens, extend_lens) in enumerate([([0], [37]), ([2048], [1]), *BATCHES]):
        check_prefill(
            cached_lens, extend_lens, page_size, WIDE_HEAD_DIM, group, dtype, num_partitions, seed=i
        )


@pytest.mark.parametrize("num_partitions", [None, 3])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("group", [1, 8])
@pytest.mark.parametrize("head_dim", [128, WIDE_HEAD_DIM])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("sliding_window", WINDOWS)
def test_sliding_window_matches_sdpa(
    sliding_window: int,
    page_size: int,
    head_dim: int,
    group: int,
    dtype: mx.Dtype,
    num_partitions: int | None,
) -> None:
    for i, (cached_lens, extend_lens) in enumerate(WINDOW_BATCHES):
        check_prefill(
            cached_lens,
            extend_lens,
            page_size,
            head_dim,
            group,
            dtype,
            num_partitions,
            seed=i,
            sliding_window=sliding_window,
        )


def test_window_partitions_follow_the_window_not_the_context() -> None:
    """A short turn far into a long context splits by the keys its window reaches."""
    full = prefill_tiles([64], [30720], num_heads=16, num_kv_heads=2)
    windowed = prefill_tiles([64], [30720], num_heads=16, num_kv_heads=2, sliding_window=1024)
    assert (full.num_partitions, windowed.num_partitions) == (8, 2)


@pytest.mark.parametrize("num_partitions", [None, 3])
def test_donated_query_buffer_keeps_the_output(num_partitions: int | None) -> None:
    """The output may take q's buffer when nothing else holds q; it must not change."""
    cached_lens, extend_lens = BATCHES[0]
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    head_dim, group = 256, 8
    k_cache, v_cache, page_table = filled_pool(seq_lens, 1, head_dim, mx.bfloat16, seed=0)
    q = mx.random.normal((sum(extend_lens), NUM_KV_HEADS * group, head_dim)).astype(mx.bfloat16)
    planned = prefill_tiles(extend_lens, cached_lens, NUM_KV_HEADS * group, NUM_KV_HEADS)
    tiles = planned if num_partitions is None else replace(planned, num_partitions=num_partitions)
    args = (k_cache, v_cache, mx.array(page_table), tiles, 1, head_dim**-0.5)
    mx.eval(q)
    kept = flash_prefill(q, *args)
    # q * 1 is q bit for bit, held by nothing but the call.
    donated = flash_prefill(q * 1, *args)
    assert mx.array_equal(kept, donated)


@pytest.mark.parametrize(
    ("cached_len", "extend_len", "num_partitions"), [(30720, 2048, 1), (30720, 64, 8)]
)
def test_partition_rule_at_the_qwen3_6_shape(
    cached_len: int, extend_len: int, num_partitions: int
) -> None:
    """A long chunk fills the GPU with its tiles; a short turn over a long prefix splits."""
    tiles = prefill_tiles([extend_len], [cached_len], num_heads=16, num_kv_heads=2)
    assert tiles.num_partitions == num_partitions
