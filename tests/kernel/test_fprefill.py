"""Paged flash-prefill against the fp32 mx.fast.scaled_dot_product_attention reference.

Random K/V pools on shuffled pages; every pool row outside a request's keys (page tails,
spare pages) holds huge keys and values, so reading one shows up in the output. The
reference (flash_prefill_ref) gathers each request's keys and runs SDPA in fp32 with the
causal mask aligned to the request's cached prefix. The lengths put tiles across the
causal diagonal (every tile of more than one query, and a cached prefix that ends inside
a key block), leave partial last tiles, and make the split-KV rule both split and not;
num_partitions forces the split on batches of unequal requests.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel.fprefill import flash_prefill
from ftmetal.kernel.ref import flash_prefill_ref

PAGE_SIZES = [1, 4, 16, 64]
HEAD_DIMS = [128, 256]
GROUPS = [1, 4, 8]
CACHED_LENS = [0, 1, 37, 2048, 5000]
EXTEND_LENS = [1, 2, 31, 32, 33, 100, 2048]
# Unequal requests in one batch: (cached_lens, extend_lens).
BATCHES = [([37, 0, 5000], [100, 2048, 33]), ([1, 2048, 0], [2, 31, 1])]
NUM_KV_HEADS = 2
SPARE_PAGES = 3
UNREAD_VALUE = 1e3
# Worst |error| over a row relative to the row's largest |value|. Past fp32 the kernel
# rounds only its bf16 output, half an ulp at the row's top binade (2^-8); one ulp leaves
# room for the fp32 summation order. fp32 inputs differ by summation order alone.
TOLERANCE = {mx.bfloat16: 2**-7, mx.float32: 1e-5}


def filled_pool(
    seq_lens: list[int], page_size: int, head_dim: int, dtype: mx.Dtype, seed: int
) -> tuple[mx.array, mx.array, np.ndarray]:
    """K/V pools with random rows on each request's shuffled pages, UNREAD_VALUE elsewhere."""
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
        is_key[slots.reshape(-1)[:seq_len]] = True
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
) -> None:
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    k_cache, v_cache, page_table = filled_pool(seq_lens, page_size, head_dim, dtype, seed)
    q = mx.random.normal((sum(extend_lens), NUM_KV_HEADS * group, head_dim)).astype(dtype)
    args = (q, k_cache, v_cache, mx.array(page_table), extend_lens, cached_lens, page_size)
    out = flash_prefill(*args, head_dim**-0.5, num_partitions=num_partitions)
    expected = flash_prefill_ref(*args, head_dim**-0.5)
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
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
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
