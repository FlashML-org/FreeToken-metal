"""Paged flash-attention prefill: a varlen batch of query tokens over the paged K/V pool.

The kernels live in metal/fprefill.metal and are encoded by the C++ FlashPrefill
primitive (csrc/fprefill.cpp): the query tokens are cut into per-request tiles, and a
threadgroup streams one kv head's visible keys once for the tile's G = Hq / Hkv query
heads (GQA packing), gathering them through the page table; the queries' own keys are
already in the pool. Split-KV partitions fill the GPU when the tiles are few (a short
turn over a long cached prefix), the partials merged by a second pass. This module lists
the tiles and picks the partitioning, once per batch (prefill_tiles), and launches the
primitive per layer (flash_prefill).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ftmetal.kernel._ext import _C
from ftmetal.kernel.fdecode import THREADGROUP_MEMORY_BYTES

FRAGMENT_ROWS = 8
# Tile shape tuned on the M4 Pro at the Qwen3.6 shape (D 256, G 8): 32 query rows (tokens
# x query heads) per threadgroup and 16-key blocks; from D 256 up two simdgroups split
# each 8-row group's head dim, since one simdgroup's fp32 [8, 256] accumulator costs
# occupancy (4.2 against 5.5 TFLOP/s; four simdgroups ran 1.3-1.6x slower at D 256 and
# 512). The block is the largest that fits THREADGROUP_MEMORY_BYTES, with K and V in one
# buffer (V staged once S is computed) where separate buffers do not fit: at the Gemma4
# global shape (D 512, G 8, bf16, a 2k chunk) shared 16-key blocks ran 4.9 TFLOP/s against
# 3.6 for separate 8-key ones; at D 256 separate 16-key blocks beat shared ones (5.8 / 5.5).
THREADGROUP_ROWS = 32
MAX_BLOCK_K = 16
SPLIT_HEAD_DIM = 256
MAX_HEAD_DIM = 512
# Split-KV up to about TARGET_THREADGROUPS threadgroups (some six waves of the ~40 the M4
# Pro runs at once), keeping at least MIN_PARTITION_TOKENS keys per partition.
TARGET_THREADGROUPS = 256
MIN_PARTITION_TOKENS = 512
MAX_PARTITIONS = 32


@dataclass(frozen=True, kw_only=True)
class PrefillTiles:
    """A batch's flash-prefill work list, shared by every full-attention layer of the step."""

    # [num_tiles, 4] int32 rows (request, first query row, query count, first query position).
    table: mx.array
    num_partitions: int


def prefill_tiles(
    extend_lens: list[int],
    cached_lens: list[int],
    num_heads: int,
    num_kv_heads: int,
    sliding_window: int | None = None,
) -> PrefillTiles:
    """Cut request b's extend_lens[b] queries (after cached_lens[b] cached keys) into tiles.

    A tile holds THREADGROUP_ROWS // G consecutive queries of one request; the split-KV
    partition count follows the tile count and the longest key range a tile reads (from
    its first query's window start under a sliding window).
    """
    assert num_heads % num_kv_heads == 0 and len(extend_lens) == len(cached_lens)
    group = num_heads // num_kv_heads
    assert group <= THREADGROUP_ROWS
    tile_queries = THREADGROUP_ROWS // group
    extend = np.asarray(extend_lens, dtype=np.int64)
    cached = np.asarray(cached_lens, dtype=np.int64)
    tiles_per_request = -(-extend // tile_queries)
    request = np.repeat(np.arange(len(extend)), tiles_per_request)
    first_tile = np.cumsum(tiles_per_request) - tiles_per_request
    q0 = (np.arange(len(request)) - first_tile[request]) * tile_queries
    count = np.minimum(tile_queries, extend[request] - q0)
    position = cached[request] + q0
    kv_end = position + count
    kv_span = kv_end if sliding_window is None else np.minimum(kv_end, sliding_window + count)
    # The longest key ranges first, so they do not trail the dispatch.
    order = np.argsort(-kv_span, kind="stable")
    row0 = (np.cumsum(extend) - extend)[request] + q0
    table = np.stack([request, row0, count, position], axis=1)[order].astype(np.int32)
    num_partitions = max(
        1,
        min(
            -(-TARGET_THREADGROUPS // (len(table) * num_kv_heads)),
            int(kv_span.max()) // MIN_PARTITION_TOKENS,
            MAX_PARTITIONS,
        ),
    )
    return PrefillTiles(table=mx.array(table), num_partitions=num_partitions)


def threadgroup_memory_bytes(
    head_dim: int, block_k: int, head_split: int, itemsize: int, kv_shared: bool
) -> int:
    """fprefill.metal's threadgroup arrays: K transposed [D, BK] and V [BK, D] with rows
    padded by 16 bytes (one buffer for both when kv_shared), and with a head split each
    simdgroup's [8, BK] fp32 partial scores."""
    key_bytes = head_dim * block_k * itemsize
    value_bytes = block_k * (head_dim + 16 // itemsize) * itemsize
    staged_bytes = max(key_bytes, value_bytes) if kv_shared else key_bytes + value_bytes
    num_simdgroups = THREADGROUP_ROWS // FRAGMENT_ROWS * head_split
    score_bytes = num_simdgroups * FRAGMENT_ROWS * block_k * 4 if head_split > 1 else 0
    return staged_bytes + score_bytes


def staging_plan(head_dim: int, itemsize: int) -> tuple[int, int, bool]:
    """(head_split, block_k, kv_shared): the largest key block whose staging fits, in
    separate K and V buffers when both fit."""
    head_split = 2 if head_dim >= SPLIT_HEAD_DIM else 1
    block_k = MAX_BLOCK_K
    while block_k >= FRAGMENT_ROWS:
        for kv_shared in (False, True):
            staged = threadgroup_memory_bytes(head_dim, block_k, head_split, itemsize, kv_shared)
            if staged <= THREADGROUP_MEMORY_BYTES:
                return head_split, block_k, kv_shared
        block_k //= 2
    raise ValueError(f"flash_prefill cannot stage head_dim {head_dim} at {itemsize}-byte items")


def flash_prefill(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    tiles: PrefillTiles,
    page_size: int,
    scale: float,
    sliding_window: int | None = None,
) -> mx.array:
    """Causal attention of each request's queries over its paged keys.

    q [T, Hq, D] holds the requests' queries back to back, cut into tiles by
    prefill_tiles; page_table [B, max_pages] int32 page ids, k_cache / v_cache
    [num_pages * page_size, Hkv, D] in q's dtype, already holding the queries' own keys.
    A sliding_window W limits a query at position p to keys (p - W, p], positions counted
    from each page-table row's first key. Returns [T, Hq, D].
    """
    num_heads, head_dim = q.shape[1:]
    num_kv_heads = k_cache.shape[1]
    assert num_heads % num_kv_heads == 0 and num_heads // num_kv_heads <= THREADGROUP_ROWS
    assert head_dim <= MAX_HEAD_DIM
    head_split, block_k, kv_shared = staging_plan(head_dim, q.dtype.size)
    return _C.flash_prefill(
        q,
        k_cache,
        v_cache,
        page_table,
        tiles.table,
        page_size,
        THREADGROUP_ROWS // FRAGMENT_ROWS,
        head_split,
        block_k,
        kv_shared,
        tiles.num_partitions,
        0 if sliding_window is None else sliding_window,
        scale,
    )


__all__ = ["PrefillTiles", "flash_prefill", "prefill_tiles"]
