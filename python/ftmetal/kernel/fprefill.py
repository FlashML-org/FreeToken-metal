"""Paged flash-attention prefill: a varlen batch of query tokens over the paged K/V pool.

The kernels live in metal/fprefill.metal and are encoded by the C++ FlashPrefill
primitive (csrc/fprefill.cpp): the query tokens are cut into per-request tiles, and a
threadgroup streams one kv head's visible keys once for the tile's G = Hq / Hkv query
heads (GQA packing), gathering them through the page table; the queries' own keys are
already in the pool. Split-KV partitions fill the GPU when the tiles are few (a short
turn over a long cached prefix), the partials merged by a second pass. This module lists
the tiles and picks the partitioning.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ftmetal.kernel._ext import _C

FRAGMENT_ROWS = 8
# Tile shape tuned on the M4 Pro at the Qwen3.6 shape (D 256, G 8): 32 query rows (tokens
# x query heads) per threadgroup and 16-key blocks; from D 256 up two simdgroups split
# each 8-row group's head dim, since one simdgroup's fp32 [8, 256] accumulator costs
# occupancy (4.2 against 5.5 TFLOP/s). fp32 inputs halve the block to fit the 32 KiB of
# threadgroup memory.
THREADGROUP_ROWS = 32
BLOCK_K = {2: 16, 4: 8}
SPLIT_HEAD_DIM = 256
MAX_HEAD_DIM = 256
# Split-KV up to about TARGET_THREADGROUPS threadgroups (some six waves of the ~40 the M4
# Pro runs at once), keeping at least MIN_PARTITION_TOKENS keys per partition.
TARGET_THREADGROUPS = 256
MIN_PARTITION_TOKENS = 512
MAX_PARTITIONS = 32


def flash_prefill(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    extend_lens: list[int],
    cached_lens: list[int],
    page_size: int,
    scale: float,
    num_partitions: int | None = None,
) -> mx.array:
    """Causal attention of each request's extend_lens[b] queries over its paged keys.

    q [T, Hq, D] holds the requests' queries back to back; request b's queries sit at
    positions cached_lens[b] .. cached_lens[b] + extend_lens[b] - 1 and its pool keys
    (page_table [B, max_pages] int32 page ids, k_cache / v_cache [num_pages * page_size,
    Hkv, D] in q's dtype) already include them. num_partitions overrides the split-KV
    partition count. Returns [T, Hq, D].
    """
    num_heads, head_dim = q.shape[1:]
    num_kv_heads = k_cache.shape[1]
    assert num_heads % num_kv_heads == 0
    group = num_heads // num_kv_heads
    assert group <= THREADGROUP_ROWS and head_dim <= MAX_HEAD_DIM
    assert q.shape[0] == sum(extend_lens) and len(extend_lens) == len(cached_lens)
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
    # The longest key ranges first, so they do not trail the dispatch.
    order = np.argsort(-kv_end, kind="stable")
    row0 = (np.cumsum(extend) - extend)[request] + q0
    tiles = np.stack([request, row0, count, position], axis=1)[order].astype(np.int32)
    if num_partitions is None:
        num_partitions = max(
            1,
            min(
                -(-TARGET_THREADGROUPS // (len(tiles) * num_kv_heads)),
                int(kv_end.max()) // MIN_PARTITION_TOKENS,
                MAX_PARTITIONS,
            ),
        )
    return _C.flash_prefill(
        q,
        k_cache,
        v_cache,
        page_table,
        mx.array(tiles),
        page_size,
        THREADGROUP_ROWS // FRAGMENT_ROWS,
        2 if head_dim >= SPLIT_HEAD_DIM else 1,
        BLOCK_K[q.dtype.size],
        num_partitions,
        scale,
    )


__all__ = ["flash_prefill"]
