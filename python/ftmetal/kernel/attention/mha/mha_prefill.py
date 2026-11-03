"""Paged flash-attention prefill: a varlen batch of query tokens over the paged K/V pool.

The kernels live in metal/fprefill.metal and are encoded by the C++ FlashPrefill
primitive (csrc/fprefill.cpp): the query tokens are cut into per-request tiles, and a
threadgroup streams one kv head's visible keys once for the tile's G = Hq / Hkv query
heads (GQA packing), gathering them through the page table; the queries' own keys are
already in the pool. Split-KV partitions fill the GPU when the tiles are few (a short
turn over a long cached prefix), the partials merged by a second pass. This module lists
the tiles and picks the partitioning, once per batch (prefill_tiles), and launches the
primitive per layer (flash_prefill). The kernel's two GEMMs run on simdgroup matrices or
as MetalPerformancePrimitives matmul2d, which reaches the matrix units of GPUs that have
them (default_mma). flash_prefill_qsa runs the same kernel block-sparse for one request:
each tile streams only the QSA blocks its queries attend, masked per query.
"""

from __future__ import annotations

import functools
import platform
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np

from ftmetal.env import read_env
from ftmetal.kernel._ext import _C
from ftmetal.kernel.fdecode import THREADGROUP_MEMORY_BYTES

FprefillMma = Literal["simdgroup", "mpp"]

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
# matmul2d's inner dimension, P V's key block, is a multiple of 16.
MPP_BLOCK_K = 16
# MLX's rule for the GPU matrix units (NAX): macOS 26.2 and Apple GPU generation 17 (18 for
# the architectures whose name ends in p).
NAX_MACOS_VERSION = (26, 2)
NAX_GENERATION = 17
NAX_P_GENERATION = 18
# QSA mode: query rows per threadgroup (tokens x G, a multiple of 8). A tile's tokens share
# their blocks' K/V reads but each attends the others' blocks masked: at Qwen3.8-Flash-Next's
# shape (G 12, D 256, QSA block 4) the union is 1.24x a query's blocks at 2 tokens, 1.52x at
# 4 and 1.84x at 8, and 2 tokens (24 rows) ran fastest (measured on the M4 Pro, 2026-10-19).
QSA_THREADGROUP_ROWS = 24


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


def has_matrix_units() -> bool:
    """Whether the GPU has matrix units (NAX) for matmul2d."""
    architecture = mx.device_info()["architecture"]
    assert isinstance(architecture, str)
    generation = int(architecture[-3:-1]) if architecture[-3:-1].isdigit() else 0
    macos = tuple(int(part) for part in platform.mac_ver()[0].split(".")[:2])
    min_generation = NAX_P_GENERATION if architecture.endswith("p") else NAX_GENERATION
    return macos >= NAX_MACOS_VERSION and generation >= min_generation


@functools.cache
def default_mma() -> FprefillMma:
    """FTMETAL_FPREFILL_MMA (simdgroup or mpp), else mpp where the GPU has matrix units."""
    value = read_env("FPREFILL_MMA")
    if value is None:
        return "mpp" if has_matrix_units() else "simdgroup"
    elif value == "simdgroup" or value == "mpp":
        return value
    else:
        raise ValueError(f"FTMETAL_FPREFILL_MMA must be simdgroup or mpp, got {value!r}")


def threadgroup_memory_bytes(
    head_dim: int, block_k: int, head_split: int, itemsize: int, kv_shared: bool, num_passes: int
) -> int:
    """fprefill.metal's threadgroup arrays: K transposed [D / num_passes, BK] and V
    [BK, D / num_passes] with rows padded by 16 bytes (one buffer for both when kv_shared),
    and with a head split each simdgroup's [8, BK] fp32 partial scores."""
    columns = head_dim // num_passes
    key_bytes = columns * block_k * itemsize
    value_bytes = block_k * (columns + 16 // itemsize) * itemsize
    staged_bytes = max(key_bytes, value_bytes) if kv_shared else key_bytes + value_bytes
    num_simdgroups = THREADGROUP_ROWS // FRAGMENT_ROWS * head_split
    score_bytes = num_simdgroups * FRAGMENT_ROWS * block_k * 4 if head_split > 1 else 0
    return staged_bytes + score_bytes


def staging_plan(head_dim: int, itemsize: int, mma: FprefillMma) -> tuple[int, int, bool, int]:
    """(head_split, block_k, kv_shared, num_passes): the largest key block whose staging
    fits, in separate K and V buffers when both fit; matmul2d keeps 16-key blocks and
    stages them in column passes instead."""
    head_split = 2 if head_dim >= SPLIT_HEAD_DIM else 1
    if mma == "mpp":
        options = [(MPP_BLOCK_K, False, 1), (MPP_BLOCK_K, True, 1), (MPP_BLOCK_K, True, head_split)]
    else:
        options = [
            (block_k, kv_shared, 1)
            for block_k in (MAX_BLOCK_K, FRAGMENT_ROWS)
            for kv_shared in (False, True)
        ]
    for block_k, kv_shared, num_passes in options:
        staged = threadgroup_memory_bytes(
            head_dim, block_k, head_split, itemsize, kv_shared, num_passes
        )
        if staged <= THREADGROUP_MEMORY_BYTES:
            return head_split, block_k, kv_shared, num_passes
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
    mma: FprefillMma | None = None,
    block_ends: mx.array | None = None,
) -> mx.array:
    """Causal attention of each request's queries over its paged keys.

    q [T, Hq, D] holds the requests' queries back to back, cut into tiles by
    prefill_tiles; page_table [B, max_pages] int32 page ids, k_cache / v_cache
    [num_pages * page_size, Hkv, D] in q's dtype, already holding the queries' own keys.
    A sliding_window W limits a query at position p to keys (p - W, p], positions counted
    from each page-table row's first key. block_ends int32 [T], when given, also shows a
    query the keys before its entry (the end of its image span, at most its request's key
    count; 0 for text), the window still applying. mma picks the GEMMs (default_mma when
    None). Returns [T, Hq, D].
    """
    num_heads, head_dim = q.shape[1:]
    num_kv_heads = k_cache.shape[1]
    assert num_heads % num_kv_heads == 0 and num_heads // num_kv_heads <= THREADGROUP_ROWS
    assert head_dim <= MAX_HEAD_DIM
    mma = default_mma() if mma is None else mma
    head_split, block_k, kv_shared, num_passes = staging_plan(head_dim, q.dtype.size, mma)
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
        num_passes,
        tiles.num_partitions,
        0 if sliding_window is None else sliding_window,
        mma == "mpp",
        scale,
        block_ends,
    )


def qsa_prefill_lists(
    visible: mx.array, first_position: int, num_heads: int, num_kv_heads: int
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """flash_prefill_qsa's kernel inputs for one request's queries at consecutive positions
    from first_position, visible bool [Q, num_blocks] the QSA blocks each attends.

    Returns tiles int32 [num_tiles, 4] (as prefill_tiles lists them, all of request 0), each
    tile's union of visible blocks int32 [num_tiles, num_blocks] ascending, the first
    counts int32 [num_tiles] of them real, and each query's visible blocks as bits uint32
    [Q, ceil(num_blocks / 32)] (bit b % 32 of word b / 32 for block b).
    """
    num_queries, num_blocks = visible.shape
    tile_queries = QSA_THREADGROUP_ROWS // (num_heads // num_kv_heads)
    num_tiles = -(-num_queries // tile_queries)
    words = mx.pad(visible, [(0, 0), (0, -num_blocks % 32)]).reshape(num_queries, -1, 32)
    bits = (words.astype(mx.uint32) << mx.arange(32, dtype=mx.uint32)).sum(axis=-1)
    padded = mx.pad(visible, [(0, num_tiles * tile_queries - num_queries), (0, 0)])
    union = padded.reshape(num_tiles, tile_queries, num_blocks).any(axis=1)
    block_ids = mx.arange(num_blocks, dtype=mx.int32)
    blocks = mx.sort(mx.where(union, block_ids, num_blocks), axis=-1)
    first_query = np.arange(num_tiles) * tile_queries
    query_count = np.minimum(tile_queries, num_queries - first_query)
    tiles = np.stack(
        [np.zeros(num_tiles), first_query, query_count, first_position + first_query], axis=1
    )
    return mx.array(tiles.astype(np.int32)), blocks, union.sum(axis=-1), bits


def flash_prefill_qsa(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    page_size: int,
    visible: mx.array,
    first_position: int,
    block_size: int,
    scale: float,
    mma: FprefillMma | None = None,
) -> mx.array:
    """Block-sparse attention of one request's queries over its paged keys.

    q [Q, Hq, D] holds the queries at consecutive positions from first_position;
    page_table [1, max_pages] int32 the request's page ids into k_cache / v_cache
    [num_pages * page_size, Hkv, D] in q's dtype, already holding the queries' own keys.
    Query i attends the keys of the block_size-key QSA blocks visible[i] marks (bool
    [Q, num_blocks], num_blocks covering its own block) up to its own position. A tile of
    QSA_THREADGROUP_ROWS // G queries streams its queries' blocks' union once. mma picks
    the GEMMs (default_mma when None). Returns [Q, Hq, D].
    """
    num_heads, head_dim = q.shape[1:]
    num_kv_heads = k_cache.shape[1]
    assert num_heads % num_kv_heads == 0 and num_heads // num_kv_heads <= QSA_THREADGROUP_ROWS
    assert head_dim <= MAX_HEAD_DIM and page_table.shape[0] == 1
    assert visible.shape[1] * block_size >= first_position + q.shape[0]
    mma = default_mma() if mma is None else mma
    head_split, block_k, kv_shared, num_passes = staging_plan(head_dim, q.dtype.size, mma)
    tiles, blocks, counts, bits = qsa_prefill_lists(
        visible, first_position, num_heads, num_kv_heads
    )
    return _C.flash_prefill_qsa(
        q,
        k_cache,
        v_cache,
        page_table,
        tiles,
        blocks,
        counts,
        bits,
        block_size,
        page_size,
        QSA_THREADGROUP_ROWS // FRAGMENT_ROWS,
        head_split,
        block_k,
        kv_shared,
        num_passes,
        1,
        mma == "mpp",
        scale,
    )


__all__ = [
    "FprefillMma",
    "PrefillTiles",
    "default_mma",
    "flash_prefill",
    "flash_prefill_qsa",
    "prefill_tiles",
    "qsa_prefill_lists",
]
