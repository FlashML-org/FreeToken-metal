"""Paged flash-decode (fdecode v4): one query token per request over the paged K/V pool.

The kernels live in metal/fdecode.metal and are encoded by the C++ FlashDecode
primitive (csrc/fdecode.cpp): one primitive per call, the partials pass and the
log-sum-exp merge in one launch pair with the partials in a temporary. Each threadgroup
streams one kv head's keys once for all G = Hq / Hkv query heads (GQA packing); split-KV
partitions fill the GPU at small batch. Divergences from the fp16 original: the dtype
is a template argument (bf16, fp16, fp32), pool rows are token slots
page_id * page_size + offset, and the page-table width, partition count and scale are
runtime values, so a growing context never recompiles. The grid's partition count
follows max_seqlen_k; the kernels apply the same rule to each request's own length and
skip the partitions past it, so a grid fixed at the longest context (a replayed graph)
does the work of the per-length one. This module only picks the partitioning.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel._ext import _C

SIMD_WIDTH = 32
SIMDGROUPS_PER_THREADGROUP = 4
# Apple GPUs cap threadgroup memory at 32 KiB; keep a margin for the compiler.
THREADGROUP_MEMORY_BYTES = 30 * 1024
# Partitioning tuned for fdecode v4 on the M4 Pro (20 GPU cores): about 1k keys per
# partition, at least 256, and at least two waves of threadgroups at small batch.
PARTITION_TOKENS = 1024
MIN_PARTITION_TOKENS = 256
MAX_PARTITIONS = 64
MIN_THREADGROUPS = 40


def flash_decode(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    seq_lens: mx.array,
    max_seqlen_k: int,
    page_size: int,
    scale: float,
) -> mx.array:
    """Attention of one query per request over its first seq_lens[b] paged keys.

    q [B, Hq, D]; k_cache / v_cache [num_pages * page_size, Hkv, D] in q's dtype;
    page_table [B, max_pages] int32 page ids; seq_lens [B] int32, each >= 1.
    max_seqlen_k bounds seq_lens and only sizes the partitioning. Returns [B, Hq, D].
    """
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    assert num_heads % num_kv_heads == 0
    assert head_dim % (2 * SIMD_WIDTH) == 0, "each lane needs at least a 2-wide vector"
    assert page_size & (page_size - 1) == 0, "page size must be a power of two"
    assert k_cache.dtype == v_cache.dtype == q.dtype
    assert page_table.dtype == seq_lens.dtype == mx.int32
    group = num_heads // num_kv_heads
    # Threadgroup memory: per simdgroup the staged [G, D] accumulator (half for 16-bit
    # inputs, fp32 otherwise) and (max, sum) pairs, plus the [G, D] query block.
    stage_size = 4 if q.dtype == mx.float32 else 2
    simdgroup_bytes = group * (head_dim * stage_size + 8)
    query_bytes = group * head_dim * q.dtype.size
    num_simdgroups = SIMDGROUPS_PER_THREADGROUP
    while num_simdgroups > 1 and (
        num_simdgroups * simdgroup_bytes + query_bytes > THREADGROUP_MEMORY_BYTES
    ):
        num_simdgroups //= 2
    num_partitions = max(1, min(MAX_PARTITIONS, -(-max_seqlen_k // PARTITION_TOKENS)))
    while batch_size * num_kv_heads * num_partitions < MIN_THREADGROUPS and num_partitions < -(
        -max_seqlen_k // MIN_PARTITION_TOKENS
    ):
        num_partitions *= 2
    return _C.flash_decode(
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        page_size,
        num_partitions,
        PARTITION_TOKENS,
        MIN_PARTITION_TOKENS,
        MAX_PARTITIONS,
        MIN_THREADGROUPS,
        num_simdgroups,
        scale,
    )


__all__ = ["flash_decode"]
