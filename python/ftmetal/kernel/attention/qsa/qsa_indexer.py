"""QSA (Qwen Sparse Attention) decode block selection and index-key pooling.

A QSA layer keeps one per-token array index_keys [pool_tokens, index_head_dim]: the slot of
the last token of each block_size-token block holds that block's pooled, normed and roped
key (the only slots qsa_block_scores reads); every other slot holds its token's raw index
key (what qsa_window_mean averages). Page tables are page ids int32 [bs, width]: position p
of request b lives at slot page_table[b, p // page_size] * page_size + p % page_size. A
decode step scores each request's complete blocks against its index queries
(qsa_block_scores), keeps the topk best (qsa_topk) and lists the token slots of those blocks
and of its incomplete tail block for flash_decode (qsa_key_table). The kernels skip the
blocks past each request's own length, so a step replayed at the page table's full width
costs what the requests' lengths do.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import metal_kernel_header, metal_source
from ftmetal.kernel.custom import metal_kernel

SCORE_SIMDGROUPS = 8
# Each simdgroup scores this many consecutive blocks: their page-table and key loads issue
# together and the queries are read once.
BLOCKS_PER_SIMDGROUP = 4
# Score threadgroups per request at most (eight per M4 Pro GPU core; PR #3's SCORE_THREADGROUPS):
# wider rows loop.
SCORE_THREADGROUPS = 160
# One threadgroup per request, the largest Metal allows.
TOPK_THREADS = 1024

SCORES_SOURCE = metal_source(Path(__file__).with_name("qsa_indexer_scores.metal"))

SCORES_KERNEL = metal_kernel(
    name="qsa_indexer_scores",
    input_names=[
        "queries",
        "index_keys",
        "page_table",
        "seq_lens",
        "page_size",
        "num_blocks_in",
        "inv_sqrt_d",
    ],
    output_names=["scores"],
    source=SCORES_SOURCE,
)

TOPK_SOURCE = metal_source(Path(__file__).with_name("qsa_indexer_topk.metal"))

TOPK_KERNEL = metal_kernel(
    name="qsa_indexer_topk",
    input_names=["scores", "seq_lens", "num_blocks_in"],
    output_names=["selected"],
    source=TOPK_SOURCE,
    header=metal_kernel_header(Path(__file__).with_name("qsa_indexer_topk.h")),
)

TABLE_SOURCE = metal_source(Path(__file__).with_name("qsa_indexer_key_table.metal"))

TABLE_KERNEL = metal_kernel(
    name="qsa_indexer_key_table",
    input_names=["selected", "page_table", "seq_lens", "page_size"],
    output_names=["key_slots", "counts"],
    source=TABLE_SOURCE,
)

WINDOW_MEAN_SOURCE = metal_source(Path(__file__).with_name("qsa_indexer_window_mean.metal"))

WINDOW_MEAN_KERNEL = metal_kernel(
    name="qsa_indexer_window_mean",
    input_names=["index_keys", "window_slots"],
    output_names=["means"],
    source=WINDOW_MEAN_SOURCE,
)


def qsa_block_scores(
    queries: mx.array,
    index_keys: mx.array,
    page_table: mx.array,
    seq_lens: mx.array,
    page_size: int,
    block_size: int,
    num_blocks: int,
) -> mx.array:
    """Block scores fp32 [bs, num_blocks] of each request's index queries [bs, heads, D].

    Block j of request b scores sum_h relu(queries[b, h] . k_j) / sqrt(D) in fp32 while it is
    complete (j < seq_lens[b] // block_size), k_j read from index_keys [pool_tokens, D] at the
    slot of the block's last token through page_table int32 [bs, width]; -inf past that.
    """
    bs, heads, head_dim = queries.shape
    assert head_dim % 32 == 0 and index_keys.shape[1] == head_dim
    blocks_per_threadgroup = SCORE_SIMDGROUPS * BLOCKS_PER_SIMDGROUP
    num_threadgroups = min(-(-num_blocks // blocks_per_threadgroup), SCORE_THREADGROUPS)
    (scores,) = SCORES_KERNEL(
        inputs=[queries, index_keys, page_table, seq_lens, page_size, num_blocks, head_dim**-0.5],
        template=[
            ("D", head_dim),
            ("HEADS", heads),
            ("BLOCK_SIZE", block_size),
            ("SIMDGROUPS", SCORE_SIMDGROUPS),
            ("BLOCKS_PER_SIMDGROUP", BLOCKS_PER_SIMDGROUP),
        ],
        grid=(num_threadgroups * SCORE_SIMDGROUPS * 32, bs, 1),
        threadgroup=(SCORE_SIMDGROUPS * 32, 1, 1),
        output_shapes=[(bs, num_blocks)],
        output_dtypes=[mx.float32],
    )
    return scores


def qsa_topk(scores: mx.array, seq_lens: mx.array, topk: int, block_size: int) -> mx.array:
    """The topk highest-scoring complete blocks int32 [bs, topk] of scores fp32 [bs, blocks].

    Ties go to the lower block index and the blocks come out ascending. A request with at
    most topk complete blocks (seq_lens[b] // block_size) gets all of them, then zeros.
    """
    assert topk >= 1
    bs, num_blocks = scores.shape
    (selected,) = TOPK_KERNEL(
        inputs=[scores, seq_lens, num_blocks],
        template=[
            ("TOPK", topk),
            ("BLOCK_SIZE", block_size),
            ("THREADS", TOPK_THREADS),
        ],
        grid=(TOPK_THREADS, bs, 1),
        threadgroup=(TOPK_THREADS, 1, 1),
        output_shapes=[(bs, topk)],
        output_dtypes=[mx.int32],
    )
    return selected


def qsa_key_table(
    selected: mx.array, page_table: mx.array, seq_lens: mx.array, page_size: int, block_size: int
) -> tuple[mx.array, mx.array]:
    """The token slots of the keys each request attends, as flash_decode takes them.

    selected int32 [bs, topk] is qsa_topk's output. Returns (key_slots int32 [bs, (topk + 1)
    * block_size], counts int32 [bs]): request b's first counts[b] entries are the slots, read
    through page_table int32 [bs, width], of its first min(complete, topk) selected blocks'
    tokens and then of its incomplete tail block's, ascending by position.
    """
    bs, topk = selected.shape
    width = (topk + 1) * block_size
    key_slots, counts = TABLE_KERNEL(
        inputs=[selected, page_table, seq_lens, page_size],
        template=[("BLOCK_SIZE", block_size)],
        grid=(width, bs, 1),
        threadgroup=(min(256, width), 1, 1),
        output_shapes=[(bs, width), (bs,)],
        output_dtypes=[mx.int32, mx.int32],
    )
    return key_slots, counts


def qsa_window_mean(index_keys: mx.array, window_slots: mx.array) -> mx.array:
    """mx.mean(index_keys[window_slots].astype(mx.float32), axis=1).astype(index_keys.dtype)
    for index_keys [pool_tokens, D] and window_slots int32 [T, window] -> [T, D], without
    the gathered [T, window, D] copy."""
    num_tokens, window = window_slots.shape
    head_dim = index_keys.shape[1]
    (means,) = WINDOW_MEAN_KERNEL(
        inputs=[index_keys, window_slots],
        template=[("T", index_keys.dtype), ("D", head_dim), ("WINDOW", window)],
        grid=(head_dim, num_tokens, 1),
        threadgroup=(min(head_dim, 256), 1, 1),
        output_shapes=[(num_tokens, head_dim)],
        output_dtypes=[index_keys.dtype],
    )
    return means


__all__ = ["qsa_block_scores", "qsa_key_table", "qsa_topk", "qsa_window_mean"]
