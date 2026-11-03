"""MoE token alignment: a counting sort of the (token, expert) pairs by expert, and its inverse.

Port of FreeToken kernel/moe_impl.py moe_align_block_size (the four-stage Triton counting
sort) as one single-threadgroup Metal kernel. Divergences: no block padding, since
gather_qmm's sorted path takes expert runs of any length; the sort is not stable within
an expert, since each output row depends only on its own input row; and it also emits
the inverse permutation, which moe_combine uses to un-sort and reduce over top-k.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import mlx.core as mx

from ftmetal.kernel._ext import metal_source
from ftmetal.kernel.custom import metal_kernel

# One threadgroup sorts every pair; each thread owns at most one expert in the scan.
ALIGN_THREADS = 1024
# The combine is bandwidth-bound: widths 1-8 and 64-512 threads ran within 5% on the M4 Pro.
COMBINE_VECTOR = 4
COMBINE_THREADS = 256

ALIGN_SOURCE = metal_source(Path(__file__).with_name("moe_align_sort.metal"))

COMBINE_SOURCE = metal_source(Path(__file__).with_name("moe_align_combine.metal"))

ALIGN_KERNEL = metal_kernel(
    name="moe_align_sort",
    input_names=["topk_ids"],
    output_names=["expert_offsets", "sorted_token_indices", "sorted_expert_ids", "pair_positions"],
    source=ALIGN_SOURCE,
)

COMBINE_KERNEL = metal_kernel(
    name="moe_align_combine",
    input_names=["expert_out", "pair_positions", "topk_weights"],
    output_names=["out"],
    source=COMBINE_SOURCE,
)


class ExpertAlignment(NamedTuple):
    # [E + 1] int32: the pairs routed to expert e sit at positions offsets[e]:offsets[e + 1].
    expert_offsets: mx.array
    # [T * top_k] int32: the hidden-state row of the pair at each sorted position.
    sorted_token_indices: mx.array
    # [T * top_k] uint32, nondecreasing: the expert of the pair at each sorted position.
    sorted_expert_ids: mx.array
    # [T * top_k] int32: the sorted position of pair t * top_k + j.
    pair_positions: mx.array


def moe_align(topk_ids: mx.array, num_experts: int) -> ExpertAlignment:
    """Sort the pairs of topk_ids [T, top_k] (int32, values in [0, num_experts)) by expert."""
    num_tokens, top_k = topk_ids.shape
    assert topk_ids.dtype == mx.int32, "the router emits int32 expert ids"
    assert num_experts <= ALIGN_THREADS, "the scan gives each thread at most one expert"
    num_pairs = num_tokens * top_k
    expert_offsets, sorted_token_indices, sorted_expert_ids, pair_positions = ALIGN_KERNEL(
        inputs=[topk_ids],
        template=[("E", num_experts), ("K", top_k), ("TG", ALIGN_THREADS)],
        grid=(ALIGN_THREADS, 1, 1),
        threadgroup=(ALIGN_THREADS, 1, 1),
        output_shapes=[(num_experts + 1,), (num_pairs,), (num_pairs,), (num_pairs,)],
        output_dtypes=[mx.int32, mx.int32, mx.uint32, mx.int32],
    )
    return ExpertAlignment(expert_offsets, sorted_token_indices, sorted_expert_ids, pair_positions)


def moe_combine(expert_out: mx.array, pair_positions: mx.array, topk_weights: mx.array) -> mx.array:
    """Un-sort expert_out [T * top_k, ..., H] and reduce it over top-k with topk_weights [T, top_k].

    Returns [T, H] in expert_out's dtype, accumulated in fp32.
    """
    num_tokens, top_k = topk_weights.shape
    hidden_size = expert_out.shape[-1]
    assert expert_out.size == num_tokens * top_k * hidden_size, "one expert row per pair"
    assert hidden_size % COMBINE_VECTOR == 0, "each thread owns COMBINE_VECTOR columns"
    columns = hidden_size // COMBINE_VECTOR
    (out,) = COMBINE_KERNEL(
        inputs=[expert_out, pair_positions, topk_weights],
        template=[
            ("T", expert_out.dtype),
            ("K", top_k),
            ("H", hidden_size),
            ("V", COMBINE_VECTOR),
        ],
        grid=(columns, num_tokens, 1),
        threadgroup=(min(columns, COMBINE_THREADS), 1, 1),
        output_shapes=[(num_tokens, hidden_size)],
        output_dtypes=[expert_out.dtype],
    )
    return out


__all__ = ["ExpertAlignment", "moe_align", "moe_combine"]
