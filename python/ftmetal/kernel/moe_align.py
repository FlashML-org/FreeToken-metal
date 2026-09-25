"""MoE token alignment: a counting sort of the (token, expert) pairs by expert, and its inverse.

Port of FreeToken kernel/moe_impl.py moe_align_block_size (the four-stage Triton counting
sort) as one single-threadgroup Metal kernel. Divergences: no block padding, since
gather_qmm's sorted path takes expert runs of any length; the sort is not stable within
an expert, since each output row depends only on its own input row; and it also emits
the inverse permutation, which moe_combine uses to un-sort and reduce over top-k.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from ftmetal.kernel.custom import metal_kernel

# One threadgroup sorts every pair; each thread owns at most one expert in the scan.
ALIGN_THREADS = 1024
# The combine is bandwidth-bound: widths 1-8 and 64-512 threads ran within 5% on the M4 Pro.
COMBINE_VECTOR = 4
COMBINE_THREADS = 256

# Histogram with threadgroup atomics, exclusive scan of the E counts (simdgroup prefix
# sums, then a scan of the simdgroup totals), then each pair claims its slot with an
# atomic cursor per expert.
ALIGN_SOURCE = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const int num_pairs = topk_ids_shape[0] * K;
    threadgroup atomic_int cursor[E];
    threadgroup int simdgroup_offsets[TG / 32];

    for (uint e = tid; e < E; e += TG) {
      atomic_store_explicit(&cursor[e], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < num_pairs; i += TG) {
      atomic_fetch_add_explicit(&cursor[topk_ids[i]], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int count = tid < E ? atomic_load_explicit(&cursor[tid], memory_order_relaxed) : 0;
    const int prefix = simd_prefix_exclusive_sum(count);
    if (lane == 31) {
      simdgroup_offsets[simdgroup] = prefix + count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
      const int total = lane < TG / 32 ? simdgroup_offsets[lane] : 0;
      const int scanned = simd_prefix_exclusive_sum(total);
      if (lane < TG / 32) {
        simdgroup_offsets[lane] = scanned;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int offset = prefix + simdgroup_offsets[simdgroup];
    if (tid < E) {
      expert_offsets[tid] = offset;
      atomic_store_explicit(&cursor[tid], offset, memory_order_relaxed);
    }
    if (tid == E - 1) {
      expert_offsets[E] = offset + count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = tid; i < num_pairs; i += TG) {
      const int expert = topk_ids[i];
      const int position = atomic_fetch_add_explicit(&cursor[expert], 1, memory_order_relaxed);
      sorted_token_indices[position] = i / K;
      sorted_expert_ids[position] = expert;
      pair_positions[i] = position;
    }
"""

# Grid (H / V, T): each thread gathers V columns of a token's K expert rows from their
# sorted positions and accumulates the routing-weighted sum in fp32.
COMBINE_SOURCE = """
    const uint token = thread_position_in_grid.y;
    const uint column = thread_position_in_grid.x * V;
    float acc[V] = {};
    for (int j = 0; j < K; ++j) {
      const size_t row = pair_positions[token * K + j];
      const float weight = static_cast<float>(topk_weights[token * K + j]);
      auto source = expert_out + row * H + column;
      for (int v = 0; v < V; ++v) {
        acc[v] += weight * static_cast<float>(source[v]);
      }
    }
    auto target = out + (size_t)token * H + column;
    for (int v = 0; v < V; ++v) {
      target[v] = static_cast<T>(acc[v]);
    }
"""

ALIGN_KERNEL = metal_kernel(
    name="moe_align",
    input_names=["topk_ids"],
    output_names=["expert_offsets", "sorted_token_indices", "sorted_expert_ids", "pair_positions"],
    source=ALIGN_SOURCE,
)

COMBINE_KERNEL = metal_kernel(
    name="moe_combine",
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
