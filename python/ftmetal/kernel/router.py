"""Fused MoE router: softmax over the experts, top-k selection and renormalization.

One simdgroup per token row on mx.fast.metal_kernel; replaces the softmax,
argpartition, take_along_axis, sum and divide dispatches of the ops path (kernel/ref.py).
"""

from __future__ import annotations

import mlx.core as mx

SIMD_WIDTH = 32

ROUTER_SOURCE = """
    const uint row = thread_position_in_grid.x / 32;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float probs[E];
    auto logits = gates + row * E;

    float row_max = -INFINITY;
    for (int e = lane; e < E; e += 32) {
      const float x = static_cast<float>(logits[e]);
      probs[e] = x;
      row_max = max(row_max, x);
    }
    row_max = simd_max(row_max);
    float total = 0.0f;
    for (int e = lane; e < E; e += 32) {
      const float p = exp(probs[e] - row_max);
      probs[e] = p;
      total += p;
    }
    total = simd_sum(total);
    const float inv_total = 1.0f / total;
    for (int e = lane; e < E; e += 32) {
      probs[e] *= inv_total;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // K serial selections; ties go to the lowest expert index.
    float picked_total = 0.0f;
    for (int k = 0; k < K; ++k) {
      float best = -1.0f;
      int best_expert = E;
      for (int e = lane; e < E; e += 32) {
        if (probs[e] > best) {
          best = probs[e];
          best_expert = e;
        }
      }
      const float group_best = simd_max(best);
      const int chosen = simd_min(best == group_best ? best_expert : E);
      picked_total += group_best;
      if (lane == 0) {
        ids[row * K + k] = chosen;
        scores[row * K + k] = static_cast<InT>(group_best);
        probs[chosen] = -1.0f;
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (NORM && lane == 0) {
      for (int k = 0; k < K; ++k) {
        const float normalized = static_cast<float>(scores[row * K + k]) / picked_total;
        scores[row * K + k] = static_cast<InT>(normalized);
      }
    }
"""

ROUTER_KERNEL = mx.fast.metal_kernel(
    name="router_topk",
    input_names=["gates"],
    output_names=["ids", "scores"],
    source=ROUTER_SOURCE,
)


def router_topk(gates: mx.array, top_k: int, norm_topk_prob: bool) -> tuple[mx.array, mx.array]:
    """gates [T, E] logits -> (expert ids int32 [T, top_k], weights [T, top_k] in gates' dtype)."""
    num_rows, num_experts = gates.shape
    ids, scores = ROUTER_KERNEL(
        inputs=[gates],
        template=[
            ("InT", gates.dtype),
            ("E", num_experts),
            ("K", top_k),
            ("NORM", norm_topk_prob),
        ],
        grid=(SIMD_WIDTH * num_rows, 1, 1),
        threadgroup=(SIMD_WIDTH, 1, 1),
        output_shapes=[(num_rows, top_k), (num_rows, top_k)],
        output_dtypes=[mx.int32, gates.dtype],
    )
    return ids, scores


__all__ = ["router_topk"]
