"""Residual add fused with the following RMSNorm: one launch returns h = x + r and rms_norm(h).

The decoder's two adds per layer each feed a norm (post-attention, and the next layer's
input norm), so the pair collapses from four dispatches to two. Numerics follow
mx.fast.rms_norm: the sum is rounded to the activation dtype, statistics are fp32.
"""

from __future__ import annotations

import mlx.core as mx

THREADS_PER_ROW = 256

ADD_RMS_NORM_SOURCE = """
    const uint row = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    threadgroup float partial[TG / 32];
    auto x_row = x + row * D;
    auto r_row = r + row * D;
    auto h_row = h + row * D;
    auto out_row = normed + row * D;

    float sum_squares = 0.0f;
    for (uint i = t; i < D; i += TG) {
      const InT sum = static_cast<InT>(static_cast<float>(x_row[i]) + static_cast<float>(r_row[i]));
      h_row[i] = sum;
      const float value = static_cast<float>(sum);
      sum_squares += value * value;
    }
    sum_squares = simd_sum(sum_squares);
    if (thread_index_in_simdgroup == 0) {
      partial[simdgroup_index_in_threadgroup] = sum_squares;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint j = 0; j < TG / 32; ++j) {
      total += partial[j];
    }
    const float inv_rms = rsqrt(total / D + eps[0]);
    for (uint i = t; i < D; i += TG) {
      const float value = static_cast<float>(h_row[i]);
      out_row[i] = static_cast<InT>(static_cast<float>(weight[i]) * value * inv_rms);
    }
"""

ADD_RMS_NORM_KERNEL = mx.fast.metal_kernel(
    name="add_rms_norm",
    input_names=["x", "r", "weight", "eps"],
    output_names=["h", "normed"],
    source=ADD_RMS_NORM_SOURCE,
)


def add_rms_norm(
    x: mx.array, r: mx.array, weight: mx.array, eps: float
) -> tuple[mx.array, mx.array]:
    """x, r [T, D] -> (x + r, rms_norm(x + r) * weight), both in x's dtype."""
    num_rows, size = x.shape
    h, normed = ADD_RMS_NORM_KERNEL(
        inputs=[x, r, weight, mx.array([eps], dtype=mx.float32)],
        template=[("InT", x.dtype), ("D", size), ("TG", THREADS_PER_ROW)],
        grid=(THREADS_PER_ROW * num_rows, 1, 1),
        threadgroup=(THREADS_PER_ROW, 1, 1),
        output_shapes=[(num_rows, size), (num_rows, size)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return h, normed


__all__ = ["add_rms_norm"]
