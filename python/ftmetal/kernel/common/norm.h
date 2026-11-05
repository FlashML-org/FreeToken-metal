// Norm helpers shared by the norm/, hc/ and moe/affine/ kernels. Adapted from MLX v0.32.3 (MIT
// License): precise_sigmoid is Sigmoid in mlx/backend/metal/kernels/unary_ops.h (308-314,
// Copyright © 2023-2024 Apple Inc.); row_inv_rms is the reduction of rms_single_row in
// mlx/backend/metal/kernels/rms_norm.metal (44-64, Copyright © 2024 Apple Inc.).
#pragma once

// MLX's sigmoid op, in fp32.
inline float precise_sigmoid(float x) {
  const float e = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? e : 1 - e;
}

// mx.fast.rms_norm's single-row reduction: 1 / rms of a row from each thread's sum of squares.
inline float row_inv_rms(
    float sum_squares,
    float size,
    float eps,
    threadgroup float* local_sums,
    threadgroup float* local_inv_mean,
    uint simd_lane,
    uint simd_group) {
  sum_squares = simd_sum(sum_squares);
  if (simd_group == 0) {
    local_sums[simd_lane] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) {
    local_sums[simd_group] = sum_squares;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    sum_squares = simd_sum(local_sums[simd_lane]);
    if (simd_lane == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(sum_squares / size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return local_inv_mean[0];
}
