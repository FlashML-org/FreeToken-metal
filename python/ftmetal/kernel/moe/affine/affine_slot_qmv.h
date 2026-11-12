// MLX's affine one-row matvec (gather_qmv) for chosen output rows of one bank slot's projection [N,
// K], with each group's scale and bias from either scale source: the slot's scales and biases [N, K
// / group_size], or its pair-code row into the layer's pair table (affine_pair_codes.h). Every
// row's per-lane sums and simd_sum are MLX's, so a row equals mx.gather_qmm's for that slot bit for
// bit on the impl MLX's gather_qmv picks for the shape (fast: N a multiple of 8 and K of its input
// step), whichever rows a simdgroup computes together. Adapted from MLX v0.32.3 (Copyright ©
// 2023-2024 Apple Inc., MIT License): qmv_fast_impl (756-821) and qmv_impl (823-982) in
// mlx/backend/metal/kernels/quantized.h, as qmv_fast_rows and qmv_rows, and their rows of a
// threadgroup as gather_qmv_rows. Divergences: the simdgroup computes the rows it is given, rows[r]
// of the slot's projection w, instead of four from its threadgroup position (the caller picks MLX's
// rows, or any others), each group's scale and bias come from a scale source in place of the scales
// and biases pointers, every lane keeps each row's simd_sum in result instead of lane 0 storing it,
// and the sizes come by value. The rest is MLX's code unchanged.
// note (jason-fxz): coupled to mlx==0.32.3's quantized.h; re-diff both impls against the new one
// on any MLX upgrade, the bit-exact tests against mx.gather_qmm are the guard.
//
// qmv_rows' last part-block on half the lanes or less (Qwen3.8's down, K 640: 128 values on lanes
// 0-15) leaves the other half idle; with QMV_HALF_REM (kernel/_ext.py's switch, PR #3's
// down_combine_gated half_rem) lane l + 16 computes the upper half of the rows on lane l's values
// and hands them down: every lane adds the same products in the same order.
#pragma once

#ifndef QMV_HALF_REM
#define QMV_HALF_REM 1
#endif

#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
#include "moe/affine/affine_pair_codes.h"

// The scale and bias of group i (row * K / group_size + group) of a slot's projection.
template <typename T>
struct AffineScales {
  const device T* scales;
  const device T* biases;

  METAL_FUNC void group(int i, thread float& scale, thread float& bias) const {
    scale = scales[i];
    bias = biases[i];
  }
};

// PairTable: the table's pointer type (metal_kernel binds a table of under 8 values in constant
// memory).
template <typename T, int CODE_BITS, typename PairTable>
struct PairCodeScales {
  const device uint32_t* codes;
  PairTable pairs;

  METAL_FUNC void group(int i, thread float& scale, thread float& bias) const {
    const uint pair = pair_code<CODE_BITS>(codes, i);
    scale = pairs[2 * pair];
    bias = pairs[2 * pair + 1];
  }
};

template <typename T, int group_size, int bits, int ROWS, typename Scales>
METAL_FUNC void qmv_fast_rows(
    const device uint32_t* w,
    Scales scales,
    const device T* x,
    const int in_vec_size,
    const thread int (&rows)[ROWS],
    thread float (&result)[ROWS],
    uint simd_lid) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  for (int row = 0; row < ROWS; row++) {
    result[row] = 0;
  }

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;

  ws += simd_lid * packs_per_thread * bytes_per_pack;
  int group = simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < ROWS; row++) {
      auto wl = (const device uint8_t*)(ws + rows[row] * in_vec_size_w);
      U s;
      U b;
      scales.group(rows[row] * in_vec_size_g + group, s, b);
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    group += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < ROWS; row++) {
    result[row] = simd_sum(result[row]);
  }
}

template <typename T, int group_size, int bits, int ROWS, typename Scales>
METAL_FUNC void qmv_rows(
    const device uint32_t* w,
    Scales scales,
    const device T* x,
    const int in_vec_size,
    const thread int (&rows)[ROWS],
    thread float (&result)[ROWS],
    uint simd_lid) {
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  for (int row = 0; row < ROWS; row++) {
    result[row] = 0;
  }

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;

  ws += simd_lid * packs_per_thread * bytes_per_pack;
  int group = simd_lid / scale_step_per_thread;
  x += simd_lid * values_per_thread;

  int k = 0;
  for (; k < in_vec_size - block_size; k += block_size) {
    U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < ROWS; row++) {
      auto wl = (const device uint8_t*)(ws + rows[row] * in_vec_size_w);
      U s;
      U b;
      scales.group(rows[row] * in_vec_size_g + group, s, b);
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    group += block_size / group_size;
    x += block_size;
  }
  constexpr int half_lanes = SIMD_SIZE / 2;
  if (QMV_HALF_REM && ROWS % 2 == 0 && half_lanes % scale_step_per_thread == 0 &&
      in_vec_size - k <= block_size / 2) {
    // Lane l + 16 reads lane l's values (pointers moved back by 16 lanes) for the upper rows.
    const int upper = simd_lid >= half_lanes;
    const int lane_shift = upper * half_lanes;
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - (int(simd_lid) - lane_shift) * values_per_thread),
        0,
        values_per_thread);
    U part[ROWS / 2];
    for (int j = 0; j < ROWS / 2; j++) {
      part[j] = 0;
    }
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(
          x - lane_shift * values_per_thread, x_thread, remaining);
      for (int j = 0; j < ROWS / 2; j++) {
        const int row = upper * (ROWS / 2) + j;
        auto wl = (const device uint8_t*)(ws - lane_shift * packs_per_thread * bytes_per_pack +
                                          rows[row] * in_vec_size_w);
        U s;
        U b;
        scales.group(rows[row] * in_vec_size_g + group - lane_shift / scale_step_per_thread, s, b);
        part[j] = qdot_safe<U, values_per_thread, bits>(wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int j = 0; j < ROWS / 2; j++) {
      const U upper_part = simd_shuffle_down(part[j], half_lanes);
      if (!upper && remaining > 0) {
        result[j] += part[j];
        result[ROWS / 2 + j] += upper_part;
      }
    }
  } else {
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
    if (remaining > 0) {
      U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);

      for (int row = 0; row < ROWS; row++) {
        auto wl = (const device uint8_t*)(ws + rows[row] * in_vec_size_w);
        U s;
        U b;
        scales.group(rows[row] * in_vec_size_g + group, s, b);
        result[row] += qdot_safe<U, values_per_thread, bits>(wl, x_thread, s, b, sum, remaining);
      }
    }
  }

  for (int row = 0; row < ROWS; row++) {
    result[row] = simd_sum(result[row]);
  }
}

// gather_qmv's tiling: a threadgroup computes QMV_TILE_ROWS output rows, each of its simdgroups
// QMV_ROWS_PER_SIMDGROUP of them.
constant constexpr int QMV_ROWS_PER_SIMDGROUP = 4;
constant constexpr int QMV_TILE_ROWS = 8;

// The rows a simdgroup of row tile tile computes of an N-row projection, as MLX's impls pick them:
// under one tile the rows past N repeat the last, else the non-fast impl's last tile moves back to
// end at N. False when the simdgroup has none.
template <bool FAST>
METAL_FUNC bool
gather_qmv_rows(uint tile, uint simdgroup, int N, thread int (&rows)[QMV_ROWS_PER_SIMDGROUP]) {
  const int out_row = int(tile) * QMV_TILE_ROWS + int(simdgroup) * QMV_ROWS_PER_SIMDGROUP;
  if (out_row >= N) {
    return false;
  }
  for (int r = 0; r < QMV_ROWS_PER_SIMDGROUP; r++) {
    if (FAST) {
      rows[r] = out_row + r;
    } else if (N < QMV_TILE_ROWS) {
      rows[r] = min(out_row + r, N - 1);
    } else {
      rows[r] = min(N - QMV_ROWS_PER_SIMDGROUP, out_row) + r;
    }
  }
  return true;
}

// qmv_fast_rows when FAST, else qmv_rows.
template <typename T, int group_size, int bits, bool FAST, int ROWS, typename Scales>
METAL_FUNC void slot_qmv_rows(
    const device uint32_t* w,
    Scales scales,
    const device T* x,
    const int in_vec_size,
    const thread int (&rows)[ROWS],
    thread float (&result)[ROWS],
    uint simd_lid) {
  if (FAST) {
    qmv_fast_rows<T, group_size, bits>(w, scales, x, in_vec_size, rows, result, simd_lid);
  } else {
    qmv_rows<T, group_size, bits>(w, scales, x, in_vec_size, rows, result, simd_lid);
  }
}
