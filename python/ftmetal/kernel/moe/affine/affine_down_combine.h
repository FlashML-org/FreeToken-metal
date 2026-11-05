// MLX's qmv over a slot's rows (affine_slot_qmv.h) and the elementwise ops of the routed experts'
// combine, as MLX computes them.
#include "common/norm.h"
#include "moe/affine/affine_slot_qmv.h"

// The routed experts' products products[j * STRIDE] (j < TOP_K) summed in the order and precision
// of (...).sum(axis=-2), which MLX runs as col_reduce_small over top_k with min(8, top_k) threads
// per column (each summing every 8th pair from zero, then the first summing theirs in turn) and as
// a copy for top_k 1. Adapted from MLX v0.32.3 (Copyright © 2023-2024 Apple Inc., MIT License):
// col_reduce_small (3-94) in mlx/backend/metal/kernels/reduction/reduce_col.h.
template <typename T, int TOP_K, int STRIDE, typename Products>
METAL_FUNC T routed_sum(Products products) {
  constexpr int COL_REDUCE_SMALL_THREADS = 8;
  constexpr int PARTIALS = TOP_K < COL_REDUCE_SMALL_THREADS ? TOP_K : COL_REDUCE_SMALL_THREADS;
  if (TOP_K == 1) {
    return products[0];
  }
  T partials[PARTIALS];
  for (int p = 0; p < PARTIALS; p++) {
    partials[p] = static_cast<T>(0);
    for (int j = p; j < TOP_K; j += PARTIALS) {
      partials[p] = products[j * STRIDE] + partials[p];
    }
  }
  T total = partials[0];
  for (int p = 1; p < PARTIALS; p++) {
    total = partials[p] + total;
  }
  return total;
}

// layers/activation.py's sigmoid_gated_add of one value: base + x * sigmoid(gate) in fp32,
// one op at a time as its compiled kernel, cast to base's dtype.
template <typename T, typename U, typename V>
inline T sigmoid_gated_add(T base, U x, V gate) {
  const float gated = static_cast<float>(x) * precise_sigmoid(static_cast<float>(gate));
  const float sum = static_cast<float>(base) + gated;
  return static_cast<T>(sum);
}
