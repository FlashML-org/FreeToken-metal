// Adapted from MLX v0.32.3 (Copyright © 2023-2024 Apple Inc., MIT License):
// affine_gather_qmv_fast (2087-2147) and affine_gather_qmv (2149-2209) in
// mlx/backend/metal/kernels/quantized.h. Divergences: a threadgroup computes ROWS output rows of
// one token, simdgroup j its pair j's down rows at the pair's slot (affine_slot_qmv.h's impl MLX's
// gather_qmv picks), each rounded to the dtype and times the pair's routing weight in the dtype, as
// (y * weights[..., None]) does; the pairs' products are summed as (...).sum(axis=-2) sums them
// (routed_sum); {{OUTPUT}} is the sum or its sigmoid_gated_add.
threadgroup T products[TOP_K * ROWS];
const int token = threadgroup_position_in_grid.z;
const int out_row = threadgroup_position_in_grid.x * ROWS;
const int expert = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const int pair = token * TOP_K + expert;
const int slot = slots[pair];
int rows[ROWS];
for (int r = 0; r < ROWS; r++) {
  rows[r] = out_row + r;
}
float result[ROWS];
slot_qmv_rows<T, GROUP_SIZE, BITS, FAST>(
    w + size_t(slot) * N * (K * BITS / 32),
    {{SCALES}},
    act + size_t(pair) * K,
    K,
    rows,
    result,
    lane);
if (lane == 0) {
  const T weight = weights[pair];
  for (int r = 0; r < ROWS; r++) {
    products[expert * ROWS + r] = static_cast<T>(result[r]) * weight;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (expert == 0 && lane < ROWS) {
  const T total = routed_sum<T, TOP_K, ROWS>(products + lane);
  const int row = out_row + lane;
  out[size_t(token) * N + row] = {{OUTPUT}};
}
