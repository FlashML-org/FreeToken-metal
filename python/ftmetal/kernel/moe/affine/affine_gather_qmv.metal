// Adapted from MLX v0.32.3 (Copyright © 2023-2024 Apple Inc., MIT License):
// affine_gather_qmv_fast (2087-2147) and affine_gather_qmv (2149-2209) in
// mlx/backend/metal/kernels/quantized.h, with the rows qmv_fast_impl (756-821) and qmv_impl
// (823-982) give a simdgroup. Divergences: threadgroup z is one (token, expert) pair, and its
// slot's weight, scales and biases and its input row replace adjust_matrix_offsets; the impls are
// affine_slot_qmv.h's; FAST takes the fast impl where MLX's gather_qmv does; a pair whose slot is
// negative is left unwritten.
const uint pair = threadgroup_position_in_grid.z;
const int slot = indices[pair];
if (slot < 0) {
  return;
}
int rows[QMV_ROWS_PER_SIMDGROUP];
if (!gather_qmv_rows<FAST>(
        threadgroup_position_in_grid.y, simdgroup_index_in_threadgroup, N, rows)) {
  return;
}
const size_t slot_groups = size_t(slot) * N * (K / GROUP_SIZE);
float result[QMV_ROWS_PER_SIMDGROUP];
slot_qmv_rows<T, GROUP_SIZE, BITS, FAST>(
    w + size_t(slot) * N * (K * BITS / 32),
    AffineScales<T>{scales + slot_groups, biases + slot_groups},
    x + size_t(pair / PAIRS_PER_INPUT) * K,
    K,
    rows,
    result,
    thread_index_in_simdgroup);
if (thread_index_in_simdgroup == 0) {
  device T* pair_y = y + size_t(pair) * N;
  for (int r = 0; r < QMV_ROWS_PER_SIMDGROUP; r++) {
    pair_y[rows[r]] = static_cast<T>(result[r]);
  }
}
