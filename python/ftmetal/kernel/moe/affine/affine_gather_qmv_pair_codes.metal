// Adapted from MLX v0.32.3 (Copyright © 2023-2024 Apple Inc., MIT License):
// affine_gather_qmv_fast (2087-2147) and affine_gather_qmv (2149-2209) in
// mlx/backend/metal/kernels/quantized.h. Divergences: threadgroup z is one (token, expert) pair,
// and its slot's weight and code rows and its input row replace adjust_matrix_offsets; FAST
// takes qmv_fast_pair_codes_impl where MLX's gather_qmv takes the fast kernel.
const uint pair = threadgroup_position_in_grid.z;
const uint slot = indices[pair];
const device uint32_t* slot_w = w + size_t(slot) * N * (K * BITS / 32);
const device uint32_t* slot_codes =
    (const device uint32_t*)(codes + size_t(slot) * CODE_ROW_BYTES);
const device T* pair_x = x + size_t(pair / PAIRS_PER_INPUT) * K;
device T* pair_y = y + size_t(pair) * N;
if (FAST) {
  qmv_fast_pair_codes_impl<T, GROUP_SIZE, BITS, CODE_BITS>(
      slot_w,
      slot_codes,
      pairs,
      pair_x,
      pair_y,
      K,
      N,
      threadgroup_position_in_grid,
      simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup);
} else {
  qmv_pair_codes_impl<T, GROUP_SIZE, BITS, CODE_BITS>(
      slot_w,
      slot_codes,
      pairs,
      pair_x,
      pair_y,
      K,
      N,
      threadgroup_position_in_grid,
      simdgroup_index_in_threadgroup,
      thread_index_in_simdgroup);
}
