// Adapted from MLX v0.32.3 (Copyright © 2023-2024 Apple Inc., MIT License): qmv_fast_impl in
// mlx/backend/metal/kernels/quantized.h (757-821), with one output row per simdgroup instead
// of four and the hyper-connection epilogue.
constexpr int packs_per_thread = BITS == 2 ? 1 : 2;
constexpr int pack_factor = get_pack_factor<BITS, 32>();
constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
constexpr int values_per_thread = pack_factor * packs_per_thread;
constexpr int block_size = values_per_thread * SIMD_SIZE;
constexpr int scale_step_per_thread = GROUP_SIZE / values_per_thread;
constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
constexpr int in_vec_size_g = K / GROUP_SIZE;
const uint simd_lid = thread_index_in_simdgroup;
const int out_row =
    threadgroup_position_in_grid.y * SIMDGROUPS + simdgroup_index_in_threadgroup;
if (out_row >= N) {
  return;
}
typedef float U;
thread U x_thread[values_per_thread];
U result = 0;
const device uint8_t* ws = (const device uint8_t*)weight + out_row * in_vec_size_w +
    simd_lid * packs_per_thread * bytes_per_pack;
const device T* sl = scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
const device T* bl = biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
const device T* xp = x + simd_lid * values_per_thread;
for (int k = 0; k < K; k += block_size) {
  U sum = load_vector<T, U, values_per_thread, BITS>(xp, x_thread);
  U s = sl[0];
  U b = bl[0];
  result += qdot<U, values_per_thread, BITS>(ws, x_thread, s, b, sum);
  ws += block_size * bytes_per_pack / pack_factor;
  sl += block_size / GROUP_SIZE;
  bl += block_size / GROUP_SIZE;
  xp += block_size;
}
result = simd_sum(result);
if (simd_lid == 0) {
  const T value = static_cast<T>(result);
  if (out_row < SILU_ROWS) {
    // silu of the stored value over hc_count in fp32, with MLX's sigmoid.
    const float scaled = static_cast<float>(value) / static_cast<float>(HC_COUNT);
    const float e = 1 / (1 + metal::precise::exp(metal::abs(scaled)));
    y[out_row] = static_cast<T>(scaled * ((scaled < 0) ? e : 1 - e));
  } else {
    y[out_row] = value;
  }
}
