// Adapted from MLX v0.32.3 (Copyright © 2023-2024 Apple Inc., MIT License): the block loop of
// qmv_fast_impl and the safe last block of qmv_impl in mlx/backend/metal/kernels/quantized.h
// (757-821, 824-982), over the HC_COUNT rows of one hidden index.
// One simdgroup per hidden index h of a token: the up projection's rows stream * HIDDEN + h,
// summed as MLX's qmv does for one token (4 bits, 8 values a lane, the last part-block through
// the safe loads) and stored in T, then hc_mix's mean. The last part-block's REMAINDER values
// fill REMAINDER_LANES lanes (input 320: 64 values, lanes 0-7); when SPLIT lane groups of that
// width divide the streams, group g computes its rows on the values of lanes
// 0..REMAINDER_LANES - 1 and shuffles them back there: the same products, added in the same order.

constexpr int pack_factor = get_pack_factor<4, 32>();
constexpr int bytes_per_pack = get_bytes_per_pack<4, 32>();
constexpr int VALUES = pack_factor;
constexpr int BLOCK = VALUES * SIMD_SIZE;
constexpr int scale_step_per_thread = GROUP_SIZE / VALUES;
constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
constexpr int in_vec_size_g = K / GROUP_SIZE;
const uint lane = thread_index_in_simdgroup;
const uint h = threadgroup_position_in_grid.x * SIMDGROUPS + simdgroup_index_in_threadgroup;
const uint token = threadgroup_position_in_grid.y;
if (h >= HIDDEN) {
  return;
}
const device uint8_t* ws[HC_COUNT];
const device T* sl[HC_COUNT];
const device T* bl[HC_COUNT];
for (int stream = 0; stream < HC_COUNT; ++stream) {
  const uint out_row = stream * HIDDEN + h;
  ws[stream] = (const device uint8_t*)weight + out_row * in_vec_size_w + lane * bytes_per_pack;
  sl[stream] = scales + out_row * in_vec_size_g + lane / scale_step_per_thread;
  bl[stream] = biases + out_row * in_vec_size_g + lane / scale_step_per_thread;
}
const device T* xp = x + token * K + lane * VALUES;
float x_thread[VALUES];
float result[HC_COUNT] = {0};
int k = 0;
for (; k < K - BLOCK; k += BLOCK) {
  const float sum = load_vector<T, float, VALUES, 4>(xp, x_thread);
  for (int stream = 0; stream < HC_COUNT; ++stream) {
    result[stream] += qdot<float, VALUES, 4>(
        ws[stream], x_thread, sl[stream][0], bl[stream][0], sum);
    ws[stream] += BLOCK * bytes_per_pack / pack_factor;
    sl[stream] += BLOCK / GROUP_SIZE;
    bl[stream] += BLOCK / GROUP_SIZE;
  }
  xp += BLOCK;
}
constexpr int REMAINDER = K - (K - 1) / BLOCK * BLOCK;
constexpr int REMAINDER_LANES = REMAINDER / VALUES;
constexpr int SPLIT =
    REMAINDER % VALUES == 0 && SIMD_SIZE % REMAINDER_LANES == 0
    ? SIMD_SIZE / REMAINDER_LANES : 1;
constexpr bool SPLIT_ROWS = SPLIT > 1 && HC_COUNT % SPLIT == 0;
if (SPLIT_ROWS) {
  constexpr int GROUP_ROWS = SPLIT_ROWS ? HC_COUNT / SPLIT : 1;
  const int lane_group = lane / REMAINDER_LANES;
  const int lane_shift = lane_group * REMAINDER_LANES;
  const int group_shift = lane_shift / scale_step_per_thread;
  const float sum = load_vector_safe<T, float, VALUES, 4>(
      xp - lane_shift * VALUES, x_thread, VALUES);
  float part[GROUP_ROWS];
  for (int j = 0; j < GROUP_ROWS; ++j) {
    const int stream = lane_group * GROUP_ROWS + j;
    part[j] = qdot_safe<float, VALUES, 4>(
        ws[stream] - lane_shift * bytes_per_pack, x_thread, sl[stream][-group_shift],
        bl[stream][-group_shift], sum, VALUES);
  }
  for (int source = 0; source < SPLIT; ++source) {
    for (int j = 0; j < GROUP_ROWS; ++j) {
      const float value = simd_shuffle_down(part[j], ushort(source * REMAINDER_LANES));
      if (lane < uint(REMAINDER_LANES)) {
        result[source * GROUP_ROWS + j] += value;
      }
    }
  }
} else {
  const int remaining = clamp(int(K - k - lane * VALUES), 0, VALUES);
  if (remaining > 0) {
    const float sum = load_vector_safe<T, float, VALUES, 4>(xp, x_thread, remaining);
    for (int stream = 0; stream < HC_COUNT; ++stream) {
      result[stream] += qdot_safe<float, VALUES, 4>(
          ws[stream], x_thread, sl[stream][0], bl[stream][0], sum, remaining);
    }
  }
}
for (int stream = 0; stream < HC_COUNT; ++stream) {
  result[stream] = simd_sum(result[stream]);
}
if (lane == 0) {
  float total = 0;
  for (int stream = 0; stream < HC_COUNT; ++stream) {
    #pragma METAL fp contract(off)
    const float up = static_cast<float>(static_cast<T>(result[stream]));
    const size_t index = (size_t(token) * HC_COUNT + stream) * HIDDEN + h;
    total += precise_sigmoid(up) * static_cast<float>(normed[index]);
  }
  mixed[size_t(token) * HIDDEN + h] = static_cast<T>(total / static_cast<float>(HC_COUNT));
}
