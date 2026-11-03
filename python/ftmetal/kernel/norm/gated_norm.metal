constexpr int READS = 4;
const uint row = threadgroup_position_in_grid.x;
const uint first = thread_position_in_threadgroup.x * READS;
const size_t base = size_t(row) * SIZE;
threadgroup float local_sums[32];
threadgroup float local_inv_mean[1];
float values[READS];
float sum_squares = 0;
for (int i = 0; i < READS; ++i) {
  const uint h = first + i;
  const float value = h < SIZE ? static_cast<float>(x[base + h]) : 0;
  values[i] = value;
  sum_squares += value * value;
}
const float inv_rms = row_inv_rms(
    sum_squares, static_cast<float>(SIZE), eps, local_sums, local_inv_mean,
    thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
for (int i = 0; i < READS; ++i) {
  const uint h = first + i;
  if (h < SIZE) {
    const float normed = static_cast<float>(weight[h]) * (values[i] * inv_rms);
    out[base + h] = static_cast<T>(normed * precise_sigmoid(static_cast<float>(z[base + h])));
  }
}
