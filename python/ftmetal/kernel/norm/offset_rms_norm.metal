// One threadgroup per (row, group) slice of SIZE values: with INJECT, the slice plus
// branch * 2 * sigmoid(logit / GROUPS) is stored first; the slice is normed and scaled by
// (1 + w); with ROTARY_DIM, its first ROTARY_DIM values are then rotated as mx.fast.rope does.
// The inject turns fma contraction off, as hc_mix does: the op chain multiplies and adds in
// separate ops, while the norm's sum of squares and the rotation contract as MLX's kernels do.

constexpr int READS = 4;
const uint row = threadgroup_position_in_grid.y;
const uint group = threadgroup_position_in_grid.x;
const uint local_index = thread_position_in_threadgroup.x;
const uint first = local_index * READS;
const size_t base = (size_t(row) * GROUPS + group) * SIZE;
threadgroup float local_sums[32];
threadgroup float local_inv_mean[1];
threadgroup T rotary_values[ROTARY_DIM > 0 ? ROTARY_DIM : 1];
float gate = 0;
if (INJECT) {
  const float logit = static_cast<float>(inject_logits[row * GROUPS + group]);
  gate = 2 * precise_sigmoid(logit / static_cast<float>(GROUPS));
}
float values[READS];
float sum_squares = 0;
for (int i = 0; i < READS; ++i) {
  const uint h = first + i;
  float value = 0;
  if (h < SIZE) {
    if (INJECT) {
      #pragma METAL fp contract(off)
      const T injected = static_cast<T>(
          static_cast<float>(hyper[base + h]) +
          static_cast<float>(branch[size_t(row) * SIZE + h]) * gate);
      hyper_out[base + h] = injected;
      value = static_cast<float>(injected);
    } else {
      value = static_cast<float>(hyper[base + h]);
    }
  }
  values[i] = value;
  sum_squares += value * value;
}
const float inv_rms = row_inv_rms(
    sum_squares, static_cast<float>(SIZE), eps, local_sums, local_inv_mean,
    thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
for (int i = 0; i < READS; ++i) {
  const uint h = first + i;
  if (h < SIZE) {
    const float scale = 1 + static_cast<float>(weight[group * SIZE + h]);
    const T value = static_cast<T>(values[i] * inv_rms * scale);
    if (h < ROTARY_DIM) {
      rotary_values[h] = value;
    } else {
      normed[base + h] = value;
    }
  }
}
if (ROTARY_DIM > 0) {
  constexpr uint HALF = ROTARY_DIM / 2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float position = static_cast<float>(positions[row / ROWS_PER_POSITION]);
  for (uint i = local_index; i < HALF; i += THREADS) {
    const float d = static_cast<float>(i) / static_cast<float>(HALF);
    const float theta = position * metal::exp2(-d * log2_base);
    const float cos_theta = metal::fast::cos(theta);
    const float sin_theta = metal::fast::sin(theta);
    const float x1 = static_cast<float>(rotary_values[i]);
    const float x2 = static_cast<float>(rotary_values[i + HALF]);
    normed[base + i] = static_cast<T>(x1 * cos_theta - x2 * sin_theta);
    normed[base + i + HALF] = static_cast<T>(x1 * sin_theta + x2 * cos_theta);
  }
}
