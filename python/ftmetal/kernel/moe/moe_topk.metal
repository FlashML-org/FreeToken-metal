const uint row = thread_position_in_grid.x / 32;
const uint lane = thread_index_in_simdgroup;
threadgroup float probs[E];
auto logits = gates + row * E;

float row_max = -INFINITY;
for (int e = lane; e < E; e += 32) {
  const float x = static_cast<float>(logits[e]);
  probs[e] = x;
  row_max = max(row_max, x);
}
row_max = simd_max(row_max);
float total = 0.0f;
for (int e = lane; e < E; e += 32) {
  const float p = exp(probs[e] - row_max);
  probs[e] = p;
  total += p;
}
total = simd_sum(total);
const float inv_total = 1.0f / total;
for (int e = lane; e < E; e += 32) {
  probs[e] *= inv_total;
}
simdgroup_barrier(mem_flags::mem_threadgroup);

// K serial selections; ties go to the lowest expert index.
float picked_total = 0.0f;
for (int k = 0; k < K; ++k) {
  float best = -1.0f;
  int best_expert = E;
  for (int e = lane; e < E; e += 32) {
    if (probs[e] > best) {
      best = probs[e];
      best_expert = e;
    }
  }
  const float group_best = simd_max(best);
  const int chosen = simd_min(best == group_best ? best_expert : E);
  picked_total += group_best;
  if (lane == 0) {
    ids[row * K + k] = chosen;
    scores[row * K + k] = static_cast<InT>(group_best);
    probs[chosen] = -1.0f;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}
if (NORM && lane == 0) {
  for (int k = 0; k < K; ++k) {
    const float normalized = static_cast<float>(scores[row * K + k]) / picked_total;
    scores[row * K + k] = static_cast<InT>(normalized);
  }
}
