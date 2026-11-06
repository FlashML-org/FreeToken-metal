// One simdgroup per row, the probabilities (softmax over the row, or each logit's own sigmoid)
// in registers: lane l holds experts l, l + 32, ...
// (PER of them). Each lane keeps its best expert (ties to its lowest); a pick takes the simdgroup's
// best (ties to the lowest expert) and only the winning lane rescans its own, so the serial
// selection reads no threadgroup memory and takes no barrier.
constexpr int PER = (E + 31) / 32;
const uint row = thread_position_in_grid.x / 32;
const int lane = thread_index_in_simdgroup;
auto logits = gates + row * E;

float probs[PER];
float row_max = -INFINITY;
for (int i = 0; i < PER; ++i) {
  const int e = lane + 32 * i;
  probs[i] = e < E ? static_cast<float>(logits[e]) : -1.0f;
  if (e < E) {
    row_max = max(row_max, probs[i]);
  }
}
if (SIGMOID) {
  for (int i = 0; i < PER; ++i) {
    if (lane + 32 * i < E) {
      const float tail = 1.0f / (1.0f + exp(abs(probs[i])));
      probs[i] = probs[i] < 0.0f ? tail : 1.0f - tail;
    }
  }
} else {
  row_max = simd_max(row_max);
  float total = 0.0f;
  for (int i = 0; i < PER; ++i) {
    if (lane + 32 * i < E) {
      const float p = exp(probs[i] - row_max);
      probs[i] = p;
      total += p;
    }
  }
  total = simd_sum(total);
  const float inv_total = 1.0f / total;
  for (int i = 0; i < PER; ++i) {
    if (lane + 32 * i < E) {
      probs[i] *= inv_total;
    }
  }
}
float best = -1.0f;
int best_expert = E;
for (int i = 0; i < PER; ++i) {
  if (probs[i] > best) {
    best = probs[i];
    best_expert = lane + 32 * i;
  }
}

// K serial selections; ties go to the lowest expert index.
float picked_total = 0.0f;
for (int k = 0; k < K; ++k) {
  const float group_best = simd_max(best);
  const int chosen = simd_min(best == group_best ? best_expert : E);
  picked_total += group_best;
  if (lane == 0) {
    ids[row * K + k] = chosen;
    scores[row * K + k] = static_cast<InT>(group_best);
  }
  if (best_expert == chosen) {
    best = -1.0f;
    best_expert = E;
    for (int i = 0; i < PER; ++i) {
      if (lane + 32 * i == chosen) {
        probs[i] = -1.0f;
      }
      if (probs[i] > best) {
        best = probs[i];
        best_expert = lane + 32 * i;
      }
    }
  }
}
if (NORM && lane == 0) {
  for (int k = 0; k < K; ++k) {
    const float normalized = static_cast<float>(scores[row * K + k]) / picked_total;
    scores[row * K + k] = static_cast<InT>(normalized);
  }
}
