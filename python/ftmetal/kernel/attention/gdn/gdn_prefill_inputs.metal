// Grid: threadgroups (conv_dim / (D * NSG), ceil(T / TPS)); a simdgroup owns one head
// row of D channels (a q, k or v head, D / 32 per lane) for TPS tokens: causal conv
// over the taps and the chunk, silu, then the rms norm and scale of q and k heads. A v
// head's simdgroup also writes the decay and beta of its TPS tokens, one per lane.
// conv_in, taps, a and b are read through their strides (views of the projections).

// The q/k norm epsilon of the decode step.
constexpr float EPS = 1e-6f;
constexpr int CPL = D / 32;
const int row = threadgroup_position_in_grid.x * NSG + simdgroup_index_in_threadgroup;
const ushort lane = thread_index_in_simdgroup;
const int t_begin = threadgroup_position_in_grid.y * TPS;
const int t_end = min(t_begin + TPS, T);
const int c0 = row * D + lane * CPL;

// window[j] holds the input at t - (K - 1) + j; positions before the chunk are taps.
float weight[K][CPL];
float window[K][CPL];
for (int i = 0; i < CPL; ++i) {
  const int c = c0 + i;
  for (int j = 0; j < K; ++j) {
    weight[j][i] = static_cast<float>(conv_weight[c * K + j]);
  }
  for (int j = 0; j < K - 1; ++j) {
    const int t = t_begin - (K - 1) + j;
    window[j][i] = static_cast<float>(
        t < 0 ? taps[(K - 1 + t) * taps_strides[0] + c * taps_strides[1]]
              : conv_in[t * conv_in_strides[0] + c * conv_in_strides[1]]);
  }
}
const float inv_scale = rsqrt(float(D));
const float norm_scale = row < Hk ? inv_scale * inv_scale : inv_scale;
for (int t = t_begin; t < t_end; ++t) {
  float value[CPL];
  float sum_sq = 0.0f;
  for (int i = 0; i < CPL; ++i) {
    window[K - 1][i] =
        static_cast<float>(conv_in[t * conv_in_strides[0] + (c0 + i) * conv_in_strides[1]]);
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
      acc += weight[j][i] * window[j][i];
    }
    value[i] = acc / (1.0f + exp(-acc));
    sum_sq += value[i] * value[i];
    for (int j = 0; j < K - 1; ++j) {
      window[j][i] = window[j + 1][i];
    }
  }
  if (row < 2 * Hk) {
    const float scale = rsqrt(simd_sum(sum_sq) / D + EPS) * norm_scale;
    device InT* dst = (row < Hk ? q : k) + ((size_t)t * Hk + row % Hk) * D + lane * CPL;
    for (int i = 0; i < CPL; ++i) {
      dst[i] = static_cast<InT>(value[i] * scale);
    }
  } else {
    device InT* dst = v + ((size_t)t * Hv + row - 2 * Hk) * D + lane * CPL;
    for (int i = 0; i < CPL; ++i) {
      dst[i] = static_cast<InT>(value[i]);
    }
  }
}

if (row >= 2 * Hk && t_begin + lane < t_end) {
  const int hv = row - 2 * Hk;
  const int t = t_begin + lane;
  const float x = static_cast<float>(a[t * a_strides[0] + hv * a_strides[1]])
                + static_cast<float>(dt_bias[hv]);
  const float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
  g[(size_t)t * Hv + hv] = exp(-exp(static_cast<float>(A_log[hv])) * softplus);
  beta[(size_t)t * Hv + hv] =
      1.0f / (1.0f + exp(-static_cast<float>(b[t * b_strides[0] + hv * b_strides[1]])));
}
