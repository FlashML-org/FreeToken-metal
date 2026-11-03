// gdn_step: one Gated DeltaNet decode step for a batch of requests, in place on
// the conv ring and the recurrent state pool. The {{...}} tokens are substituted
// per configuration before compilation. Buffer slots: 0 conv_pool [S, KCONV, CONV_DIM]
// (ring of the last KCONV inputs by position mod KCONV), 1 state_pool [S, HV, DV, DK]
// fp32, 2 y [B, HV, DV], 3 conv_in [B, CONV_DIM], 4 conv_weight [CONV_DIM, KCONV],
// 5 a [B, HV], 6 b [B, HV], 7 A_log [HV], 8 dt_bias [HV], 9 slots [B], 10 positions [B].
// Grid: threadgroups (ceil(DV / ROWS_TG), HV, B) of NSG simdgroups; a simdgroup owns
// one state row at a time, its 32 lanes split the DK columns.
#include <metal_stdlib>
using namespace metal;

#define HK {{HK}}
#define HV {{HV}}
#define DK {{DK}}
#define DV {{DV}}
#define KCONV {{KCONV}}
#define NSG {{NSG}}
#define ROWS_TG {{ROWS_TG}}
#define LPD (DK / 32)
#define GR (HV / HK)
#define KEY_DIM (HK * DK)
#define CONV_DIM (2 * KEY_DIM + HV * DV)
typedef {{T}} T;

constant float QK_NORM_EPS = 1e-6f;

inline float silu(float x) {
  return x / (1.0f + exp(-x));
}

inline float softplus(float x) {
  return max(x, 0.0f) + log(1.0f + exp(-abs(x)));
}

inline int ring_cell(int position) {
  int cell = position % KCONV;
  return cell < 0 ? cell + KCONV : cell;
}

// Causal depthwise conv of channel c at position p: the first KCONV - 1 taps read the
// ring, the last tap is this step's input.
inline float conv_at(
    device const T* ring, device const T* weight, float current, int c, int p) {
  float acc = float(weight[c * KCONV + KCONV - 1]) * current;
  for (int j = 0; j < KCONV - 1; ++j) {
    int cell = ring_cell(p - (KCONV - 1) + j);
    acc += float(weight[c * KCONV + j]) * float(ring[cell * CONV_DIM + c]);
  }
  return acc;
}

[[kernel]] void gdn_step(
    device T* conv_pool [[buffer(0)]],
    device float* state_pool [[buffer(1)]],
    device T* y [[buffer(2)]],
    device const T* conv_in [[buffer(3)]],
    device const T* conv_weight [[buffer(4)]],
    device const T* a [[buffer(5)]],
    device const T* b [[buffer(6)]],
    device const T* A_log [[buffer(7)]],
    device const T* dt_bias [[buffer(8)]],
    device const int* slots [[buffer(9)]],
    device const int* positions [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint ln [[thread_index_in_simdgroup]]) {
  const int tile = tg.x;
  const int hv = tg.y;
  const int bi = tg.z;
  const int hk = hv / GR;
  const int slot = slots[bi];
  const int p = positions[bi];
  device const T* ring = conv_pool + size_t(slot) * KCONV * CONV_DIM;
  device const T* x = conv_in + size_t(bi) * CONV_DIM;

  // q and k of this head: conv, silu, rms norm; each lane holds LPD channels.
  float q[LPD];
  float k[LPD];
  float q_sq = 0.0f;
  float k_sq = 0.0f;
  for (int i = 0; i < LPD; ++i) {
    const int cq = hk * DK + ln * LPD + i;
    const int ck = KEY_DIM + cq;
    q[i] = silu(conv_at(ring, conv_weight, float(x[cq]), cq, p));
    k[i] = silu(conv_at(ring, conv_weight, float(x[ck]), ck, p));
    q_sq += q[i] * q[i];
    k_sq += k[i] * k[i];
  }
  q_sq = simd_sum(q_sq);
  k_sq = simd_sum(k_sq);
  const float inv_scale = rsqrt(float(DK));
  const float q_scale = rsqrt(q_sq / DK + QK_NORM_EPS) * inv_scale * inv_scale;
  const float k_scale = rsqrt(k_sq / DK + QK_NORM_EPS) * inv_scale;
  for (int i = 0; i < LPD; ++i) {
    q[i] *= q_scale;
    k[i] *= k_scale;
  }

  const float gate = float(a[bi * HV + hv]) + float(dt_bias[hv]);
  const float decay = exp(-exp(float(A_log[hv])) * softplus(gate));
  const float beta = 1.0f / (1.0f + exp(-float(b[bi * HV + hv])));

  // This position's ring cell held position p - KCONV, which no reader needs this
  // step, so the ring is updated in the same launch: q/k channels once per head
  // (first tile of the first v-head of the group), v channels by their row's owner.
  device T* ring_out = conv_pool + (size_t(slot) * KCONV + ring_cell(p)) * CONV_DIM;
  if (tile == 0 && hv % GR == 0) {
    for (int i = 0; i < LPD; ++i) {
      const int cq = hk * DK + ln * LPD + i;
      ring_out[cq] = x[cq];
      ring_out[KEY_DIM + cq] = x[KEY_DIM + cq];
    }
  }

  const int row_end = min(tile * ROWS_TG + ROWS_TG, DV);
  for (int r = tile * ROWS_TG + sg; r < row_end; r += NSG) {
    const int cv = 2 * KEY_DIM + hv * DV + r;
    const float v = silu(conv_at(ring, conv_weight, float(x[cv]), cv, p));
    device float* row = state_pool + ((size_t(slot) * HV + hv) * DV + r) * DK + ln * LPD;
    float s[LPD];
    float kv = 0.0f;
#if LPD % 4 == 0
    for (int i = 0; i < LPD / 4; ++i) {
      const float4 loaded = *(device const float4*)(row + 4 * i) * decay;
      s[4 * i] = loaded.x;
      s[4 * i + 1] = loaded.y;
      s[4 * i + 2] = loaded.z;
      s[4 * i + 3] = loaded.w;
    }
    for (int i = 0; i < LPD; ++i) {
      kv += s[i] * k[i];
    }
#else
    for (int i = 0; i < LPD; ++i) {
      s[i] = row[i] * decay;
      kv += s[i] * k[i];
    }
#endif
    kv = simd_sum(kv);
    const float delta = (v - kv) * beta;
    float out = 0.0f;
    for (int i = 0; i < LPD; ++i) {
      s[i] += delta * k[i];
      out += s[i] * q[i];
    }
#if LPD % 4 == 0
    for (int i = 0; i < LPD / 4; ++i) {
      *(device float4*)(row + 4 * i) = float4(s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]);
    }
#else
    for (int i = 0; i < LPD; ++i) {
      row[i] = s[i];
    }
#endif
    out = simd_sum(out);
    if (ln == 0) {
      y[(size_t(bi) * HV + hv) * DV + r] = T(out);
      ring_out[cv] = x[cv];
    }
  }
}
