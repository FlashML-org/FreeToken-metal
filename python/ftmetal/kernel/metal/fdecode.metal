// fdecode: paged flash-decode, one query token per request over the paged K/V pool.
// The {{...}} tokens are substituted per configuration before compilation.
// fdecode_partials, grid (32 * NSG * P, HKV, B): a threadgroup streams one kv head's
// keys of one partition once for all G = Hq / Hkv query heads; each simdgroup takes a
// contiguous slice and keeps an online softmax per query head whose accumulator is the
// running weighted mean of V (bounded by max |v|, so a half accumulator cannot
// overflow); the slices merge through threadgroup memory and simdgroup 0 writes the
// partition's unnormalized [G, D] accumulator and its (max, sum) in the log2 domain.
// fdecode_merge, grid (32, HKV, B): one simdgroup per (request, kv head) merges the
// P partials of each query head by log-sum-exp.
#include <metal_stdlib>
using namespace metal;

#define D {{D}}
#define G {{G}}
#define HKV {{HKV}}
#define BS {{BS}}
#define NSG {{NSG}}
#define LPD (D / 32)
#define VW (LPD % 4 == 0 ? 4 : 2)
#define NV (LPD / VW)
typedef {{T}} T;
typedef {{ACC_T}} AccT;
typedef {{STAGE_T}} StageT;
using InVec = vec<T, VW>;
using FloatVec = vec<float, VW>;
using AccVec = vec<AccT, VW>;

// Buffers: 0 q [B, HKV * G, D], 1 k_cache / 2 v_cache [num_pages * BS, HKV, D],
// 3 page_table [B, max_pages] int32, 4 seq_lens [B] int32, 5 part_acc [B, HKV, P, G, D]
// fp32, 6 part_ml [B, HKV, P, G, 2] fp32, 7 max_pages, 8 scale * log2(e).
[[kernel]] void fdecode_partials(
    device const T* q [[buffer(0)]],
    device const T* k_cache [[buffer(1)]],
    device const T* v_cache [[buffer(2)]],
    device const int* page_table [[buffer(3)]],
    device const int* seq_lens [[buffer(4)]],
    device float* part_acc [[buffer(5)]],
    device float* part_ml [[buffer(6)]],
    constant const int& max_pages [[buffer(7)]],
    constant const float& scale_log2e [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tptg [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint partition = tgpig.x;
  const uint kv_head = tgpig.y;
  const uint b = tgpig.z;
  const uint num_partitions = tptg.x;
  const size_t part_row = ((size_t)b * HKV + kv_head) * num_partitions + partition;

  const uint seq_len = seq_lens[b];
  const uint per_part = (seq_len + num_partitions - 1) / num_partitions;
  const uint t0 = partition * per_part;
  const uint t1 = min(seq_len, t0 + per_part);
  if (t0 >= t1) {
    if (sg == 0 && lane == 0) {
      for (int g = 0; g < G; ++g) {
        part_ml[(part_row * G + g) * 2] = -INFINITY;
        part_ml[(part_row * G + g) * 2 + 1] = 0.0f;
      }
    }
    return;
  }
  const uint per_sg = (t1 - t0 + NSG - 1) / NSG;
  const uint s0 = t0 + sg * per_sg;
  const uint s1 = min(t1, s0 + per_sg);

  threadgroup T q_tg[G * D];
  for (uint i = sg * 32 + lane; i < G * D; i += NSG * 32) {
    q_tg[i] = q[((size_t)b * HKV + kv_head) * G * D + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m_g[G], l_g[G];
  AccVec acc[G][NV];
  for (int g = 0; g < G; ++g) {
    m_g[g] = -INFINITY;
    l_g[g] = 0.0f;
    for (int d = 0; d < NV; ++d) acc[g][d] = AccVec(0);
  }

  const uint lane_off = lane * LPD;
  for (uint t = s0; t < s1; ++t) {
    const uint page = page_table[(size_t)b * max_pages + t / BS];
    const size_t base = (((size_t)page * BS + t % BS) * HKV + kv_head) * D + lane_off;
    const device InVec* k_vec = (const device InVec*)(k_cache + base);
    FloatVec kv[NV];
    for (int d = 0; d < NV; ++d) kv[d] = FloatVec(k_vec[d]);
    float w[G];
    for (int g = 0; g < G; ++g) {
      const threadgroup InVec* q_vec = (const threadgroup InVec*)(q_tg + g * D + lane_off);
      FloatVec dot = kv[0] * FloatVec(q_vec[0]);
      for (int d = 1; d < NV; ++d) dot += kv[d] * FloatVec(q_vec[d]);
      float partial = 0.0f;
      for (int j = 0; j < VW; ++j) partial += dot[j];
      w[g] = simd_sum(partial) * scale_log2e;
    }
    const device InVec* v_vec = (const device InVec*)(v_cache + base);
    for (int d = 0; d < NV; ++d) kv[d] = FloatVec(v_vec[d]);
    for (int g = 0; g < G; ++g) {
      const float m_new = max(m_g[g], w[g]);
      const float corr = fast::exp2(m_g[g] - m_new);
      const float p = fast::exp2(w[g] - m_new);
      const float l_new = l_g[g] * corr + p;
      const float inv_l = fast::divide(1.0f, l_new);
      const AccT keep = (AccT)(l_g[g] * corr * inv_l);
      const AccT take = (AccT)(p * inv_l);
      for (int d = 0; d < NV; ++d) acc[g][d] = acc[g][d] * keep + take * AccVec(kv[d]);
      l_g[g] = l_new;
      m_g[g] = m_new;
    }
  }

  threadgroup float tg_ml[NSG * G * 2];
  threadgroup StageT tg_acc[NSG * G * D];
  for (int g = 0; g < G; ++g) {
    if (lane == 0) {
      tg_ml[(sg * G + g) * 2] = m_g[g];
      tg_ml[(sg * G + g) * 2 + 1] = l_g[g];
    }
    for (int d = 0; d < NV; ++d)
      for (int j = 0; j < VW; ++j)
        tg_acc[(sg * G + g) * D + lane_off + d * VW + j] = (StageT)acc[g][d][j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    for (int g = 0; g < G; ++g) {
      float m_max = -INFINITY;
      for (int s = 0; s < NSG; ++s) m_max = max(m_max, tg_ml[(s * G + g) * 2]);
      float l_sum = 0.0f, weight[NSG];
      for (int s = 0; s < NSG; ++s) {
        const float m_s = tg_ml[(s * G + g) * 2];
        weight[s] = (m_s == -INFINITY) ? 0.0f : fast::exp2(m_s - m_max) * tg_ml[(s * G + g) * 2 + 1];
        l_sum += weight[s];
      }
      const size_t out_off = (part_row * G + g) * D + lane_off;
      for (int i = 0; i < LPD; ++i) {
        float o = 0.0f;
        for (int s = 0; s < NSG; ++s) o += (float)tg_acc[(s * G + g) * D + lane_off + i] * weight[s];
        part_acc[out_off + i] = o;
      }
      if (lane == 0) {
        part_ml[(part_row * G + g) * 2] = m_max;
        part_ml[(part_row * G + g) * 2 + 1] = l_sum;
      }
    }
  }
}

// Buffers: 0 part_acc, 1 part_ml (as above), 2 out [B, HKV * G, D], 3 num_partitions.
[[kernel]] void fdecode_merge(
    device const float* part_acc [[buffer(0)]],
    device const float* part_ml [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const int& num_partitions [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint kv_head = tgpig.y;
  const uint b = tgpig.z;
  const uint lane_off = lane * LPD;
  const size_t head_row = (size_t)b * HKV + kv_head;

  for (int g = 0; g < G; ++g) {
    float m_max = -INFINITY;
    for (int p = 0; p < num_partitions; ++p)
      m_max = max(m_max, part_ml[((head_row * num_partitions + p) * G + g) * 2]);
    float l_sum = 0.0f;
    float o[LPD];
    for (int i = 0; i < LPD; ++i) o[i] = 0.0f;
    for (int p = 0; p < num_partitions; ++p) {
      const size_t ml_off = ((head_row * num_partitions + p) * G + g) * 2;
      const float m_p = part_ml[ml_off];
      if (m_p == -INFINITY) continue;
      const float weight = fast::exp2(m_p - m_max);
      l_sum += part_ml[ml_off + 1] * weight;
      const size_t acc_off = ((head_row * num_partitions + p) * G + g) * D + lane_off;
      for (int i = 0; i < LPD; ++i) o[i] += part_acc[acc_off + i] * weight;
    }
    const float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    const size_t out_off = (head_row * G + g) * D + lane_off;
    for (int i = 0; i < LPD; ++i) out[out_off + i] = static_cast<T>(o[i] * inv);
  }
}
