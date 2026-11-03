// fdecode: paged flash-decode, one query token per request over the paged K/V pool.
// The {{...}} tokens are substituted per configuration before compilation.
// The grid holds P partitions per (request, kv head); a request of seq_len keys uses the
// first request_partitions(seq_len) of them, so a grid fixed at the longest context (a
// replayed graph) costs a short request only the partitions it would get on its own.
// fdecode_partials, grid (32 * NSG * P, HKV * HG, B): a threadgroup streams one kv
// head's keys of one partition once for GH = G / HG of its G = Hq / Hkv query heads
// (HG > 1 trades re-reading the keys for fewer registers per lane at a wide head);
// each simdgroup takes a contiguous slice and keeps an online softmax per query head
// whose accumulator is the running weighted mean of V (bounded by max |v|, so a half
// accumulator cannot overflow); the slices merge through threadgroup memory and
// simdgroup 0 writes the partition's unnormalized [GH, D] accumulator and its (max,
// sum) in the log2 domain.
// fdecode_merge, grid (32 * G, HKV, B): one simdgroup per (request, query head) merges
// the request's partials by log-sum-exp.
#include <metal_stdlib>
using namespace metal;

#define D {{D}}
#define G {{G}}
#define HKV {{HKV}}
#define BS {{BS}}
#define NSG {{NSG}}
#define HG {{HG}}
#define GH (G / HG)
#define PARTITION_TOKENS {{PARTITION_TOKENS}}
#define MIN_PARTITION_TOKENS {{MIN_PARTITION_TOKENS}}
#define MAX_PARTITIONS {{MAX_PARTITIONS}}
#define MIN_THREADGROUPS {{MIN_THREADGROUPS}}
#define LPD (D / 32)
#define VW (LPD % 4 == 0 ? 4 : 2)
#define NV (LPD / VW)
typedef {{T}} T;
typedef {{ACC_T}} AccT;
typedef {{STAGE_T}} StageT;
using InVec = vec<T, VW>;
using FloatVec = vec<float, VW>;
using AccVec = vec<AccT, VW>;

// The host's partition count (kernel/fdecode.py) for max_seqlen_k = seq_len, capped at
// the grid's.
inline uint request_partitions(uint seq_len, uint batch_size, uint grid_partitions) {
  uint partitions =
      clamp((seq_len + PARTITION_TOKENS - 1) / PARTITION_TOKENS, 1u, (uint)MAX_PARTITIONS);
  const uint min_partition_limit = (seq_len + MIN_PARTITION_TOKENS - 1) / MIN_PARTITION_TOKENS;
  while (batch_size * HKV * HG * partitions < MIN_THREADGROUPS &&
         partitions < min_partition_limit) {
    partitions *= 2;
  }
  return min(partitions, grid_partitions);
}

// Buffers: 0 q [B, HKV * G, D], 1 k_cache / 2 v_cache [num_pages * BS, HKV, D],
// 3 page_table [B, max_pages] int32, 4 seq_lens [B] int32, 5 part_acc [B, HKV, P, G, D]
// fp32, 6 part_ml [B, HKV, P, G, 2] fp32, 7 max_pages, 8 scale * log2(e).
[[kernel, max_total_threads_per_threadgroup(NSG * 32)]] void fdecode_partials(
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
  const uint kv_head = tgpig.y / HG;
  const uint head0 = tgpig.y % HG * GH;
  const uint b = tgpig.z;
  const uint seq_len = seq_lens[b];
  const uint num_partitions = request_partitions(seq_len, tptg.z, tptg.x);
  if (partition >= num_partitions) {
    return;
  }
  const size_t part_row = ((size_t)b * HKV + kv_head) * tptg.x + partition;
  const uint per_part = (seq_len + num_partitions - 1) / num_partitions;
  const uint t0 = partition * per_part;
  const uint t1 = min(seq_len, t0 + per_part);
  if (t0 >= t1) {
    if (sg == 0 && lane == 0) {
      for (int g = 0; g < GH; ++g) {
        part_ml[(part_row * G + head0 + g) * 2] = -INFINITY;
        part_ml[(part_row * G + head0 + g) * 2 + 1] = 0.0f;
      }
    }
    return;
  }
  const uint per_sg = (t1 - t0 + NSG - 1) / NSG;
  const uint s0 = t0 + sg * per_sg;
  const uint s1 = min(t1, s0 + per_sg);

  threadgroup T q_tg[GH * D];
  for (uint i = sg * 32 + lane; i < GH * D; i += NSG * 32) {
    q_tg[i] = q[(((size_t)b * HKV + kv_head) * G + head0) * D + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m_g[GH], l_g[GH];
  AccVec acc[GH][NV];
  for (int g = 0; g < GH; ++g) {
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
    float w[GH];
    for (int g = 0; g < GH; ++g) {
      const threadgroup InVec* q_vec = (const threadgroup InVec*)(q_tg + g * D + lane_off);
      FloatVec dot = kv[0] * FloatVec(q_vec[0]);
      for (int d = 1; d < NV; ++d) dot += kv[d] * FloatVec(q_vec[d]);
      float partial = 0.0f;
      for (int j = 0; j < VW; ++j) partial += dot[j];
      w[g] = simd_sum(partial) * scale_log2e;
    }
    const device InVec* v_vec = (const device InVec*)(v_cache + base);
    for (int d = 0; d < NV; ++d) kv[d] = FloatVec(v_vec[d]);
    for (int g = 0; g < GH; ++g) {
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

  if (NSG == 1) {
    // One simdgroup has nothing to merge: its accumulator is the partition's.
    for (int g = 0; g < GH; ++g) {
      const size_t out_off = (part_row * G + head0 + g) * D + lane_off;
      for (int d = 0; d < NV; ++d)
        for (int j = 0; j < VW; ++j) part_acc[out_off + d * VW + j] = (float)acc[g][d][j] * l_g[g];
      if (lane == 0) {
        part_ml[(part_row * G + head0 + g) * 2] = m_g[g];
        part_ml[(part_row * G + head0 + g) * 2 + 1] = l_g[g];
      }
    }
    return;
  }
  threadgroup float tg_ml[NSG * GH * 2];
  threadgroup StageT tg_acc[NSG > 1 ? NSG * GH * D : 1];
  for (int g = 0; g < GH; ++g) {
    if (lane == 0) {
      tg_ml[(sg * GH + g) * 2] = m_g[g];
      tg_ml[(sg * GH + g) * 2 + 1] = l_g[g];
    }
    for (int d = 0; d < NV; ++d)
      for (int j = 0; j < VW; ++j)
        tg_acc[(sg * GH + g) * D + lane_off + d * VW + j] = (StageT)acc[g][d][j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    for (int g = 0; g < GH; ++g) {
      float m_max = -INFINITY;
      for (int s = 0; s < NSG; ++s) m_max = max(m_max, tg_ml[(s * GH + g) * 2]);
      float l_sum = 0.0f, weight[NSG];
      for (int s = 0; s < NSG; ++s) {
        const float m_s = tg_ml[(s * GH + g) * 2];
        weight[s] =
            (m_s == -INFINITY) ? 0.0f : fast::exp2(m_s - m_max) * tg_ml[(s * GH + g) * 2 + 1];
        l_sum += weight[s];
      }
      const size_t out_off = (part_row * G + head0 + g) * D + lane_off;
      for (int i = 0; i < LPD; ++i) {
        float o = 0.0f;
        for (int s = 0; s < NSG; ++s) o += (float)tg_acc[(s * GH + g) * D + lane_off + i] * weight[s];
        part_acc[out_off + i] = o;
      }
      if (lane == 0) {
        part_ml[(part_row * G + head0 + g) * 2] = m_max;
        part_ml[(part_row * G + head0 + g) * 2 + 1] = l_sum;
      }
    }
  }
}

// Buffers: 0 part_acc, 1 part_ml (as above), 2 seq_lens, 3 out [B, HKV * G, D],
// 4 the grid's partition count P.
[[kernel]] void fdecode_merge(
    device const float* part_acc [[buffer(0)]],
    device const float* part_ml [[buffer(1)]],
    device const int* seq_lens [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant const int& grid_partitions [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tptg [[threadgroups_per_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint g = tgpig.x;
  const uint kv_head = tgpig.y;
  const uint b = tgpig.z;
  const uint num_partitions = request_partitions(seq_lens[b], tptg.z, grid_partitions);
  const uint lane_off = lane * LPD;
  const size_t head_row = (size_t)b * HKV + kv_head;

  float m_max = -INFINITY;
  for (uint p = 0; p < num_partitions; ++p)
    m_max = max(m_max, part_ml[((head_row * grid_partitions + p) * G + g) * 2]);
  float l_sum = 0.0f;
  float o[LPD];
  for (int i = 0; i < LPD; ++i) o[i] = 0.0f;
  for (uint p = 0; p < num_partitions; ++p) {
    const size_t ml_off = ((head_row * grid_partitions + p) * G + g) * 2;
    const float m_p = part_ml[ml_off];
    if (m_p == -INFINITY) continue;
    const float weight = fast::exp2(m_p - m_max);
    l_sum += part_ml[ml_off + 1] * weight;
    const size_t acc_off = ((head_row * grid_partitions + p) * G + g) * D + lane_off;
    for (int i = 0; i < LPD; ++i) o[i] += part_acc[acc_off + i] * weight;
  }
  const float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  const size_t out_off = (head_row * G + g) * D + lane_off;
  for (int i = 0; i < LPD; ++i) out[out_off + i] = static_cast<T>(o[i] * inv);
}
