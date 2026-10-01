// fprefill: paged flash-attention prefill, a varlen batch of query tokens over the paged
// K/V pool. The {{...}} tokens are substituted per configuration before compilation.
// A tile is one request's run of up to M / G consecutive query tokens; the host lists the
// tiles (request, first query row, query count, position of the first query). A
// threadgroup owns one tile and one kv head: M query rows (token-major, a token's G query
// heads adjacent) in 8-row groups, each group on DS simdgroups that split the head dim.
// It streams the tile's visible keys [0, position of its last query] in BK-key blocks:
// K and V rows are gathered into threadgroup memory through the page table one key at a
// time (any page size), K transposed; S = Q K^T and O += P V (P rounded to the input
// type) run on 8x8 simdgroup matrices with fp32 accumulators, the DS partial products of S
// summed through threadgroup memory; the online softmax keeps each row's (max, sum) in
// fp32 in the log2 domain. Only blocks that cross the causal diagonal are masked.
// fprefill, grid (32 * NSG * num_tiles, HKV, 1): one pass, writes the output.
// fprefill_partials, grid (32 * NSG * num_tiles, HKV, P): split-KV, each partition takes
// an equal share of the tile's key blocks and writes its unnormalized [M, D] accumulator
// and per-row (max, sum); a partition with no keys writes max -inf.
// fprefill_merge, grid (32 * M, HKV, num_tiles): one simdgroup per query row merges the
// row's partitions by log-sum-exp.
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

#define D {{D}}
#define G {{G}}
#define HKV {{HKV}}
#define BS {{BS}}
#define NSG {{NSG}}
#define DS {{DS}}
#define BK {{BK}}
#define M (8 * NSG / DS)
#define DH (D / DS)
#define TDH (DH / 8)
#define TK (BK / 8)
#define UNROLL _Pragma("clang loop unroll(full)")
typedef {{T}} T;

// V rows are padded by 16 bytes against bank conflicts. K is staged transposed, [D][BK],
// so that S = Q K^T loads it untransposed; its rows stay unpadded: a K fragment's 2-way
// bank conflict costs less than the threadgroup memory a pad takes.
constant constexpr int PAD = 16 / sizeof(T);
constant constexpr int LDK = BK;
constant constexpr int LDV = D + PAD;
// Staging loads and stores move 16 bytes.
constant constexpr int VEC = 16 / sizeof(T);
using Vec2 = vec<T, 2>;
using LoadVec = vec<T, VEC>;

// The (column, row) of a lane's two adjacent elements in an 8x8 simdgroup matrix.
inline short2 fragment_coord(ushort lane) {
  const short quad = lane / 4;
  const short row = (quad & 4) + (lane / 2) % 4;
  const short column = (quad & 2) * 2 + (lane % 2) * 2;
  return short2(column, row);
}

// The four lanes holding one fragment row differ in lane bits 0 and 3.
inline float fragment_row_max(float x) {
  x = max(x, simd_shuffle_xor(x, 1));
  return max(x, simd_shuffle_xor(x, 8));
}

inline float fragment_row_sum(float x) {
  x += simd_shuffle_xor(x, 1);
  return x + simd_shuffle_xor(x, 8);
}

// Buffers: 0 q [num_tokens, HKV * G, D], 1 k_cache / 2 v_cache [num_pages * BS, HKV, D],
// 3 page_table [B, max_pages] int32, 4 tiles [num_tiles] int4, 5 out [num_tokens, HKV * G,
// D] (one pass) or part_acc [num_tiles, HKV, P, M, D] fp32 with 6 part_ml [num_tiles,
// HKV, P, M, 2] fp32 (split-KV), 7 max_pages, 8 scale * log2(e).
template <bool SPLIT>
inline void fprefill_tile(
    device const T* q,
    device const T* k_cache,
    device const T* v_cache,
    device const int* page_table,
    device const int4* tiles,
    device T* out,
    device float* part_acc,
    device float* part_ml,
    int max_pages,
    float scale_log2e,
    uint tile_index,
    uint kv_head,
    uint partition,
    uint num_partitions,
    uint sg,
    uint lane,
    threadgroup T* k_tg,
    threadgroup T* v_tg,
    threadgroup float2* s_tg) {
  const int4 tile = tiles[tile_index];
  const uint request = tile.x;
  const uint q_row0 = tile.y;
  const uint q_count = tile.z;
  const uint q_pos0 = tile.w;
  const uint kv_end = q_pos0 + q_count;
  const uint num_blocks = (kv_end + BK - 1) / BK;
  uint block_begin = 0;
  uint block_end = num_blocks;
  if (SPLIT) {
    const uint per_partition = (num_blocks + num_partitions - 1) / num_partitions;
    block_begin = min(partition * per_partition, num_blocks);
    block_end = min(block_begin + per_partition, num_blocks);
  }

  const uint row_group = sg / DS;
  const uint column0 = sg % DS * DH;
  const short2 coord = fragment_coord(lane);
  const uint row = row_group * 8 + coord.y;
  const bool row_valid = row < q_count * G;
  // Padding rows repeat the last query; they are computed and never written.
  const uint token = min(row / G, q_count - 1);
  const uint head = kv_head * G + row % G;
  const uint position = q_pos0 + token;
  const uint group_first_position = q_pos0 + min(row_group * 8 / G, q_count - 1);
  const size_t q_offset = ((size_t)(q_row0 + token) * HKV * G + head) * D + column0 + coord.x;

  Vec2 q_frag[TDH];
  UNROLL for (int d = 0; d < TDH; ++d) q_frag[d] = *(const device Vec2*)(q + q_offset + d * 8);

  float m = -INFINITY;
  float l = 0.0f;
  float2 o[TDH];
  UNROLL for (int d = 0; d < TDH; ++d) o[d] = float2(0.0f);

  const device int* page_row = page_table + (size_t)request * max_pages;
  for (uint block = block_begin; block < block_end; ++block) {
    const uint k0 = block * BK;
    // Lanes take consecutive keys so the transposed K stores spread over the banks.
    for (uint i = sg * 32 + lane; i < BK * D / VEC; i += NSG * 32) {
      const uint key_row = i % BK;
      const uint column = i / BK * VEC;
      const uint key = k0 + key_row;
      LoadVec k_vec = LoadVec(0);
      LoadVec v_vec = LoadVec(0);
      if (key < kv_end) {
        const size_t slot = (size_t)page_row[key / BS] * BS + key % BS;
        const size_t offset = (slot * HKV + kv_head) * D + column;
        k_vec = *(const device LoadVec*)(k_cache + offset);
        v_vec = *(const device LoadVec*)(v_cache + offset);
      }
      UNROLL for (int j = 0; j < VEC; ++j) k_tg[(column + j) * LDK + key_row] = k_vec[j];
      *(threadgroup LoadVec*)(v_tg + key_row * LDV + column) = v_vec;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 s[TK];
    UNROLL for (int k = 0; k < TK; ++k) s[k] = float2(0.0f);
    UNROLL for (int d = 0; d < TDH; ++d) {
      simdgroup_matrix<T, 8, 8> q_mat;
      reinterpret_cast<thread Vec2&>(q_mat.thread_elements()) = q_frag[d];
      UNROLL for (int k = 0; k < TK; ++k) {
        simdgroup_matrix<T, 8, 8> k_mat;
        simdgroup_load(k_mat, k_tg + (column0 + d * 8) * LDK + k * 8, LDK);
        simdgroup_matrix<float, 8, 8> s_mat;
        reinterpret_cast<thread float2&>(s_mat.thread_elements()) = s[k];
        simdgroup_multiply_accumulate(s_mat, q_mat, k_mat, s_mat);
        s[k] = reinterpret_cast<thread float2&>(s_mat.thread_elements());
      }
    }
    if (DS > 1) {
      // Each simdgroup adds the other column slices' partial scores in the same order,
      // so all DS copies of S, and of the softmax state, agree bit for bit.
      UNROLL for (int k = 0; k < TK; ++k) s_tg[(sg * TK + k) * 32 + lane] = s[k];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      UNROLL for (int k = 0; k < TK; ++k) {
        s[k] = s_tg[((row_group * DS) * TK + k) * 32 + lane];
        for (int part = 1; part < DS; ++part) {
          s[k] += s_tg[((row_group * DS + part) * TK + k) * 32 + lane];
        }
      }
    }

    const bool is_diagonal = k0 + BK - 1 > group_first_position;
    float block_max = -INFINITY;
    UNROLL for (int k = 0; k < TK; ++k) {
      UNROLL for (int j = 0; j < 2; ++j) {
        float score = s[k][j] * scale_log2e;
        if (is_diagonal && k0 + k * 8 + coord.x + j > position) score = -INFINITY;
        s[k][j] = score;
        block_max = max(block_max, score);
      }
    }
    const float m_new = max(m, fragment_row_max(block_max));
    // A row that has seen no visible key keeps max -inf; exponents use 0 instead.
    const float m_base = m_new == -INFINITY ? 0.0f : m_new;
    const float correction = fast::exp2(m - m_base);
    float block_sum = 0.0f;
    UNROLL for (int k = 0; k < TK; ++k) {
      s[k] = fast::exp2(s[k] - m_base);
      block_sum += s[k][0] + s[k][1];
    }
    l = l * correction + fragment_row_sum(block_sum);
    m = m_new;
    UNROLL for (int d = 0; d < TDH; ++d) o[d] *= correction;

    UNROLL for (int k = 0; k < TK; ++k) {
      // P in the input type makes PV the same T x T -> fp32 product as Q K^T.
      simdgroup_matrix<T, 8, 8> p_mat;
      reinterpret_cast<thread Vec2&>(p_mat.thread_elements()) = Vec2(s[k]);
      UNROLL for (int d = 0; d < TDH; ++d) {
        simdgroup_matrix<T, 8, 8> v_mat;
        simdgroup_load(v_mat, v_tg + k * 8 * LDV + column0 + d * 8, LDV);
        simdgroup_matrix<float, 8, 8> o_mat;
        reinterpret_cast<thread float2&>(o_mat.thread_elements()) = o[d];
        simdgroup_multiply_accumulate(o_mat, p_mat, v_mat, o_mat);
        o[d] = reinterpret_cast<thread float2&>(o_mat.thread_elements());
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (SPLIT) {
    const size_t part_row =
        (((size_t)tile_index * HKV + kv_head) * num_partitions + partition) * M + row;
    device float* acc_row = part_acc + part_row * D + column0 + coord.x;
    UNROLL for (int d = 0; d < TDH; ++d) *(device float2*)(acc_row + d * 8) = o[d];
    if (column0 == 0 && coord.x == 0) {
      part_ml[part_row * 2] = m;
      part_ml[part_row * 2 + 1] = l;
    }
  } else if (row_valid) {
    const float inv_l = 1.0f / l;
    UNROLL for (int d = 0; d < TDH; ++d) {
      *(device Vec2*)(out + q_offset + d * 8) = Vec2(o[d] * inv_l);
    }
  }
}

[[kernel, max_total_threads_per_threadgroup(NSG * 32)]] void fprefill(
    device const T* q [[buffer(0)]],
    device const T* k_cache [[buffer(1)]],
    device const T* v_cache [[buffer(2)]],
    device const int* page_table [[buffer(3)]],
    device const int4* tiles [[buffer(4)]],
    device T* out [[buffer(5)]],
    constant const int& max_pages [[buffer(7)]],
    constant const float& scale_log2e [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup T k_tg[D * LDK];
  threadgroup T v_tg[BK * LDV];
  threadgroup float2 s_tg[DS > 1 ? NSG * TK * 32 : 1];
  fprefill_tile<false>(
      q, k_cache, v_cache, page_table, tiles, out, nullptr, nullptr, max_pages, scale_log2e,
      tgpig.x, tgpig.y, 0, 1, sg, lane, k_tg, v_tg, s_tg);
}

[[kernel, max_total_threads_per_threadgroup(NSG * 32)]] void fprefill_partials(
    device const T* q [[buffer(0)]],
    device const T* k_cache [[buffer(1)]],
    device const T* v_cache [[buffer(2)]],
    device const int* page_table [[buffer(3)]],
    device const int4* tiles [[buffer(4)]],
    device float* part_acc [[buffer(5)]],
    device float* part_ml [[buffer(6)]],
    constant const int& max_pages [[buffer(7)]],
    constant const float& scale_log2e [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tptg [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup T k_tg[D * LDK];
  threadgroup T v_tg[BK * LDV];
  threadgroup float2 s_tg[DS > 1 ? NSG * TK * 32 : 1];
  fprefill_tile<true>(
      q, k_cache, v_cache, page_table, tiles, nullptr, part_acc, part_ml, max_pages,
      scale_log2e, tgpig.x, tgpig.y, tgpig.z, tptg.z, sg, lane, k_tg, v_tg, s_tg);
}

// Buffers: 0 part_acc, 1 part_ml (as above), 2 tiles, 3 out [num_tokens, HKV * G, D],
// 4 the partition count P.
[[kernel]] void fprefill_merge(
    device const float* part_acc [[buffer(0)]],
    device const float* part_ml [[buffer(1)]],
    device const int4* tiles [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant const int& num_partitions [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int LPD = D / 32;
  const uint row = tgpig.x;
  const uint kv_head = tgpig.y;
  const uint tile_index = tgpig.z;
  const int4 tile = tiles[tile_index];
  if (row >= (uint)tile.z * G) {
    return;
  }
  const size_t first_row = ((size_t)tile_index * HKV + kv_head) * num_partitions * M + row;
  float m_max = -INFINITY;
  for (int p = 0; p < num_partitions; ++p) m_max = max(m_max, part_ml[(first_row + p * M) * 2]);
  float l_sum = 0.0f;
  float o[LPD];
  UNROLL for (int i = 0; i < LPD; ++i) o[i] = 0.0f;
  for (int p = 0; p < num_partitions; ++p) {
    const size_t part_row = first_row + p * M;
    const float m_p = part_ml[part_row * 2];
    if (m_p == -INFINITY) continue;
    const float weight = fast::exp2(m_p - m_max);
    l_sum += part_ml[part_row * 2 + 1] * weight;
    UNROLL for (int i = 0; i < LPD; ++i) {
      o[i] += part_acc[part_row * D + lane * LPD + i] * weight;
    }
  }
  const float inv_l = l_sum > 0.0f ? 1.0f / l_sum : 0.0f;
  const size_t out_offset =
      ((size_t)(tile.y + row / G) * HKV * G + kv_head * G + row % G) * D + lane * LPD;
  UNROLL for (int i = 0; i < LPD; ++i) out[out_offset + i] = static_cast<T>(o[i] * inv_l);
}
