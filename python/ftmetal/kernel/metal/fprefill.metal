// fprefill: paged flash-attention prefill, a varlen batch of query tokens over the paged
// K/V pool. The {{...}} tokens are substituted per configuration before compilation.
// A tile is one request's run of up to M / G consecutive query tokens; the host lists the
// tiles (request, first query row, query count, position of the first query). A
// threadgroup owns one tile and one kv head: M query rows (token-major, a token's G query
// heads adjacent) in 8-row groups, each group on DS simdgroups that split the head dim.
// It streams the tile's visible keys [0, position of its last query] in BK-key blocks:
// K and V rows are gathered into threadgroup memory through the page table one key at a
// time (any page size), K transposed; S = Q K^T and O += P V run with fp32 accumulators,
// the DS partial products of S summed through threadgroup memory; the online softmax
// keeps each row's (max, sum) in fp32 in the log2 domain. Only blocks that cross the
// causal diagonal are masked. With a sliding window W > 0 a query at position p sees keys
// (p - W, p]: the blocks before the tile's first window are skipped and only blocks
// crossing a window start are masked. Positions count from the page table row's first
// key, so a caller may start the row at the oldest key the batch's windows reach. With
// KV_SHARED, V is staged into K's buffer once S is computed; NPASS > 1 stages each block
// in NPASS column slices, a simdgroup multiplying while its own slice is staged.
// With BLOCKS, a query also sees the keys before its block end (buffer 9, per query token:
// where its image span ends in row positions, 0 for text), so a tile streams up to its
// largest block end; the window still applies. Without it the kernel is the causal one.
// MMA picks how the two GEMMs run: on 8x8 simdgroup matrices (P rounded to the input
// type), or as MetalPerformancePrimitives matmul2d on cooperative tensors (P in fp32),
// which uses the GPU's matrix units where it has them. Everything else is shared: a
// lane's Q, S and O elements live in arrays and their (column, row) comes from the GEMM's
// fragment layout.
// fprefill, grid (32 * NSG * num_tiles, HKV, 1): one pass, writes the output.
// fprefill_partials, grid (32 * NSG * num_tiles, HKV, P): split-KV, each partition takes
// an equal share of the tile's key blocks and writes its unnormalized [M, D] accumulator
// and per-row (max, sum); a partition with no keys writes max -inf.
// fprefill_merge, grid (32 * M, HKV, num_tiles): one simdgroup per query row merges the
// row's partitions by log-sum-exp.
#define MMA_SIMDGROUP 0
#define MMA_MPP 1
#define MMA {{MMA}}
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
#if MMA == MMA_MPP
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

#define D {{D}}
#define G {{G}}
#define HKV {{HKV}}
#define BS {{BS}}
#define NSG {{NSG}}
#define DS {{DS}}
#define BK {{BK}}
#define W {{W}}
#define KV_SHARED {{KV_SHARED}}
#define NPASS {{NPASS}}
#define BLOCKS {{BLOCKS}}
#define M (8 * NSG / DS)
#define DH (D / DS)
// Columns staged per pass.
#define DP (D / NPASS)
// A lane's elements of its simdgroup's 8-row tiles: S [8, BK], Q and O [8, DH].
#define S_LANE (BK / 4)
#define O_LANE (DH / 4)
#define UNROLL _Pragma("clang loop unroll(full)")
typedef {{T}} T;

// V rows are padded by 16 bytes against bank conflicts. K is staged transposed, [DP][BK],
// so that S = Q K^T loads it untransposed; its rows stay unpadded: a K fragment's 2-way
// bank conflict costs less than the threadgroup memory a pad takes.
constant constexpr int PAD = 16 / sizeof(T);
constant constexpr int LDK = BK;
constant constexpr int LDV = DP + PAD;
constant constexpr int K_SIZE = DP * LDK;
constant constexpr int V_SIZE = BK * LDV;
constant constexpr int KV_SHARED_SIZE = K_SIZE > V_SIZE ? K_SIZE : V_SIZE;
// Staging loads and stores move 16 bytes; KEY_LANES lanes cover 128 bytes of a key's row
// (64 when DP / VEC is not a multiple of 8).
constant constexpr int VEC = 16 / sizeof(T);
constant constexpr int KEY_LANES = DP / VEC % 8 == 0 ? 8 : 4;
using Vec2 = vec<T, 2>;
using LoadVec = vec<T, VEC>;

#if MMA == MMA_MPP
using namespace mpp::tensor_ops;
// P V runs in slices of at most 128 columns: on gen-16 GPUs (macOS 26.5) a simdgroup-scope
// matmul2d with a cooperative-tensor left operand or destination returns garbage past 128
// output columns (e.g. M 8, N 256, K 16, bf16 x bf16 -> fp32); N <= 128 is exact.
constexpr int pv_columns() {
  for (int n = 128; n > 16; n -= 16) {
    if (DH % n == 0) return n;
  }
  return 16;
}
constant constexpr int PV_N = pv_columns();
constexpr constant auto QK_DESC = matmul2d_descriptor(
    8, BK, DH, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
constexpr constant auto PV_DESC = matmul2d_descriptor(
    8, PV_N, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
using QKOp = matmul2d<QK_DESC, execution_simdgroup>;
using PVOp = matmul2d<PV_DESC, execution_simdgroup>;
using QTile = decltype(QKOp().get_left_input_cooperative_tensor<T, T, float>());
using KSlice = tensor<threadgroup T, extents<int, BK, DH>, tensor_inline>;
using STile = decltype(QKOp().get_destination_cooperative_tensor<QTile, KSlice, float>());
using PTile = decltype(PVOp().get_left_input_cooperative_tensor<float, T, float>());
using VSlice = tensor<threadgroup T, extents<int, PV_N, BK>, tensor_inline>;
using OTile = decltype(PVOp().get_destination_cooperative_tensor<PTile, VSlice, float>());

// The (column, row) of a lane's i-th Q, S or O element in the cooperative tensors' layout,
// O spanning DH / PV_N of them. Like the 8x8 layout, an 8-row tile gives each lane one row,
// the lanes of a row differ in lane bits 0 and 3, and elements 2j and 2j + 1 are adjacent.
template <typename Tile>
inline short2 element_coord(ushort, int i) {
  if (is_same_v<Tile, OTile>) {
    const auto coord = OTile().get_multidimensional_index(i % (PV_N / 4));
    return short2(i / (PV_N / 4) * PV_N + coord[0], coord[1]);
  } else {
    const auto coord = Tile().get_multidimensional_index(i);
    return short2(coord[0], coord[1]);
  }
}
#else
struct QTile;
struct STile;
struct OTile;

// The (column, row) of a lane's i-th Q, S or O element: pairs along a row of the 8x8
// fragments, the fragments in column order.
template <typename Tile>
inline short2 element_coord(ushort lane, int i) {
  const short quad = lane / 4;
  const short row = (quad & 4) + (lane / 2) % 4;
  const short column = (quad & 2) * 2 + (lane % 2) * 2;
  return short2(i / 2 * 8 + column + i % 2, row);
}
#endif

// The four lanes holding one fragment row differ in lane bits 0 and 3.
inline float fragment_row_max(float x) {
  x = max(x, simd_shuffle_xor(x, 1));
  return max(x, simd_shuffle_xor(x, 8));
}

inline float fragment_row_sum(float x) {
  x += simd_shuffle_xor(x, 1);
  return x + simd_shuffle_xor(x, 8);
}

// s [8, BK] = q [8, DH] k_tg [DH, BK], k_tg pointing at the simdgroup's first column.
inline void multiply_qk(thread const T* q, threadgroup const T* k_tg, thread float* s) {
#if MMA == MMA_MPP
  QKOp qk;
  QTile q_tile;
  STile s_tile;
  UNROLL for (int i = 0; i < O_LANE; ++i) q_tile[i] = q[i];
  UNROLL for (int i = 0; i < S_LANE; ++i) s_tile[i] = 0.0f;
  KSlice k_slice((threadgroup T*)k_tg, extents<int, BK, DH>(), array<int, 2>{1, LDK});
  qk.run(q_tile, k_slice, s_tile);
  UNROLL for (int i = 0; i < S_LANE; ++i) s[i] = s_tile[i];
#else
  UNROLL for (int i = 0; i < S_LANE; ++i) s[i] = 0.0f;
  UNROLL for (int d = 0; d < DH / 8; ++d) {
    simdgroup_matrix<T, 8, 8> q_mat;
    reinterpret_cast<thread Vec2&>(q_mat.thread_elements()) = Vec2(q[2 * d], q[2 * d + 1]);
    UNROLL for (int k = 0; k < BK / 8; ++k) {
      simdgroup_matrix<T, 8, 8> k_mat;
      simdgroup_load(k_mat, k_tg + d * 8 * LDK + k * 8, LDK);
      simdgroup_matrix<float, 8, 8> s_mat;
      reinterpret_cast<thread float2&>(s_mat.thread_elements()) = float2(s[2 * k], s[2 * k + 1]);
      simdgroup_multiply_accumulate(s_mat, q_mat, k_mat, s_mat);
      const float2 s_pair = reinterpret_cast<thread float2&>(s_mat.thread_elements());
      s[2 * k] = s_pair.x;
      s[2 * k + 1] = s_pair.y;
    }
  }
#endif
}

// o [8, DH] += p [8, BK] v_tg [BK, DH], v_tg pointing at the simdgroup's first column.
inline void multiply_pv(thread const float* p, threadgroup const T* v_tg, thread float* o) {
#if MMA == MMA_MPP
  PVOp pv;
  STile s_tile;
  UNROLL for (int i = 0; i < S_LANE; ++i) s_tile[i] = p[i];
  PTile p_tile = pv.get_left_input_cooperative_tensor<float, T, float>(s_tile);
  UNROLL for (int n = 0; n < DH / PV_N; ++n) {
    OTile o_tile;
    UNROLL for (int i = 0; i < PV_N / 4; ++i) o_tile[i] = o[n * PV_N / 4 + i];
    VSlice v_slice(
        (threadgroup T*)v_tg + n * PV_N, extents<int, PV_N, BK>(), array<int, 2>{1, LDV});
    pv.run(p_tile, v_slice, o_tile);
    UNROLL for (int i = 0; i < PV_N / 4; ++i) o[n * PV_N / 4 + i] = o_tile[i];
  }
#else
  UNROLL for (int k = 0; k < BK / 8; ++k) {
    // P in the input type makes PV the same T x T -> fp32 product as Q K^T.
    simdgroup_matrix<T, 8, 8> p_mat;
    reinterpret_cast<thread Vec2&>(p_mat.thread_elements()) = Vec2(float2(p[2 * k], p[2 * k + 1]));
    UNROLL for (int d = 0; d < DH / 8; ++d) {
      simdgroup_matrix<T, 8, 8> v_mat;
      simdgroup_load(v_mat, v_tg + k * 8 * LDV + d * 8, LDV);
      simdgroup_matrix<float, 8, 8> o_mat;
      reinterpret_cast<thread float2&>(o_mat.thread_elements()) = float2(o[2 * d], o[2 * d + 1]);
      simdgroup_multiply_accumulate(o_mat, p_mat, v_mat, o_mat);
      const float2 o_pair = reinterpret_cast<thread float2&>(o_mat.thread_elements());
      o[2 * d] = o_pair.x;
      o[2 * d + 1] = o_pair.y;
    }
  }
#endif
}

// Offset of a key's row for one kv head in the [num_pages * BS, HKV, D] pool.
inline size_t key_row_offset(const device int* page_row, uint key, uint kv_head) {
  return (((size_t)page_row[key / BS] * BS + key % BS) * HKV + kv_head) * D;
}

// Stage the i-th 16-byte vector of a pass's columns of a block's V rows (32 lanes along a
// key), zero past kv_end.
inline void stage_value(
    device const T* v_cache,
    const device int* page_row,
    uint kv_head,
    uint k0,
    uint kv_end,
    uint column_offset,
    uint i,
    threadgroup T* v_tg) {
  const uint v_row = i / (DP / VEC);
  const uint v_column = i % (DP / VEC) * VEC;
  LoadVec v_vec = LoadVec(0);
  if (k0 + v_row < kv_end) {
    v_vec = *(const device LoadVec*)(
        v_cache + key_row_offset(page_row, k0 + v_row, kv_head) + column_offset + v_column);
  }
  *(threadgroup LoadVec*)(v_tg + v_row * LDV + v_column) = v_vec;
}

// Buffers: 0 q [num_tokens, HKV * G, D], 1 k_cache / 2 v_cache [num_pages * BS, HKV, D],
// 3 page_table [B, max_pages] int32, 4 tiles [num_tiles] int4, 5 out [num_tokens, HKV * G,
// D] (one pass) or part_acc [num_tiles, HKV, P, M, D] fp32 with 6 part_ml [num_tiles,
// HKV, P, M, 2] fp32 (split-KV), 7 max_pages, 8 scale * log2(e), 9 block_ends
// [num_tokens] int32 (BLOCKS only).
#if BLOCKS
#define BLOCK_ENDS_ARG , device const int* block_ends [[buffer(9)]]
#define BLOCK_ENDS block_ends
#else
#define BLOCK_ENDS_ARG
#define BLOCK_ENDS nullptr
#endif

template <bool SPLIT>
inline void fprefill_tile(
    device const T* q,
    device const T* k_cache,
    device const T* v_cache,
    device const int* page_table,
    device const int4* tiles,
    device const int* block_ends,
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
  uint kv_end = q_pos0 + q_count;
  if (BLOCKS) {
    for (uint t = 0; t < q_count; ++t) kv_end = max(kv_end, (uint)block_ends[q_row0 + t]);
  }
  const uint num_blocks = (kv_end + BK - 1) / BK;
  // The first query's window starts the tile's keys.
  const uint first_block = W > 0 && q_pos0 + 1 > W ? (q_pos0 + 1 - W) / BK : 0;
  uint block_begin = first_block;
  uint block_end = num_blocks;
  if (SPLIT) {
    const uint per_partition = (num_blocks - first_block + num_partitions - 1) / num_partitions;
    block_begin = min(first_block + partition * per_partition, num_blocks);
    block_end = min(block_begin + per_partition, num_blocks);
  }

  const uint row_group = sg / DS;
  const uint column0 = sg % DS * DH;
  // The pass that stages this simdgroup's columns.
  const uint own_pass = column0 / DP;
  const uint row = row_group * 8 + element_coord<STile>(lane, 0).y;
  const bool row_valid = row < q_count * G;
  // Padding rows repeat the last query; they are computed and never written.
  const uint token = min(row / G, q_count - 1);
  const uint head = kv_head * G + row % G;
  const uint position = q_pos0 + token;
  // Keys before it are visible past the diagonal; 0 hides none of them.
  const uint row_block_end = BLOCKS ? (uint)block_ends[q_row0 + token] : 0;
  const uint group_first_position = q_pos0 + min(row_group * 8 / G, q_count - 1);
  const uint group_last_position = q_pos0 + min((row_group * 8 + 7) / G, q_count - 1);
  const size_t q_offset = ((size_t)(q_row0 + token) * HKV * G + head) * D + column0;

  T q_lane[O_LANE];
  UNROLL for (int i = 0; i < O_LANE; i += 2) {
    const Vec2 pair = *(const device Vec2*)(q + q_offset + element_coord<QTile>(lane, i).x);
    q_lane[i] = pair.x;
    q_lane[i + 1] = pair.y;
  }

  float m = -INFINITY;
  float l = 0.0f;
  float o[O_LANE];
  UNROLL for (int i = 0; i < O_LANE; ++i) o[i] = 0.0f;
  float s[S_LANE];

  const device int* page_row = page_table + (size_t)request * max_pages;
  const uint thread_index = sg * 32 + lane;
  for (uint block = block_begin; block < block_end; ++block) {
    const uint k0 = block * BK;
    for (uint pass = 0; pass < NPASS; ++pass) {
      // Each simdgroup load reads whole lines of few keys: KEY_LANES lanes per K key (keys
      // vary fastest across lane groups for the transposed stores), 32 lanes along a V key.
      for (uint i = thread_index; i < BK * DP / VEC; i += NSG * 32) {
        const uint k_row = i / KEY_LANES % BK;
        const uint k_column = i / (KEY_LANES * BK) * (KEY_LANES * VEC) + i % KEY_LANES * VEC;
        LoadVec k_vec = LoadVec(0);
        if (k0 + k_row < kv_end) {
          k_vec = *(const device LoadVec*)(
              k_cache + key_row_offset(page_row, k0 + k_row, kv_head) + pass * DP + k_column);
        }
        if (!KV_SHARED) {
          stage_value(v_cache, page_row, kv_head, k0, kv_end, pass * DP, i, v_tg);
        }
        UNROLL for (int j = 0; j < VEC; ++j) k_tg[(k_column + j) * LDK + k_row] = k_vec[j];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (NPASS == 1 || pass == own_pass) {
        multiply_qk(q_lane, k_tg + (column0 - pass * DP) * LDK, s);
      }
      if (NPASS > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    if (DS > 1) {
      // Each simdgroup adds the other column slices' partial scores in the same order,
      // so all DS copies of S, and of the softmax state, agree bit for bit.
      UNROLL for (int i = 0; i < S_LANE / 2; ++i) {
        s_tg[(sg * S_LANE / 2 + i) * 32 + lane] = float2(s[2 * i], s[2 * i + 1]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      UNROLL for (int i = 0; i < S_LANE / 2; ++i) {
        float2 sum = s_tg[(row_group * DS * S_LANE / 2 + i) * 32 + lane];
        for (int part = 1; part < DS; ++part) {
          sum += s_tg[((row_group * DS + part) * S_LANE / 2 + i) * 32 + lane];
        }
        s[2 * i] = sum.x;
        s[2 * i + 1] = sum.y;
      }
    }

    // Blocks crossing the causal diagonal or a window start are masked.
    const bool needs_mask =
        k0 + BK - 1 > group_first_position || (W > 0 && k0 + W <= group_last_position);
    float block_max = -INFINITY;
    UNROLL for (int i = 0; i < S_LANE; ++i) {
      float score = s[i] * scale_log2e;
      const uint key = k0 + element_coord<STile>(lane, i).x;
      if (needs_mask &&
          ((key > position && key >= row_block_end) || (W > 0 && key + W <= position))) {
        score = -INFINITY;
      }
      s[i] = score;
      block_max = max(block_max, score);
    }
    const float m_new = max(m, fragment_row_max(block_max));
    // A row that has seen no visible key keeps max -inf; exponents use 0 instead.
    const float m_base = m_new == -INFINITY ? 0.0f : m_new;
    const float correction = fast::exp2(m - m_base);
    float block_sum = 0.0f;
    UNROLL for (int i = 0; i < S_LANE; ++i) {
      s[i] = fast::exp2(s[i] - m_base);
      block_sum += s[i];
    }
    l = l * correction + fragment_row_sum(block_sum);
    m = m_new;
    UNROLL for (int i = 0; i < O_LANE; ++i) o[i] *= correction;

    for (uint pass = 0; pass < NPASS; ++pass) {
      if (KV_SHARED) {
        // Every simdgroup has read K for S (and for DS > 1 summed S) before V replaces it.
        if (NPASS == 1) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = thread_index; i < BK * DP / VEC; i += NSG * 32) {
          stage_value(v_cache, page_row, kv_head, k0, kv_end, pass * DP, i, v_tg);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (NPASS == 1 || pass == own_pass) {
        multiply_pv(s, v_tg + column0 - pass * DP, o);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if (SPLIT) {
    const size_t part_row =
        (((size_t)tile_index * HKV + kv_head) * num_partitions + partition) * M + row;
    device float* acc_row = part_acc + part_row * D + column0;
    UNROLL for (int i = 0; i < O_LANE; i += 2) {
      *(device float2*)(acc_row + element_coord<OTile>(lane, i).x) = float2(o[i], o[i + 1]);
    }
    if (column0 == 0 && element_coord<OTile>(lane, 0).x == 0) {
      part_ml[part_row * 2] = m;
      part_ml[part_row * 2 + 1] = l;
    }
  } else if (row_valid) {
    const float inv_l = 1.0f / l;
    UNROLL for (int i = 0; i < O_LANE; i += 2) {
      *(device Vec2*)(out + q_offset + element_coord<OTile>(lane, i).x) =
          Vec2(float2(o[i], o[i + 1]) * inv_l);
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
    constant const float& scale_log2e [[buffer(8)]] BLOCK_ENDS_ARG,
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup T kv_tg[KV_SHARED ? KV_SHARED_SIZE : K_SIZE + V_SIZE];
  threadgroup float2 s_tg[DS > 1 ? NSG * S_LANE / 2 * 32 : 1];
  fprefill_tile<false>(
      q, k_cache, v_cache, page_table, tiles, BLOCK_ENDS, out, nullptr, nullptr, max_pages,
      scale_log2e, tgpig.x, tgpig.y, 0, 1, sg, lane, kv_tg, kv_tg + (KV_SHARED ? 0 : K_SIZE),
      s_tg);
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
    constant const float& scale_log2e [[buffer(8)]] BLOCK_ENDS_ARG,
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tptg [[threadgroups_per_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup T kv_tg[KV_SHARED ? KV_SHARED_SIZE : K_SIZE + V_SIZE];
  threadgroup float2 s_tg[DS > 1 ? NSG * S_LANE / 2 * 32 : 1];
  fprefill_tile<true>(
      q, k_cache, v_cache, page_table, tiles, BLOCK_ENDS, nullptr, part_acc, part_ml, max_pages,
      scale_log2e, tgpig.x, tgpig.y, tgpig.z, tptg.z, sg, lane, kv_tg,
      kv_tg + (KV_SHARED ? 0 : K_SIZE), s_tg);
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
