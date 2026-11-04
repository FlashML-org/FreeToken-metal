// gather_qmm_sorted_pair_codes: gather_qmm_sorted (affine_gather_qmm_sorted.metal) over pair-code
// expert rows (affine_pair_codes.h): each logical expert's weight and codes at its byte offsets,
// every group's scale and bias the pair table's at the group's code_bits-bit code. The {{...}}
// tokens are substituted per configuration before compilation, and each #include is inlined
// (_ext.py).
// Adapted from MLX v0.32.3 (Copyright © 2023-2024 Apple Inc., MIT License):
// affine_gather_qmm_rhs (2426-2574) and QuantizedBlockLoader (571-697) in
// mlx/backend/metal/kernels/quantized.h, with the includes of
// mlx/backend/metal/kernels/quantized.metal (4-7). Divergences: those of
// affine_gather_qmm_sorted.metal, the expert's codes (byte offsets [E, 2]: weight, codes) and
// the pair table (buffer 3) in place of its scales and biases, and PairCodeBlockLoader:
// QuantizedBlockLoader for reduction_dim = 1 with a code index in place of the scales and
// biases pointers, which looks a group's scale and bias up once, on the group's first K tile,
// where MLX reads them again every tile (the same values; 1.5-2 % of a group-64 call on the
// M4 Pro).
// note (jason-fxz): this is coupled to the internals of mlx==0.32.3 as
// affine_gather_qmm_sorted.metal is, and to QuantizedBlockLoader. Re-check them against the new
// quantized.h on any MLX upgrade; the bit-exact tests against gather_qmm_sorted are the guard.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
#include "moe/affine/affine_pair_codes.h"

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short tgp_size,
    short group_size,
    short bits,
    short code_bits>
struct PairCodeBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  static_assert(
      group_size % (n_reads * pack_factor) == 0,
      "The group size must be a multiple of the columns read per thread.");

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  // Not in MLX: the expert's codes, the pair table, this thread's row's current code and its
  // scale and bias, looked up on the group's first tile.
  const device uint32_t* codes;
  const device T* pairs;
  uint code;
  T scale;
  T bias;

  PairCodeBlockLoader(
      const device uint8_t* src_,
      const device uint32_t* codes_,
      uint first_code,
      const device T* pairs_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(BCOLS_PACKED * bytes_per_pack),
        group_step_cnt(0),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        codes(codes_),
        pairs(pairs_),
        code(first_code + bi * src_ld / group_size) {}

  void look_up_pair() thread {
    if (group_step_cnt == 0) {
      const uint pair = pair_code<code_bits>(codes, code);
      scale = pairs[2 * pair];
      bias = pairs[2 * pair + 1];
    }
  }

  void load_unsafe() thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    look_up_pair();
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    look_up_pair();
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() thread {
    src += tile_stride;
    if (group_steps > 1) {
      group_step_cnt++;
      if (group_step_cnt == group_steps) {
        group_step_cnt = 0;
        code++;
      }
    } else {
      code++;
    }
  }
};

template <
    typename T,
    int group_size,
    int bits,
    int code_bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void gather_qmm_sorted_pair_codes(
    const device T* x [[buffer(0)]],
    const device uint8_t* w [[buffer(1)]],
    const device uint8_t* codes [[buffer(2)]],
    // Not in MLX: the layer's pair table [P, 2] of (scale, bias), in MLX's biases slot.
    const device T* pairs [[buffer(3)]],
    const device int32_t* offsets [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    // MLX passes the bank rows here; we pass the layer's logical experts, so the tile scan
    // in schedule_row_tile costs what a resident layer's does, whatever the bank size.
    const constant int& num_groups [[buffer(9)]],
    // Not in MLX: each logical expert's byte offsets into w and codes.
    const device int64_t* expert_bytes [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = PairCodeBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      code_bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  int y_row;
  int group;
  short tgp_bm;
  if (!schedule_row_tile<BM>(
          offsets, num_groups, M, tid.y, simd_lane_id, y_row, group, tgp_bm)) {
    return;
  }
  // Not in MLX: group is a logical expert here, its weight and codes live at its byte offsets.
  const device int64_t* bytes = expert_bytes + 2 * group;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);

  // Move x and output to the correct block
  auto wl = w + bytes[0];
  auto expert_codes = (const device uint32_t*)(codes + bytes[1]);
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += y_col_long * K_w;

  // Prepare threadgroup mma operation
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  // Prepare threadgroup loading operations
  thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
  // The tile's first output row's first code, where MLX offsets scales and biases by y_col * K_g.
  thread loader_w_t loader_w(
      wl,
      expert_codes,
      uint(y_col) * K_g,
      pairs,
      K,
      Ws,
      simd_group_id,
      simd_lane_id);

  // Tile aligned so check outside of the hot loop
  if (tgp_bm == BM && (align_N || tgp_bn == BN)) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (!align_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(y, N);
  }

  // Tile partially aligned check rows
  else if (align_N || tgp_bn == BN) {
    gemm_loop_unaligned<false, true, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (!align_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_safe(y, N, short2(BN, tgp_bm));
  }

  // Tile partially aligned check cols
  else if (tgp_bm == BM) {
    gemm_loop_unaligned<true, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (!align_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_safe(y, N, short2(tgp_bn, BM));
  }

  // Nothing aligned so check both rows and cols
  else {
    gemm_loop_unaligned<false, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (!align_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_safe(y, N, short2(tgp_bn, tgp_bm));
  }
}

instantiate_kernel(
    "gather_qmm_sorted_pair_codes",
    gather_qmm_sorted_pair_codes,
    {{T}},
    {{GROUP_SIZE}},
    {{BITS}},
    {{CODE_BITS}},
    {{BM}},
    {{BN}},
    {{BK}},
    {{WM}},
    {{WN}})
