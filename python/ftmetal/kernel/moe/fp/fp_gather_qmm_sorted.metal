// fp_gather_qmm_sorted: y[b] = x[b] @ dequantize(expert e's weights).T for the rows b of each
// logical expert e, which the caller sorted by expert: offsets[e] is e's first row. The
// {{...}} tokens are substituted per configuration before compilation, and each
// #include "mlx/..." is replaced by that header from the installed mlx wheel (_ext.py).
// Adapted from MLX v0.32.3 (Copyright © 2025-2026 Apple Inc., MIT License):
// fp_gather_qmm_rhs in mlx/backend/metal/kernels/fp_quantized.h (2037-2186), with the
// includes of mlx/backend/metal/kernels/fp_quantized.metal (4-7). Divergences: specialized
// to transpose = true (the nt kernel) without the nvfp4 global scale (its buffer 3 stays
// unbound), and the weights and scales of the logical expert that schedule_row_tile finds
// are read at its byte offsets (buffer 10) into two byte buffers, wherever and in whatever
// bank its rows lie; the tile scan stays over the logical experts, whatever the bank size.
// note (jason-fxz): this is coupled to the internals of mlx==0.32.3: schedule_row_tile,
// the float QuantizedBlockLoader, BlockMMA, the gemm_loop helpers, the align_N / align_K
// function constants and the buffer order. Re-check them against the new fp_quantized.h on
// any MLX upgrade; the bit-exact tests against mx.gather_qmm are the guard.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"

template <typename T, int group_size, int bits, int BM, int BN, int BK, int WM, int WN>
[[kernel]] void fp_gather_qmm_sorted(
    const device T* x [[buffer(0)]],
    const device uint8_t* w [[buffer(1)]],
    const device uint8_t* scales [[buffer(2)]],
    const device int32_t* offsets [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    // MLX passes the bank rows here; we pass the layer's logical experts, so the tile scan
    // in schedule_row_tile costs what a resident layer's does, whatever the bank size.
    const constant int& num_groups [[buffer(9)]],
    // Not in MLX: each logical expert's byte offsets into w and scales.
    const device int64_t* expert_bytes [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::
      BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      true,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

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
  // Not in MLX: group is a logical expert here, its weights live at its byte offsets.
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
  auto expert_scales = scales + bytes[1];
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += y_col_long * K_w;
  expert_scales += y_col_long * K_g;

  // Prepare threadgroup mma operation
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  // Prepare threadgroup loading operations
  thread loader_x_t loader_x(x, K, Xs, simd_group_id, simd_lane_id);
  // MLX offsets by group * stride; everything from here on is MLX's kernel unchanged.
  thread loader_w_t loader_w(
      wl, expert_scales, K, Ws, simd_group_id, simd_lane_id);

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
    "fp_gather_qmm_sorted",
    fp_gather_qmm_sorted,
    {{T}},
    {{GROUP_SIZE}},
    {{BITS}},
    {{BM}},
    {{BN}},
    {{BK}},
    {{WM}},
    {{WN}})
