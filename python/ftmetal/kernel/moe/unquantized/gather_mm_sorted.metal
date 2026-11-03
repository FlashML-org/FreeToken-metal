// gather_mm_sorted: y[b] = x[b] @ (expert e's weight).T for the rows b of each logical expert
// e, which the caller sorted by expert: offsets[e] is e's first row. The {{...}} tokens are
// substituted per configuration before compilation, and each #include "mlx/..." is replaced
// by that header from the installed mlx wheel (_ext.py).
// Adapted from MLX v0.32.3 (Copyright © 2024 Apple Inc., MIT License): gather_mm_rhs in
// mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather.h (10-190), with the
// includes of mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather.metal (4-6).
// Divergences: specialized to transpose_a = false, transpose_b = true (the nt kernel: each
// expert's weight is [N, K]) and float accumulation; GEMMParams gives way to the M, N and K
// of gather_qmm_sorted's buffer order (its leading dimensions are K, K and N, its aligned K
// iterations K / BK, and its column-tile bound check goes: the grid has exactly that many
// column tiles); and the weight of the logical expert that schedule_row_tile finds is read
// at its byte offset (buffer 10) into a byte buffer, wherever and in whatever bank its rows
// lie; the tile scan stays over the logical experts, whatever the bank size.
// note (jason-fxz): this is coupled to the internals of mlx==0.32.3: schedule_row_tile,
// GEMMKernel (its loaders, BlockMMA and gemm_loop), LoopAlignment, the align_N / align_K
// function constants and the buffer order. Re-check them against the new
// steel_gemm_gather.h on any MLX upgrade; the bit-exact tests against mx.gather_mm are the
// guard.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather.h"

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void gather_mm_sorted(
    const device T* A [[buffer(0)]],
    const device uint8_t* B [[buffer(1)]],
    const device int32_t* offsets [[buffer(4)]],
    device T* C [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    // MLX passes the bank rows here; we pass the layer's logical experts, so the tile scan
    // in schedule_row_tile costs what a resident layer's does, whatever the bank size.
    const constant int& num_groups [[buffer(9)]],
    // Not in MLX: each logical expert's byte offset into B.
    const device int64_t* expert_bytes [[buffer(10)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using gemm_kernel =
      GEMMKernel<T, T, BM, BN, BK, WM, WN, false, true, true, true, float>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  int c_row;
  int group;
  short tgp_bm;
  if (!schedule_row_tile<BM>(
          offsets, num_groups, M, tid.y, simd_lane_id, c_row, group, tgp_bm)) {
    return;
  }

  // Prepare threadgroup memory
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  const int c_col = tid.x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  // Prepare threadgroup bounds
  const short tgp_bn = align_N ? BN : short(min(BN, N - c_col));

  // Not in MLX: group is a logical expert here, its weight lives at its byte offset.
  auto Bl = (const device T*)(B + expert_bytes[group]);
  A += c_row_long * K;
  Bl += c_col_long * K;
  C += c_row_long * N + c_col_long;

  // Prepare threadgroup mma operation
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  // Prepare threadgroup loading operations
  thread loader_a_t loader_a(A, K, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(Bl, K, Bs, simd_group_id, simd_lane_id);

  // Prepare iterations
  const int gemm_k_iterations = K / BK;

  // Do unaligned K iterations first
  if (!align_K) {
    const int k_last = gemm_k_iterations * BK;
    const int k_remain = K - k_last;
    const size_t k_jump_a = size_t(k_last);
    const size_t k_jump_b = size_t(k_last);

    // Move loader source ahead to end
    loader_a.src += k_jump_a;
    loader_b.src += k_jump_b;

    // Load tile
    const short2 tile_dims_A = short2(k_remain, tgp_bm);
    const short2 tile_dims_B = short2(k_remain, tgp_bn);

    loader_a.load_safe(tile_dims_A);
    loader_b.load_safe(tile_dims_B);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Do matmul
    mma_op.mma(As, Bs);

    // Reset source back to start
    loader_a.src -= k_jump_a;
    loader_b.src -= k_jump_b;
  }

  const short lbk = 0;

  // Tile aligned don't check
  if ((tgp_bm == BM) && (align_N || tgp_bn == BN)) {
    gemm_kernel::gemm_loop(
        As,
        Bs,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        lbk,
        LoopAlignment<true, true, true>{});
  }

  // Tile partially aligned check rows
  else if (align_N || tgp_bn == BN) {
    gemm_kernel::gemm_loop(
        As,
        Bs,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        lbk,
        LoopAlignment<false, true, true>{});
  }

  // Tile partially aligned check cols
  else if (tgp_bm == BM) {
    gemm_kernel::gemm_loop(
        As,
        Bs,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        lbk,
        LoopAlignment<true, false, true>{});
  }

  // Nothing aligned so check both rows and cols
  else {
    gemm_kernel::gemm_loop(
        As,
        Bs,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        lbk,
        LoopAlignment<false, false, true>{});
  }

  // Store results to device memory
  if (tgp_bm == BM && (align_N || tgp_bn == BN)) {
    mma_op.store_result(C, N);
  } else {
    mma_op.store_result_safe(C, N, short2(tgp_bn, tgp_bm));
  }
}

instantiate_kernel(
    "gather_mm_sorted",
    gather_mm_sorted,
    {{T}},
    {{BM}},
    {{BN}},
    {{BK}},
    {{WM}},
    {{WN}})
