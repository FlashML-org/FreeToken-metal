// Expert-sorted gather matmuls, each expert's weights at its byte offsets: quantized
// (gather_qmm_sorted, affine or a float block mode; gather_qmm_sorted_pair_codes, affine over
// pair-code rows) and unquantized (gather_mm_sorted).
// Adapted from MLX v0.32.3 (MIT): the host sides of gather_qmm_rhs in
// mlx/backend/metal/quantized.cpp (1661-1823) and of gather_mm_rhs in
// mlx/backend/metal/matmul.cpp (2213-2345), non-NAX branches: tiles, function constants,
// buffer order and grid. Divergences: the row-tile offsets come from the caller (moe_align)
// instead of MLX's gather_mm_offsets pass over the bank-row ids, one more buffer holds each
// expert's byte offsets into the weight, scales and biases buffers, and the unquantized
// kernel takes the quantized kernels' buffer order (M, N, K instead of gather_mm_rhs's
// GEMMParams), so the three share one dispatch.
// note (jason-fxz): this is coupled to the internals of mlx==0.32.3 (the kernels in
// moe/affine/affine_gather_qmm_sorted.metal, moe/affine/affine_gather_qmm_sorted_pair_codes.metal,
// moe/fp/fp_gather_qmm_sorted.metal and moe/unquantized/gather_mm_sorted.metal and this host
// side). Re-check them against the new
// quantized.cpp, matmul.cpp, quantized.h, fp_quantized.h and steel_gemm_gather.h on any MLX
// upgrade; the bit-exact tests are the guard.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "moe/affine/affine_pair_codes_shared.h"

namespace mx = mlx::core;

namespace ftmetal {

// Register the Metal sources of the four kernels, headers inlined; compiled on first use.
void init_gather_sorted_libraries(
    const std::string& affine_source,
    const std::string& affine_pair_codes_source,
    const std::string& fp_source,
    const std::string& unquantized_source);

// x [B, 1, K] in float32, float16 or bfloat16, its rows sorted by logical expert; w, scales
// and biases: row-contiguous buffers of any dtype, expert e's [n, K * bits / 32] uint32
// weight (transposed), [n, K / group_size] scales (in x's dtype for affine, uint8 for a float
// block mode) and, affine only, biases in x's dtype at byte offsets expert_bytes[e] int64
// [E, 3] (weight, scales, biases) or [E, 2] (weight, scales); expert_offsets [E + 1] int32,
// expert e's rows at [expert_offsets[e], expert_offsets[e + 1]) and expert_offsets[E] == B.
// Returns [B, 1, n] in x's dtype. mode affine (group size 32, 64 or 128, bits 2-6 or 8), or
// mxfp4, nvfp4 or mxfp8 with MLX's one (bits, group size) for each, no biases and K a
// multiple of 32. Neither the row nor the byte offsets are checked: they stay on the device.
mx::array gather_qmm_sorted(
    const mx::array& x,
    const mx::array& w,
    const mx::array& scales,
    const std::optional<mx::array>& biases,
    const mx::array& expert_offsets,
    const mx::array& expert_bytes,
    int n,
    int group_size,
    int bits,
    const std::string& mode,
    mx::StreamOrDevice s = {});

// gather_qmm_sorted for affine experts in pair-code rows (moe/affine/affine_pair_codes.h): w and
// codes are row-contiguous buffers of any dtype holding expert e's [n, K * bits / 32] uint32
// weight (transposed) and its n * K / group_size code_bits-bit codes (4-byte aligned) at byte
// offsets expert_bytes[e] int64 [E, 2] (weight, codes); pairs [P, 2] in x's dtype, row
// contiguous, the (scale, bias) of each code. Group size 32, 64 or 128, bits 2-6 or 8, code_bits
// 1 to MAX_PAIR_CODE_BITS; the rest as gather_qmm_sorted's. Neither the codes nor the offsets
// are checked.
mx::array gather_qmm_sorted_pair_codes(
    const mx::array& x,
    const mx::array& w,
    const mx::array& codes,
    const mx::array& pairs,
    const mx::array& expert_offsets,
    const mx::array& expert_bytes,
    int n,
    int group_size,
    int bits,
    int code_bits,
    mx::StreamOrDevice s = {});

// gather_qmm_sorted for unquantized experts: w holds expert e's [n, K] weight in x's dtype at
// byte offset expert_bytes[e] int64 [E, 1].
mx::array gather_mm_sorted(
    const mx::array& x,
    const mx::array& w,
    const mx::array& expert_offsets,
    const mx::array& expert_bytes,
    int n,
    mx::StreamOrDevice s = {});

// The donor kernel a GatherSorted runs, each from its own Metal source: affine_gather_qmm_rhs
// (with its scales and biases or over pair-code rows), fp_gather_qmm_rhs or gather_mm_rhs.
// The quantized kernels' K tile (MLX's gather_qmm_rhs BK): a float block mode's K must be
// whole tiles, as the float tile reads its last partial one past the row.
constexpr int QMM_TILE_K = 32;

enum class GatherSortedKernel { Affine, AffinePairCodes, Fp, Unquantized };

struct GatherSortedConfig {
  GatherSortedKernel kernel;
  // The output columns: two calls on the same inputs that differ in n are not equivalent.
  int n;
  // The quantized kernels' only.
  int group_size;
  int bits;
  // The pair-code kernel's only.
  int code_bits = 0;

  bool operator==(const GatherSortedConfig&) const = default;
};

class GatherSorted : public mx::Primitive {
 public:
  GatherSorted(mx::Stream stream, GatherSortedConfig config)
      : mx::Primitive(stream), config_(config) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "GatherSorted";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  GatherSortedConfig config_;
};

} // namespace ftmetal
