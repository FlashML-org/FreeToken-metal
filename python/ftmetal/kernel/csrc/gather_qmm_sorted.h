// Expert-sorted affine quantized gather matmul, each expert's weights at its byte offsets.
// Adapted from MLX v0.32.3 (MIT): the host side of gather_qmm_rhs in
// mlx/backend/metal/quantized.cpp (1661-1823), non-NAX branch: tiles, function constants,
// buffer order and grid. Divergences: the row-tile offsets come from the caller (moe_align)
// instead of MLX's gather_mm_offsets pass over the bank-row ids, and one more buffer holds
// each expert's byte offsets into the weight, scales and biases buffers.
// note (jason-fxz): this is coupled to the internals of mlx==0.32.3 (the kernel in
// metal/gather_qmm_sorted.metal and this host side). Re-check both against the new
// quantized.cpp and quantized.h on any MLX upgrade; the bit-exact tests are the guard.
#pragma once

#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ftmetal {

// Register the Metal source of the kernel, MLX headers inlined; compiled on first use.
void init_gather_qmm_sorted_library(const std::string& source);

// x [B, 1, K] in float32, float16 or bfloat16, its rows sorted by logical expert; w,
// scales and biases: row-contiguous buffers of any dtype, expert e's [n, K * bits / 32]
// uint32 weight (transposed) and [n, K / group_size] scales and biases in x's dtype at byte
// offsets expert_bytes[e] int64 [E, 3] (weight, scales, biases); expert_offsets [E + 1]
// int32, expert e's rows at [expert_offsets[e], expert_offsets[e + 1]) and
// expert_offsets[E] == B. Returns [B, 1, n] in x's dtype. affine only, group size 32, 64 or
// 128, bits 2-6 or 8. Neither the row nor the byte offsets are checked: they stay on the
// device.
mx::array gather_qmm_sorted(
    const mx::array& x,
    const mx::array& w,
    const mx::array& scales,
    const mx::array& biases,
    const mx::array& expert_offsets,
    const mx::array& expert_bytes,
    int n,
    int group_size,
    int bits,
    mx::StreamOrDevice s = {});

struct GatherQMMSortedConfig {
  int group_size;
  int bits;

  bool operator==(const GatherQMMSortedConfig&) const = default;
};

class GatherQMMSorted : public mx::Primitive {
 public:
  GatherQMMSorted(mx::Stream stream, GatherQMMSortedConfig config)
      : mx::Primitive(stream), config_(config) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;

  const char* name() const override {
    return "GatherQMMSorted";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  GatherQMMSortedConfig config_;
};

} // namespace ftmetal
