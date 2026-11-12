// Token sampling as one primitive: the radix-select rounds and the draw of
// sample_tokens.metal, encoded as one dispatch chain over a temporary workspace.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ftmetal {

// Register the Metal source template of the sampling kernels; compiled per logits dtype.
void init_sample_tokens_library(const std::string& source);

// logits [B, V] (bf16, fp16 or fp32); temperatures [B] fp32 > 0; top_k [B] int32, off at
// 0 or >= V; top_p [B] fp32, off at >= 1. A missing top_k or top_p turns that filter off
// for every row and drops its rounds. Each row is split into num_groups chunks. The draw's
// uniform is (bits >> 8) / 2^24 of random_bits [B] uint32, drawn from MLX's global key
// sequence when missing. Returns int32 [B] token ids.
mx::array sample_tokens(
    const mx::array& logits,
    const mx::array& temperatures,
    const std::optional<mx::array>& top_k,
    const std::optional<mx::array>& top_p,
    int num_groups,
    const std::optional<mx::array>& random_bits = std::nullopt,
    mx::StreamOrDevice s = {});

struct SampleConfig {
  bool has_top_k;
  bool has_top_p;
  int num_groups;

  bool operator==(const SampleConfig&) const = default;
};

class SampleTokens : public mx::Primitive {
 public:
  SampleTokens(mx::Stream stream, SampleConfig config) : mx::Primitive(stream), config_(config) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "SampleTokens";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  SampleConfig config_;
};

} // namespace ftmetal
