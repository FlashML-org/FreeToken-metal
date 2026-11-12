// What the Metal-backed primitives share: the affine quantizations MLX instantiates, templating a
// kernel source per configuration, and the paged-attention inputs and split-KV partials of
// mha_decode and mha_prefill.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"

namespace mx = mlx::core;

namespace ftmetal {

// The Metal name of a floating-point dtype; kernel names the primitive in the error.
inline std::string metal_type(mx::Dtype dtype, const std::string& kernel) {
  switch (dtype) {
    case mx::float16:
      return "half";
    case mx::bfloat16:
      return "bfloat";
    case mx::float32:
      return "float";
    default:
      throw std::invalid_argument(
          "ftmetal " + kernel + ": unsupported dtype " + mx::type_to_name(dtype));
  }
}

// The affine (group size, bits) MLX instantiates.
inline void check_affine_quantization(const std::string& op_name, int group_size, int bits) {
  bool is_instantiated = (group_size == 32 || group_size == 64 || group_size == 128) &&
      (bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8);
  if (!is_instantiated) {
    throw std::invalid_argument(
        "ftmetal " + op_name + ": affine group size must be 32, 64 or 128 and bits 2-6 or 8, " +
        "not (" + std::to_string(group_size) + ", " + std::to_string(bits) + ")");
  }
}

inline void replace_all(std::string& text, const std::string& token, const std::string& value) {
  for (size_t pos = text.find(token); pos != std::string::npos; pos = text.find(token, pos)) {
    text.replace(pos, token.size(), value);
    pos += value.size();
  }
}

// {q, k_cache, v_cache, page_table, lengths}: the k/v pools stay as they are (see
// kvcache/pool_rows.h), the rest is made contiguous; eval_gpu checks the pools with
// check_pools_row_contiguous.
inline std::vector<mx::array> paged_attention_inputs(
    const mx::array& q,
    const mx::array& k_cache,
    const mx::array& v_cache,
    const mx::array& page_table,
    const mx::array& lengths,
    mx::Stream stream) {
  return {
      mx::contiguous(q, false, stream),
      k_cache,
      v_cache,
      mx::contiguous(page_table, false, stream),
      mx::contiguous(lengths, false, stream)};
}

inline void check_pools_row_contiguous(
    const std::vector<mx::array>& inputs,
    const std::string& kernel) {
  if (!inputs[1].flags().row_contiguous || !inputs[2].flags().row_contiguous) {
    throw std::runtime_error("ftmetal " + kernel + ": k/v cache must be row contiguous");
  }
}

// Per-partition unnormalized fp32 accumulators [rows..., head_dim] and (max, sum) pairs
// [rows..., 2]; they live only in this step's command buffer.
struct SplitPartials {
  mx::array acc;
  mx::array ml;
};

inline SplitPartials
split_partials(const mx::Shape& rows, int head_dim, mx::metal::CommandEncoder& encoder) {
  mx::Shape acc_shape = rows;
  acc_shape.push_back(head_dim);
  mx::Shape ml_shape = rows;
  ml_shape.push_back(2);
  SplitPartials partials{
      mx::array(acc_shape, mx::float32, nullptr, {}),
      mx::array(ml_shape, mx::float32, nullptr, {})};
  partials.acc.set_data(mx::allocator::malloc(partials.acc.nbytes()));
  partials.ml.set_data(mx::allocator::malloc(partials.ml.nbytes()));
  encoder.add_temporary(partials.acc);
  encoder.add_temporary(partials.ml);
  return partials;
}

} // namespace ftmetal
