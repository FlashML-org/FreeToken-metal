// One Gated DeltaNet decode step per request, in place on the conv ring and the
// recurrent state pool (see gdn_decode.metal). Returns (conv_pool, state_pool, y);
// the first two alias their inputs, callers rebind them.
#pragma once

#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ftmetal {

// Register the Metal source template of the step kernel; compiled per configuration.
void init_gdn_decode_library(const std::string& source);

std::vector<mx::array> gdn_decode(
    const mx::array& conv_pool,
    const mx::array& state_pool,
    const mx::array& conv_in,
    const mx::array& conv_weight,
    const mx::array& a,
    const mx::array& b,
    const mx::array& A_log,
    const mx::array& dt_bias,
    const mx::array& slots,
    const mx::array& positions,
    mx::StreamOrDevice s = {});

struct GDNDecodeShape {
  int num_k_heads;
  int num_v_heads;
  int key_dim;
  int value_dim;
  int conv_kernel;
};

class GDNDecode : public mx::Primitive {
 public:
  GDNDecode(mx::Stream stream, GDNDecodeShape shape) : mx::Primitive(stream), shape_(shape) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;

  const char* name() const override {
    return "GDNDecode";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  GDNDecodeShape shape_;
};

} // namespace ftmetal
