// AffineSwigluPending: the offload decode's gated activation of one MoE layer's routed pairs,
// after moe_lookup_issue and the resident gate_up pass (affine_gather_qmv, or
// affine_gather_qmv_pair_codes with skip_pending). The resident pairs' act is silu_and_mul of their
// gate_up rows; the pending pairs' threadgroups wait inside the kernel for the request issued names
// (common/mailbox.h), resolve their slots from slot_of as moe_lookup would have, and compute their
// gate_up rows themselves. after is an input that is never read: the shared expert's output, so
// that MLX encodes the shared expert before this kernel, which then waits while it runs. slot_of,
// the mailbox and the banks are written by the CPU behind MLX's back and bound in place.
#pragma once

#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ftmetal {

// Register the Metal source of the kernel; compiled per configuration on first use.
void init_affine_swiglu_pending_library(const std::string& source);

// x [T, 1, 1, H] in a float dtype; w uint32 [S, 2I, H * bits / 32], the gate_up bank of the layer's
// pool; scales_or_codes and biases_or_pairs the bank's scales and biases [S, 2I, H / group_size] in
// x's dtype (code_bits 0), or its pair-code rows uint8 [S, row bytes] and the layer's pair table
// [2^code_bits, 2] in x's dtype; gate_up_resident [T, top_k, 1, 2I], the resident pass's; slots
// int32 [T, top_k] and issued int32 [ISSUED_WORDS], moe_lookup_issue's; slot_of int32 [L *
// num_experts] and mailbox uint32 [MAILBOX_WORDS], the offload cache's; is_fast: the gate_up rows
// on MLX's fast impl, as mx.gather_qmm picks it for H. Returns {act [T, top_k, 1, I] in x's dtype,
// slots int32 [T, top_k]: every pair's pool slot, the mailbox}: the mailbox aliases the argument,
// and the next post takes it so that it runs after every wait.
std::vector<mx::array> affine_swiglu_pending(
    const mx::array& x,
    const mx::array& w,
    const mx::array& scales_or_codes,
    const mx::array& biases_or_pairs,
    const mx::array& gate_up_resident,
    const mx::array& slots,
    const mx::array& issued,
    const mx::array& slot_of,
    const mx::array& mailbox,
    const mx::array& after,
    int layer_id,
    int slot_base,
    int num_experts,
    int group_size,
    int bits,
    int code_bits,
    bool is_fast,
    mx::StreamOrDevice s = {});

struct AffineSwigluPendingConfig {
  int layer_id;
  int slot_base;
  int num_experts;
  int group_size;
  int bits;
  // 0: plain scales and biases.
  int code_bits;
  bool is_fast;

  bool operator==(const AffineSwigluPendingConfig&) const = default;
};

class AffineSwigluPending : public mx::Primitive {
 public:
  AffineSwigluPending(mx::Stream stream, AffineSwigluPendingConfig config)
      : mx::Primitive(stream), config_(config) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;

  const char* name() const override {
    return "AffineSwigluPending";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  AffineSwigluPendingConfig config_;
};

} // namespace ftmetal
