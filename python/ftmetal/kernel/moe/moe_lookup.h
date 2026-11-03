// MoeLookup: one MoE layer's routed experts -> bank slots of the offload cache, on the GPU.
// The distinct experts of the real rows are looked up in slot_of; when some are absent the
// kernel posts them to the mailbox and spins until the CPU (ExpertService's coordinator)
// has read them into victim slots, then looks them up again (see moe_lookup.metal).
// mailbox, last_use and hit_stats are updated in place: the outputs alias the inputs (see
// kvcache/pool_rows.h) and callers rebind them. slot_of and the banks are written by the CPU behind
// MLX's back and read coherently.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/mailbox.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ftmetal {

// Register the Metal source of the lookup kernel; compiled on first use.
void init_moe_lookup_library(const std::string& source);

// topk_ids int32 [T, top_k]; moe_step int32 [2] = {real rows, step}; slot_of int32
// [L * E]; mailbox uint32 [MAILBOX_WORDS]; last_use uint32 [S]; hit_stats uint32 [L, 2]
// (lookups, hit experts). Returns {slots int32 [T, top_k], slot_map int32 [E], mailbox,
// last_use, hit_stats}, the slots less slot_base (the first slot of the layer's pool).
// Padding rows and unused experts get a slot the step already uses.
std::vector<mx::array> moe_lookup(
    const mx::array& topk_ids,
    const mx::array& moe_step,
    const mx::array& slot_of,
    const mx::array& mailbox,
    const mx::array& last_use,
    const mx::array& hit_stats,
    int layer_id,
    int slot_base,
    MoeLookupKind kind,
    mx::StreamOrDevice s = {});

struct MoeLookupConfig {
  int layer_id;
  int num_experts;
  MoeLookupKind kind;
  int slot_base;

  bool operator==(const MoeLookupConfig&) const = default;
};

class MoeLookup : public mx::Primitive {
 public:
  MoeLookup(mx::Stream stream, MoeLookupConfig config) : mx::Primitive(stream), config_(config) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;

  const char* name() const override {
    return "MoeLookup";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  MoeLookupConfig config_;
};

} // namespace ftmetal
