// MoeLookup: one MoE layer's routed experts -> bank slots of the offload cache, on the GPU.
// The distinct experts of the real rows are looked up in slot_of; when some are absent the
// kernel posts them to the mailbox and spins until the CPU (ExpertService's coordinator)
// has read them into victim slots, then looks them up again (see metal/moe_lookup.metal).
// mailbox, last_use and hit_stats are updated in place: the outputs alias the inputs (see
// rows.h) and callers rebind them. slot_of and the banks are written by the CPU behind
// MLX's back and read coherently.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ftmetal {

// The mailbox: uint32 words in 128-byte lines; line 0 the handshake and the shutdown word,
// line 1 the request header, lines 2-9 the request's distinct expert ids (uint16, the absent
// first), line 10 the reply. Shared with metal/moe_lookup.metal.
enum class MailboxWord : int {
  State = 0,
  // The last request's sequence number, incremented by the GPU per request.
  Sequence = 1,
  // The sequence number of the last request answered.
  Acknowledged = 2,
  // Nonzero once the service shut down; written by the CPU only, never cleared. The GPU
  // posts nothing once it is set, and stops waiting when it appears.
  Shutdown = 3,
  Kind = 32,
  Layer = 33,
  Step = 34,
  NumRows = 35,
  NumIds = 36,
  NumMisses = 37,
  Ids = 64,
  NumLoaded = 320,
  NumEvicted = 321,
  Error = 322,
  // uint64 mach_absolute_time stamps.
  ClaimTime = 324,
  DoneTime = 326,
};

constexpr int MAILBOX_WORDS = 1024;
// The id lines hold this many uint16 ids, one threadgroup thread per expert.
constexpr int MAX_LOOKUP_EXPERTS = 512;

enum class MailboxState : uint32_t {
  Idle = 0,
  Posted = 1,
  Claimed = 2,
  Done = 3,
  Failed = 4,
  Shutdown = 5,
};

// Decode and short prefills alike read their misses into the LRU; a probe posts no ids and
// is answered without touching the cache.
enum class MoeLookupKind : uint32_t {
  Decode = 1,
  PrefillCopyMiss = 2,
  Probe = 3,
};

enum class MailboxError : uint32_t {
  None = 0,
  // A read from the checkpoint failed.
  Io = 1,
  // The watchdog found the request unanswered.
  Timeout = 2,
  // The request broke the protocol (sequence, counts, ids) or the host ordering.
  Protocol = 3,
  Shutdown = 4,
  // The layer's pool cannot hold the request's experts.
  Capacity = 5,
};

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
