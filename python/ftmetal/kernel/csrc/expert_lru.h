// SlotLru: the bookkeeping of the routed-expert slot cache, a token-granular LRU over S
// bank slots shared by every MoE layer. An expert is named by its flat id
// layer_id * num_experts + expert. slot_of [num_layers * num_experts] int32 (-1: absent)
// and last_use [num_slots] uint32 (the step a slot was last used, 0: never) are mx.arrays
// the GPU reads, written here in place through their data pointers; id_of_slot (-1: free)
// lives only here. Slots [0, window_slots) are the two prefill windows of num_experts
// slots each (window_slots 0: none); between prefills they are ordinary slots.
#pragma once

#include <compare>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "mlx/array.h"

namespace mx = mlx::core;

namespace ftmetal {

// The CPU address of an evaluated, row-contiguous array the caller writes in place; name
// says which argument is wrong.
void* shared_data(const mx::array& array, const std::string& name);

// (expert, slot): one expert of a layer to read into one bank slot.
using ExpertLoad = std::pair<int32_t, int32_t>;

struct SlotAssignment {
  std::vector<int32_t> slots;
  std::vector<ExpertLoad> loads;
};

class SlotLru {
 public:
  SlotLru(
      int num_layers,
      int num_experts,
      int num_slots,
      int window_slots,
      mx::array slot_of,
      mx::array last_use);

  // Empty the cache, then place flat_ids in slots window_slots, window_slots + 1, ...;
  // returns their slots.
  std::vector<int32_t> fill(const std::vector<int32_t>& flat_ids);

  // The slot of each of one layer's distinct experts at step (from 1), loading the misses
  // into the least recently used slots that this call does not use.
  SlotAssignment resolve(int layer_id, const std::vector<int32_t>& expert_ids, uint32_t step);

  // Drop every expert held in a window slot.
  void invalidate_windows();

  // Every expert of the layer: its slot when resident outside the windows, else its slot in
  // window layer_id % 2, which the loads list. The LRU is not touched.
  SlotAssignment prefill_slot_map(int layer_id) const;

 private:
  // Victim order: least recently used first, then a free slot (flat id -1), then the lowest
  // flat id, then the lowest slot.
  struct Candidate {
    uint32_t last_use;
    int32_t flat_id;
    int32_t slot;

    auto operator<=>(const Candidate&) const = default;
  };

  void check_layer(int layer_id, const char* method) const;

  int num_layers_;
  int num_experts_;
  int num_slots_;
  int window_slots_;
  mx::array slot_of_;
  mx::array last_use_;
  int32_t* slot_of_data_;
  uint32_t* last_use_data_;
  std::vector<int32_t> id_of_slot_;
  // Per-call marks: a slot this call uses, an expert this call names.
  uint32_t call_ = 0;
  std::vector<uint32_t> slot_call_;
  std::vector<uint32_t> expert_call_;
  std::vector<size_t> misses_;
  std::vector<Candidate> candidates_;
};

} // namespace ftmetal
