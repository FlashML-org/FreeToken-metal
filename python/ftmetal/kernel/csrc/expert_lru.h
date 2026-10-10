// SlotLru: the bookkeeping of the routed-expert slot cache. Each MoE layer owns a run of
// bank slots and keeps a token-granular LRU over them. An expert is named by its flat id
// layer_id * num_experts + expert. slot_of [num_layers * num_experts] int32 (-1: absent)
// and last_use [num_slots] uint32 (the step a slot was last used, 0: never) are mx.arrays
// the GPU reads, written here in place through their data pointers; id_of_slot (-1: free)
// lives only here. Slots [0, 2 * window_size) are the two prefill windows (window_size 0:
// none), which hold an overlap prefill layer's absent experts and nothing else; the layers'
// runs follow, layer 0 first, layer_slots[l] slots each. A miss evicts a slot of its own
// layer, which no other layer reads: a step that resolves each layer once (one command
// buffer) never rewrites a slot it has read, whatever the victim order.
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
  // Loads into a slot that held another expert.
  int64_t num_evicted = 0;
};

class SlotLru {
 public:
  SlotLru(
      int num_layers,
      int num_experts,
      int window_size,
      const std::vector<int32_t>& layer_slots,
      mx::array slot_of,
      mx::array last_use);

  // 2 * window_size + the sum of layer_slots.
  int num_slots() const {
    return num_slots_;
  }

  // Empty the cache, then place each of flat_ids in the next slot of its layer's run;
  // returns their slots.
  std::vector<int32_t> fill(const std::vector<int32_t>& flat_ids);

  // The slot of each of one layer's distinct experts at step (from 1), loading the misses
  // into the least recently used slots of the layer's run that this call does not use.
  SlotAssignment resolve(int layer_id, const std::vector<int32_t>& expert_ids, uint32_t step);

  // Every expert of the layer: its slot when resident, else the next slot of window
  // layer_id % 2, which the loads list. The LRU is not touched.
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
  int window_size_;
  // layer_base_[l]: the first slot of layer l's run; layer_base_[num_layers]: num_slots.
  std::vector<int32_t> layer_base_;
  int num_slots_;
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
