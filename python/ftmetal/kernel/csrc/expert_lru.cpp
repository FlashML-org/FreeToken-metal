#include "expert_lru.h"

#include <algorithm>
#include <stdexcept>

namespace ftmetal {

void* shared_data(const mx::array& array, const std::string& name) {
  if (!array.is_available()) {
    throw std::invalid_argument(name + " must be evaluated before it is shared; eval it first");
  }
  if (!array.flags().row_contiguous || array.data_size() != array.size()) {
    throw std::invalid_argument(name + " must be a dense row-contiguous array");
  }
  return const_cast<mx::array&>(array).data<void>();
}

SlotLru::SlotLru(
    int num_layers,
    int num_experts,
    int num_slots,
    int window_slots,
    mx::array slot_of,
    mx::array last_use)
    : num_layers_(num_layers),
      num_experts_(num_experts),
      num_slots_(num_slots),
      window_slots_(window_slots),
      slot_of_(std::move(slot_of)),
      last_use_(std::move(last_use)),
      slot_of_data_(static_cast<int32_t*>(shared_data(slot_of_, "slot_of"))),
      last_use_data_(static_cast<uint32_t*>(shared_data(last_use_, "last_use"))),
      id_of_slot_(num_slots > 0 ? num_slots : 0, -1),
      slot_call_(num_slots > 0 ? num_slots : 0, 0),
      expert_call_(num_experts > 0 ? num_experts : 0, 0) {
  if (num_layers < 1 || num_experts < 1 || num_slots < 1) {
    throw std::invalid_argument("SlotLru: num_layers, num_experts and num_slots must be positive");
  }
  if ((window_slots != 0 && window_slots != 2 * num_experts) || window_slots > num_slots) {
    throw std::invalid_argument(
        "SlotLru: window_slots must be 0 or 2 * num_experts, within num_slots");
  }
  if (slot_of_.dtype() != mx::int32 ||
      slot_of_.size() != static_cast<size_t>(num_layers) * num_experts) {
    throw std::invalid_argument("SlotLru: slot_of must be int32 [num_layers * num_experts]");
  }
  if (last_use_.dtype() != mx::uint32 || last_use_.size() != static_cast<size_t>(num_slots)) {
    throw std::invalid_argument("SlotLru: last_use must be uint32 [num_slots]");
  }
  std::fill(slot_of_data_, slot_of_data_ + slot_of_.size(), -1);
  std::fill(last_use_data_, last_use_data_ + num_slots, 0u);
  candidates_.reserve(num_slots);
}

void SlotLru::check_layer(int layer_id, const char* method) const {
  if (layer_id < 0 || layer_id >= num_layers_) {
    throw std::out_of_range(
        std::string("SlotLru.") + method + ": layer " + std::to_string(layer_id) +
        " is not in [0, " + std::to_string(num_layers_) + ")");
  }
}

std::vector<int32_t> SlotLru::fill(const std::vector<int32_t>& flat_ids) {
  if (flat_ids.size() > static_cast<size_t>(num_slots_ - window_slots_)) {
    throw std::invalid_argument(
        "SlotLru.fill: " + std::to_string(flat_ids.size()) + " experts do not fit in the " +
        std::to_string(num_slots_ - window_slots_) + " slots outside the windows");
  }
  std::fill(slot_of_data_, slot_of_data_ + slot_of_.size(), -1);
  std::fill(last_use_data_, last_use_data_ + num_slots_, 0u);
  std::fill(id_of_slot_.begin(), id_of_slot_.end(), -1);
  int32_t num_ids = num_layers_ * num_experts_;
  std::vector<int32_t> slots(flat_ids.size());
  for (size_t j = 0; j < flat_ids.size(); ++j) {
    int32_t flat_id = flat_ids[j];
    if (flat_id < 0 || flat_id >= num_ids || slot_of_data_[flat_id] >= 0) {
      throw std::invalid_argument(
          "SlotLru.fill: flat id " + std::to_string(flat_id) + " is out of range or repeated");
    }
    slots[j] = window_slots_ + static_cast<int32_t>(j);
    id_of_slot_[slots[j]] = flat_id;
    slot_of_data_[flat_id] = slots[j];
  }
  return slots;
}

SlotAssignment SlotLru::resolve(
    int layer_id,
    const std::vector<int32_t>& expert_ids,
    uint32_t step) {
  check_layer(layer_id, "resolve");
  if (step == 0) {
    throw std::invalid_argument("SlotLru.resolve: steps count from 1; 0 marks a slot never used");
  }
  if (++call_ == 0) {
    std::fill(slot_call_.begin(), slot_call_.end(), 0u);
    std::fill(expert_call_.begin(), expert_call_.end(), 0u);
    call_ = 1;
  }
  SlotAssignment assignment;
  assignment.slots.resize(expert_ids.size());
  misses_.clear();
  int32_t* layer_slots = slot_of_data_ + static_cast<size_t>(layer_id) * num_experts_;
  for (size_t i = 0; i < expert_ids.size(); ++i) {
    int32_t expert = expert_ids[i];
    if (expert < 0 || expert >= num_experts_ || expert_call_[expert] == call_) {
      throw std::invalid_argument(
          "SlotLru.resolve: expert " + std::to_string(expert) + " is out of range or repeated");
    }
    expert_call_[expert] = call_;
    int32_t slot = layer_slots[expert];
    if (slot >= 0) {
      assignment.slots[i] = slot;
      last_use_data_[slot] = step;
      slot_call_[slot] = call_;
    } else {
      misses_.push_back(i);
    }
  }
  if (misses_.empty()) {
    return assignment;
  }
  candidates_.clear();
  for (int32_t slot = 0; slot < num_slots_; ++slot) {
    if (slot_call_[slot] != call_) {
      candidates_.push_back({last_use_data_[slot], id_of_slot_[slot], slot});
    }
  }
  if (candidates_.size() < misses_.size()) {
    throw std::runtime_error(
        "SlotLru.resolve: a cache of " + std::to_string(num_slots_) + " slots cannot hold layer " +
        std::to_string(layer_id) + "'s " + std::to_string(expert_ids.size()) + " experts");
  }
  auto victims_end = candidates_.begin() + static_cast<std::ptrdiff_t>(misses_.size());
  std::partial_sort(candidates_.begin(), victims_end, candidates_.end());
  // A slot used earlier in this step may still be read by this step's GPU work, whose
  // cache could then serve the old row after the new one is written.
  const Candidate& newest = *(victims_end - 1);
  if (newest.last_use >= step) {
    throw std::runtime_error(
        "SlotLru.resolve: layer " + std::to_string(layer_id) + " would evict a slot used at step " +
        std::to_string(newest.last_use) + ", not before step " + std::to_string(step) +
        "; the cache is too small for one step's experts");
  }
  for (size_t j = 0; j < misses_.size(); ++j) {
    const Candidate& victim = candidates_[j];
    size_t i = misses_[j];
    int32_t flat_id = layer_id * num_experts_ + expert_ids[i];
    if (victim.flat_id >= 0) {
      slot_of_data_[victim.flat_id] = -1;
    }
    id_of_slot_[victim.slot] = flat_id;
    slot_of_data_[flat_id] = victim.slot;
    last_use_data_[victim.slot] = step;
    assignment.slots[i] = victim.slot;
    assignment.loads.emplace_back(expert_ids[i], victim.slot);
  }
  return assignment;
}

void SlotLru::invalidate_windows() {
  for (int32_t slot = 0; slot < window_slots_; ++slot) {
    if (id_of_slot_[slot] >= 0) {
      slot_of_data_[id_of_slot_[slot]] = -1;
      id_of_slot_[slot] = -1;
    }
    last_use_data_[slot] = 0;
  }
}

SlotAssignment SlotLru::prefill_slot_map(int layer_id) const {
  check_layer(layer_id, "prefill_slot_map");
  if (window_slots_ != 2 * num_experts_) {
    throw std::logic_error("SlotLru.prefill_slot_map: this cache has no prefill windows");
  }
  SlotAssignment assignment;
  assignment.slots.resize(num_experts_);
  const int32_t* layer_slots = slot_of_data_ + static_cast<size_t>(layer_id) * num_experts_;
  int32_t window_base = (layer_id % 2) * num_experts_;
  for (int32_t expert = 0; expert < num_experts_; ++expert) {
    int32_t slot = layer_slots[expert];
    if (slot >= window_slots_) {
      assignment.slots[expert] = slot;
    } else if (slot >= 0) {
      throw std::logic_error(
          "SlotLru.prefill_slot_map: expert " + std::to_string(expert) + " of layer " +
          std::to_string(layer_id) + " sits in window slot " + std::to_string(slot) +
          "; invalidate the windows first");
    } else {
      assignment.slots[expert] = window_base + expert;
      assignment.loads.emplace_back(expert, window_base + expert);
    }
  }
  return assignment;
}

} // namespace ftmetal
