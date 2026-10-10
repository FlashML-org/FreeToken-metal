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

namespace {

// The first slot of each layer's run after the windows, and the slot count last.
std::vector<int32_t> layer_bases(
    int num_layers,
    int num_experts,
    int window_size,
    const std::vector<int32_t>& layer_slots) {
  if (num_layers < 1 || num_experts < 1) {
    throw std::invalid_argument("SlotLru: num_layers and num_experts must be positive");
  }
  if (window_size < 0 || window_size > num_experts) {
    throw std::invalid_argument("SlotLru: window_size must be in [0, num_experts]");
  }
  if (layer_slots.size() != static_cast<size_t>(num_layers)) {
    throw std::invalid_argument("SlotLru: layer_slots must hold one count per layer");
  }
  std::vector<int32_t> bases{2 * window_size};
  for (int32_t count : layer_slots) {
    if (count < 1 || count > num_experts) {
      throw std::invalid_argument("SlotLru: every layer needs 1 to num_experts slots");
    }
    bases.push_back(bases.back() + count);
  }
  return bases;
}

} // namespace

SlotLru::SlotLru(
    int num_layers,
    int num_experts,
    int window_size,
    const std::vector<int32_t>& layer_slots,
    mx::array slot_of,
    mx::array last_use)
    : num_layers_(num_layers),
      num_experts_(num_experts),
      window_size_(window_size),
      layer_base_(layer_bases(num_layers, num_experts, window_size, layer_slots)),
      num_slots_(layer_base_.back()),
      slot_of_(std::move(slot_of)),
      last_use_(std::move(last_use)),
      slot_of_data_(static_cast<int32_t*>(shared_data(slot_of_, "slot_of"))),
      last_use_data_(static_cast<uint32_t*>(shared_data(last_use_, "last_use"))),
      id_of_slot_(num_slots_, -1),
      slot_call_(num_slots_, 0),
      expert_call_(num_experts, 0) {
  if (slot_of_.dtype() != mx::int32 ||
      slot_of_.size() != static_cast<size_t>(num_layers) * num_experts) {
    throw std::invalid_argument("SlotLru: slot_of must be int32 [num_layers * num_experts]");
  }
  if (last_use_.dtype() != mx::uint32 || last_use_.size() != static_cast<size_t>(num_slots_)) {
    throw std::invalid_argument("SlotLru: last_use must be uint32 [num_slots]");
  }
  std::fill(slot_of_data_, slot_of_data_ + slot_of_.size(), -1);
  std::fill(last_use_data_, last_use_data_ + num_slots_, 0u);
  candidates_.reserve(num_experts);
}

void SlotLru::check_layer(int layer_id, const char* method) const {
  if (layer_id < 0 || layer_id >= num_layers_) {
    throw std::out_of_range(
        std::string("SlotLru.") + method + ": layer " + std::to_string(layer_id) +
        " is not in [0, " + std::to_string(num_layers_) + ")");
  }
}

std::vector<int32_t> SlotLru::fill(const std::vector<int32_t>& flat_ids) {
  std::fill(slot_of_data_, slot_of_data_ + slot_of_.size(), -1);
  std::fill(last_use_data_, last_use_data_ + num_slots_, 0u);
  std::fill(id_of_slot_.begin(), id_of_slot_.end(), -1);
  int32_t num_ids = num_layers_ * num_experts_;
  std::vector<int32_t> next_slot(layer_base_.begin(), layer_base_.end() - 1);
  std::vector<int32_t> slots(flat_ids.size());
  for (size_t j = 0; j < flat_ids.size(); ++j) {
    int32_t flat_id = flat_ids[j];
    if (flat_id < 0 || flat_id >= num_ids || slot_of_data_[flat_id] >= 0) {
      throw std::invalid_argument(
          "SlotLru.fill: flat id " + std::to_string(flat_id) + " is out of range or repeated");
    }
    int32_t layer_id = flat_id / num_experts_;
    if (next_slot[layer_id] == layer_base_[layer_id + 1]) {
      throw std::invalid_argument(
          "SlotLru.fill: layer " + std::to_string(layer_id) + " has more experts than its " +
          std::to_string(layer_base_[layer_id + 1] - layer_base_[layer_id]) + " slots");
    }
    slots[j] = next_slot[layer_id]++;
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
  for (int32_t slot = layer_base_[layer_id]; slot < layer_base_[layer_id + 1]; ++slot) {
    if (slot_call_[slot] != call_) {
      candidates_.push_back({last_use_data_[slot], id_of_slot_[slot], slot});
    }
  }
  if (candidates_.size() < misses_.size()) {
    throw std::runtime_error(
        "SlotLru.resolve: layer " + std::to_string(layer_id) + "'s " +
        std::to_string(layer_base_[layer_id + 1] - layer_base_[layer_id]) +
        " slots cannot hold its " + std::to_string(expert_ids.size()) +
        " experts; the scheduler caps a step's rows to fit");
  }
  auto victims_end = candidates_.begin() + static_cast<std::ptrdiff_t>(misses_.size());
  std::partial_sort(candidates_.begin(), victims_end, candidates_.end());
  // Only this layer reads its slots, once per step: a slot it used at this step means the
  // layer resolved twice in one step, whose GPU caches could serve the evicted row.
  const Candidate& newest = *(victims_end - 1);
  if (newest.last_use >= step) {
    throw std::runtime_error(
        "SlotLru.resolve: layer " + std::to_string(layer_id) + " would evict a slot it used at "
        "step " + std::to_string(newest.last_use) + ", not before step " + std::to_string(step));
  }
  for (size_t j = 0; j < misses_.size(); ++j) {
    const Candidate& victim = candidates_[j];
    size_t i = misses_[j];
    int32_t flat_id = layer_id * num_experts_ + expert_ids[i];
    if (victim.flat_id >= 0) {
      slot_of_data_[victim.flat_id] = -1;
      ++assignment.num_evicted;
    }
    id_of_slot_[victim.slot] = flat_id;
    slot_of_data_[flat_id] = victim.slot;
    last_use_data_[victim.slot] = step;
    assignment.slots[i] = victim.slot;
    assignment.loads.emplace_back(expert_ids[i], victim.slot);
  }
  return assignment;
}

SlotAssignment SlotLru::prefill_slot_map(int layer_id) const {
  check_layer(layer_id, "prefill_slot_map");
  const int32_t* layer_slots = slot_of_data_ + static_cast<size_t>(layer_id) * num_experts_;
  int32_t absent = static_cast<int32_t>(std::count(layer_slots, layer_slots + num_experts_, -1));
  if (absent > window_size_) {
    throw std::logic_error(
        "SlotLru.prefill_slot_map: windows of " + std::to_string(window_size_) +
        " slots cannot hold layer " + std::to_string(layer_id) + "'s " + std::to_string(absent) +
        " absent experts");
  }
  SlotAssignment assignment;
  assignment.slots.resize(num_experts_);
  int32_t window_slot = (layer_id % 2) * window_size_;
  for (int32_t expert = 0; expert < num_experts_; ++expert) {
    int32_t slot = layer_slots[expert];
    if (slot >= 0) {
      assignment.slots[expert] = slot;
    } else {
      assignment.slots[expert] = window_slot;
      assignment.loads.emplace_back(expert, window_slot++);
    }
  }
  return assignment;
}

} // namespace ftmetal
