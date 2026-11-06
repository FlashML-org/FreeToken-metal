#include "service/expert/expert_lru.h"

#include <algorithm>
#include <cmath>
#include <functional>
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

// Throw unless the layout's slots and pools are in range.
void check_layout(
    int num_layers,
    int num_experts,
    int buffer_size,
    const std::vector<int32_t>& layer_pools,
    const std::vector<int32_t>& pool_bounds,
    const std::vector<int32_t>& pool_volatile) {
  if (num_layers < 1 || num_experts < 1) {
    throw std::invalid_argument("SlotLru: num_layers and num_experts must be positive");
  }
  if (buffer_size < 0 || buffer_size > num_experts) {
    throw std::invalid_argument("SlotLru: buffer_size must be in [0, num_experts]");
  }
  if (layer_pools.size() != static_cast<size_t>(num_layers) || pool_bounds.size() < 2 ||
      pool_volatile.size() + 1 != pool_bounds.size()) {
    throw std::invalid_argument(
        "SlotLru: layer_pools must hold one pool per layer, pool_bounds a pool or more and "
        "pool_volatile one count per pool");
  }
  int32_t num_pools = static_cast<int32_t>(pool_volatile.size());
  for (int32_t pool : layer_pools) {
    if (pool < 0 || pool >= num_pools) {
      throw std::invalid_argument("SlotLru: a layer's pool is out of range");
    }
  }
  if (pool_bounds[0] != 0 ||
      std::adjacent_find(pool_bounds.begin(), pool_bounds.end(), std::greater_equal<int32_t>()) !=
          pool_bounds.end()) {
    throw std::invalid_argument("SlotLru: pool_bounds must rise from slot 0");
  }
  for (int32_t pool = 0; pool < num_pools; ++pool) {
    int32_t num_slots = pool_bounds[pool + 1] - pool_bounds[pool];
    if (pool_volatile[pool] < 0 || pool_volatile[pool] > num_slots) {
      throw std::invalid_argument("SlotLru: a pool's volatile slots are out of range");
    }
  }
}

} // namespace

SlotLru::SlotLru(
    int num_layers,
    int num_experts,
    int buffer_size,
    const std::vector<int32_t>& layer_pools,
    const std::vector<int32_t>& pool_bounds,
    const std::vector<int32_t>& pool_volatile,
    mx::array slot_of,
    mx::array last_use,
    double lrfu_half_life)
    : num_layers_(num_layers),
      num_experts_(num_experts),
      buffer_size_(buffer_size),
      layer_pools_(layer_pools),
      pool_bounds_(pool_bounds),
      pool_volatile_(pool_volatile),
      slot_of_(std::move(slot_of)),
      last_use_(std::move(last_use)),
      slot_of_data_(static_cast<int32_t*>(shared_data(slot_of_, "slot_of"))),
      last_use_data_(static_cast<uint32_t*>(shared_data(last_use_, "last_use"))),
      expert_call_(num_experts, 0) {
  check_layout(num_layers, num_experts, buffer_size, layer_pools, pool_bounds, pool_volatile);
  if (slot_of_.dtype() != mx::int32 ||
      slot_of_.size() != static_cast<size_t>(num_layers) * num_experts) {
    throw std::invalid_argument("SlotLru: slot_of must be int32 [num_layers * num_experts]");
  }
  if (last_use_.dtype() != mx::uint32 || last_use_.size() != static_cast<size_t>(num_slots())) {
    throw std::invalid_argument("SlotLru: last_use must be uint32 [num_slots]");
  }
  id_of_slot_.assign(num_slots(), -1);
  is_held_.assign(num_slots(), 0);
  slot_call_.assign(num_slots(), 0);
  std::fill(slot_of_data_, slot_of_data_ + slot_of_.size(), -1);
  std::fill(last_use_data_, last_use_data_ + num_slots(), 0u);
  candidates_.reserve(num_slots());
  if (!(lrfu_half_life >= 0)) {
    throw std::invalid_argument("SlotLru: lrfu_half_life must be >= 0");
  }
  half_life_ = lrfu_half_life;
  score_.assign(num_slots(), 0.0);
  counted_use_.assign(num_slots(), 0u);
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
  std::fill(last_use_data_, last_use_data_ + num_slots(), 0u);
  std::fill(id_of_slot_.begin(), id_of_slot_.end(), -1);
  std::fill(is_held_.begin(), is_held_.end(), 0);
  if (half_life_ != 0) {
    std::fill(score_.begin(), score_.end(), 0.0);
    std::fill(counted_use_.begin(), counted_use_.end(), 0u);
  }
  int32_t num_ids = num_layers_ * num_experts_;
  std::vector<int32_t> next_slot(pool_bounds_.begin(), pool_bounds_.end() - 1);
  std::vector<int32_t> slots(flat_ids.size());
  for (size_t j = 0; j < flat_ids.size(); ++j) {
    int32_t flat_id = flat_ids[j];
    if (flat_id < 0 || flat_id >= num_ids || slot_of_data_[flat_id] >= 0) {
      throw std::invalid_argument(
          "SlotLru.fill: flat id " + std::to_string(flat_id) + " is out of range or repeated");
    }
    int32_t pool = layer_pools_[flat_id / num_experts_];
    if (next_slot[pool] == pool_bounds_[pool + 1]) {
      throw std::invalid_argument(
          "SlotLru.fill: pool " + std::to_string(pool) + " has more experts than its " +
          std::to_string(pool_bounds_[pool + 1] - pool_bounds_[pool]) + " slots");
    }
    slots[j] = next_slot[pool]++;
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
  count_uses(step);
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
  int32_t pool = layer_pools_[layer_id];
  int32_t pool_begin = pool_bounds_[pool];
  int32_t pool_end = pool_bounds_[pool + 1];
  for (int32_t slot = pool_begin; slot < pool_end; ++slot) {
    // LRFU's order could put a slot an earlier layer used this step first; LRU's never does.
    if (slot_call_[slot] != call_ && !is_held_[slot] &&
        (half_life_ == 0 || last_use_data_[slot] < step)) {
      candidates_.push_back({last_use_data_[slot], id_of_slot_[slot], slot});
    }
  }
  if (candidates_.size() < misses_.size()) {
    throw std::runtime_error(
        "SlotLru.resolve: pool " + std::to_string(pool) + "'s " +
        std::to_string(pool_end - pool_begin) + " slots cannot hold layer " +
        std::to_string(layer_id) + "'s " + std::to_string(expert_ids.size()) +
        " experts; the scheduler caps a forward's rows to fit");
  }
  auto victims_end = candidates_.begin() + static_cast<std::ptrdiff_t>(misses_.size());
  std::partial_sort(
      candidates_.begin(), victims_end, candidates_.end(),
      [this](const Candidate& a, const Candidate& b) { return is_victim_before(a, b); });
  // The cache splits a forward into steps whose experts fit every pool at once, so the
  // victims were all last used before this step.
  const Candidate& newest = *(victims_end - 1);
  if (newest.last_use >= step) {
    throw std::runtime_error(
        "SlotLru.resolve: layer " + std::to_string(layer_id) + " would evict a slot its pool " +
        "used at step " + std::to_string(newest.last_use) + ", not before step " +
        std::to_string(step));
  }
  for (size_t j = 0; j < misses_.size(); ++j) {
    const Candidate& victim = candidates_[j];
    size_t i = misses_[j];
    int32_t flat_id = layer_id * num_experts_ + expert_ids[i];
    if (victim.flat_id >= 0) {
      slot_of_data_[victim.flat_id] = -1;
      ++assignment.num_evicted;
      assignment.evicted.emplace_back(victim.flat_id, victim.slot);
    }
    id_of_slot_[victim.slot] = flat_id;
    slot_of_data_[flat_id] = victim.slot;
    last_use_data_[victim.slot] = step;
    score_load(victim.slot, step);
    assignment.slots[i] = victim.slot;
    assignment.loads.emplace_back(expert_ids[i], victim.slot);
  }
  return assignment;
}

int32_t SlotLru::claim_free_slot(int32_t pool, uint32_t step) {
  for (int32_t slot = pool_bounds_[pool]; slot < pool_bounds_[pool + 1]; ++slot) {
    if (id_of_slot_[slot] < 0 && !is_held_[slot] && last_use_data_[slot] < step &&
        slot_call_[slot] != call_) {
      is_held_[slot] = 1;
      return slot;
    }
  }
  return -1;
}

void SlotLru::move_load(int32_t from, int32_t to) {
  int32_t flat_id = id_of_slot_[from];
  if (flat_id < 0 || id_of_slot_[to] >= 0 || !is_held_[to] || is_held_[from]) {
    throw std::logic_error(
        "SlotLru.move_load: slot " + std::to_string(from) + " holds no load or slot " +
        std::to_string(to) + " is no spare");
  }
  id_of_slot_[to] = flat_id;
  slot_of_data_[flat_id] = to;
  last_use_data_[to] = last_use_data_[from];
  slot_call_[to] = slot_call_[from];
  if (half_life_ != 0) {
    score_[to] = score_[from];
    counted_use_[to] = counted_use_[from];
  }
  is_held_[to] = 0;
  id_of_slot_[from] = -1;
  // Never read in this step (its evicted expert was a victim, the load moved before the GPU saw
  // it): once released it may take a load of this step again.
  last_use_data_[from] = 0;
  is_held_[from] = 1;
}

void SlotLru::reset_scores() {
  if (half_life_ == 0) {
    return;
  }
  count_uses(0);
  for (int32_t slot = 0; slot < num_slots(); ++slot) {
    score_[slot] = counted_use_[slot] / half_life_;
  }
}

void SlotLru::count_uses(uint32_t step) {
  // Once per step: a slot is used at most once in a step, and last_use keeps the last.
  if (half_life_ == 0 || (step != 0 && step == counted_step_)) {
    return;
  }
  counted_step_ = step;
  for (int32_t slot = 0; slot < num_slots(); ++slot) {
    uint32_t last_use = last_use_data_[slot];
    if (last_use > counted_use_[slot] && id_of_slot_[slot] >= 0) {
      // score * 2^-(age / half-life) + 1, in log2 with the decay to step 0 folded in.
      double now = last_use / half_life_;
      score_[slot] = std::log2(std::exp2(score_[slot] - now) + 1.0) + now;
      counted_use_[slot] = last_use;
    }
  }
}

void SlotLru::score_load(int32_t slot, uint32_t step) {
  if (half_life_ != 0) {
    score_[slot] = step / half_life_;
    counted_use_[slot] = step;
  }
}

bool SlotLru::is_victim_before(const Candidate& a, const Candidate& b) const {
  if (half_life_ == 0) {
    return a < b;
  }
  // Free slots first, then the lowest score, then LRU's order.
  bool a_free = a.flat_id < 0;
  bool b_free = b.flat_id < 0;
  if (a_free != b_free) {
    return a_free;
  }
  double a_score = a_free ? 0.0 : score_[a.slot];
  double b_score = b_free ? 0.0 : score_[b.slot];
  if (a_score != b_score) {
    return a_score < b_score;
  }
  return a < b;
}

std::vector<SlotExpert> SlotLru::invalidate_prefill_buffers() {
  candidates_.clear();
  for (size_t pool = 0; pool < pool_volatile_.size(); ++pool) {
    for (int32_t slot = pool_bounds_[pool]; slot < pool_bounds_[pool] + pool_volatile_[pool];
         ++slot) {
      if (id_of_slot_[slot] >= 0) {
        candidates_.push_back({last_use_data_[slot], id_of_slot_[slot], slot});
        slot_of_data_[id_of_slot_[slot]] = -1;
        id_of_slot_[slot] = -1;
      }
      last_use_data_[slot] = 0;
    }
  }
  std::sort(candidates_.begin(), candidates_.end(), [](const Candidate& a, const Candidate& b) {
    return a.last_use > b.last_use || (a.last_use == b.last_use && a.slot < b.slot);
  });
  std::vector<SlotExpert> dropped;
  dropped.reserve(candidates_.size());
  for (const Candidate& candidate : candidates_) {
    dropped.emplace_back(candidate.flat_id, candidate.slot);
  }
  return dropped;
}

SlotAssignment SlotLru::prefill_slot_map(int layer_id) const {
  check_layer(layer_id, "prefill_slot_map");
  const int32_t* layer_slots = slot_of_data_ + static_cast<size_t>(layer_id) * num_experts_;
  int32_t pool = layer_pools_[layer_id];
  int32_t volatile_end = pool_bounds_[pool] + pool_volatile_[pool];
  int32_t absent = 0;
  for (int32_t expert = 0; expert < num_experts_; ++expert) {
    int32_t slot = layer_slots[expert];
    if (slot < 0) {
      ++absent;
    } else if (slot < volatile_end) {
      throw std::logic_error(
          "SlotLru.prefill_slot_map: expert " + std::to_string(expert) + " of layer " +
          std::to_string(layer_id) + " sits in slot " + std::to_string(slot) +
          " under the prefill buffers; invalidate them first");
    }
  }
  if (absent > buffer_size_) {
    throw std::logic_error(
        "SlotLru.prefill_slot_map: prefill buffers of " + std::to_string(buffer_size_) +
        " experts cannot hold layer " + std::to_string(layer_id) + "'s " +
        std::to_string(absent) + " absent experts");
  }
  SlotAssignment assignment;
  assignment.slots.resize(num_experts_);
  int32_t row = 0;
  for (int32_t expert = 0; expert < num_experts_; ++expert) {
    int32_t slot = layer_slots[expert];
    if (slot >= 0) {
      assignment.slots[expert] = slot;
    } else {
      assignment.slots[expert] = -1 - row;
      assignment.loads.emplace_back(expert, row++);
    }
  }
  return assignment;
}

} // namespace ftmetal
