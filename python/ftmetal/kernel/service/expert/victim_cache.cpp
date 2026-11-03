#include "victim_cache.h"

#include <mach/mach.h>
#include <mach/mach_vm.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ftmetal {

namespace {

// Set an entry's purgeable state; returns the one before (VM_PURGABLE_EMPTY: purged).
int set_state(uint8_t* data, int state) {
  int previous = state;
  kern_return_t result = mach_vm_purgable_control(
      mach_task_self(),
      reinterpret_cast<mach_vm_address_t>(data),
      VM_PURGABLE_SET_STATE,
      &previous);
  if (result != KERN_SUCCESS) {
    throw std::runtime_error(
        std::string("VictimCache: vm_purgable_control: ") + mach_error_string(result));
  }
  return previous & VM_PURGABLE_STATE_MASK;
}

} // namespace

VictimCache::VictimCache(
    int num_experts,
    const std::vector<int32_t>& layer_pools,
    const std::vector<int64_t>& entry_bytes,
    const std::vector<int64_t>& pool_slots,
    int64_t budget_bytes)
    : num_experts_(num_experts),
      layer_pools_(layer_pools),
      pools_(entry_bytes.size()),
      entry_of_(layer_pools.size() * num_experts, NO_ENTRY) {
  if (pool_slots.size() != entry_bytes.size() || budget_bytes < 0) {
    throw std::invalid_argument(
        "VictimCache: entry_bytes and pool_slots must hold one count per pool, and the budget "
        "must not be negative");
  }
  std::vector<int64_t> pool_experts(pools_.size(), 0);
  for (int32_t pool : layer_pools_) {
    pool_experts[pool] += num_experts;
  }
  __int128 slot_bytes = 0;
  for (size_t pool = 0; pool < pools_.size(); ++pool) {
    slot_bytes += static_cast<__int128>(pool_slots[pool]) * entry_bytes[pool];
  }
  for (size_t pool = 0; pool < pools_.size(); ++pool) {
    pools_[pool].entry_bytes = entry_bytes[pool];
    if (slot_bytes > 0 && entry_bytes[pool] > 0) {
      __int128 share = static_cast<__int128>(budget_bytes) * pool_slots[pool] *
          entry_bytes[pool] / slot_bytes;
      // Never more entries than the pool's experts outside its slots.
      pools_[pool].capacity = std::min(
          static_cast<int64_t>(share / entry_bytes[pool]),
          std::max<int64_t>(pool_experts[pool] - pool_slots[pool], 0));
    }
  }
  for (size_t pool = 0; pool < pools_.size(); ++pool) {
    for (int64_t i = 0; i < pools_[pool].capacity; ++i) {
      mach_vm_address_t address = 0;
      kern_return_t result = mach_vm_allocate(
          mach_task_self(),
          &address,
          pools_[pool].entry_bytes,
          VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE);
      if (result != KERN_SUCCESS) {
        deallocate();
        throw std::runtime_error(
            std::string("VictimCache: allocating an entry: ") + mach_error_string(result));
      }
      entries_.push_back({reinterpret_cast<uint8_t*>(address), static_cast<int32_t>(pool)});
      // The whole pages mach_vm_allocate maps; the budget is sized in unrounded entries.
      int64_t page = static_cast<int64_t>(vm_page_size);
      num_bytes_ += (pools_[pool].entry_bytes + page - 1) / page * page;
    }
  }
  // The lowest entries first off each free list.
  for (int32_t entry = static_cast<int32_t>(entries_.size()) - 1; entry >= 0; --entry) {
    free_entry(entry);
  }
}

VictimCache::~VictimCache() {
  deallocate();
}

void VictimCache::deallocate() {
  for (const Entry& entry : entries_) {
    mach_vm_deallocate(
        mach_task_self(),
        reinterpret_cast<mach_vm_address_t>(entry.data),
        pools_[entry.pool].entry_bytes);
  }
  entries_.clear();
}

void VictimCache::unlink(int32_t entry) {
  Entry& linked = entries_[entry];
  Pool& pool = pools_[linked.pool];
  if (linked.previous != NO_ENTRY) {
    entries_[linked.previous].next = linked.next;
  } else {
    pool.newest = linked.next;
  }
  if (linked.next != NO_ENTRY) {
    entries_[linked.next].previous = linked.previous;
  } else {
    pool.oldest = linked.previous;
  }
  linked.previous = NO_ENTRY;
  linked.next = NO_ENTRY;
}

void VictimCache::push_newest(int32_t entry) {
  Entry& linked = entries_[entry];
  Pool& pool = pools_[linked.pool];
  linked.previous = NO_ENTRY;
  linked.next = pool.newest;
  if (pool.newest != NO_ENTRY) {
    entries_[pool.newest].previous = entry;
  } else {
    pool.oldest = entry;
  }
  pool.newest = entry;
}

void VictimCache::free_entry(int32_t entry) {
  Entry& freed = entries_[entry];
  if (freed.flat_id >= 0) {
    entry_of_[freed.flat_id] = NO_ENTRY;
  }
  freed.flat_id = -1;
  freed.is_pinned = false;
  set_state(freed.data, VM_PURGABLE_VOLATILE);
  Pool& pool = pools_[freed.pool];
  freed.next = pool.free;
  pool.free = entry;
}

int32_t VictimCache::pin(int32_t flat_id, bool& is_purged) {
  is_purged = false;
  int32_t entry = entry_of_[flat_id];
  if (entry == NO_ENTRY) {
    return NO_ENTRY;
  }
  Entry& held = entries_[entry];
  if (held.is_pinned) {
    throw std::logic_error(
        "VictimCache.pin: the entry of flat id " + std::to_string(flat_id) + " is pinned already");
  }
  unlink(entry);
  if (set_state(held.data, VM_PURGABLE_NONVOLATILE) == VM_PURGABLE_EMPTY) {
    is_purged = true;
    free_entry(entry);
    return NO_ENTRY;
  }
  held.is_pinned = true;
  push_newest(entry);
  return entry;
}

void VictimCache::unpin(int32_t entry) {
  set_state(entries_[entry].data, VM_PURGABLE_VOLATILE);
  entries_[entry].is_pinned = false;
}

int32_t VictimCache::claim(int32_t flat_id) {
  if (entry_of_[flat_id] != NO_ENTRY) {
    throw std::logic_error(
        "VictimCache.claim: flat id " + std::to_string(flat_id) + " is held already");
  }
  Pool& pool = pools_[pool_of(flat_id)];
  int32_t entry = pool.free;
  if (entry != NO_ENTRY) {
    pool.free = entries_[entry].next;
  } else {
    entry = pool.oldest;
    while (entry != NO_ENTRY && entries_[entry].is_pinned) {
      entry = entries_[entry].previous;
    }
    if (entry == NO_ENTRY) {
      return NO_ENTRY;
    }
    unlink(entry);
    entry_of_[entries_[entry].flat_id] = NO_ENTRY;
  }
  Entry& claimed = entries_[entry];
  set_state(claimed.data, VM_PURGABLE_NONVOLATILE);
  claimed.flat_id = flat_id;
  claimed.is_pinned = true;
  entry_of_[flat_id] = entry;
  push_newest(entry);
  return entry;
}

void VictimCache::release(int32_t entry) {
  unlink(entry);
  free_entry(entry);
}

void VictimCache::clear() {
  for (int32_t entry = 0; entry < static_cast<int32_t>(entries_.size()); ++entry) {
    if (entries_[entry].is_pinned) {
      throw std::logic_error("VictimCache.clear: an entry is pinned");
    } else if (entries_[entry].flat_id >= 0) {
      release(entry);
    }
  }
}

void VictimCache::purge() {
  for (const Entry& entry : entries_) {
    if (entry.is_pinned) {
      throw std::logic_error("VictimCache.purge: an entry is pinned");
    }
    set_state(entry.data, VM_PURGABLE_EMPTY);
  }
}

std::vector<int32_t> VictimCache::flat_ids() const {
  std::vector<int32_t> ids;
  for (const Entry& entry : entries_) {
    if (entry.flat_id >= 0) {
      ids.push_back(entry.flat_id);
    }
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

} // namespace ftmetal
