// VictimCache: host memory below the routed-expert slot cache. An expert evicted from its slot
// keeps its row bytes, every bank's row of its layer back to back, in an entry of its pool's
// set until a later miss copies them back instead of reading the checkpoint; an expert is in
// a slot or here, never both (a prefill buffer row is no slot). The byte budget is split
// over the pools by their slots' bytes. Entries are memory outside the GPU's wired budget, of
// one of two kinds (VictimMemory). Purgeable: each entry a purgeable region of its own,
// volatile unless pinned for a copy; under memory pressure the kernel discards volatile
// entries instead of compressing or swapping them, and pinning one (nonvolatile) tells whether
// it was discarded. Anonymous: every entry in one anonymous mapping, as much of it wired
// (mlock) as the OS allows and the rest pageable, never discarded: no lost hits and no state
// calls, at the cost of memory the system cannot take back. Each pool's entries keep an LRU of
// their own. Called from one thread at a time; a pinned entry's bytes may be copied on any
// thread.
#pragma once

#include <cstdint>
#include <vector>

namespace ftmetal {

// No entry holds the expert, or the pool has none to give.
constexpr int32_t NO_ENTRY = -1;

enum class VictimMemory : uint32_t {
  Purgeable = 0,
  Anonymous = 1,
};

class VictimCache {
 public:
  // layer_pools: each layer's pool; entry_bytes and pool_slots: each pool's; budget_bytes:
  // every entry's bytes together, at most.
  VictimCache(
      int num_experts,
      const std::vector<int32_t>& layer_pools,
      const std::vector<int64_t>& entry_bytes,
      const std::vector<int64_t>& pool_slots,
      int64_t budget_bytes,
      VictimMemory memory = VictimMemory::Purgeable);
  ~VictimCache();

  VictimCache(const VictimCache&) = delete;
  VictimCache& operator=(const VictimCache&) = delete;

  int64_t num_entries() const {
    return static_cast<int64_t>(entries_.size());
  }

  int64_t num_bytes() const {
    return num_bytes_;
  }

  // Anonymous memory's wired bytes (0 when purgeable).
  int64_t locked_bytes() const {
    return locked_bytes_;
  }

  int32_t num_pools() const {
    return static_cast<int32_t>(pools_.size());
  }

  int64_t capacity(int32_t pool) const {
    return pools_[pool].capacity;
  }

  int32_t pool_of(int32_t flat_id) const {
    return layer_pools_[flat_id / num_experts_];
  }

  uint8_t* data(int32_t entry) const {
    return entries_[entry].data;
  }

  // Pin flat_id's entry nonvolatile and make it its pool's most recently used; NO_ENTRY when
  // none holds it. An entry the kernel discarded is freed instead, and is_purged set.
  int32_t pin(int32_t flat_id, bool& is_purged);

  // Volatile again, after a copy.
  void unpin(int32_t entry);

  // A pinned entry to store flat_id in, its pool's most recently used: a free one, else the
  // least recently used unpinned one, whose expert it forgets; NO_ENTRY when there is none.
  int32_t claim(int32_t flat_id);

  // Forget a pinned entry's expert; the entry is free and volatile again.
  void release(int32_t entry);

  // Free every entry.
  void clear();

  // For tests: discard every entry's bytes as memory pressure would (purgeable entries only;
  // anonymous ones are never discarded), and the flat ids held.
  void purge();
  std::vector<int32_t> flat_ids() const;

 private:
  struct Entry {
    uint8_t* data = nullptr;
    int32_t pool = 0;
    // The expert held, -1 when free.
    int32_t flat_id = -1;
    // The pool's LRU neighbours, previous the more recently used; next alone links the free.
    int32_t previous = NO_ENTRY;
    int32_t next = NO_ENTRY;
    bool is_pinned = false;
  };

  struct Pool {
    int64_t entry_bytes = 0;
    int64_t capacity = 0;
    int32_t newest = NO_ENTRY;
    int32_t oldest = NO_ENTRY;
    int32_t free = NO_ENTRY;
  };

  void deallocate();
  // vm_purgable_control on a purgeable entry; an anonymous one keeps its bytes (nonvolatile).
  int set_state(uint8_t* data, int state);
  void allocate_anonymous();
  void allocate_purgeable();
  void unlink(int32_t entry);
  void push_newest(int32_t entry);
  // Unindex the entry's expert and put it on its pool's free list, volatile.
  void free_entry(int32_t entry);

  int num_experts_;
  std::vector<int32_t> layer_pools_;
  std::vector<Pool> pools_;
  std::vector<Entry> entries_;
  int64_t num_bytes_ = 0;
  bool is_anonymous_ = false;
  // Anonymous: the mapping and its wired bytes.
  uint8_t* region_ = nullptr;
  size_t region_bytes_ = 0;
  int64_t locked_bytes_ = 0;
  // [num_layers * num_experts]: the entry holding each flat id.
  std::vector<int32_t> entry_of_;
};

} // namespace ftmetal
