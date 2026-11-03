// ExpertService: the CPU side of the routed-expert slot cache. It owns the SlotLru and a
// pool of IO threads that read expert rows straight from the checkpoint into the bank
// slots: raw read-around (each read widened to 4 KiB boundaries, F_NOCACHE, no read-ahead)
// into a per-thread aligned staging buffer, then one memcpy per expert into its row. A run
// of consecutive experts whose bytes follow on in one file is one read; a part the
// checkpoint does not store is filled as the read is submitted instead. Idle threads block; a
// caller waits for its reads by spinning on a completion counter. A coordinator thread
// answers the GPU lookups' mailbox requests (see moe/moe_lookup.h) with the same reads, and a
// watchdog thread fails a request the coordinator leaves unanswered. With a victim cache
// (victim_cache.h), the experts leaving their slots are copied into its entries and a miss it
// holds is copied back instead of read: the reads of the other misses start first, their row
// writes held until the evicted rows are stored, and the hits land after the stores, since a
// hit may take an evicted slot. An overlap prefill copies the absent experts the cache holds
// into its prefill buffer on the IO threads, keeping them cached, and the experts dropped from
// the slots under the buffers as it begins are stored first. Only where a slot's bytes come
// from changes: the GPU side never sees the cache. Public methods other than arm,
// check_healthy and shutdown are called from one thread at a time.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/mailbox.h"
#include "mlx/array.h"
#include "service/expert/expert_lru.h"
#include "service/expert/victim_cache.h"

namespace mx = mlx::core;

namespace ftmetal {

// Each IO thread's aligned staging buffer, which one read fills.
constexpr int64_t EXPERT_STAGING_BYTES = 8 << 20;

// One checkpoint part of one bank, per MoE layer: each expert's expert_bytes bytes land at
// row_offset in a row of bank bank_index.
struct ExpertSegment {
  int64_t bank_index;
  int64_t row_offset;
  int64_t expert_bytes;
};

// The file_index of a part the checkpoint does not store: every byte of it is file_offset.
constexpr int64_t FILL_FILE = -1;

// Where one expert's bytes of one segment start: file_offset of file file_index, or FILL_FILE.
struct ExpertLocation {
  int64_t file_index;
  int64_t file_offset;
};

// Per MoE layer. hits and misses count the experts of ensure_experts and of the answered
// lookup requests; experts_read and bytes_read (read-around included) count their reads from
// the checkpoint and the prefill buffers', not the initial fill; read_wait_ns is the time
// spent waiting for them (the victim cache's copies overlap it). lookups and lookup_hits are
// the GPU's counts: lookups run, experts found resident.
struct ExpertServiceStats {
  std::vector<int64_t> hits;
  std::vector<int64_t> misses;
  // Reads of the misses of one lookup (ensure_experts call or mailbox request): one IO wait.
  std::vector<int64_t> requests;
  std::vector<int64_t> experts_read;
  std::vector<int64_t> bytes_read;
  std::vector<int64_t> read_wait_ns;
  // The coordinator's time from claiming a mailbox request to its slots being ready.
  std::vector<int64_t> serve_ns;
  std::vector<int64_t> lookups;
  std::vector<int64_t> lookup_hits;
  // The victim cache's: misses copied from it, absent experts copied from it into a prefill
  // buffer, experts leaving their slots stored in it, and entries found purged.
  std::vector<int64_t> victim_hits;
  std::vector<int64_t> victim_buffer_hits;
  std::vector<int64_t> victim_stores;
  std::vector<int64_t> victim_drops;
  // Time spent copying to and from its entries, over every thread; its entries and bytes.
  int64_t victim_copy_ns = 0;
  int64_t victim_entries = 0;
  int64_t victim_bytes = 0;
};

// A read from the checkpoint failed; the cache maps slots whose rows are incomplete.
struct ReadError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The reads of some of one layer's experts, into slots or into the rows of one prefill
// buffer.
struct ReadBatch {
  int layer_id = -1;
  // The prefill buffer read into, -1 for slots.
  int buffer = -1;
  int64_t experts_read = 0;
  // The slot (or buffer row) of each expert being read, -1 for the others.
  std::vector<int32_t> index_of_expert;
  std::atomic<int64_t> remaining{0};
  std::atomic<int64_t> bytes_read{0};
  std::mutex error_mutex;
  std::string error;
  // While set, the reads hold their row writes (the rows' evicted experts are being stored),
  // and submit leaves its fills, (segment index, expert), to held_fills.
  std::atomic<bool> is_held{false};
  std::vector<std::pair<int64_t, int32_t>> held_fills;
};

// The bytes [begin, end) of file file_index for one segment of the batch's layer, all inside
// one run of the batch's experts, expert e of the run at base + e * expert_bytes; or, with
// entry set, a copy between that victim cache entry and the batch layer's row at index (a
// slot, or a row of the batch's buffer), into the entry when is_store.
struct ReadTask {
  ReadBatch* batch = nullptr;
  int segment = 0;
  int64_t file_index = 0;
  int64_t base = 0;
  int64_t begin = 0;
  int64_t end = 0;
  uint8_t* entry = nullptr;
  int32_t index = 0;
  bool is_store = false;
};

// The prefill buffer of the layers of one parity: the layer it holds or is reading.
struct PrefillBuffer {
  int layer_id = -1;
  bool is_waited = true;
  std::vector<int32_t> slot_map;
  std::unique_ptr<ReadBatch> batch = std::make_unique<ReadBatch>();
  // The victim cache entries being copied into it, pinned until it is waited.
  std::vector<int32_t> entries;
};

class ExpertService {
 public:
  // buffer_size, layer_pools, pool_bounds and pool_volatile lay out the slots (SlotLru);
  // mailbox uint32 [MAILBOX_WORDS] and hit_stats uint32 [num_layers, 2] are the lookups';
  // banks: one evaluated row-contiguous byte buffer per bank; segments: num_layers *
  // segments-per-layer entries, layer-major, and locations: each segment's num_experts
  // entries, in segment order; destinations: per layer and bank, the byte
  // offset of slot 0's row and the row's bytes (slot s at offset + s * row); buffer_bytes:
  // each prefill buffer's bytes in each bank, from byte 0; victim_cache_bytes: the victim
  // cache's budget (0: none). The watchdog fails a request left unanswered for
  // stall_timeout_s; without run_coordinator nobody answers (a stalled coordinator, for
  // tests).
  ExpertService(
      int num_layers,
      int num_experts,
      int buffer_size,
      const std::vector<int32_t>& layer_pools,
      const std::vector<int32_t>& pool_bounds,
      const std::vector<int32_t>& pool_volatile,
      mx::array slot_of,
      mx::array last_use,
      mx::array mailbox,
      mx::array hit_stats,
      std::vector<mx::array> banks,
      const std::vector<std::string>& files,
      std::vector<ExpertSegment> segments,
      std::vector<ExpertLocation> locations,
      std::vector<int64_t> destinations,
      std::vector<int64_t> buffer_bytes,
      int64_t victim_cache_bytes,
      int io_threads,
      double stall_timeout_s,
      bool run_coordinator);
  ~ExpertService();

  ExpertService(const ExpertService&) = delete;
  ExpertService& operator=(const ExpertService&) = delete;

  // SlotLru::fill, then read every placed expert.
  void fill_initial(const std::vector<int32_t>& flat_ids);

  // SlotLru::resolve, then read the misses; returns each expert's slot.
  std::vector<int32_t> ensure_experts(
      int layer_id,
      const std::vector<int32_t>& expert_ids,
      uint32_t step);

  // Drain the prefill buffers' reads, forget the layers they hold and drop the experts their
  // slots hold, into the victim cache.
  void begin_prefill();

  // Start reading the layer's non-resident experts into prefill buffer layer_id % 2; returns its
  // slot map (SlotLru::prefill_slot_map).
  std::vector<int32_t> prefetch_prefill_layer(int layer_id);

  // Wait for the layer's prefill buffer reads.
  void wait_prefill_layer(int layer_id);

  ExpertServiceStats stats();

  // Keep the coordinator spinning, instead of dozing, until 50 ms after the later of now and
  // its last answer; the engine arms it before submitting a forward with lookups.
  void arm();

  // Raise the first failure (a read, a broken or unanswered request, the shutdown); the
  // engine calls it once per forward.
  void check_healthy() const;

  // Answer a waiting lookup with SHUTDOWN and refuse later ones, then stop the threads.
  void shutdown();

  // For tests: discard the victim cache's bytes as memory pressure would, and the flat ids it
  // holds.
  void purge_victim_cache();
  std::vector<int32_t> victim_cache_ids();

 private:
  std::atomic_ref<uint32_t> mailbox_word(MailboxWord word) const;
  void check_layer(int layer_id, const char* method) const;
  void fail(MailboxError error, const std::string& message);
  // SlotLru::resolve, then read the misses; call_mutex_ held.
  SlotAssignment serve(int layer_id, const std::vector<int32_t>& expert_ids, uint32_t step);
  void coordinate();
  void answer();
  void watch();
  // Read loads (expert, slot) into slots, or (expert, row) into prefill buffer buffer, and run
  // copies (victim cache copy tasks) with them.
  void submit(
      ReadBatch& batch,
      int layer_id,
      const std::vector<ExpertLoad>& loads,
      int buffer = -1,
      std::vector<ReadTask> copies = {});
  void finish_buffer(PrefillBuffer& buffer, int64_t wait_ns);
  void work(int thread_index);
  void read(const ReadTask& task, uint8_t* staging);
  // Fill the segment's bytes of the expert, which the checkpoint does not store.
  void fill(const ReadBatch& batch, int64_t segment_index, int32_t expert);
  // Where the segment's bytes of the expert land in the batch's slot or buffer row.
  uint8_t* expert_row(const ReadBatch& batch, const ExpertSegment& segment, int64_t expert) const;
  // The layer's row at index in the bank: a slot, or (buffer >= 0) a row of that buffer.
  uint8_t* bank_row(int layer_id, int64_t bank, int buffer, int64_t index) const;
  // Copy between the entry and the layer's row at index in every bank, into the entry when
  // is_store.
  void copy_entry(uint8_t* entry, int layer_id, int buffer, int32_t index, bool is_store);
  // The loads of the layer the victim cache does not hold; each one it holds is pinned and
  // added to hits as (entry, slot or row).
  std::vector<ExpertLoad> split_victim_hits(
      int layer_id,
      const std::vector<ExpertLoad>& loads,
      std::vector<std::pair<int32_t, int32_t>>& hits);

  SlotLru lru_;
  int num_layers_;
  int num_experts_;
  std::vector<mx::array> banks_;
  std::vector<uint8_t*> bank_data_;
  // [num_layers][num_banks] (slot 0's row offset, row bytes), flat.
  std::vector<int64_t> destinations_;
  std::vector<int64_t> buffer_bytes_;
  std::vector<std::string> files_;
  std::vector<int> file_descriptors_;
  std::vector<ExpertSegment> segments_;
  std::vector<ExpertLocation> locations_;
  int segments_per_layer_;

  std::mutex call_mutex_;
  ReadBatch ensure_batch_;
  PrefillBuffer buffers_[2];
  ExpertServiceStats stats_;
  // Null without a victim cache.
  std::unique_ptr<VictimCache> victims_;
  std::atomic<int64_t> victim_copy_ns_{0};

  std::mutex queue_mutex_;
  std::condition_variable queue_ready_;
  std::deque<ReadTask> queue_;
  bool is_stopping_ = false;
  std::vector<std::unique_ptr<uint8_t, void (*)(void*)>> staging_;
  std::vector<std::thread> threads_;

  mx::array mailbox_;
  uint32_t* mailbox_data_;
  mx::array hit_stats_;
  uint32_t* hit_stats_data_;
  std::chrono::nanoseconds stall_timeout_;
  // The first failure; every later call and request fails with it.
  std::atomic<bool> has_failed_{false};
  mutable std::mutex failure_mutex_;
  std::string failure_;
  MailboxError failure_error_ = MailboxError::None;
  // Requests the coordinator finished, answered or not (the watchdog's progress signal).
  std::atomic<int64_t> num_answered_{0};
  // steady_clock nanoseconds of the last arm.
  std::atomic<int64_t> armed_at_ns_{0};
  std::atomic<bool> is_dozing_{false};
  std::atomic<bool> is_stopping_coordination_{false};
  std::mutex wake_mutex_;
  std::condition_variable coordinator_wake_;
  std::condition_variable watchdog_wake_;
  std::thread coordinator_;
  std::thread watchdog_;
  std::once_flag shutdown_once_;
};

} // namespace ftmetal
