// ExpertService: the CPU side of the routed-expert slot cache. It owns the SlotLru and a
// pool of IO threads that read expert rows straight from the checkpoint into the bank
// slots: raw read-around (each read widened to 4 KiB boundaries, F_NOCACHE, no read-ahead)
// into a per-thread aligned staging buffer, then one memcpy per expert into its row. A run
// of consecutive experts of one checkpoint segment is one read. Idle threads block; a
// caller waits for its reads by spinning on a completion counter. Public methods are
// called from one thread at a time.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "expert_lru.h"

namespace ftmetal {

// One checkpoint part of one bank, per MoE layer: expert e's bytes are
// [file_offset + e * expert_bytes, file_offset + (e + 1) * expert_bytes) of file
// file_index, and land at row_offset in a row of bank bank_index.
struct ExpertSegment {
  int64_t file_index;
  int64_t file_offset;
  int64_t expert_bytes;
  int64_t bank_index;
  int64_t row_offset;
};

// Per MoE layer. hits and misses count ensure_experts' experts; experts_read and bytes_read
// (read-around included) count ensure_experts' and the prefill windows' reads, not the
// initial fill; read_wait_ns is the time callers spent waiting for them.
struct ExpertServiceStats {
  std::vector<int64_t> hits;
  std::vector<int64_t> misses;
  std::vector<int64_t> experts_read;
  std::vector<int64_t> bytes_read;
  std::vector<int64_t> read_wait_ns;
};

// The reads of some of one layer's experts.
struct ReadBatch {
  int layer_id = -1;
  int64_t experts_read = 0;
  // Bank slot of each expert being read, -1 for the others.
  std::vector<int32_t> slot_of_expert;
  std::atomic<int64_t> remaining{0};
  std::atomic<int64_t> bytes_read{0};
  std::mutex error_mutex;
  std::string error;
};

// The bytes [begin, end) of one segment of the batch's layer, all inside one run of the
// batch's experts.
struct ReadTask {
  ReadBatch* batch = nullptr;
  int segment = 0;
  int64_t begin = 0;
  int64_t end = 0;
};

// The window of the prefill layers of one parity: the layer it holds or is reading.
struct PrefillWindow {
  int layer_id = -1;
  bool is_waited = true;
  std::vector<int32_t> slot_map;
  std::unique_ptr<ReadBatch> batch = std::make_unique<ReadBatch>();
};

class ExpertService {
 public:
  // banks: one evaluated row-contiguous [num_slots, ...] array per bank; segments:
  // num_layers * segments-per-layer entries, layer-major.
  ExpertService(
      int num_layers,
      int num_experts,
      int num_slots,
      int window_slots,
      mx::array slot_of,
      mx::array last_use,
      std::vector<mx::array> banks,
      const std::vector<std::string>& files,
      std::vector<ExpertSegment> segments,
      int io_threads);
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

  // Drain the windows' reads, drop the experts the window slots hold, forget the windows.
  void begin_prefill();

  // Start reading the layer's non-resident experts into window layer_id % 2; returns its
  // slot map (SlotLru::prefill_slot_map).
  std::vector<int32_t> prefetch_prefill_layer(int layer_id);

  // Wait for the layer's window reads.
  void wait_prefill_layer(int layer_id);

  ExpertServiceStats stats();

 private:
  void shutdown();
  void check_healthy() const;
  void check_layer(int layer_id, const char* method) const;
  void submit(ReadBatch& batch, int layer_id, const std::vector<ExpertLoad>& loads);
  void finish_window(PrefillWindow& window, int64_t wait_ns);
  void work(int thread_index);
  void read(const ReadTask& task, uint8_t* staging);

  SlotLru lru_;
  int num_layers_;
  int num_experts_;
  int window_slots_;
  std::vector<mx::array> banks_;
  std::vector<uint8_t*> bank_data_;
  std::vector<int64_t> bank_row_bytes_;
  std::vector<std::string> files_;
  std::vector<int> file_descriptors_;
  std::vector<ExpertSegment> segments_;
  int segments_per_layer_;

  std::mutex call_mutex_;
  // Set when a read into slots the LRU already maps fails; every later call raises it.
  std::string failure_;
  ReadBatch ensure_batch_;
  PrefillWindow windows_[2];
  ExpertServiceStats stats_;

  std::mutex queue_mutex_;
  std::condition_variable queue_ready_;
  std::deque<ReadTask> queue_;
  bool is_stopping_ = false;
  std::vector<std::unique_ptr<uint8_t, void (*)(void*)>> staging_;
  std::vector<std::thread> threads_;
};

} // namespace ftmetal
