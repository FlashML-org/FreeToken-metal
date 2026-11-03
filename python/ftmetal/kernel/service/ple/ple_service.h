// The CPU side of the PLE table lookups. PleRowStore reads table rows straight from the
// checkpoint into a direct-mapped cache of whole rows: a row's three parts (weight, scales,
// biases) lie in three tensors, each read by an uncached pread of the 4 KiB blocks it lies in
// (a mapped table faults 16 KiB pages per part of a ~100-byte row and keeps them in the page
// cache). A gather copies its cached rows on the caller's thread, then reads the rest on the
// caller and a pool of IO threads. PleService answers ple_post's mailbox posts (see
// ple_lookup.h) on a coordinator thread: it hashes each row's n-grams and gathers the table
// rows into the staging buffers. A watchdog thread fails a post left unanswered.
#pragma once

#include <array>
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

#include "ple_lookup.h"

namespace ftmetal {

// A row's part is at most this long: the 4 KiB read boundary, so it spans at most 2 blocks.
constexpr int64_t MAX_PLE_PART_BYTES = 4096;

// A read from the checkpoint failed.
struct PleReadError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The rows of one gather the cache did not hold, read by the caller and the IO threads.
struct PleGatherBatch {
  const int64_t* ids = nullptr;
  uint8_t* parts[3] = {};
  // Indices into ids of the rows to read.
  std::vector<int64_t> misses;
  std::atomic<int64_t> next{0};
  // IO threads that were asked to help and may still touch the batch.
  std::atomic<int> num_helpers{0};
  std::mutex error_mutex;
  std::string error;
};

class PleRowStore {
 public:
  // parts: per shard and part (weight, scales, biases), the file index, the byte offset of
  // the shard's row 0 and the row stride, [num_shards * 3 * 3] flat; shard_starts
  // [num_shards + 1]: each shard's first table row, then the table's rows; row_bytes: each
  // part's bytes per row; cache_rows: the cache's rows, a power of 2; io_threads: the
  // threads that read beside a gather's caller (0: the caller reads alone).
  PleRowStore(
      const std::vector<std::string>& files,
      const std::vector<int64_t>& parts,
      std::vector<int64_t> shard_starts,
      std::vector<int64_t> row_bytes,
      int64_t cache_rows,
      int io_threads);
  ~PleRowStore();

  PleRowStore(const PleRowStore&) = delete;
  PleRowStore& operator=(const PleRowStore&) = delete;

  // Rows ids[i] (in [0, num_rows)) into parts[p] + i * row_bytes(p); throws PleReadError.
  // Safe to call from several threads at once.
  void gather(const int64_t* ids, int64_t num_ids, uint8_t* const parts[3]) const;

  int64_t num_rows() const {
    return shard_starts_.back();
  }
  int64_t row_bytes(int part) const {
    return row_bytes_[part];
  }
  int64_t hits() const {
    return hits_.load(std::memory_order_relaxed);
  }
  int64_t misses() const {
    return misses_.load(std::memory_order_relaxed);
  }
  // Read-around included.
  int64_t bytes_read() const {
    return bytes_read_.load(std::memory_order_relaxed);
  }

 private:
  int64_t cache_slot(int64_t row) const;
  // Copies the row out of the cache; false if the cache does not hold it.
  bool copy_cached_row(int64_t row, uint8_t* const destinations[3]) const;
  // Reads the row from the checkpoint into destinations and the cache.
  void read_row(int64_t row, uint8_t* const destinations[3]) const;
  // Read the batch's misses until none is left to claim.
  void help(PleGatherBatch& batch) const;
  void work();
  // Stops and joins the IO threads, closes the files.
  void shut_down();

  std::vector<std::string> files_;
  std::vector<int> file_descriptors_;
  // [shard * 3 + part]: (file index, offset of row 0, row stride).
  std::vector<std::array<int64_t, 3>> parts_;
  std::vector<int64_t> shard_starts_;
  std::vector<int64_t> row_bytes_;
  int64_t record_bytes_ = 0;
  int64_t cache_mask_;
  // Cache slot -> the row it holds (-1: none), and that row's parts back to back; a slot is
  // read and written under its stripe's spin lock.
  std::unique_ptr<int64_t[]> cached_rows_;
  std::unique_ptr<uint8_t[]> records_;
  mutable std::unique_ptr<std::atomic_flag[]> stripe_locks_;
  mutable std::atomic<int64_t> hits_{0};
  mutable std::atomic<int64_t> misses_{0};
  mutable std::atomic<int64_t> bytes_read_{0};

  int num_io_threads_;
  mutable std::mutex queue_mutex_;
  mutable std::condition_variable queue_ready_;
  // One entry per IO thread asked to help a batch.
  mutable std::deque<PleGatherBatch*> queue_;
  bool is_stopping_ = false;
  std::vector<std::thread> threads_;
};

class PleService {
 public:
  // mailbox uint32 [PLE_MAILBOX_WORDS]; staging: the weight, scales and biases buffers, uint8
  // [MAX_PLE_ROWS, heads, store.row_bytes(p)], evaluated; store: the table, which must
  // outlive the service; multipliers [3] and head_sizes, head_offsets [heads]: the n-gram
  // hash, the first half of the heads bigrams, the rest trigrams. The watchdog fails a post
  // left unanswered for stall_timeout_s; without run_coordinator nobody answers (a stalled
  // coordinator, for tests).
  PleService(
      mx::array mailbox,
      std::vector<mx::array> staging,
      const PleRowStore& store,
      std::vector<int64_t> multipliers,
      std::vector<int64_t> head_sizes,
      std::vector<int64_t> head_offsets,
      int64_t eos_token_id,
      double stall_timeout_s,
      bool run_coordinator);
  ~PleService();

  PleService(const PleService&) = delete;
  PleService& operator=(const PleService&) = delete;

  int num_heads() const {
    return static_cast<int>(head_sizes_.size());
  }

  // The table row of each head of each (token, previous, before previous) triple:
  // ids[r * heads + h] for triples[3 * r ...], r < num_triples.
  void ngram_ids(const int64_t* triples, int64_t num_triples, int64_t* ids) const;

  // Keep the coordinator spinning, instead of dozing, until 50 ms after the later of now and
  // its last answer; the engine arms it before submitting a step that posts.
  void arm();

  // Raise the first failure (a read, a broken or unanswered post, the shutdown): a step
  // whose wait it failed computed with zero rows.
  void check_healthy() const;

  // Answer a waiting ple_wait with SHUTDOWN and refuse later posts, then stop the threads.
  void shutdown();

 private:
  std::atomic_ref<uint32_t> mailbox_word(PleMailboxWord word) const;
  void fail(MailboxError error, const std::string& message);
  void coordinate();
  void answer();
  void watch();

  mx::array mailbox_;
  std::vector<mx::array> staging_;
  uint32_t* mailbox_data_;
  uint8_t* staging_data_[3];
  const PleRowStore& store_;
  std::vector<int64_t> multipliers_;
  std::vector<int64_t> head_sizes_;
  std::vector<int64_t> head_offsets_;
  int64_t eos_token_id_;
  std::chrono::nanoseconds stall_timeout_;

  // The first failure; every later post fails with it.
  std::atomic<bool> has_failed_{false};
  mutable std::mutex failure_mutex_;
  std::string failure_;
  MailboxError failure_error_ = MailboxError::None;
  // Posts the coordinator finished, answered or not (the watchdog's progress signal).
  std::atomic<int64_t> num_answered_{0};
  // steady_clock nanoseconds of the last arm.
  std::atomic<int64_t> armed_at_ns_{0};
  std::atomic<bool> is_dozing_{false};
  std::atomic<bool> is_stopping_{false};
  std::mutex wake_mutex_;
  std::condition_variable coordinator_wake_;
  std::condition_variable watchdog_wake_;
  std::thread coordinator_;
  std::thread watchdog_;
  std::once_flag shutdown_once_;
};

} // namespace ftmetal
