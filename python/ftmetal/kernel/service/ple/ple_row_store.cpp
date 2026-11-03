// PleRowStore (see ple_service.h): the row cache, the uncached reads and the IO threads.
#include "ple_service.h"

#include <fcntl.h>
#include <pthread/qos.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <functional>

namespace ftmetal {

namespace {

// Uncached (F_NOCACHE) reads go straight to the device at this boundary.
const int64_t READ_ALIGNMENT = 4096;
// The cache slots share this many spin locks.
const int64_t CACHE_LOCK_STRIPES = 4096;
// Fibonacci hashing spreads neighbouring rows over the cache.
const uint64_t CACHE_HASH_MULTIPLIER = 0x9E3779B97F4A7C15ull;
const int CACHE_HASH_SHIFT = 20;

std::string error_text(int error_number) {
  char text[256];
  strerror_r(error_number, text, sizeof(text));
  return text;
}

void lock_stripe(std::atomic_flag& lock) {
  while (lock.test_and_set(std::memory_order_acquire)) {
    __builtin_arm_yield();
  }
}

} // namespace

PleRowStore::PleRowStore(
    const std::vector<std::string>& files,
    const std::vector<int64_t>& parts,
    std::vector<int64_t> shard_starts,
    std::vector<int64_t> row_bytes,
    int64_t cache_rows,
    int io_threads)
    : files_(files),
      shard_starts_(std::move(shard_starts)),
      row_bytes_(std::move(row_bytes)),
      cache_mask_(cache_rows - 1),
      num_io_threads_(io_threads) {
  int64_t num_shards = static_cast<int64_t>(shard_starts_.size()) - 1;
  if (num_shards < 1 || static_cast<int64_t>(parts.size()) != 9 * num_shards ||
      row_bytes_.size() != 3) {
    throw std::invalid_argument(
        "PleRowStore: parts must be int64 [num_shards, 3, 3], shard_starts [num_shards + 1] "
        "and row_bytes the three parts'");
  }
  if (shard_starts_[0] != 0 ||
      std::adjacent_find(shard_starts_.begin(), shard_starts_.end(), std::greater_equal<>()) !=
          shard_starts_.end()) {
    throw std::invalid_argument("PleRowStore: shard_starts must rise from 0");
  }
  if (cache_rows < 1 || (cache_rows & cache_mask_) != 0) {
    throw std::invalid_argument("PleRowStore: cache_rows must be a power of 2");
  }
  if (io_threads < 0) {
    throw std::invalid_argument("PleRowStore: io_threads must not be negative");
  }
  for (int64_t bytes : row_bytes_) {
    if (bytes < 1 || bytes > MAX_PLE_PART_BYTES) {
      throw std::invalid_argument(
          "PleRowStore: a part's row must hold 1 to " + std::to_string(MAX_PLE_PART_BYTES) +
          " bytes");
    }
    record_bytes_ += bytes;
  }
  for (int64_t i = 0; i < 3 * num_shards; ++i) {
    int64_t file_index = parts[3 * i];
    int64_t offset = parts[3 * i + 1];
    int64_t stride = parts[3 * i + 2];
    if (file_index < 0 || file_index >= static_cast<int64_t>(files_.size()) || offset < 0 ||
        stride < row_bytes_[i % 3]) {
      throw std::invalid_argument(
          "PleRowStore: shard " + std::to_string(i / 3) + " part " + std::to_string(i % 3) +
          " names no file, a negative offset or a stride shorter than its row");
    }
    parts_.push_back({file_index, offset, stride});
  }
  try {
    for (const std::string& file : files_) {
      int descriptor = open(file.c_str(), O_RDONLY | O_CLOEXEC);
      if (descriptor < 0) {
        throw std::runtime_error("PleRowStore: opening " + file + ": " + error_text(errno));
      }
      file_descriptors_.push_back(descriptor);
      if (fcntl(descriptor, F_NOCACHE, 1) != 0 || fcntl(descriptor, F_RDAHEAD, 0) != 0) {
        throw std::runtime_error(
            "PleRowStore: uncached reads of " + file + ": " + error_text(errno));
      }
    }
    // A file cut short under its header fails here, not at the first read of a step.
    for (int64_t i = 0; i < 3 * num_shards; ++i) {
      const auto& [file_index, offset, stride] = parts_[i];
      struct stat status;
      if (fstat(file_descriptors_[file_index], &status) != 0) {
        throw std::runtime_error(
            "PleRowStore: stat of " + files_[file_index] + ": " + error_text(errno));
      }
      int64_t shard_rows = shard_starts_[i / 3 + 1] - shard_starts_[i / 3];
      int64_t end = offset + (shard_rows - 1) * stride + row_bytes_[i % 3];
      if (end > status.st_size) {
        throw std::runtime_error(
            "PleRowStore: " + files_[file_index] + " ends before shard " +
            std::to_string(i / 3) + " part " + std::to_string(i % 3) + "'s last row at " +
            std::to_string(end));
      }
    }
    cached_rows_ = std::make_unique<int64_t[]>(cache_rows);
    std::fill(cached_rows_.get(), cached_rows_.get() + cache_rows, int64_t{-1});
    records_ = std::make_unique<uint8_t[]>(cache_rows * record_bytes_);
    stripe_locks_ = std::make_unique<std::atomic_flag[]>(CACHE_LOCK_STRIPES);
    for (int i = 0; i < io_threads; ++i) {
      threads_.emplace_back(&PleRowStore::work, this);
    }
  } catch (...) {
    // No destructor runs for a constructor that throws.
    shut_down();
    throw;
  }
}

PleRowStore::~PleRowStore() {
  shut_down();
}

void PleRowStore::shut_down() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    is_stopping_ = true;
  }
  queue_ready_.notify_all();
  for (std::thread& thread : threads_) {
    thread.join();
  }
  for (int descriptor : file_descriptors_) {
    close(descriptor);
  }
}

int64_t PleRowStore::cache_slot(int64_t row) const {
  return static_cast<int64_t>(
      (static_cast<uint64_t>(row) * CACHE_HASH_MULTIPLIER >> CACHE_HASH_SHIFT) & cache_mask_);
}

bool PleRowStore::copy_cached_row(int64_t row, uint8_t* const destinations[3]) const {
  int64_t slot = cache_slot(row);
  std::atomic_flag& lock = stripe_locks_[slot % CACHE_LOCK_STRIPES];
  const uint8_t* record = records_.get() + slot * record_bytes_;
  lock_stripe(lock);
  bool is_cached = cached_rows_[slot] == row;
  if (is_cached) {
    for (int64_t p = 0, at = 0; p < 3; at += row_bytes_[p], ++p) {
      std::memcpy(destinations[p], record + at, row_bytes_[p]);
    }
  }
  lock.clear(std::memory_order_release);
  return is_cached;
}

void PleRowStore::read_row(int64_t row, uint8_t* const destinations[3]) const {
  int64_t shard = std::upper_bound(shard_starts_.begin(), shard_starts_.end(), row) -
      shard_starts_.begin() - 1;
  int64_t local_row = row - shard_starts_[shard];
  alignas(READ_ALIGNMENT) uint8_t blocks[2 * READ_ALIGNMENT];
  for (int p = 0; p < 3; ++p) {
    const auto& [file_index, offset, stride] = parts_[3 * shard + p];
    int64_t begin = offset + local_row * stride;
    int64_t aligned_begin = begin / READ_ALIGNMENT * READ_ALIGNMENT;
    int64_t aligned_end =
        (begin + row_bytes_[p] + READ_ALIGNMENT - 1) / READ_ALIGNMENT * READ_ALIGNMENT;
    // The widened end may pass the end of the file; only the wanted bytes must arrive.
    int64_t done = 0;
    while (done < begin + row_bytes_[p] - aligned_begin) {
      ssize_t count = pread(
          file_descriptors_[file_index],
          blocks + done,
          aligned_end - aligned_begin - done,
          aligned_begin + done);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count <= 0) {
        throw PleReadError(
            "PleRowStore: reading " + files_[file_index] + " at " +
            std::to_string(aligned_begin + done) + ": " +
            (count < 0 ? error_text(errno) : "unexpected end of file"));
      }
      done += count;
    }
    bytes_read_.fetch_add(done, std::memory_order_relaxed);
    std::memcpy(destinations[p], blocks + (begin - aligned_begin), row_bytes_[p]);
  }
  misses_.fetch_add(1, std::memory_order_relaxed);
  int64_t slot = cache_slot(row);
  std::atomic_flag& lock = stripe_locks_[slot % CACHE_LOCK_STRIPES];
  uint8_t* record = records_.get() + slot * record_bytes_;
  lock_stripe(lock);
  cached_rows_[slot] = row;
  for (int64_t p = 0, at = 0; p < 3; at += row_bytes_[p], ++p) {
    std::memcpy(record + at, destinations[p], row_bytes_[p]);
  }
  lock.clear(std::memory_order_release);
}

void PleRowStore::help(PleGatherBatch& batch) const {
  int64_t num_misses = static_cast<int64_t>(batch.misses.size());
  for (int64_t j = batch.next.fetch_add(1, std::memory_order_relaxed); j < num_misses;
       j = batch.next.fetch_add(1, std::memory_order_relaxed)) {
    int64_t i = batch.misses[j];
    uint8_t* destinations[3] = {
        batch.parts[0] + i * row_bytes_[0],
        batch.parts[1] + i * row_bytes_[1],
        batch.parts[2] + i * row_bytes_[2]};
    try {
      read_row(batch.ids[i], destinations);
    } catch (const PleReadError& read_error) {
      std::lock_guard<std::mutex> lock(batch.error_mutex);
      if (batch.error.empty()) {
        batch.error = read_error.what();
      }
    }
  }
}

void PleRowStore::work() {
  // A ple_wait spins on the GPU until these reads land.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  for (;;) {
    PleGatherBatch* batch;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_ready_.wait(lock, [this] { return is_stopping_ || !queue_.empty(); });
      if (is_stopping_) {
        return;
      }
      batch = queue_.front();
      queue_.pop_front();
    }
    help(*batch);
    // The last touch: the caller may return once every helper has left.
    batch->num_helpers.fetch_sub(1, std::memory_order_release);
  }
}

void PleRowStore::gather(const int64_t* ids, int64_t num_ids, uint8_t* const parts[3]) const {
  PleGatherBatch batch;
  batch.ids = ids;
  std::copy(parts, parts + 3, batch.parts);
  for (int64_t i = 0; i < num_ids; ++i) {
    uint8_t* destinations[3] = {
        parts[0] + i * row_bytes_[0], parts[1] + i * row_bytes_[1], parts[2] + i * row_bytes_[2]};
    if (!copy_cached_row(ids[i], destinations)) {
      batch.misses.push_back(i);
    }
  }
  int64_t num_misses = static_cast<int64_t>(batch.misses.size());
  hits_.fetch_add(num_ids - num_misses, std::memory_order_relaxed);
  if (num_misses == 0) {
    return;
  }
  int num_helpers = static_cast<int>(std::min<int64_t>(num_io_threads_, num_misses - 1));
  if (num_helpers > 0) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      batch.num_helpers.store(num_helpers, std::memory_order_relaxed);
      queue_.insert(queue_.end(), num_helpers, &batch);
    }
    queue_ready_.notify_all();
  }
  help(batch);
  if (num_helpers > 0) {
    // Helpers that never started are not waited for.
    size_t num_unstarted;
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      num_unstarted = std::erase(queue_, &batch);
    }
    batch.num_helpers.fetch_sub(static_cast<int>(num_unstarted), std::memory_order_relaxed);
    while (batch.num_helpers.load(std::memory_order_acquire) != 0) {
    }
  }
  if (!batch.error.empty()) {
    throw PleReadError(batch.error);
  }
}

} // namespace ftmetal
