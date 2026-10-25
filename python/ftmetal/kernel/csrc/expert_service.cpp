#include "expert_service.h"

#include <fcntl.h>
#include <pthread/qos.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <tuple>

namespace ftmetal {

namespace {

// Reads are widened to this boundary, which F_NOCACHE transfers go straight to the device at.
const int64_t READ_ALIGNMENT = 4096;
// The VM page, so a staging buffer starts on a page of its own.
const size_t STAGING_ALIGNMENT = 16384;
// The longest wanted range of one read, so that its widened range fits the staging buffer.
const int64_t MAX_READ_BYTES = EXPERT_STAGING_BYTES - 2 * READ_ALIGNMENT;

void wait_for(const ReadBatch& batch) {
  while (batch.remaining.load(std::memory_order_acquire) != 0) {
  }
}

int64_t nanoseconds_since(std::chrono::steady_clock::time_point started) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now() - started)
      .count();
}

std::string error_text(int error_number) {
  char text[256];
  strerror_r(error_number, text, sizeof(text));
  return text;
}

} // namespace

ExpertService::ExpertService(
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
    int io_threads,
    double stall_timeout_s,
    bool run_coordinator)
    : lru_(
          num_layers,
          num_experts,
          buffer_size,
          layer_pools,
          pool_bounds,
          pool_volatile,
          std::move(slot_of),
          std::move(last_use)),
      num_layers_(num_layers),
      num_experts_(num_experts),
      banks_(std::move(banks)),
      destinations_(std::move(destinations)),
      buffer_bytes_(std::move(buffer_bytes)),
      files_(files),
      segments_(std::move(segments)),
      locations_(std::move(locations)),
      segments_per_layer_(static_cast<int>(segments_.size()) / num_layers),
      mailbox_(std::move(mailbox)),
      mailbox_data_(static_cast<uint32_t*>(shared_data(mailbox_, "mailbox"))),
      hit_stats_(std::move(hit_stats)),
      hit_stats_data_(static_cast<uint32_t*>(shared_data(hit_stats_, "hit_stats"))),
      stall_timeout_(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(stall_timeout_s))) {
  if (io_threads < 1) {
    throw std::invalid_argument("ExpertService: io_threads must be positive");
  }
  if (!(stall_timeout_s > 0)) {
    throw std::invalid_argument("ExpertService: stall_timeout_s must be positive");
  }
  if (mailbox_.dtype() != mx::uint32 || mailbox_.size() != MAILBOX_WORDS) {
    throw std::invalid_argument("ExpertService: mailbox must be uint32 [1024]");
  }
  if (hit_stats_.dtype() != mx::uint32 || hit_stats_.shape() != mx::Shape{num_layers, 2}) {
    throw std::invalid_argument("ExpertService: hit_stats must be uint32 [num_layers, 2]");
  }
  if (segments_.empty() || segments_.size() % num_layers != 0) {
    throw std::invalid_argument("ExpertService: segments must hold the same count per layer");
  }
  if (locations_.size() != segments_.size() * num_experts) {
    throw std::invalid_argument("ExpertService: locations must hold num_experts per segment");
  }
  int64_t num_banks = static_cast<int64_t>(banks_.size());
  if (destinations_.size() != static_cast<size_t>(num_layers * num_banks * 2) ||
      buffer_bytes_.size() != static_cast<size_t>(num_banks)) {
    throw std::invalid_argument(
        "ExpertService: destinations must be [num_layers, num_banks, 2] and buffer_bytes "
        "[num_banks]");
  }
  for (int64_t b = 0; b < num_banks; ++b) {
    std::string name = "bank " + std::to_string(b);
    bank_data_.push_back(static_cast<uint8_t*>(shared_data(banks_[b], name)));
    if (2 * buffer_bytes_[b] > static_cast<int64_t>(banks_[b].nbytes())) {
      throw std::invalid_argument("ExpertService: the prefill buffers outgrow a bank");
    }
  }
  // Every slot of a layer's pool, and a whole layer in each prefill buffer, inside the bank.
  for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
    int32_t pool = layer_pools[layer_id];
    for (int64_t b = 0; b < num_banks; ++b) {
      int64_t offset = destinations_[2 * (layer_id * num_banks + b)];
      int64_t row = destinations_[2 * (layer_id * num_banks + b) + 1];
      if (row < 1 || offset + pool_bounds[pool] * row < 0 ||
          offset + pool_bounds[pool + 1] * row > static_cast<int64_t>(banks_[b].nbytes()) ||
          buffer_size * row > buffer_bytes_[b]) {
        throw std::invalid_argument(
            "ExpertService: layer " + std::to_string(layer_id) + "'s rows of bank " +
            std::to_string(b) + " fall outside it or its prefill buffers");
      }
    }
  }
  // Each layer's segments must tile every bank row exactly, so a read expert is whole.
  for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
    std::vector<std::tuple<int64_t, int64_t, int64_t>> pieces;
    for (int s = 0; s < segments_per_layer_; ++s) {
      int64_t segment_index = layer_id * segments_per_layer_ + s;
      const ExpertSegment& segment = segments_[segment_index];
      bool is_valid = segment.bank_index >= 0 && segment.bank_index < num_banks &&
          segment.expert_bytes >= 1;
      for (int e = 0; e < num_experts; ++e) {
        const ExpertLocation& location = locations_[segment_index * num_experts + e];
        is_valid = is_valid &&
            (location.file_index == FILL_FILE
                 ? location.file_offset >= 0 && location.file_offset <= 0xFF
                 : location.file_index >= 0 &&
                     location.file_index < static_cast<int64_t>(files.size()) &&
                     location.file_offset >= 0);
      }
      if (!is_valid) {
        throw std::invalid_argument(
            "ExpertService: segment " + std::to_string(s) + " of layer " +
            std::to_string(layer_id) + " names no bank, an empty range or an expert in no file");
      }
      pieces.emplace_back(segment.bank_index, segment.row_offset, segment.expert_bytes);
    }
    std::sort(pieces.begin(), pieces.end());
    std::vector<int64_t> covered(num_banks, 0);
    bool is_tiled = true;
    for (const auto& [bank_index, row_offset, row_bytes] : pieces) {
      is_tiled = is_tiled && row_offset == covered[bank_index];
      covered[bank_index] += row_bytes;
    }
    for (int64_t b = 0; b < num_banks; ++b) {
      is_tiled = is_tiled && covered[b] == destinations_[2 * (layer_id * num_banks + b) + 1];
    }
    if (!is_tiled) {
      throw std::invalid_argument(
          "ExpertService: the segments of layer " + std::to_string(layer_id) +
          " do not tile every bank row exactly once");
    }
  }
  stats_.hits.assign(num_layers, 0);
  stats_.misses.assign(num_layers, 0);
  stats_.requests.assign(num_layers, 0);
  stats_.experts_read.assign(num_layers, 0);
  stats_.bytes_read.assign(num_layers, 0);
  stats_.read_wait_ns.assign(num_layers, 0);
  stats_.serve_ns.assign(num_layers, 0);
  std::fill(mailbox_data_, mailbox_data_ + MAILBOX_WORDS, 0u);
  try {
    for (const std::string& file : files_) {
      int descriptor = open(file.c_str(), O_RDONLY | O_CLOEXEC);
      if (descriptor < 0) {
        throw std::runtime_error("ExpertService: opening " + file + ": " + error_text(errno));
      }
      file_descriptors_.push_back(descriptor);
      if (fcntl(descriptor, F_NOCACHE, 1) != 0 || fcntl(descriptor, F_RDAHEAD, 0) != 0) {
        throw std::runtime_error(
            "ExpertService: uncached reads of " + file + ": " + error_text(errno));
      }
    }
    // A file cut short under its header fails here, not at the first miss mid-serve.
    std::vector<int64_t> file_bytes;
    for (size_t f = 0; f < files_.size(); ++f) {
      struct stat status;
      if (fstat(file_descriptors_[f], &status) != 0) {
        throw std::runtime_error("ExpertService: stat of " + files_[f] + ": " + error_text(errno));
      }
      file_bytes.push_back(status.st_size);
    }
    for (size_t i = 0; i < locations_.size(); ++i) {
      const ExpertLocation& location = locations_[i];
      int64_t expert_bytes = segments_[i / num_experts].expert_bytes;
      if (location.file_index != FILL_FILE &&
          location.file_offset + expert_bytes > file_bytes[location.file_index]) {
        throw std::runtime_error(
            "ExpertService: " + files_[location.file_index] + " ends before expert " +
            std::to_string(i % num_experts) + "'s bytes at " +
            std::to_string(location.file_offset));
      }
    }
    for (int i = 0; i < io_threads; ++i) {
      void* memory = nullptr;
      if (posix_memalign(&memory, STAGING_ALIGNMENT, EXPERT_STAGING_BYTES) != 0) {
        throw std::bad_alloc();
      }
      staging_.emplace_back(static_cast<uint8_t*>(memory), std::free);
    }
    for (int i = 0; i < io_threads; ++i) {
      threads_.emplace_back(&ExpertService::work, this, i);
    }
    watchdog_ = std::thread(&ExpertService::watch, this);
    if (run_coordinator) {
      coordinator_ = std::thread(&ExpertService::coordinate, this);
    }
  } catch (...) {
    shutdown();
    throw;
  }
}

ExpertService::~ExpertService() {
  shutdown();
}

void ExpertService::shutdown() {
  std::call_once(shutdown_once_, [this] {
    fail(MailboxError::Shutdown, "the service was shut down");
    is_stopping_coordination_.store(true, std::memory_order_release);
    // The shutdown word first: a lookup that has not posted yet posts nothing, and one that
    // posts anyway (it read the word just before) leaves its wait on it. State last: a waiting
    // GPU leaves on SHUTDOWN, and a coordinator still answering finds its request taken and
    // leaves the reply alone.
    mailbox_word(MailboxWord::Shutdown).store(1, std::memory_order_release);
    mailbox_word(MailboxWord::Error)
        .store(static_cast<uint32_t>(MailboxError::Shutdown), std::memory_order_relaxed);
    mailbox_word(MailboxWord::Acknowledged)
        .store(
            mailbox_word(MailboxWord::Sequence).load(std::memory_order_acquire),
            std::memory_order_release);
    mailbox_word(MailboxWord::State)
        .store(static_cast<uint32_t>(MailboxState::Shutdown), std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(wake_mutex_);
    }
    coordinator_wake_.notify_all();
    watchdog_wake_.notify_all();
    for (std::thread* thread : {&coordinator_, &watchdog_}) {
      if (thread->joinable()) {
        thread->join();
      }
    }
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      is_stopping_ = true;
    }
    queue_ready_.notify_all();
    for (std::thread& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    for (int descriptor : file_descriptors_) {
      close(descriptor);
    }
    file_descriptors_.clear();
  });
}

void ExpertService::check_layer(int layer_id, const char* method) const {
  if (layer_id < 0 || layer_id >= num_layers_) {
    throw std::out_of_range(
        std::string("ExpertService.") + method + ": layer " + std::to_string(layer_id) +
        " is not in [0, " + std::to_string(num_layers_) + ")");
  }
}

void ExpertService::submit(
    ReadBatch& batch,
    int layer_id,
    const std::vector<ExpertLoad>& loads,
    int buffer) {
  batch.layer_id = layer_id;
  batch.buffer = buffer;
  batch.experts_read = static_cast<int64_t>(loads.size());
  batch.index_of_expert.assign(num_experts_, -1);
  for (const auto& [expert, index] : loads) {
    batch.index_of_expert[expert] = index;
  }
  batch.bytes_read.store(0, std::memory_order_relaxed);
  batch.error.clear();
  std::vector<ReadTask> tasks;
  for (int s = 0; s < segments_per_layer_; ++s) {
    int64_t segment_index = layer_id * segments_per_layer_ + s;
    const ExpertSegment& segment = segments_[segment_index];
    const ExpertLocation* locations = &locations_[segment_index * num_experts_];
    for (int32_t first = 0; first < num_experts_;) {
      const ExpertLocation& start = locations[first];
      if (batch.index_of_expert[first] < 0) {
        ++first;
        continue;
      } else if (start.file_index == FILL_FILE) {
        std::memset(
            expert_row(batch, segment, first),
            static_cast<int>(start.file_offset),
            segment.expert_bytes);
        ++first;
        continue;
      }
      int32_t last = first + 1;
      while (last < num_experts_ && batch.index_of_expert[last] >= 0 &&
             locations[last].file_index == start.file_index &&
             locations[last].file_offset ==
                 start.file_offset + (last - first) * segment.expert_bytes) {
        ++last;
      }
      int64_t base = start.file_offset - first * segment.expert_bytes;
      int64_t run_end = base + last * segment.expert_bytes;
      for (int64_t begin = start.file_offset; begin < run_end; begin += MAX_READ_BYTES) {
        tasks.push_back(
            {&batch, s, start.file_index, base, begin, std::min(begin + MAX_READ_BYTES, run_end)});
      }
      first = last;
    }
  }
  batch.remaining.store(static_cast<int64_t>(tasks.size()), std::memory_order_release);
  if (tasks.empty()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.insert(queue_.end(), tasks.begin(), tasks.end());
  }
  queue_ready_.notify_all();
}

void ExpertService::work(int thread_index) {
  // A lookup spins on the GPU until these reads land.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  uint8_t* staging = staging_[thread_index].get();
  for (;;) {
    ReadTask task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_ready_.wait(lock, [this] { return is_stopping_ || !queue_.empty(); });
      if (is_stopping_) {
        return;
      }
      task = queue_.front();
      queue_.pop_front();
    }
    read(task, staging);
    task.batch->remaining.fetch_sub(1, std::memory_order_release);
  }
}

void ExpertService::read(const ReadTask& task, uint8_t* staging) {
  ReadBatch& batch = *task.batch;
  const ExpertSegment& segment = segments_[batch.layer_id * segments_per_layer_ + task.segment];
  int64_t aligned_begin = task.begin / READ_ALIGNMENT * READ_ALIGNMENT;
  int64_t aligned_end = (task.end + READ_ALIGNMENT - 1) / READ_ALIGNMENT * READ_ALIGNMENT;
  int descriptor = file_descriptors_[task.file_index];
  // The widened end may pass the end of the file; only the wanted bytes must arrive.
  int64_t done = 0;
  while (done < task.end - aligned_begin) {
    ssize_t count = pread(
        descriptor, staging + done, aligned_end - aligned_begin - done, aligned_begin + done);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      std::string reason = count < 0 ? error_text(errno) : "unexpected end of file";
      std::lock_guard<std::mutex> lock(batch.error_mutex);
      if (batch.error.empty()) {
        batch.error = "ExpertService: reading " + files_[task.file_index] + " at " +
            std::to_string(aligned_begin + done) + ": " + reason;
      }
      return;
    }
    done += count;
  }
  batch.bytes_read.fetch_add(done, std::memory_order_relaxed);
  int64_t first = (task.begin - task.base) / segment.expert_bytes;
  int64_t last = (task.end - 1 - task.base) / segment.expert_bytes;
  for (int64_t expert = first; expert <= last; ++expert) {
    int64_t expert_begin = task.base + expert * segment.expert_bytes;
    int64_t piece_begin = std::max(task.begin, expert_begin);
    int64_t piece_end = std::min(task.end, expert_begin + segment.expert_bytes);
    std::memcpy(
        expert_row(batch, segment, expert) + (piece_begin - expert_begin),
        staging + (piece_begin - aligned_begin),
        piece_end - piece_begin);
  }
}

uint8_t* ExpertService::expert_row(
    const ReadBatch& batch,
    const ExpertSegment& segment,
    int64_t expert) const {
  size_t destination = 2 * (batch.layer_id * banks_.size() + segment.bank_index);
  int64_t row_bytes = destinations_[destination + 1];
  int64_t row_zero = batch.buffer < 0 ? destinations_[destination]
                                      : batch.buffer * buffer_bytes_[segment.bank_index];
  return bank_data_[segment.bank_index] + row_zero + batch.index_of_expert[expert] * row_bytes +
      segment.row_offset;
}

void ExpertService::finish_buffer(PrefillBuffer& buffer, int64_t wait_ns) {
  buffer.is_waited = true;
  ReadBatch& batch = *buffer.batch;
  stats_.read_wait_ns[batch.layer_id] += wait_ns;
  stats_.experts_read[batch.layer_id] += batch.experts_read;
  stats_.bytes_read[batch.layer_id] += batch.bytes_read.load(std::memory_order_relaxed);
}

void ExpertService::fill_initial(const std::vector<int32_t>& flat_ids) {
  std::lock_guard<std::mutex> lock(call_mutex_);
  check_healthy();
  for (PrefillBuffer& buffer : buffers_) {
    if (!buffer.is_waited) {
      auto started = std::chrono::steady_clock::now();
      wait_for(*buffer.batch);
      finish_buffer(buffer, nanoseconds_since(started));
    }
    buffer.layer_id = -1;
  }
  std::vector<int32_t> slots = lru_.fill(flat_ids);
  std::vector<std::vector<ExpertLoad>> loads(num_layers_);
  for (size_t j = 0; j < flat_ids.size(); ++j) {
    loads[flat_ids[j] / num_experts_].emplace_back(flat_ids[j] % num_experts_, slots[j]);
  }
  std::vector<std::unique_ptr<ReadBatch>> batches;
  for (int layer_id = 0; layer_id < num_layers_; ++layer_id) {
    if (!loads[layer_id].empty()) {
      batches.push_back(std::make_unique<ReadBatch>());
      submit(*batches.back(), layer_id, loads[layer_id]);
    }
  }
  std::string error;
  for (const auto& batch : batches) {
    wait_for(*batch);
    if (error.empty()) {
      error = batch->error;
    }
  }
  if (!error.empty()) {
    fail(MailboxError::Io, error);
    throw ReadError(error);
  }
}

std::vector<int32_t> ExpertService::ensure_experts(
    int layer_id,
    const std::vector<int32_t>& expert_ids,
    uint32_t step) {
  std::lock_guard<std::mutex> lock(call_mutex_);
  check_healthy();
  return std::move(serve(layer_id, expert_ids, step).slots);
}

SlotAssignment ExpertService::serve(
    int layer_id,
    const std::vector<int32_t>& expert_ids,
    uint32_t step) {
  for (const PrefillBuffer& buffer : buffers_) {
    if (!buffer.is_waited) {
      throw std::logic_error(
          "ExpertService: prefill layer " + std::to_string(buffer.layer_id) +
          " is still being read into its prefill buffer; wait for it first");
    }
  }
  SlotAssignment assignment = lru_.resolve(layer_id, expert_ids, step);
  int64_t num_loads = static_cast<int64_t>(assignment.loads.size());
  stats_.hits[layer_id] += static_cast<int64_t>(expert_ids.size()) - num_loads;
  stats_.misses[layer_id] += num_loads;
  if (num_loads == 0) {
    return assignment;
  }
  stats_.requests[layer_id] += 1;
  auto started = std::chrono::steady_clock::now();
  submit(ensure_batch_, layer_id, assignment.loads);
  wait_for(ensure_batch_);
  stats_.read_wait_ns[layer_id] += nanoseconds_since(started);
  stats_.experts_read[layer_id] += num_loads;
  stats_.bytes_read[layer_id] += ensure_batch_.bytes_read.load(std::memory_order_relaxed);
  if (!ensure_batch_.error.empty()) {
    fail(MailboxError::Io, ensure_batch_.error);
    throw ReadError(ensure_batch_.error);
  }
  return assignment;
}

void ExpertService::begin_prefill() {
  std::lock_guard<std::mutex> lock(call_mutex_);
  check_healthy();
  for (PrefillBuffer& buffer : buffers_) {
    if (!buffer.is_waited) {
      auto started = std::chrono::steady_clock::now();
      wait_for(*buffer.batch);
      finish_buffer(buffer, nanoseconds_since(started));
    }
    buffer.layer_id = -1;
  }
  lru_.invalidate_prefill_buffers();
}

std::vector<int32_t> ExpertService::prefetch_prefill_layer(int layer_id) {
  std::lock_guard<std::mutex> lock(call_mutex_);
  check_healthy();
  check_layer(layer_id, "prefetch_prefill_layer");
  PrefillBuffer& buffer = buffers_[layer_id % 2];
  if (buffer.layer_id == layer_id) {
    return buffer.slot_map;
  }
  if (!buffer.is_waited) {
    throw std::logic_error(
        "ExpertService.prefetch_prefill_layer: prefill buffer " + std::to_string(layer_id % 2) +
        " still reads layer " + std::to_string(buffer.layer_id) + ", which was never waited");
  }
  SlotAssignment assignment = lru_.prefill_slot_map(layer_id);
  buffer.layer_id = layer_id;
  buffer.slot_map = std::move(assignment.slots);
  buffer.is_waited = false;
  submit(*buffer.batch, layer_id, assignment.loads, layer_id % 2);
  return buffer.slot_map;
}

void ExpertService::wait_prefill_layer(int layer_id) {
  std::lock_guard<std::mutex> lock(call_mutex_);
  check_healthy();
  check_layer(layer_id, "wait_prefill_layer");
  PrefillBuffer& buffer = buffers_[layer_id % 2];
  if (buffer.layer_id != layer_id) {
    throw std::logic_error(
        "ExpertService.wait_prefill_layer: layer " + std::to_string(layer_id) +
        " is not prefetched");
  }
  if (buffer.is_waited) {
    return;
  }
  auto started = std::chrono::steady_clock::now();
  wait_for(*buffer.batch);
  finish_buffer(buffer, nanoseconds_since(started));
  if (!buffer.batch->error.empty()) {
    buffer.layer_id = -1;
    throw std::runtime_error(buffer.batch->error);
  }
}

ExpertServiceStats ExpertService::stats() {
  ExpertServiceStats stats;
  {
    std::lock_guard<std::mutex> lock(call_mutex_);
    stats = stats_;
  }
  for (int layer_id = 0; layer_id < num_layers_; ++layer_id) {
    // The GPU's counts, read without synchronizing: an older value is fine.
    stats.lookups.push_back(
        std::atomic_ref<uint32_t>(hit_stats_data_[2 * layer_id]).load(std::memory_order_relaxed));
    stats.lookup_hits.push_back(std::atomic_ref<uint32_t>(hit_stats_data_[2 * layer_id + 1])
                                    .load(std::memory_order_relaxed));
  }
  return stats;
}

} // namespace ftmetal
