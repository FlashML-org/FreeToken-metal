#include "service/expert/expert_service.h"

#include <fcntl.h>
#include <pthread/qos.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <ranges>
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
    int64_t victim_cache_bytes,
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
  if (victim_cache_bytes < 0) {
    throw std::invalid_argument("ExpertService: victim_cache_bytes must not be negative");
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
  // A record layer: each stored segment of every expert in one file at one offset from the
  // expert's record start, the records evenly spaced and no longer than their spacing (a
  // checkpoint's parts, each a tensor of its own, are not).
  records_.assign(num_layers, RecordLayout{});
  for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
    const ExpertLocation* layer_locations =
        &locations_[static_cast<size_t>(layer_id) * segments_per_layer_ * num_experts];
    std::vector<int> stored;
    bool is_record = true;
    for (int s = 0; s < segments_per_layer_; ++s) {
      bool is_fill = layer_locations[s * num_experts].file_index == FILL_FILE;
      for (int e = 0; e < num_experts; ++e) {
        is_record = is_record &&
            (layer_locations[s * num_experts + e].file_index == FILL_FILE) == is_fill;
      }
      if (!is_fill) {
        stored.push_back(s);
      }
    }
    if (!is_record || stored.empty()) {
      continue;
    }
    RecordLayout record;
    record.file_index = layer_locations[stored[0] * num_experts].file_index;
    std::vector<int64_t> starts(num_experts, INT64_MAX);
    std::vector<int64_t> ends(num_experts, 0);
    for (int s : stored) {
      int64_t expert_bytes = segments_[layer_id * segments_per_layer_ + s].expert_bytes;
      for (int e = 0; e < num_experts; ++e) {
        const ExpertLocation& location = layer_locations[s * num_experts + e];
        is_record = is_record && location.file_index == record.file_index;
        starts[e] = std::min(starts[e], location.file_offset);
        ends[e] = std::max(ends[e], location.file_offset + expert_bytes);
      }
    }
    record.first = starts[0];
    record.bytes = ends[0] - starts[0];
    record.stride = num_experts > 1 ? starts[1] - starts[0] : record.bytes;
    for (int s : stored) {
      int64_t record_offset = layer_locations[s * num_experts].file_offset - starts[0];
      for (int e = 0; e < num_experts; ++e) {
        is_record = is_record &&
            layer_locations[s * num_experts + e].file_offset - starts[e] == record_offset &&
            starts[e] == record.first + e * record.stride;
      }
    }
    if (is_record && record.bytes <= record.stride && record.bytes <= MAX_READ_BYTES) {
      records_[layer_id] = record;
    }
  }
  stats_.hits.assign(num_layers, 0);
  stats_.misses.assign(num_layers, 0);
  stats_.requests.assign(num_layers, 0);
  stats_.experts_read.assign(num_layers, 0);
  stats_.bytes_read.assign(num_layers, 0);
  stats_.read_wait_ns.assign(num_layers, 0);
  stats_.serve_ns.assign(num_layers, 0);
  stats_.victim_hits.assign(num_layers, 0);
  stats_.victim_buffer_hits.assign(num_layers, 0);
  stats_.victim_stores.assign(num_layers, 0);
  stats_.victim_drops.assign(num_layers, 0);
  if (victim_cache_bytes > 0) {
    // An entry holds a row of every bank, whose bytes a pool's layers share.
    size_t num_pools = pool_bounds.size() - 1;
    std::vector<int64_t> entry_bytes(num_pools, 0);
    std::vector<int64_t> pool_slots(num_pools);
    for (size_t pool = 0; pool < num_pools; ++pool) {
      pool_slots[pool] = pool_bounds[pool + 1] - pool_bounds[pool];
    }
    for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
      int64_t layer_bytes = 0;
      for (int64_t b = 0; b < num_banks; ++b) {
        layer_bytes += destinations_[2 * (layer_id * num_banks + b) + 1];
      }
      int64_t& pool_bytes = entry_bytes[layer_pools[layer_id]];
      if (pool_bytes != 0 && pool_bytes != layer_bytes) {
        throw std::invalid_argument(
            "ExpertService: the layers of pool " + std::to_string(layer_pools[layer_id]) +
            " have rows of different bytes");
      }
      pool_bytes = layer_bytes;
    }
    victims_ = std::make_unique<VictimCache>(
        num_experts, layer_pools, entry_bytes, pool_slots, victim_cache_bytes);
    if (victims_->num_entries() == 0) {
      // A budget below one entry per pool is no cache: the plain path, without the hold.
      victims_.reset();
    }
  }
  std::fill(mailbox_data_, mailbox_data_ + MAILBOX_WORDS, 0u);
  mailbox_word(MailboxWord::SpinLimit).store(MAILBOX_SPIN_LIMIT, std::memory_order_release);
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
    int buffer,
    std::vector<ReadTask> copies) {
  batch.layer_id = layer_id;
  batch.buffer = buffer;
  batch.experts_read = static_cast<int64_t>(loads.size());
  batch.index_of_expert.assign(num_experts_, -1);
  for (const auto& [expert, index] : loads) {
    batch.index_of_expert[expert] = index;
  }
  batch.bytes_read.store(0, std::memory_order_relaxed);
  batch.error.clear();
  batch.held_fills.clear();
  bool is_held = batch.is_held.load(std::memory_order_relaxed);
  const RecordLayout& record = records_[layer_id];
  std::vector<ReadTask> tasks;
  for (int s = 0; s < segments_per_layer_; ++s) {
    int64_t segment_index = layer_id * segments_per_layer_ + s;
    const ExpertSegment& segment = segments_[segment_index];
    const ExpertLocation* locations = &locations_[segment_index * num_experts_];
    if (record.stride > 0 && locations[0].file_index != FILL_FILE) {
      // Read with the records below.
      continue;
    }
    for (int32_t first = 0; first < num_experts_;) {
      const ExpertLocation& start = locations[first];
      if (batch.index_of_expert[first] < 0) {
        ++first;
        continue;
      } else if (start.file_index == FILL_FILE) {
        if (is_held) {
          batch.held_fills.emplace_back(segment_index, first);
        } else {
          fill(batch, segment_index, first);
        }
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
  if (record.stride > 0) {
    // The records one read takes: the last one's bytes, a stride for each of the others.
    int32_t records_per_read =
        static_cast<int32_t>(1 + (MAX_READ_BYTES - record.bytes) / record.stride);
    for (int32_t first = 0; first < num_experts_;) {
      if (batch.index_of_expert[first] < 0) {
        ++first;
        continue;
      }
      int32_t last = first + 1;
      while (last < num_experts_ && last - first < records_per_read &&
             batch.index_of_expert[last] >= 0) {
        ++last;
      }
      int64_t begin = record.first + first * record.stride;
      int64_t end = begin + (last - 1 - first) * record.stride + record.bytes;
      tasks.push_back({&batch, -1, record.file_index, record.first, begin, end});
      first = last;
    }
  }
  for (ReadTask& copy : copies) {
    copy.batch = &batch;
    tasks.push_back(copy);
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
    if (task.entry != nullptr) {
      copy_entry(task.entry, task.batch->layer_id, task.batch->buffer, task.index, task.is_store);
    } else {
      read(task, staging);
    }
    task.batch->remaining.fetch_sub(1, std::memory_order_release);
  }
}

void ExpertService::read(const ReadTask& task, uint8_t* staging) {
  ReadBatch& batch = *task.batch;
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
  while (batch.is_held.load(std::memory_order_acquire)) {
    __builtin_arm_yield();
  }
  if (task.segment < 0) {
    const RecordLayout& record = records_[batch.layer_id];
    int64_t first = (task.begin - task.base) / record.stride;
    int64_t last = (task.end - 1 - task.base) / record.stride;
    for (int64_t expert = first; expert <= last; ++expert) {
      for (int s = 0; s < segments_per_layer_; ++s) {
        int64_t segment_index = batch.layer_id * segments_per_layer_ + s;
        const ExpertLocation& location = locations_[segment_index * num_experts_ + expert];
        if (location.file_index != FILL_FILE) {
          const ExpertSegment& segment = segments_[segment_index];
          std::memcpy(
              expert_row(batch, segment, expert),
              staging + (location.file_offset - aligned_begin),
              segment.expert_bytes);
        }
      }
    }
  } else {
    const ExpertSegment& segment = segments_[batch.layer_id * segments_per_layer_ + task.segment];
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
}

void ExpertService::fill(const ReadBatch& batch, int64_t segment_index, int32_t expert) {
  const ExpertSegment& segment = segments_[segment_index];
  std::memset(
      expert_row(batch, segment, expert),
      static_cast<int>(locations_[segment_index * num_experts_ + expert].file_offset),
      segment.expert_bytes);
}

uint8_t* ExpertService::expert_row(
    const ReadBatch& batch,
    const ExpertSegment& segment,
    int64_t expert) const {
  return bank_row(
             batch.layer_id, segment.bank_index, batch.buffer, batch.index_of_expert[expert]) +
      segment.row_offset;
}

uint8_t* ExpertService::bank_row(int layer_id, int64_t bank, int buffer, int64_t index) const {
  size_t destination = 2 * (layer_id * banks_.size() + bank);
  int64_t row_zero = buffer < 0 ? destinations_[destination] : buffer * buffer_bytes_[bank];
  return bank_data_[bank] + row_zero + index * destinations_[destination + 1];
}

void ExpertService::copy_entry(
    uint8_t* entry,
    int layer_id,
    int buffer,
    int32_t index,
    bool is_store) {
  auto started = std::chrono::steady_clock::now();
  for (size_t b = 0; b < banks_.size(); ++b) {
    int64_t row_bytes = destinations_[2 * (layer_id * banks_.size() + b) + 1];
    uint8_t* row = bank_row(layer_id, static_cast<int64_t>(b), buffer, index);
    if (is_store) {
      std::memcpy(entry, row, row_bytes);
    } else {
      std::memcpy(row, entry, row_bytes);
    }
    entry += row_bytes;
  }
  victim_copy_ns_.fetch_add(nanoseconds_since(started), std::memory_order_relaxed);
}

std::vector<ExpertLoad> ExpertService::split_victim_hits(
    int layer_id,
    const std::vector<ExpertLoad>& loads,
    std::vector<std::pair<int32_t, int32_t>>& hits) {
  std::vector<ExpertLoad> reads;
  for (const auto& [expert, index] : loads) {
    bool is_purged = false;
    int32_t entry = victims_->pin(layer_id * num_experts_ + expert, is_purged);
    stats_.victim_drops[layer_id] += is_purged;
    if (entry == NO_ENTRY) {
      reads.emplace_back(expert, index);
    } else {
      hits.emplace_back(entry, index);
    }
  }
  return reads;
}

void ExpertService::finish_buffer(PrefillBuffer& buffer, int64_t wait_ns) {
  buffer.is_waited = true;
  ReadBatch& batch = *buffer.batch;
  stats_.read_wait_ns[batch.layer_id] += wait_ns;
  stats_.experts_read[batch.layer_id] += batch.experts_read;
  stats_.bytes_read[batch.layer_id] += batch.bytes_read.load(std::memory_order_relaxed);
  if (victims_) {
    for (int32_t entry : buffer.entries) {
      victims_->unpin(entry);
    }
    stats_.victim_buffer_hits[batch.layer_id] += static_cast<int64_t>(buffer.entries.size());
    buffer.entries.clear();
  }
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
  if (victims_) {
    victims_->clear();
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
  if (!victims_) {
    submit(ensure_batch_, layer_id, assignment.loads);
  } else {
    // A read's or a hit's slot may be an evicted one: the reads' row writes wait for the
    // evicted rows to be stored, and the hits are copied in after.
    try {
      std::vector<std::pair<int32_t, int32_t>> hits;
      std::vector<ExpertLoad> reads = split_victim_hits(layer_id, assignment.loads, hits);
      ensure_batch_.is_held.store(true, std::memory_order_relaxed);
      submit(ensure_batch_, layer_id, reads);
      for (const auto& [flat_id, slot] : assignment.evicted) {
        int32_t entry = victims_->claim(flat_id);
        if (entry != NO_ENTRY) {
          copy_entry(victims_->data(entry), layer_id, -1, slot, true);
          victims_->unpin(entry);
          stats_.victim_stores[layer_id] += 1;
        }
      }
      ensure_batch_.is_held.store(false, std::memory_order_release);
      for (const auto& [segment_index, expert] : ensure_batch_.held_fills) {
        fill(ensure_batch_, segment_index, expert);
      }
      for (const auto& [entry, slot] : hits) {
        copy_entry(victims_->data(entry), layer_id, -1, slot, false);
        victims_->release(entry);
      }
      stats_.victim_hits[layer_id] += static_cast<int64_t>(hits.size());
    } catch (const std::exception& error) {
      // slot_of already maps the loads, whose rows may be incomplete: no IO thread may wait
      // on the hold, the batch's reads end before the batch is reused, and the cache fails.
      ensure_batch_.is_held.store(false, std::memory_order_release);
      wait_for(ensure_batch_);
      fail(MailboxError::Io, std::string("ExpertService: victim cache: ") + error.what());
      throw;
    }
  }
  wait_for(ensure_batch_);
  // The checkpoint's reads alone; a miss the victim cache served is no read.
  if (ensure_batch_.experts_read > 0) {
    stats_.read_wait_ns[layer_id] += nanoseconds_since(started);
  }
  stats_.experts_read[layer_id] += ensure_batch_.experts_read;
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
  std::vector<SlotExpert> dropped = lru_.invalidate_prefill_buffers();
  if (victims_) {
    // Their rows hold them until the next prefetch. Each pool keeps its most recently used,
    // stored least recently used first so they end up its newest entries.
    std::vector<int64_t> taken(victims_->num_pools(), 0);
    std::vector<SlotExpert> kept;
    for (const auto& [flat_id, slot] : dropped) {
      int32_t pool = victims_->pool_of(flat_id);
      if (taken[pool] < victims_->capacity(pool)) {
        ++taken[pool];
        kept.emplace_back(flat_id, slot);
      }
    }
    std::vector<std::vector<ReadTask>> stores(num_layers_);
    std::vector<int32_t> entries;
    std::vector<std::unique_ptr<ReadBatch>> batches;
    try {
      for (const auto& [flat_id, slot] : std::views::reverse(kept)) {
        int32_t entry = victims_->claim(flat_id);
        if (entry != NO_ENTRY) {
          int layer_id = flat_id / num_experts_;
          stores[layer_id].push_back(
              {.entry = victims_->data(entry), .index = slot, .is_store = true});
          entries.push_back(entry);
          stats_.victim_stores[layer_id] += 1;
        }
      }
      for (int layer_id = 0; layer_id < num_layers_; ++layer_id) {
        if (!stores[layer_id].empty()) {
          batches.push_back(std::make_unique<ReadBatch>());
          submit(*batches.back(), layer_id, {}, -1, std::move(stores[layer_id]));
        }
      }
      for (const auto& batch : batches) {
        wait_for(*batch);
      }
      for (int32_t entry : entries) {
        victims_->unpin(entry);
      }
    } catch (const std::exception& error) {
      // The submitted stores end before their batches go; the cache fails.
      for (const auto& batch : batches) {
        wait_for(*batch);
      }
      fail(MailboxError::Io, std::string("ExpertService: victim cache: ") + error.what());
      throw;
    }
  }
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
  if (!victims_) {
    submit(*buffer.batch, layer_id, assignment.loads, layer_id % 2);
  } else {
    // A cached expert is copied in and stays cached, pinned until the buffer is waited.
    try {
      std::vector<std::pair<int32_t, int32_t>> hits;
      std::vector<ExpertLoad> reads = split_victim_hits(layer_id, assignment.loads, hits);
      std::vector<ReadTask> copies;
      for (const auto& [entry, row] : hits) {
        copies.push_back({.entry = victims_->data(entry), .index = row});
        buffer.entries.push_back(entry);
      }
      submit(*buffer.batch, layer_id, reads, layer_id % 2, std::move(copies));
    } catch (const std::exception& error) {
      // The pool already maps the layer's experts to buffer rows nothing will fill.
      fail(MailboxError::Io, std::string("ExpertService: victim cache: ") + error.what());
      throw;
    }
  }
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
    if (victims_) {
      stats.victim_entries = victims_->num_entries();
      stats.victim_bytes = victims_->num_bytes();
    }
  }
  stats.victim_copy_ns = victim_copy_ns_.load(std::memory_order_relaxed);
  for (int layer_id = 0; layer_id < num_layers_; ++layer_id) {
    // The GPU's counts, read without synchronizing: an older value is fine.
    stats.lookups.push_back(
        std::atomic_ref<uint32_t>(hit_stats_data_[2 * layer_id]).load(std::memory_order_relaxed));
    stats.lookup_hits.push_back(std::atomic_ref<uint32_t>(hit_stats_data_[2 * layer_id + 1])
                                    .load(std::memory_order_relaxed));
  }
  return stats;
}

void ExpertService::purge_victim_cache() {
  std::lock_guard<std::mutex> lock(call_mutex_);
  if (victims_) {
    victims_->purge();
  }
}

std::vector<int32_t> ExpertService::victim_cache_ids() {
  std::lock_guard<std::mutex> lock(call_mutex_);
  return victims_ ? victims_->flat_ids() : std::vector<int32_t>();
}

} // namespace ftmetal
