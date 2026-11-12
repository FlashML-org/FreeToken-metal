// PleService (see ple_service.h): the n-gram hash, the coordinator thread that answers
// ple_post's posts, the watchdog thread that fails a post left unanswered, and the failure
// record every step checks.
#include "service/ple/ple_service.h"

#include <pthread/qos.h>

#include <algorithm>
#include <optional>

#include "service/expert/expert_lru.h"

namespace ftmetal {

namespace {

// As ExpertService's coordinator: it spins this long after its last answer or arm, then
// dozes in sleeps doubling from FIRST_DOZE to LONGEST_DOZE; spinning, it reads the clock once
// per POLLS_PER_CLOCK_READ empty polls.
const auto HOT_WINDOW = std::chrono::milliseconds(50);
const auto FIRST_DOZE = std::chrono::microseconds(100);
const auto LONGEST_DOZE = std::chrono::microseconds(2000);
const uint32_t POLLS_PER_CLOCK_READ = 1024;
const auto WATCHDOG_INTERVAL = std::chrono::seconds(2);

using Clock = std::chrono::steady_clock;

constexpr uint32_t state_code(MailboxState state) {
  return static_cast<uint32_t>(state);
}

bool is_pending(uint32_t state) {
  return state == state_code(MailboxState::Posted) || state == state_code(MailboxState::Claimed);
}

} // namespace

PleService::PleService(
    mx::array mailbox,
    std::vector<mx::array> staging,
    const PleRowStore& store,
    std::vector<int64_t> multipliers,
    std::vector<int64_t> head_sizes,
    std::vector<int64_t> head_offsets,
    int64_t eos_token_id,
    double stall_timeout_s,
    bool run_coordinator)
    : mailbox_(std::move(mailbox)),
      staging_(std::move(staging)),
      store_(store),
      multipliers_(std::move(multipliers)),
      head_sizes_(std::move(head_sizes)),
      head_offsets_(std::move(head_offsets)),
      eos_token_id_(eos_token_id),
      stall_timeout_(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::duration<double>(stall_timeout_s))) {
  if (mailbox_.dtype() != mx::uint32 || mailbox_.shape() != mx::Shape{PLE_MAILBOX_WORDS}) {
    throw std::invalid_argument(
        "PleService: mailbox must be uint32 [" + std::to_string(PLE_MAILBOX_WORDS) + "]");
  }
  if (!(stall_timeout_s > 0)) {
    throw std::invalid_argument("PleService: stall_timeout_s must be positive");
  }
  if (multipliers_.size() != 3) {
    throw std::invalid_argument("PleService: the hash takes 3 multipliers");
  }
  int heads = num_heads();
  if (heads < 2 || heads % 2 != 0 || head_offsets_.size() != head_sizes_.size()) {
    throw std::invalid_argument(
        "PleService: as many bigram as trigram heads, each with a size and an offset");
  }
  for (int h = 0; h < heads; ++h) {
    if (head_sizes_[h] < 1 || head_offsets_[h] < 0 ||
        head_offsets_[h] + head_sizes_[h] > store_.num_rows()) {
      throw std::invalid_argument(
          "PleService: head " + std::to_string(h) + "'s rows fall outside the table's " +
          std::to_string(store_.num_rows()));
    }
  }
  const char* names[3] = {"weight staging", "scales staging", "biases staging"};
  if (staging_.size() != 3) {
    throw std::invalid_argument(
        "PleService: staging must be the weight, scales and biases buffers");
  }
  for (int p = 0; p < 3; ++p) {
    mx::Shape shape{MAX_PLE_ROWS, heads, static_cast<int>(store_.row_bytes(p))};
    if (staging_[p].dtype() != mx::uint8 || staging_[p].shape() != shape) {
      throw std::invalid_argument(
          std::string("PleService: ") + names[p] + " must be uint8 [" +
          std::to_string(MAX_PLE_ROWS) + ", " + std::to_string(heads) + ", " +
          std::to_string(store_.row_bytes(p)) + "]");
    }
    staging_data_[p] = static_cast<uint8_t*>(shared_data(staging_[p], names[p]));
  }
  mailbox_data_ = static_cast<uint32_t*>(shared_data(mailbox_, "mailbox"));
  std::fill(mailbox_data_, mailbox_data_ + PLE_MAILBOX_WORDS, 0u);
  mailbox_word(PleMailboxWord::SpinLimit).store(MAILBOX_SPIN_LIMIT, std::memory_order_release);
  try {
    watchdog_ = std::thread(&PleService::watch, this);
    if (run_coordinator) {
      coordinator_ = std::thread(&PleService::coordinate, this);
    }
  } catch (...) {
    shutdown();
    throw;
  }
}

PleService::~PleService() {
  shutdown();
}

std::atomic_ref<uint32_t> PleService::mailbox_word(PleMailboxWord word) const {
  return std::atomic_ref<uint32_t>(mailbox_data_[static_cast<int>(word)]);
}

void PleService::ngram_ids(const int64_t* triples, int64_t num_triples, int64_t* ids) const {
  // Products wrap in int64; the remainder is taken non-negative.
  const uint64_t m0 = static_cast<uint64_t>(multipliers_[0]);
  const uint64_t m1 = static_cast<uint64_t>(multipliers_[1]);
  const uint64_t m2 = static_cast<uint64_t>(multipliers_[2]);
  const int heads = num_heads();
  for (int64_t r = 0; r < num_triples; ++r) {
    const int64_t token = triples[3 * r];
    const int64_t previous = triples[3 * r + 1];
    // A token before an eos is not part of the n-gram: the eos stands in for it.
    const int64_t before_previous = previous == eos_token_id_ ? eos_token_id_ : triples[3 * r + 2];
    const uint64_t bigram =
        (static_cast<uint64_t>(token) * m0) ^ (static_cast<uint64_t>(previous) * m1);
    const uint64_t trigram = bigram ^ (static_cast<uint64_t>(before_previous) * m2);
    for (int h = 0; h < heads; ++h) {
      const int64_t mixed = static_cast<int64_t>(h < heads / 2 ? bigram : trigram);
      int64_t remainder = mixed % head_sizes_[h];
      if (remainder < 0) {
        remainder += head_sizes_[h];
      }
      ids[r * heads + h] = remainder + head_offsets_[h];
    }
  }
}

void PleService::fail(MailboxError error, const std::string& message) {
  std::lock_guard<std::mutex> lock(failure_mutex_);
  if (failure_.empty()) {
    failure_ = message;
    failure_error_ = error;
    has_failed_.store(true, std::memory_order_release);
  }
}

void PleService::check_healthy() const {
  if (has_failed_.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    throw std::runtime_error("PleService failed earlier: " + failure_);
  }
  if (mailbox_word(PleMailboxWord::Abandoned).load(std::memory_order_acquire) != 0) {
    throw std::runtime_error(
        "PleService: a GPU lookup gave up waiting for the CPU past its spin limit");
  }
}

void PleService::arm() {
  armed_at_ns_.store(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
          .count());
  if (is_dozing_.load()) {
    {
      std::lock_guard<std::mutex> lock(wake_mutex_);
    }
    coordinator_wake_.notify_one();
  }
}

void PleService::shutdown() {
  std::call_once(shutdown_once_, [this] {
    // Recorded first: every wait the shutdown releases with zero rows is a failure.
    fail(MailboxError::Shutdown, "the service was shut down");
    is_stopping_.store(true, std::memory_order_release);
    // The shutdown word first: a post that has not happened yet posts nothing, and one that
    // posts anyway (it read the word just before) leaves its wait on it. State last: a waiting
    // GPU leaves on SHUTDOWN, and a coordinator still answering finds its post taken and
    // leaves the reply alone.
    mailbox_word(PleMailboxWord::Shutdown).store(1, std::memory_order_release);
    mailbox_word(PleMailboxWord::Error)
        .store(static_cast<uint32_t>(MailboxError::Shutdown), std::memory_order_relaxed);
    mailbox_word(PleMailboxWord::Acknowledged)
        .store(
            mailbox_word(PleMailboxWord::Sequence).load(std::memory_order_acquire),
            std::memory_order_release);
    mailbox_word(PleMailboxWord::State)
        .store(state_code(MailboxState::Shutdown), std::memory_order_release);
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
  });
}

void PleService::coordinate() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  auto is_armed = [this](Clock::time_point now) {
    auto armed_at = Clock::time_point(
        std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(armed_at_ns_.load())));
    return now - armed_at < HOT_WINDOW;
  };
  auto last_answered = Clock::now();
  auto doze = FIRST_DOZE;
  uint32_t empty_polls = 0;
  bool is_dozing = false;
  while (!is_stopping_.load(std::memory_order_acquire)) {
    if (mailbox_word(PleMailboxWord::State).load(std::memory_order_acquire) ==
        state_code(MailboxState::Posted)) {
      answer();
      last_answered = Clock::now();
      empty_polls = 0;
      is_dozing = false;
      continue;
    }
    if (!is_dozing) {
      if (++empty_polls % POLLS_PER_CLOCK_READ != 0) {
        __builtin_arm_yield();
        continue;
      }
      auto now = Clock::now();
      if (now - last_answered < HOT_WINDOW || is_armed(now)) {
        continue;
      }
      is_dozing = true;
      doze = FIRST_DOZE;
    }
    // arm() reads is_dozing_ after storing its time, so it either sees this store or the
    // wait's predicate sees its time.
    is_dozing_.store(true);
    {
      std::unique_lock<std::mutex> lock(wake_mutex_);
      coordinator_wake_.wait_for(lock, doze, [&] {
        return is_stopping_.load(std::memory_order_acquire) || is_armed(Clock::now());
      });
    }
    is_dozing_.store(false);
    if (is_armed(Clock::now())) {
      is_dozing = false;
    } else {
      doze = std::min(doze * 2, LONGEST_DOZE);
    }
  }
}

void PleService::answer() {
  std::atomic_ref<uint32_t> state = mailbox_word(PleMailboxWord::State);
  uint32_t expected = state_code(MailboxState::Posted);
  // The watchdog may have failed the post since it was seen.
  if (!state.compare_exchange_strong(
          expected, state_code(MailboxState::Claimed), std::memory_order_acq_rel)) {
    return;
  }
  auto read = [this](PleMailboxWord word) {
    return mailbox_word(word).load(std::memory_order_relaxed);
  };
  uint32_t sequence = read(PleMailboxWord::Sequence);
  uint32_t acknowledged = read(PleMailboxWord::Acknowledged);
  uint32_t num_rows = read(PleMailboxWord::NumRows);
  std::string post = "PLE post " + std::to_string(sequence);
  MailboxError error = MailboxError::None;
  std::string message;
  if (has_failed_.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    error = failure_error_;
  } else if (sequence != (acknowledged + 1 == 0 ? 1 : acknowledged + 1)) {
    error = MailboxError::Protocol;
    message = "protocol error: " + post + " follows post " + std::to_string(acknowledged);
  } else if (num_rows < 1 || num_rows > static_cast<uint32_t>(MAX_PLE_ROWS)) {
    error = MailboxError::Protocol;
    message = "protocol error: " + post + " has " + std::to_string(num_rows) + " rows";
  } else {
    uint32_t* rows = mailbox_data_ + static_cast<int>(PleMailboxWord::Rows);
    std::vector<int64_t> triples(3 * num_rows);
    for (uint32_t i = 0; i < 3 * num_rows; ++i) {
      // The GPU wrote int32 tokens.
      triples[i] =
          static_cast<int32_t>(std::atomic_ref<uint32_t>(rows[i]).load(std::memory_order_relaxed));
    }
    std::vector<int64_t> ids(num_rows * num_heads());
    ngram_ids(triples.data(), num_rows, ids.data());
    try {
      store_.gather(ids.data(), static_cast<int64_t>(ids.size()), staging_data_);
    } catch (const PleReadError& read_error) {
      error = MailboxError::Io;
      message = post + ": " + read_error.what();
    }
  }
  if (!message.empty()) {
    fail(error, message);
  }
  expected = state_code(MailboxState::Claimed);
  MailboxState reply = error == MailboxError::None ? MailboxState::Done : MailboxState::Failed;
  if (state.compare_exchange_strong(expected, state_code(reply), std::memory_order_acq_rel)) {
    mailbox_word(PleMailboxWord::Error)
        .store(static_cast<uint32_t>(error), std::memory_order_relaxed);
    // The GPU leaves once the acknowledgement matches: after the rows and the reply.
    mailbox_word(PleMailboxWord::Acknowledged).store(sequence, std::memory_order_release);
  }
  num_answered_.fetch_add(1, std::memory_order_release);
}

void PleService::watch() {
  struct Suspect {
    uint32_t sequence;
    int64_t num_answered;
    Clock::time_point first_seen;
  };
  std::optional<Suspect> suspect;
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(wake_mutex_);
      if (watchdog_wake_.wait_for(lock, WATCHDOG_INTERVAL, [this] {
            return is_stopping_.load(std::memory_order_acquire);
          })) {
        return;
      }
    }
    std::atomic_ref<uint32_t> state = mailbox_word(PleMailboxWord::State);
    uint32_t observed = state.load(std::memory_order_acquire);
    uint32_t sequence = mailbox_word(PleMailboxWord::Sequence).load(std::memory_order_acquire);
    int64_t num_answered = num_answered_.load(std::memory_order_acquire);
    auto now = Clock::now();
    // A post is stuck only while the coordinator answers nothing.
    if (!is_pending(observed)) {
      suspect.reset();
    } else if (!suspect || suspect->sequence != sequence || suspect->num_answered != num_answered) {
      suspect = Suspect{sequence, num_answered, now};
    } else if (now - suspect->first_seen >= stall_timeout_) {
      auto age_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - suspect->first_seen);
      suspect.reset();
      while (is_pending(observed) &&
             mailbox_word(PleMailboxWord::Sequence).load(std::memory_order_acquire) == sequence) {
        if (state.compare_exchange_weak(
                observed, state_code(MailboxState::Failed), std::memory_order_acq_rel)) {
          fail(
              MailboxError::Timeout,
              "PLE post " + std::to_string(sequence) + " was left unanswered for over " +
                  std::to_string(age_ms.count()) + " ms while the coordinator answered nothing");
          mailbox_word(PleMailboxWord::Error)
              .store(static_cast<uint32_t>(MailboxError::Timeout), std::memory_order_relaxed);
          mailbox_word(PleMailboxWord::Acknowledged).store(sequence, std::memory_order_release);
          break;
        }
      }
    }
  }
}

} // namespace ftmetal
