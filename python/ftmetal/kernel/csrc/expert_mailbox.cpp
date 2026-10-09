// ExpertService's mailbox side (see moe_lookup.h): the coordinator thread that answers the
// GPU lookups' requests with the IO side's reads, the watchdog thread that fails a request
// left unanswered, and the failure record every call checks.
#include "expert_service.h"

#include <mach/mach_time.h>
#include <pthread/qos.h>

#include <algorithm>
#include <optional>

namespace ftmetal {

namespace {

// The coordinator spins this long after its last answer or arm, then dozes in sleeps doubling
// from FIRST_DOZE to LONGEST_DOZE; spinning, it reads the clock once per POLLS_PER_CLOCK_READ
// empty polls.
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

std::atomic_ref<uint32_t> ExpertService::mailbox_word(MailboxWord word) const {
  return std::atomic_ref<uint32_t>(mailbox_data_[static_cast<int>(word)]);
}

void ExpertService::fail(MailboxError error, const std::string& message) {
  std::lock_guard<std::mutex> lock(failure_mutex_);
  if (failure_.empty()) {
    failure_ = message;
    failure_error_ = error;
    has_failed_.store(true, std::memory_order_release);
  }
}

void ExpertService::check_healthy() const {
  if (has_failed_.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    throw std::runtime_error("ExpertService failed earlier: " + failure_);
  }
}

void ExpertService::arm() {
  armed_at_ns_.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                         Clock::now().time_since_epoch())
                         .count());
  if (is_dozing_.load()) {
    {
      std::lock_guard<std::mutex> lock(wake_mutex_);
    }
    coordinator_wake_.notify_one();
  }
}

void ExpertService::coordinate() {
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  auto is_armed = [this](Clock::time_point now) {
    auto armed_at = Clock::time_point(std::chrono::duration_cast<Clock::duration>(
        std::chrono::nanoseconds(armed_at_ns_.load())));
    return now - armed_at < HOT_WINDOW;
  };
  auto last_answered = Clock::now();
  auto doze = FIRST_DOZE;
  uint32_t empty_polls = 0;
  bool is_dozing = false;
  while (!is_stopping_coordination_.load(std::memory_order_acquire)) {
    if (mailbox_word(MailboxWord::State).load(std::memory_order_acquire) ==
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
        return is_stopping_coordination_.load(std::memory_order_acquire) ||
            is_armed(Clock::now());
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

void ExpertService::answer() {
  std::atomic_ref<uint32_t> state = mailbox_word(MailboxWord::State);
  uint32_t expected = state_code(MailboxState::Posted);
  // The watchdog may have failed the request since it was seen.
  if (!state.compare_exchange_strong(
          expected, state_code(MailboxState::Claimed), std::memory_order_acq_rel)) {
    return;
  }
  uint64_t claim_time = mach_absolute_time();
  auto read = [this](MailboxWord word) {
    return mailbox_word(word).load(std::memory_order_relaxed);
  };
  uint32_t sequence = read(MailboxWord::Sequence);
  uint32_t acknowledged = read(MailboxWord::Acknowledged);
  uint32_t kind = read(MailboxWord::Kind);
  uint32_t layer_id = read(MailboxWord::Layer);
  uint32_t step = read(MailboxWord::Step);
  uint32_t num_ids = read(MailboxWord::NumIds);
  uint32_t num_misses = read(MailboxWord::NumMisses);
  std::string request = "lookup request " + std::to_string(sequence) + " (layer " +
      std::to_string(layer_id) + ", step " + std::to_string(step) + ")";
  bool is_known_kind = kind == static_cast<uint32_t>(MoeLookupKind::Decode) ||
      kind == static_cast<uint32_t>(MoeLookupKind::PrefillCopyMiss) ||
      kind == static_cast<uint32_t>(MoeLookupKind::Probe);
  bool is_probe = kind == static_cast<uint32_t>(MoeLookupKind::Probe);
  MailboxError error = MailboxError::None;
  std::string message;
  int64_t num_loaded = 0;
  int64_t num_evicted = 0;
  if (has_failed_.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    error = failure_error_;
  } else if (sequence != acknowledged + 1) {
    error = MailboxError::Protocol;
    message = "protocol error: " + request + " follows request " + std::to_string(acknowledged);
  } else if (
      !is_known_kind || layer_id >= static_cast<uint32_t>(num_layers_) ||
      num_ids > static_cast<uint32_t>(num_experts_) || num_misses > num_ids ||
      (is_probe && num_ids != 0)) {
    error = MailboxError::Protocol;
    message = "protocol error: " + request + " has kind " + std::to_string(kind) + ", " +
        std::to_string(num_misses) + " of " + std::to_string(num_ids) + " experts absent";
  } else if (!is_probe) {
    const uint16_t* posted =
        reinterpret_cast<const uint16_t*>(mailbox_data_ + static_cast<int>(MailboxWord::Ids));
    std::vector<int32_t> expert_ids(posted, posted + num_ids);
    try {
      std::lock_guard<std::mutex> lock(call_mutex_);
      SlotAssignment assignment = serve(static_cast<int>(layer_id), expert_ids, step);
      bool is_same_split = assignment.loads.size() == num_misses;
      for (size_t j = 0; is_same_split && j < num_misses; ++j) {
        is_same_split = assignment.loads[j].first == expert_ids[j];
      }
      if (!is_same_split) {
        throw std::logic_error(
            "the GPU found " + std::to_string(num_misses) + " experts absent, the cache " +
            std::to_string(assignment.loads.size()));
      }
      num_loaded = static_cast<int64_t>(assignment.loads.size());
      num_evicted = assignment.num_evicted;
    } catch (const ReadError& read_error) {
      error = MailboxError::Io;
      message = read_error.what();
    } catch (const std::logic_error& logic_error) {
      error = MailboxError::Protocol;
      message = "protocol error: " + request + ": " + logic_error.what();
    } catch (const std::runtime_error& runtime_error) {
      error = MailboxError::Capacity;
      message = request + ": " + runtime_error.what();
    }
  }
  if (!message.empty()) {
    fail(error, message);
  }
  expected = state_code(MailboxState::Claimed);
  MailboxState answer = error == MailboxError::None ? MailboxState::Done : MailboxState::Failed;
  if (state.compare_exchange_strong(expected, state_code(answer), std::memory_order_acq_rel)) {
    mailbox_word(MailboxWord::NumLoaded)
        .store(static_cast<uint32_t>(num_loaded), std::memory_order_relaxed);
    mailbox_word(MailboxWord::NumEvicted)
        .store(static_cast<uint32_t>(num_evicted), std::memory_order_relaxed);
    mailbox_word(MailboxWord::Error)
        .store(static_cast<uint32_t>(error), std::memory_order_relaxed);
    auto stamp = [this](MailboxWord word, uint64_t time) {
      std::atomic_ref<uint64_t>(
          *reinterpret_cast<uint64_t*>(mailbox_data_ + static_cast<int>(word)))
          .store(time, std::memory_order_relaxed);
    };
    stamp(MailboxWord::ClaimTime, claim_time);
    stamp(MailboxWord::DoneTime, mach_absolute_time());
    // The GPU leaves once the acknowledgement matches: after the slots, rows and reply.
    mailbox_word(MailboxWord::Acknowledged).store(sequence, std::memory_order_release);
  }
  num_answered_.fetch_add(1, std::memory_order_release);
}

void ExpertService::watch() {
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
            return is_stopping_coordination_.load(std::memory_order_acquire);
          })) {
        return;
      }
    }
    std::atomic_ref<uint32_t> state = mailbox_word(MailboxWord::State);
    uint32_t observed = state.load(std::memory_order_acquire);
    uint32_t sequence = mailbox_word(MailboxWord::Sequence).load(std::memory_order_acquire);
    int64_t num_answered = num_answered_.load(std::memory_order_acquire);
    auto now = Clock::now();
    // A request is stuck only while the coordinator answers nothing: a loaded one that
    // progresses between samples is alive.
    if (!is_pending(observed)) {
      suspect.reset();
    } else if (
        !suspect || suspect->sequence != sequence || suspect->num_answered != num_answered) {
      suspect = Suspect{sequence, num_answered, now};
    } else if (now - suspect->first_seen >= stall_timeout_) {
      auto age_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - suspect->first_seen);
      suspect.reset();
      while (is_pending(observed) &&
             mailbox_word(MailboxWord::Sequence).load(std::memory_order_acquire) == sequence) {
        if (state.compare_exchange_weak(
                observed, state_code(MailboxState::Failed), std::memory_order_acq_rel)) {
          fail(
              MailboxError::Timeout,
              "lookup request " + std::to_string(sequence) + " (layer " +
                  std::to_string(mailbox_word(MailboxWord::Layer).load()) +
                  ") was left unanswered for over " + std::to_string(age_ms.count()) +
                  " ms while the coordinator answered nothing");
          mailbox_word(MailboxWord::Error)
              .store(static_cast<uint32_t>(MailboxError::Timeout), std::memory_order_relaxed);
          mailbox_word(MailboxWord::Acknowledged).store(sequence, std::memory_order_release);
          break;
        }
      }
    }
  }
}

} // namespace ftmetal
