// The GPU side of the mailbox handshake (common/mailbox.h) the lookup kernels share: whether the
// mailbox still takes posts, and the bounded wait for a request's answer. The including kernel
// defines thread_scope_system (coherent(system) needs the Metal internals) before it.
#pragma once

#include <metal_atomic>
#include <metal_stdlib>

#include "common/mailbox.h"

inline void system_fence() {
  metal::atomic_thread_fence(
      metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
}

// Whether a post may go out: nobody answers after a shutdown, nor once a wait gave up.
inline bool is_mailbox_open(volatile coherent(system) device const uint* mailbox) {
  return mailbox[uint(MailboxWord::Shutdown)] == 0 && mailbox[uint(MailboxWord::Abandoned)] == 0;
}

// Whether the CPU answered the request of sequence DONE: spins until it acknowledges it (State
// then holds the answer), the service shuts down or a wait gives up. After the spin limit
// unanswered this wait gives up itself, setting the abandoned word.
inline bool wait_for_answer(volatile coherent(system) device uint* mailbox, uint sequence) {
  const uint spin_limit = mailbox[uint(MailboxWord::SpinLimit)];
  for (uint spins = 0;; ++spins) {
    system_fence();
    if (mailbox[uint(MailboxWord::Acknowledged)] == sequence) {
      system_fence();
      return mailbox[uint(MailboxWord::State)] == uint(MailboxState::Done);
    }
    if (!is_mailbox_open(mailbox)) {
      return false;
    }
    if (spins >= spin_limit) {
      mailbox[uint(MailboxWord::Abandoned)] = 1;
      system_fence();
      return false;
    }
  }
}
