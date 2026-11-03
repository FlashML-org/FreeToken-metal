// PleLookup (see ple_lookup.h). ple_post (one thread) writes each batch row's token and
// n-gram context to the mailbox, advances the context pool and posts; ple_wait (one
// threadgroup) spins until the CPU has answered the ticket's post, then copies the staged
// rows out with coherent loads. Memory the CPU shares is accessed volatile coherent(system),
// with system-scope fences around the handshake. Buffers: ple_post 0 input_ids, 1 slots,
// 2 context, 3 mailbox, 4 input_ids copy, 5 ticket, 6 params; ple_wait 0 ticket, 1 mailbox,
// 2-4 weight, scales and biases staging, 5-7 the parts out, 8 params.
#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
    static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}

#include <metal_atomic>
#include <metal_stdlib>
using namespace metal;

#include "common/mailbox.h"

constant constexpr uint WORD_STATE = uint(PleMailboxWord::State);
constant constexpr uint WORD_SEQUENCE = uint(PleMailboxWord::Sequence);
constant constexpr uint WORD_ACKNOWLEDGED = uint(PleMailboxWord::Acknowledged);
constant constexpr uint WORD_SHUTDOWN = uint(PleMailboxWord::Shutdown);
constant constexpr uint WORD_NUM_ROWS = uint(PleMailboxWord::NumRows);
constant constexpr uint WORD_ROWS = uint(PleMailboxWord::Rows);
constant constexpr uint STATE_IDLE = uint(MailboxState::Idle);
constant constexpr uint STATE_POSTED = uint(MailboxState::Posted);
constant constexpr uint STATE_DONE = uint(MailboxState::Done);
constant constexpr uint STATE_FAILED = uint(MailboxState::Failed);
constant constexpr uint STATE_SHUTDOWN = uint(MailboxState::Shutdown);

inline void system_fence() {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
}

[[kernel]] void ple_post(
    device const int* input_ids [[buffer(0)]],
    device const int* slots [[buffer(1)]],
    device int* context [[buffer(2)]],
    volatile coherent(system) device uint* mailbox [[buffer(3)]],
    device int* input_ids_copy [[buffer(4)]],
    device uint* ticket [[buffer(5)]],
    constant const PostParams& params [[buffer(6)]]) {
  const int num_rows = params.num_rows;
  // Every row's context is read before any is written: padding rows share a slot.
  for (int r = 0; r < num_rows; ++r) {
    const int slot = slots[r];
    mailbox[WORD_ROWS + 3 * r] = uint(input_ids[r]);
    mailbox[WORD_ROWS + 3 * r + 1] = uint(context[2 * slot + 1]);
    mailbox[WORD_ROWS + 3 * r + 2] = uint(context[2 * slot]);
    input_ids_copy[r] = input_ids[r];
  }
  for (int r = 0; r < num_rows; ++r) {
    const int slot = slots[r];
    context[2 * slot] = int(mailbox[WORD_ROWS + 3 * r + 1]);
    context[2 * slot + 1] = input_ids[r];
  }
  uint sequence = 0;
  // After a shutdown nobody answers: post nothing.
  if (mailbox[WORD_SHUTDOWN] == 0) {
    mailbox[WORD_NUM_ROWS] = uint(num_rows);
    system_fence();
    sequence = mailbox[WORD_SEQUENCE] + 1;
    // 0 is a refused post's ticket: the count wraps past it.
    sequence = sequence == 0 ? 1 : sequence;
    mailbox[WORD_SEQUENCE] = sequence;
    system_fence();
    mailbox[WORD_STATE] = STATE_POSTED;
    system_fence();
  }
  ticket[0] = sequence;
}

// The first bytes of staging, or zeros, by word where the part allows it.
inline void copy_part(
    volatile coherent(system) device const uchar* staging,
    device uchar* part,
    int bytes,
    bool is_answered,
    uint thread_index,
    uint num_threads) {
  if (bytes % 4 == 0) {
    auto staging_words = reinterpret_cast<volatile coherent(system) device const uint*>(staging);
    auto part_words = reinterpret_cast<device uint*>(part);
    for (int i = thread_index; i < bytes / 4; i += num_threads) {
      part_words[i] = is_answered ? staging_words[i] : 0u;
    }
  } else {
    for (int i = thread_index; i < bytes; i += num_threads) {
      part[i] = is_answered ? staging[i] : uchar(0);
    }
  }
}

[[kernel]] void ple_wait(
    device const uint* ticket [[buffer(0)]],
    volatile coherent(system) device uint* mailbox [[buffer(1)]],
    volatile coherent(system) device const uchar* weight_staging [[buffer(2)]],
    volatile coherent(system) device const uchar* scales_staging [[buffer(3)]],
    volatile coherent(system) device const uchar* biases_staging [[buffer(4)]],
    device uchar* weight [[buffer(5)]],
    device uchar* scales [[buffer(6)]],
    device uchar* biases [[buffer(7)]],
    constant const WaitParams& params [[buffer(8)]],
    uint thread_index [[thread_position_in_threadgroup]],
    uint num_threads [[threads_per_threadgroup]]) {
  threadgroup bool is_answered;
  if (thread_index == 0) {
    const uint sequence = ticket[0];
    bool answered = false;
    // A refused post (sequence 0) has nothing to wait for.
    if (sequence != 0) {
      uint state;
      while (true) {
        system_fence();
        state = mailbox[WORD_STATE];
        if ((state == STATE_DONE || state == STATE_FAILED) &&
            mailbox[WORD_ACKNOWLEDGED] == sequence) {
          break;
        }
        if (mailbox[WORD_SHUTDOWN] != 0) {
          state = STATE_SHUTDOWN;
          break;
        }
      }
      system_fence();
      answered = state == STATE_DONE;
      mailbox[WORD_STATE] = mailbox[WORD_SHUTDOWN] != 0 ? STATE_SHUTDOWN : STATE_IDLE;
      system_fence();
    }
    is_answered = answered;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  system_fence();
  const bool answered = is_answered;
  copy_part(weight_staging, weight, params.part_bytes[0], answered, thread_index, num_threads);
  copy_part(scales_staging, scales, params.part_bytes[1], answered, thread_index, num_threads);
  copy_part(biases_staging, biases, params.part_bytes[2], answered, thread_index, num_threads);
}
