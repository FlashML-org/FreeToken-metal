// MoeLookup (see moe_lookup.h): one threadgroup of MAX_EXPERTS threads, thread e owning expert e.
// The real rows' experts are marked in a threadgroup bitmap and looked up in slot_of; when some are
// absent thread 0 posts them, absent first, waits for the CPU's answer (common/mailbox_wait.h) and
// their slots are looked up again. Memory the CPU shares is accessed volatile coherent(system),
// with system-scope fences around the handshake. Buffers: 0 topk_ids, 1 moe_step, 2 slot_of, 3
// mailbox, 4 last_use, 5 hit_stats, 6 slots, 7 slot_map, 8 params.
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

#include "common/mailbox_wait.h"

constant constexpr uint WORD_STATE = uint(MailboxWord::State);
constant constexpr uint WORD_SEQUENCE = uint(MailboxWord::Sequence);
constant constexpr uint WORD_SHUTDOWN = uint(MailboxWord::Shutdown);
constant constexpr uint WORD_ABANDONED = uint(MailboxWord::Abandoned);
constant constexpr uint WORD_KIND = uint(MailboxWord::Kind);
constant constexpr uint WORD_LAYER = uint(MailboxWord::Layer);
constant constexpr uint WORD_STEP = uint(MailboxWord::Step);
constant constexpr uint WORD_NUM_ROWS = uint(MailboxWord::NumRows);
constant constexpr uint WORD_NUM_IDS = uint(MailboxWord::NumIds);
constant constexpr uint WORD_NUM_MISSES = uint(MailboxWord::NumMisses);
constant constexpr uint WORD_IDS = uint(MailboxWord::Ids);
constant constexpr uint STATE_IDLE = uint(MailboxState::Idle);
constant constexpr uint STATE_POSTED = uint(MailboxState::Posted);
constant constexpr uint STATE_SHUTDOWN = uint(MailboxState::Shutdown);
constant constexpr uint KIND_PROBE = uint(MoeLookupKind::Probe);
constant constexpr uint MAX_EXPERTS = MAX_LOOKUP_EXPERTS;

constant constexpr uint SIMD_WIDTH = 32;
constant constexpr uint NUM_SIMDGROUPS = MAX_EXPERTS / SIMD_WIDTH;

[[kernel]] void moe_lookup(
    device const int* topk_ids [[buffer(0)]],
    device const int* moe_step [[buffer(1)]],
    volatile coherent(system) device const int* slot_of [[buffer(2)]],
    volatile coherent(system) device uint* mailbox [[buffer(3)]],
    volatile coherent(system) device uint* last_use [[buffer(4)]],
    device uint* hit_stats [[buffer(5)]],
    device int* slots [[buffer(6)]],
    device int* slot_map [[buffer(7)]],
    constant const LookupParams& params [[buffer(8)]],
    uint expert [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint used_bits[MAX_EXPERTS / 32];
  threadgroup int expert_slot[MAX_EXPERTS];
  threadgroup uint simdgroup_misses[NUM_SIMDGROUPS];
  threadgroup uint simdgroup_hits[NUM_SIMDGROUPS];
  threadgroup ushort posted_ids[MAX_EXPERTS];
  threadgroup bool is_answered;

  const int num_experts = params.num_experts;
  const int top_k = params.top_k;
  const bool is_probe = params.kind == KIND_PROBE;
  const int num_real_rows = is_probe ? 0 : clamp(moe_step[0], 0, params.num_rows);
  const uint step = uint(moe_step[1]);
  if (expert < MAX_EXPERTS / 32) {
    atomic_store_explicit(&used_bits[expert], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = expert; i < num_real_rows * top_k; i += MAX_EXPERTS) {
    const int id = topk_ids[i];
    if (id >= 0 && id < num_experts) {
      atomic_fetch_or_explicit(&used_bits[id / 32], 1u << (id % 32), memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint flat_id = uint(params.layer_id * num_experts) + expert;
  const bool is_used = int(expert) < num_experts &&
      ((atomic_load_explicit(&used_bits[expert / 32], memory_order_relaxed) >> (expert % 32)) & 1u);
  const int resident_slot = is_used ? slot_of[flat_id] : -1;
  const bool is_hit = resident_slot >= 0;
  const bool is_miss = is_used && !is_hit;
  if (is_hit) {
    last_use[resident_slot] = step;
    system_fence();
  }
  expert_slot[expert] = resident_slot;

  // Posted order: the absent experts ascending, then the resident ones ascending.
  const uint miss_rank = simd_prefix_exclusive_sum(uint(is_miss));
  const uint hit_rank = simd_prefix_exclusive_sum(uint(is_hit));
  if (lane == SIMD_WIDTH - 1) {
    simdgroup_misses[simdgroup] = miss_rank + uint(is_miss);
    simdgroup_hits[simdgroup] = hit_rank + uint(is_hit);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint miss_base = 0;
  uint hit_base = 0;
  uint num_misses = 0;
  uint num_hits = 0;
  for (uint g = 0; g < NUM_SIMDGROUPS; ++g) {
    if (g < simdgroup) {
      miss_base += simdgroup_misses[g];
      hit_base += simdgroup_hits[g];
    }
    num_misses += simdgroup_misses[g];
    num_hits += simdgroup_hits[g];
  }

  if (num_misses > 0 || is_probe) {
    const uint num_ids = num_misses + num_hits;
    if (is_miss) {
      posted_ids[miss_base + miss_rank] = ushort(expert);
    }
    if (is_hit) {
      posted_ids[num_misses + hit_base + hit_rank] = ushort(expert);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = expert; 2 * w < num_ids; w += MAX_EXPERTS) {
      const uint high = 2 * w + 1 < num_ids ? uint(posted_ids[2 * w + 1]) : 0u;
      mailbox[WORD_IDS + w] = uint(posted_ids[2 * w]) | (high << 16);
    }
    system_fence();
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (expert == 0) {
      bool answered = false;
      // After a shutdown, or once a wait gave up, nobody answers: post nothing. The shutdown
      // word, unlike the state, is never overwritten here, so a shutdown racing this post still
      // releases the wait.
      if (is_mailbox_open(mailbox)) {
        mailbox[WORD_KIND] = params.kind;
        mailbox[WORD_LAYER] = uint(params.layer_id);
        mailbox[WORD_STEP] = step;
        mailbox[WORD_NUM_ROWS] = uint(num_real_rows);
        mailbox[WORD_NUM_IDS] = num_ids;
        mailbox[WORD_NUM_MISSES] = num_misses;
        system_fence();
        const uint sequence = mailbox[WORD_SEQUENCE] + 1;
        mailbox[WORD_SEQUENCE] = sequence;
        system_fence();
        mailbox[WORD_STATE] = STATE_POSTED;
        answered = wait_for_answer(mailbox, sequence);
        // A wait that gave up leaves the handshake to the CPU, if it is still there.
        if (mailbox[WORD_ABANDONED] == 0) {
          mailbox[WORD_STATE] = mailbox[WORD_SHUTDOWN] != 0 ? STATE_SHUTDOWN : STATE_IDLE;
          system_fence();
        }
      }
      is_answered = answered;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    system_fence();
    if (is_miss) {
      expert_slot[expert] = is_answered ? slot_of[flat_id] : -1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Padding rows, unused experts and unanswered misses read the slot of the first real
  // pair: the step already uses it, so no later layer of the step can rewrite it while the
  // GPU caches may hold it.
  const int first_id = num_real_rows > 0 ? topk_ids[0] : -1;
  const int first_slot = first_id >= 0 && first_id < num_experts ? expert_slot[first_id] : -1;
  const int fallback_slot = max(first_slot, params.slot_base) - params.slot_base;
  if (expert == 0 && !is_probe) {
    hit_stats[2 * params.layer_id] += 1;
    hit_stats[2 * params.layer_id + 1] += num_hits;
  }
  for (int i = expert; i < params.num_rows * top_k; i += MAX_EXPERTS) {
    const int id = topk_ids[i];
    const bool is_real = i < num_real_rows * top_k && id >= 0 && id < num_experts;
    const int slot = is_real ? expert_slot[id] : -1;
    slots[i] = slot >= 0 ? slot - params.slot_base : fallback_slot;
  }
  if (int(expert) < num_experts) {
    const int slot = expert_slot[expert];
    slot_map[expert] = slot >= 0 && is_used ? slot - params.slot_base : fallback_slot;
  }
}
