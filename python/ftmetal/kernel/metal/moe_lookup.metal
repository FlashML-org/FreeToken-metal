// MoeLookup (see csrc/moe_lookup.h): one threadgroup of MAX_EXPERTS threads, thread e owning
// expert e. The real rows' experts are marked in a threadgroup bitmap and looked up in
// slot_of; when some are absent thread 0 posts them, absent first, and spins until the CPU
// answers, then their slots are looked up again. Memory the CPU shares is accessed
// volatile coherent(system), with system-scope fences around the handshake. Buffers: 0
// topk_ids, 1 moe_step, 2 slot_of, 3 mailbox, 4 last_use, 5 hit_stats, 6 slots, 7 slot_map,
// 8 params.
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

// Shared with csrc/moe_lookup.h.
constant constexpr uint WORD_STATE = 0;
constant constexpr uint WORD_SEQUENCE = 1;
constant constexpr uint WORD_ACKNOWLEDGED = 2;
constant constexpr uint WORD_KIND = 32;
constant constexpr uint WORD_LAYER = 33;
constant constexpr uint WORD_STEP = 34;
constant constexpr uint WORD_NUM_ROWS = 35;
constant constexpr uint WORD_NUM_IDS = 36;
constant constexpr uint WORD_NUM_MISSES = 37;
constant constexpr uint WORD_IDS = 64;
constant constexpr uint STATE_IDLE = 0;
constant constexpr uint STATE_POSTED = 1;
constant constexpr uint STATE_DONE = 3;
constant constexpr uint STATE_FAILED = 4;
constant constexpr uint STATE_SHUTDOWN = 5;
constant constexpr uint KIND_PROBE = 3;
constant constexpr uint MAX_EXPERTS = 256;

constant constexpr uint SIMD_WIDTH = 32;
constant constexpr uint NUM_SIMDGROUPS = MAX_EXPERTS / SIMD_WIDTH;

struct LookupParams {
  int layer_id;
  int num_experts;
  int num_rows;
  int top_k;
  uint kind;
};

inline void system_fence() {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_system);
}

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
      // After a shutdown nobody answers: post nothing.
      if (mailbox[WORD_STATE] != STATE_SHUTDOWN) {
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
        uint state;
        while (true) {
          system_fence();
          state = mailbox[WORD_STATE];
          if (state == STATE_SHUTDOWN ||
              ((state == STATE_DONE || state == STATE_FAILED) &&
               mailbox[WORD_ACKNOWLEDGED] == sequence)) {
            break;
          }
        }
        system_fence();
        answered = state == STATE_DONE;
        if (state != STATE_SHUTDOWN) {
          mailbox[WORD_STATE] = STATE_IDLE;
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
  const int fallback_slot = max(first_slot, 0);
  if (expert == 0 && !is_probe) {
    hit_stats[2 * params.layer_id] += 1;
    hit_stats[2 * params.layer_id + 1] += num_hits;
  }
  for (int i = expert; i < params.num_rows * top_k; i += MAX_EXPERTS) {
    const int id = topk_ids[i];
    const bool is_real = i < num_real_rows * top_k && id >= 0 && id < num_experts;
    const int slot = is_real ? expert_slot[id] : -1;
    slots[i] = slot >= 0 ? slot : fallback_slot;
  }
  if (int(expert) < num_experts) {
    const int slot = expert_slot[expert];
    slot_map[expert] = slot >= 0 && is_used ? slot : fallback_slot;
  }
}
