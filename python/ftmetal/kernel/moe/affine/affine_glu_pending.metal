// AffineGluPending (see affine_glu_pending.h): threadgroup z is one (token, expert) pair, each
// simdgroup ACT_ROWS_PER_SIMDGROUP rows of its act. A pair moe_lookup_issue resolved takes its gate
// and up rows from gate_up_resident, in its first threadgroup; a pending one waits (thread 0 on the
// mailbox, common/mailbox_wait.h), resolves its slot as moe_lookup would have and computes its gate
// and up rows at it. act is the Act template argument's, silu_and_mul's or gelu_tanh_and_mul's: the
// rows in the dtype, then Act(gate) * up in fp32 as MLX computes it, cast back. Memory the CPU
// shares is accessed volatile coherent(system) after a system-scope fence; the bank rows of pending
// slots, which nothing reads before the answer, are read after it. The {{...}} tokens are
// substituted per configuration before compilation, and each #include is inlined (_ext.py).
// Buffers: 0 x, 1 w, 2 scales or codes, 3 biases or pairs, 4 gate_up_resident, 5 slots, 6 issued, 7
// slot_of, 8 mailbox, 9 act, 10 resolved slots, 11 params.
// Adapted from MLX v0.32.3 (Copyright © 2023-2024 Apple Inc., MIT License): the pending pairs'
// gate_up rows are those of affine_gather_qmv_fast (2087-2147) and affine_gather_qmv (2149-2209)
// in mlx/backend/metal/kernels/quantized.h, on affine_slot_qmv.h's impls, a simdgroup computing
// its act rows' gate and up rows together; act_mul's gelu_tanh is nn.gelu_approx in
// python/mlx/nn/layers/activations.py (186).
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

#include "mlx/backend/metal/kernels/utils.h"
#include "common/mailbox_wait.h"
#include "common/norm.h"
#include "moe/affine/affine_glu_pending_shared.h"
#include "moe/affine/affine_slot_qmv.h"

// MLX's steel headers disable the internals that coherent(system) needs.
#pragma METAL internals : enable

using namespace metal;

constant constexpr int ISSUED_SEQUENCE = int(IssuedWord::Sequence);
constant constexpr int ISSUED_NUM_MISSES = int(IssuedWord::NumMisses);

// The activations layers/activation.py serves, of a gate value in fp32. Each op rounds as in MLX's
// compiled kernels, which contract none into an fma.
struct Silu {
  float operator()(float gate) thread {
#pragma clang fp contract(off)
    return gate * precise_sigmoid(gate);
  }
};

// nn.gelu_approx's terms in its order, on MLX's ops and constants.
struct GeluTanh {
  float operator()(float gate) thread {
#pragma clang fp contract(off)
    const float inner = 0.7978846f * (gate + 0.044715f * metal::pow(gate, 3.0f));
    return 0.5f * gate * (1.0f + metal::precise::tanh(inner));
  }
};

// layers/activation.py's silu_mul or gelu_tanh_mul of one gate and up value.
template <typename Act, typename T>
inline T act_mul(T gate, T up) {
  return static_cast<T>(Act()(static_cast<float>(gate)) * static_cast<float>(up));
}

template <typename T, int GROUP_SIZE, int BITS, int CODE_BITS, bool FAST, typename Act>
[[kernel]] void affine_glu_pending(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device uint8_t* scales_or_codes [[buffer(2)]],
    const device T* biases_or_pairs [[buffer(3)]],
    const device T* gate_up_resident [[buffer(4)]],
    const device int* slots [[buffer(5)]],
    const device int* issued [[buffer(6)]],
    volatile coherent(system) device const int* slot_of [[buffer(7)]],
    volatile coherent(system) device uint* mailbox [[buffer(8)]],
    device T* act [[buffer(9)]],
    device int* resolved_slots [[buffer(10)]],
    constant const GluPendingParams& params [[buffer(11)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int ROWS = ACT_ROWS_PER_SIMDGROUP;
  constexpr int THREADS = SIMD_SIZE * GLU_PENDING_SIMDGROUPS;
  threadgroup bool is_answered;

  const int pair = threadgroup_id.z;
  const int intermediate_size = params.intermediate_size;
  const int act_row = (int(threadgroup_id.y) * GLU_PENDING_SIMDGROUPS + int(simdgroup)) * ROWS;
  const bool is_slot_writer = threadgroup_id.y == 0 && thread_index == 0;
  device T* pair_act = act + size_t(pair) * intermediate_size;
  const int issued_slot = slots[pair];
  // A resident pair's act is elementwise: the pair's first threadgroup computes all of it.
  if (issued_slot >= 0) {
    if (threadgroup_id.y == 0) {
      if (thread_index == 0) {
        resolved_slots[pair] = issued_slot;
      }
      const device T* gate_up = gate_up_resident + size_t(pair) * 2 * intermediate_size;
      for (int row = thread_index; row < intermediate_size; row += THREADS) {
        pair_act[row] = act_mul<Act>(gate_up[row], gate_up[intermediate_size + row]);
      }
    }
    return;
  }

  if (thread_index == 0) {
    // A refused post (or none) has nothing to wait for.
    is_answered = issued[ISSUED_NUM_MISSES] != 0 &&
        wait_for_answer(mailbox, uint(issued[ISSUED_SEQUENCE]));
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  system_fence();
  const bool answered = is_answered;
  // moe_lookup's slots: an answered miss's from slot_of; padding pairs and unanswered misses the
  // first pair's, itself resolved first, or the pool's first slot.
  const int num_experts = params.num_experts;
  const int first_slot = slots[0];
  const int layer_slot = params.layer_id * num_experts;
  int fallback_slot = 0;
  if (first_slot >= 0) {
    fallback_slot = first_slot;
  } else if (answered && first_slot >= -num_experts) {
    fallback_slot =
        max(slot_of[layer_slot + (-1 - first_slot)], params.slot_base) - params.slot_base;
  }
  int slot = fallback_slot;
  if (issued_slot >= -num_experts) {
    const int expert_slot = answered ? slot_of[layer_slot + (-1 - issued_slot)] : -1;
    if (expert_slot >= 0) {
      slot = expert_slot - params.slot_base;
    }
  }
  if (is_slot_writer) {
    resolved_slots[pair] = slot;
  }
  if (act_row >= intermediate_size) {
    return;
  }

  const int hidden_size = params.hidden_size;
  const int output_size = 2 * intermediate_size;
  int rows[2 * ROWS];
  for (int r = 0; r < ROWS; r++) {
    rows[r] = act_row + r;
    rows[ROWS + r] = intermediate_size + act_row + r;
  }
  float result[2 * ROWS];
  const device uint32_t* slot_w = w + size_t(slot) * output_size * (hidden_size * BITS / 32);
  const device T* pair_x = x + size_t(pair / params.top_k) * hidden_size;
  if constexpr (CODE_BITS == 0) {
    const size_t slot_groups = size_t(slot) * output_size * (hidden_size / GROUP_SIZE);
    const AffineScales<T> scales{
        (const device T*)scales_or_codes + slot_groups, biases_or_pairs + slot_groups};
    slot_qmv_rows<T, GROUP_SIZE, BITS, FAST>(
        slot_w, scales, pair_x, hidden_size, rows, result, lane);
  } else {
    const PairCodeScales<T, CODE_BITS, const device T*> scales{
        (const device uint32_t*)(scales_or_codes + size_t(slot) * params.code_row_bytes),
        biases_or_pairs};
    slot_qmv_rows<T, GROUP_SIZE, BITS, FAST>(
        slot_w, scales, pair_x, hidden_size, rows, result, lane);
  }
  if (lane == 0) {
    for (int r = 0; r < ROWS; r++) {
      pair_act[act_row + r] =
          act_mul<Act>(static_cast<T>(result[r]), static_cast<T>(result[ROWS + r]));
    }
  }
}

instantiate_kernel(
    "affine_glu_pending",
    affine_glu_pending,
    {{T}},
    {{GROUP_SIZE}},
    {{BITS}},
    {{CODE_BITS}},
    {{FAST}},
    {{ACT}})
