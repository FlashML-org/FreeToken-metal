// Token sampling: softmax(logits / T), top-k and top-p filtering, inverse-CDF draw.
// {{T}} and {{KEY_BITS}} are substituted per logits dtype; function constants specialize
// each dispatch of the chain csrc/sample.cpp encodes.
// Thresholds come from a radix select over an order-preserving key of each logit's bit
// pattern (16 bits for bf16 / fp16, 32 for fp32), eight bits per round: top-k keeps every
// logit >= the k-th largest (counts per bin), top-p every logit >= the largest value whose
// descending softmax mass reaches p times the mass top-k kept (mass per bin); boundary ties
// are kept. A row is split into G chunks, one threadgroup each. sample_round first selects
// the previous round's bin: every threadgroup of the row sums the G per-chunk histograms that
// round left in the workspace, in chunk order, so all of them reach the same bracket without
// waiting on each other, and every sum is deterministic. It then bins its chunk inside the
// bracket. sample_find, one threadgroup per row, selects the last round's bin, forms each
// chunk's kept mass and draws by inverse CDF inside the chunk that holds u * total.
#include <metal_stdlib>
using namespace metal;

typedef {{T}} T;
#define KEY_BITS {{KEY_BITS}}

#if KEY_BITS == 16
typedef ushort Bits;
#else
typedef uint Bits;
#endif
constant constexpr uint SIGN_BIT = 1u << (KEY_BITS - 1);
constant constexpr uint KEY_MASK = KEY_BITS == 32 ? 0xFFFFFFFFu : 0xFFFFu;

constant constexpr int KIND_NONE = 0;
constant constexpr int KIND_COUNT = 1;
constant constexpr int KIND_MASS = 2;
constant constexpr uint NUM_BINS = 256;
constant constexpr uint ROUND_THREADS = 256;
constant constexpr uint FIND_THREADS = 1024;
// Logits each thread loads before binning them, so their memory latencies overlap.
constant constexpr uint LOADS_PER_THREAD = 8;
constant constexpr uint MAX_GROUPS = 256;
constant constexpr uint STATE_SLOTS = 10;
constant constexpr uint STATE_WORDS = 8;

// Position in the chain; the round this dispatch selects (prev_*) and the one it bins (cur_*).
constant int dispatch_index [[function_constant(0)]];
constant int prev_kind [[function_constant(1)]];
constant int prev_shift [[function_constant(2)]];
constant bool prev_root [[function_constant(3)]];
constant int cur_kind [[function_constant(4)]];
constant int cur_shift [[function_constant(5)]];
constant bool cur_root [[function_constant(6)]];
constant bool cur_final [[function_constant(7)]];
constant bool has_top_k [[function_constant(8)]];
constant bool has_top_p [[function_constant(9)]];

// Workspace, fp32 / uint32 words: group_max [B, G]; state [B, STATE_SLOTS, STATE_WORDS];
// counts and masses [2, B, G, NUM_BINS] (dispatches alternate parities); above [2, B, G].
struct Workspace {
  device float* group_max;
  device uint* state;
  device uint* counts;
  device float* masses;
  device float* above;
};

inline Workspace workspace_at(device uint* base, uint rows, uint groups) {
  const uint cells = rows * groups;
  Workspace w;
  w.group_max = (device float*)base;
  w.state = base + cells;
  w.counts = w.state + rows * STATE_SLOTS * STATE_WORDS;
  w.masses = (device float*)(w.counts + 2 * cells * NUM_BINS);
  w.above = w.masses + 2 * cells * NUM_BINS;
  return w;
}

inline uint chunk_cell(uint parity, uint row, uint group, uint rows, uint groups) {
  return (parity * rows + row) * groups + group;
}

// A row's radix-select progress after a dispatch's select.
struct RowState {
  uint lo;            // bracket bottom key
  uint above_count;   // logits above the bracket, top-k rounds
  float above_mass;   // their mass, top-p rounds
  float target_mass;  // p times the mass top-k kept
  float row_max;
  uint selected_bin;  // the bin the select took, relative to the previous bracket
};

inline RowState load_state(device const uint* slot) {
  return RowState{
      slot[0],
      slot[1],
      as_type<float>(slot[2]),
      as_type<float>(slot[3]),
      as_type<float>(slot[4]),
      slot[5]};
}

inline void store_state(device uint* slot, RowState state) {
  slot[0] = state.lo;
  slot[1] = state.above_count;
  slot[2] = as_type<uint>(state.above_mass);
  slot[3] = as_type<uint>(state.target_mass);
  slot[4] = as_type<uint>(state.row_max);
  slot[5] = state.selected_bin;
}

// Larger logits get larger keys; -0 shares +0's key so equal logits tie.
inline uint sort_key(T value) {
  uint bits = as_type<Bits>(value);
  if ((bits & (KEY_MASK ^ SIGN_BIT)) == 0) {
    bits = 0;
  }
  return (bits & SIGN_BIT) ? (~bits & KEY_MASK) : (bits | SIGN_BIT);
}

// Unnormalized softmax weight; the row max weighs exactly 1 (also when it is -inf).
inline float token_mass(float x, float row_max, float scale) {
  if (x == row_max) {
    return 1.0f;
  }
  return x == -INFINITY ? 0.0f : fast::exp2((x - row_max) * scale);
}

// Selects the previous round's bin from its per-chunk histograms and returns the row's
// bracket after it. Every thread of the threadgroup must call it.
template <uint NT>
RowState select_bracket(
    Workspace ws,
    uint row,
    uint rows,
    uint groups,
    uint vocab,
    device const int* top_k,
    device const float* top_p,
    threadgroup uint* scratch,
    threadgroup uint* suffix,
    threadgroup uint* simd_words,
    uint tid,
    uint sg,
    uint lane) {
  constexpr uint PARTS = NT / NUM_BINS;
  constexpr uint SIMDGROUPS = NT / 32;
  RowState state =
      load_state(ws.state + (row * STATE_SLOTS + dispatch_index - 1) * STATE_WORDS);
  if (dispatch_index == 1) {
    if (sg == 0) {
      float row_max = -INFINITY;
      for (uint g = lane; g < groups; g += 32) {
        row_max = max(row_max, ws.group_max[row * groups + g]);
      }
      row_max = simd_max(row_max);
      if (lane == 0) {
        simd_words[0] = as_type<uint>(row_max);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    state.row_max = as_type<float>(simd_words[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (prev_kind == KIND_NONE) {
    return state;
  }

  // Bin totals over the row's chunks, in chunk order.
  const uint parity = (dispatch_index - 1) & 1;
  const uint bin = tid % NUM_BINS;
  const uint part = tid / NUM_BINS;
  const uint first_cell = chunk_cell(parity, row, 0, rows, groups);
  if (prev_kind == KIND_COUNT) {
    uint total = 0;
    for (uint g0 = part; g0 < groups; g0 += PARTS * LOADS_PER_THREAD) {
      uint counts[LOADS_PER_THREAD];
      for (uint u = 0; u < LOADS_PER_THREAD; ++u) {
        const uint g = g0 + u * PARTS;
        counts[u] = g < groups ? ws.counts[(first_cell + g) * NUM_BINS + bin] : 0;
      }
      for (uint u = 0; u < LOADS_PER_THREAD; ++u) {
        total += counts[u];
      }
    }
    scratch[tid] = total;
  } else {
    float total = 0.0f;
    for (uint g0 = part; g0 < groups; g0 += PARTS * LOADS_PER_THREAD) {
      float masses[LOADS_PER_THREAD];
      for (uint u = 0; u < LOADS_PER_THREAD; ++u) {
        const uint g = g0 + u * PARTS;
        masses[u] = g < groups ? ws.masses[(first_cell + g) * NUM_BINS + bin] : 0.0f;
      }
      for (uint u = 0; u < LOADS_PER_THREAD; ++u) {
        total += masses[u];
      }
    }
    scratch[tid] = as_type<uint>(total);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < NUM_BINS) {
    if (prev_kind == KIND_COUNT) {
      uint total = 0;
      for (uint q = 0; q < PARTS; ++q) {
        total += scratch[q * NUM_BINS + tid];
      }
      scratch[tid] = total;
    } else {
      float total = 0.0f;
      for (uint q = 0; q < PARTS; ++q) {
        total += as_type<float>(scratch[q * NUM_BINS + tid]);
      }
      scratch[tid] = as_type<uint>(total);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // suffix[j] = sum of bins >= j: thread t scans bin NUM_BINS - 1 - t.
  const uint j = NUM_BINS - 1 - (tid % NUM_BINS);
  uint count_sum = 0;
  float mass_sum = 0.0f;
  if (prev_kind == KIND_COUNT) {
    count_sum = simd_prefix_inclusive_sum(tid < NUM_BINS ? scratch[j] : 0u);
    if (lane == 31) {
      simd_words[sg] = count_sum;
    }
  } else {
    mass_sum = simd_prefix_inclusive_sum(tid < NUM_BINS ? as_type<float>(scratch[j]) : 0.0f);
    if (lane == 31) {
      simd_words[sg] = as_type<uint>(mass_sum);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < NUM_BINS) {
    if (prev_kind == KIND_COUNT) {
      for (uint q = 0; q < sg; ++q) {
        count_sum += simd_words[q];
      }
      suffix[j] = count_sum;
    } else {
      for (uint q = 0; q < sg; ++q) {
        mass_sum += as_type<float>(simd_words[q]);
      }
      suffix[j] = as_type<uint>(mass_sum);
    }
  }
  if (tid == 0) {
    suffix[NUM_BINS] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The largest bin whose suffix reaches the target; bin 0 (the whole bracket) when none
  // does, which is how a row with the filter off keeps everything.
  uint target_count = 0xFFFFFFFFu;
  if (prev_kind == KIND_COUNT) {
    const int k = has_top_k ? top_k[row] : 0;
    target_count = (k > 0 && uint(k) < vocab) ? uint(k) : 0xFFFFFFFFu;
  } else if (prev_root) {
    const float p = has_top_p ? top_p[row] : 1.0f;
    const float kept = state.above_mass + as_type<float>(suffix[0]);
    state.target_mass = p < 1.0f ? p * kept : INFINITY;
  }
  int best = -1;
  if (tid < NUM_BINS) {
    bool reached;
    if (prev_kind == KIND_COUNT) {
      reached = state.above_count + suffix[j] >= target_count;
    } else {
      const float mass = state.above_mass + as_type<float>(suffix[j]);
      reached = mass >= state.target_mass && mass > 0.0f;
    }
    best = reached ? int(j) : -1;
  }
  best = simd_max(best);
  if (lane == 0) {
    simd_words[32 + sg] = as_type<uint>(best);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int selected = 0;
  for (uint q = 0; q < SIMDGROUPS; ++q) {
    selected = max(selected, as_type<int>(simd_words[32 + q]));
  }
  state.lo += uint(selected) << prev_shift;
  if (prev_kind == KIND_COUNT) {
    state.above_count += suffix[selected + 1];
  } else {
    state.above_mass += as_type<float>(suffix[selected + 1]);
  }
  state.selected_bin = uint(selected);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return state;
}

// Adds one element per active lane to this simdgroup's bins. Threadgroup memory has no float
// atomics: lanes sharing a bin are summed by one simd reduction that lane 0 adds, so every
// bin's sum runs in a fixed order.
inline void add_to_bins(
    bool active,
    uint bin,
    float mass,
    bool with_counts,
    bool with_masses,
    threadgroup uint* bin_counts,
    threadgroup float* bin_masses,
    uint lane) {
  while (true) {
    const uint pending = uint(static_cast<simd_vote::vote_t>(simd_ballot(active)));
    if (pending == 0) {
      break;
    }
    const uint leader_bin = simd_shuffle(bin, ushort(ctz(pending)));
    const bool match = active && bin == leader_bin;
    if (with_counts) {
      const uint count = popcount(uint(static_cast<simd_vote::vote_t>(simd_ballot(match))));
      if (lane == 0) {
        bin_counts[leader_bin] += count;
      }
    }
    if (with_masses) {
      const float sum = simd_sum(match ? mass : 0.0f);
      if (lane == 0) {
        bin_masses[leader_bin] += sum;
      }
    }
    active = active && !match;
  }
}

// Buffers: 0 logits [B, V], 1 temperatures [B] fp32, 2 top_k [B] int32, 3 top_p [B] fp32,
// 4 workspace, 5 vocab, 6 chunk, 7 groups. Grid (G, B) threadgroups of ROUND_THREADS.
[[kernel]] void sample_round(
    device const T* logits [[buffer(0)]],
    device const float* temperatures [[buffer(1)]],
    device const int* top_k [[buffer(2)]],
    device const float* top_p [[buffer(3)]],
    device uint* workspace [[buffer(4)]],
    constant const uint& vocab [[buffer(5)]],
    constant const uint& chunk [[buffer(6)]],
    constant const uint& groups [[buffer(7)]],
    uint2 position [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint SIMDGROUPS = ROUND_THREADS / 32;
  const uint group = position.x;
  const uint row = position.y;
  const uint rows = grid.y;
  const Workspace ws = workspace_at(workspace, rows, groups);
  threadgroup uint bin_counts[SIMDGROUPS * NUM_BINS];
  threadgroup float bin_masses[SIMDGROUPS * NUM_BINS];
  threadgroup uint suffix[NUM_BINS + 1];
  threadgroup uint simd_words[64];
  threadgroup float simd_floats[2 * SIMDGROUPS];

  RowState state = {0, 0, 0.0f, INFINITY, -INFINITY, 0};
  if (dispatch_index > 0) {
    state = select_bracket<ROUND_THREADS>(
        ws, row, rows, groups, vocab, top_k, top_p, bin_counts, suffix, simd_words, tid, sg,
        lane);
  }
  if (tid == 0 && group == 0) {
    store_state(ws.state + (row * STATE_SLOTS + dispatch_index) * STATE_WORDS, state);
  }
  for (uint i = tid; i < SIMDGROUPS * NUM_BINS; i += ROUND_THREADS) {
    bin_counts[i] = 0;
    bin_masses[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool with_counts = cur_kind == KIND_COUNT;
  const bool with_masses = cur_kind == KIND_MASS || cur_final;
  // Without top-p a mass round only feeds the draw: all of its mass counts as above the bracket.
  const bool binless = cur_kind == KIND_MASS && !has_top_p;
  const uint width_shift = cur_shift + 8;
  const float scale = M_LOG2E_F / temperatures[row];
  const uint start = min(vocab, group * chunk);
  const uint end = min(vocab, start + chunk);
  const device T* row_logits = logits + (ulong)row * vocab;
  float local_max = -INFINITY;
  float above_mass = 0.0f;
  for (uint base = start; base < end; base += ROUND_THREADS * LOADS_PER_THREAD) {
    T raws[LOADS_PER_THREAD];
    for (uint u = 0; u < LOADS_PER_THREAD; ++u) {
      const uint i = base + u * ROUND_THREADS + tid;
      raws[u] = i < end ? row_logits[i] : T(0);
    }
    for (uint u = 0; u < LOADS_PER_THREAD; ++u) {
      const bool valid = base + u * ROUND_THREADS + tid < end;
      const float x = float(raws[u]);
      if (dispatch_index == 0 && valid) {
        local_max = max(local_max, x);
      }
      if (cur_kind == KIND_NONE) {
        continue;
      }
      const uint key = sort_key(raws[u]);
      const uint offset = key - state.lo;
      const bool at_or_above = valid && key >= state.lo;
      const bool in_bracket =
          !binless && at_or_above && (cur_root || (offset >> width_shift) == 0);
      float mass = 0.0f;
      if (with_masses && at_or_above) {
        mass = token_mass(x, state.row_max, scale);
      }
      if (cur_final && at_or_above && !in_bracket) {
        above_mass += mass;
      }
      const uint bin = in_bracket ? offset >> cur_shift : 0;
      if (with_masses) {
        add_to_bins(
            in_bracket,
            bin,
            mass,
            with_counts,
            with_masses,
            bin_counts + sg * NUM_BINS,
            bin_masses + sg * NUM_BINS,
            lane);
      } else if (in_bracket) {
        // Counts do not depend on the order of the adds.
        atomic_fetch_add_explicit(
            (threadgroup atomic_uint*)(bin_counts + sg * NUM_BINS + bin), 1u, memory_order_relaxed);
      }
    }
  }

  local_max = simd_max(local_max);
  above_mass = simd_sum(above_mass);
  if (lane == 0) {
    simd_floats[sg] = local_max;
    simd_floats[SIMDGROUPS + sg] = above_mass;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint cell = chunk_cell(dispatch_index & 1, row, group, rows, groups);
  if (with_counts) {
    uint count = 0;
    for (uint s = 0; s < SIMDGROUPS; ++s) {
      count += bin_counts[s * NUM_BINS + tid];
    }
    ws.counts[cell * NUM_BINS + tid] = count;
  }
  if (with_masses) {
    float mass = 0.0f;
    for (uint s = 0; s < SIMDGROUPS; ++s) {
      mass += bin_masses[s * NUM_BINS + tid];
    }
    ws.masses[cell * NUM_BINS + tid] = mass;
  }
  if (tid == 0) {
    float group_max = -INFINITY;
    float group_above = 0.0f;
    for (uint s = 0; s < SIMDGROUPS; ++s) {
      group_max = max(group_max, simd_floats[s]);
      group_above += simd_floats[SIMDGROUPS + s];
    }
    if (dispatch_index == 0) {
      ws.group_max[row * groups + group] = group_max;
    }
    if (cur_final) {
      ws.above[cell] = group_above;
    }
  }
}

// Buffers 0-7 as sample_round, 8 random_bits [B] uint32, 9 tokens [B] int32.
// Grid (1, B) threadgroups of FIND_THREADS.
[[kernel]] void sample_find(
    device const T* logits [[buffer(0)]],
    device const float* temperatures [[buffer(1)]],
    device const int* top_k [[buffer(2)]],
    device const float* top_p [[buffer(3)]],
    device uint* workspace [[buffer(4)]],
    constant const uint& vocab [[buffer(5)]],
    constant const uint& chunk [[buffer(6)]],
    constant const uint& groups [[buffer(7)]],
    device const uint* random_bits [[buffer(8)]],
    device int* tokens [[buffer(9)]],
    uint2 position [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint SIMDGROUPS = FIND_THREADS / 32;
  const uint row = position.y;
  const uint rows = grid.y;
  const Workspace ws = workspace_at(workspace, rows, groups);
  threadgroup uint scratch[FIND_THREADS];
  threadgroup uint suffix[NUM_BINS + 1];
  threadgroup uint simd_words[64];
  threadgroup float kept[MAX_GROUPS];
  threadgroup float simd_sums[SIMDGROUPS];
  threadgroup int simd_best[SIMDGROUPS];
  threadgroup uint draw[3];

  const RowState state = select_bracket<FIND_THREADS>(
      ws, row, rows, groups, vocab, top_k, top_p, scratch, suffix, simd_words, tid, sg, lane);
  const uint threshold = state.lo;

  // Kept mass per chunk: the mass above the last bracket plus its bins from the selected one.
  const uint first_cell = chunk_cell((dispatch_index - 1) & 1, row, 0, rows, groups);
  for (uint g = sg; g < groups; g += SIMDGROUPS) {
    float mass = 0.0f;
    for (uint b = lane; b < NUM_BINS; b += 32) {
      if (b >= state.selected_bin) {
        mass += ws.masses[(first_cell + g) * NUM_BINS + b];
      }
    }
    mass = simd_sum(mass);
    if (lane == 0) {
      kept[g] = mass + ws.above[first_cell + g];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The chunk holding the draw: the last one with mass that starts at or before u * total.
  if (tid == 0) {
    float total = 0.0f;
    for (uint g = 0; g < groups; ++g) {
      total += kept[g];
    }
    const float target = float(random_bits[row] >> 8) * 0x1.0p-24f * total;
    uint owner = 0;
    float offset = 0.0f;
    float running = 0.0f;
    for (uint g = 0; g < groups; ++g) {
      if (kept[g] > 0.0f && running <= target) {
        owner = g;
        offset = running;
      }
      running += kept[g];
    }
    draw[0] = owner;
    draw[1] = as_type<uint>(offset);
    draw[2] = as_type<uint>(target);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint owner = draw[0];
  const float target = as_type<float>(draw[2]);

  // Inside the chunk: each thread owns a contiguous run; the last run with mass that starts
  // at or before the target walks to the token, or to its last kept token when fp32
  // rounding leaves the target past the run's end.
  const uint start = min(vocab, owner * chunk);
  const uint end = min(vocab, start + chunk);
  const uint run = (end - start + FIND_THREADS - 1) / FIND_THREADS;
  const uint run_start = min(end, start + tid * run);
  const uint run_end = min(end, run_start + run);
  const float scale = M_LOG2E_F / temperatures[row];
  const device T* row_logits = logits + (ulong)row * vocab;
  float run_mass = 0.0f;
  for (uint i = run_start; i < run_end; ++i) {
    const T raw = row_logits[i];
    if (sort_key(raw) >= threshold) {
      run_mass += token_mass(float(raw), state.row_max, scale);
    }
  }
  const float before = simd_prefix_exclusive_sum(run_mass);
  if (lane == 31) {
    simd_sums[sg] = before + run_mass;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float running = as_type<float>(draw[1]);
  for (uint q = 0; q < sg; ++q) {
    running += simd_sums[q];
  }
  running += before;
  const int candidate = simd_max(run_mass > 0.0f && running <= target ? int(tid) : -1);
  if (lane == 0) {
    simd_best[sg] = candidate;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int chosen = -1;
  for (uint q = 0; q < SIMDGROUPS; ++q) {
    chosen = max(chosen, simd_best[q]);
  }
  if (int(tid) == chosen) {
    int token = -1;
    int last_kept = -1;
    for (uint i = run_start; i < run_end; ++i) {
      const T raw = row_logits[i];
      if (sort_key(raw) < threshold) {
        continue;
      }
      const float mass = token_mass(float(raw), state.row_max, scale);
      if (mass > 0.0f) {
        last_kept = int(i);
        running += mass;
        if (running > target) {
          token = int(i);
          break;
        }
      }
    }
    tokens[row] = token >= 0 ? token : last_kept;
  } else if (chosen < 0 && tid == 0) {
    // No kept mass anywhere (NaN logits): any in-range id beats garbage.
    tokens[row] = int(start);
  }
}
