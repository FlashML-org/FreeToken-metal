// One threadgroup of THREADS per request: the topk-th largest of the
// complete blocks' order-preserving uint keys from a 4096-bin histogram of their high 12 bits,
// then a bisection of the low 20 bits over the threshold bin's keys (gathered in threadgroup
// memory, or re-read when more than 4096 share the bin): three passes over the row. The blocks
// above that key, then its ties lowest index first, are written in index order. Keys:
// score_order_key (qsa_indexer_topk.h).
static_assert(THREADS * 4 == 4096, "each thread sums four histogram bins");
const uint row = threadgroup_position_in_grid.y;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint simdgroup = simdgroup_index_in_threadgroup;
const uint num_blocks = uint(num_blocks_in);
const uint complete = metal::min(uint(seq_lens[row]) / BLOCK_SIZE, num_blocks);
device int* out = selected + row * TOPK;
if (complete <= uint(TOPK)) {
  for (uint i = tid; i < uint(TOPK); i += THREADS) {
    out[i] = i < complete ? int(i) : 0;
  }
  return;
}
// auto: MLX binds an input of under 8 elements as constant, not device.
auto row_scores = scores + row * num_blocks;
// The histogram, then (once the threshold bin is known) that bin's keys.
threadgroup atomic_uint bins[4096];
threadgroup uint sums[THREADS / 32];
threadgroup uint shared_state[3];
for (uint b = tid; b < 4096; b += THREADS) {
  atomic_store_explicit(&bins[b], 0u, memory_order_relaxed);
}
if (tid == 0) {
  shared_state[2] = 0;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < complete; i += THREADS) {
  const uint key = score_order_key(row_scores[i]);
  atomic_fetch_add_explicit(&bins[key >> 20], 1u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// Thread t holds bins 4095 - 4 t .. 4092 - 4 t, highest first.
uint counts[4];
uint total = 0;
for (uint j = 0; j < 4; ++j) {
  counts[j] = atomic_load_explicit(&bins[4095u - 4u * tid - j], memory_order_relaxed);
  total += counts[j];
}
uint above = simd_prefix_exclusive_sum(total);
if (lane == 31) {
  sums[simdgroup] = above + total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint g = 0; g < simdgroup; ++g) {
  above += sums[g];
}
if (above < uint(TOPK) && uint(TOPK) <= above + total) {
  for (uint j = 0; j < 4; ++j) {
    if (above + counts[j] >= uint(TOPK)) {
      shared_state[0] = 4095u - 4u * tid - j;
      shared_state[1] = uint(TOPK) - above;
      break;
    }
    above += counts[j];
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint bin = shared_state[0];
const uint need = shared_state[1];
threadgroup uint* candidates = reinterpret_cast<threadgroup uint*>(bins);
for (uint i = tid; i < complete; i += THREADS) {
  const uint key = score_order_key(row_scores[i]);
  if ((key >> 20) == bin) {
    const uint slot = atomic_fetch_add_explicit(
        reinterpret_cast<threadgroup atomic_uint*>(&shared_state[2]), 1u, memory_order_relaxed);
    if (slot < 4096u) {
      candidates[slot] = key;
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint num_candidates = shared_state[2];
// The need-th largest key of the bin: the largest prefix with at least need keys at or above.
uint prefix = bin << 20;
if (num_candidates <= 4096u) {
  if (simdgroup == 0) {
    for (int bit = 19; bit >= 0; --bit) {
      const uint candidate = prefix | (1u << bit);
      uint at_least = 0;
      for (uint c = lane; c < num_candidates; c += 32) {
        at_least += candidates[c] >= candidate;
      }
      if (simd_sum(at_least) >= need) {
        prefix = candidate;
      }
    }
    if (lane == 0) {
      shared_state[0] = prefix;
    }
  }
} else {
  threadgroup uint partial[2][THREADS / 32];
  for (int bit = 19; bit >= 0; --bit) {
    const uint candidate = prefix | (1u << bit);
    uint at_least = 0;
    for (uint i = tid; i < complete; i += THREADS) {
      const uint key = score_order_key(row_scores[i]);
      at_least += (key >> 20) == bin && key >= candidate;
    }
    at_least = simd_sum(at_least);
    if (lane == 0) {
      partial[bit & 1][simdgroup] = at_least;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = 0;
    for (uint g = 0; g < THREADS / 32; ++g) {
      count += partial[bit & 1][g];
    }
    if (count >= need) {
      prefix = candidate;
    }
  }
  if (tid == 0) {
    shared_state[0] = prefix;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
prefix = shared_state[0];
// prefix is the k-th largest key; ties equal to it are taken, lowest index first, each
// thread over a contiguous range so the selection comes out ascending.
const uint per = (complete + THREADS - 1) / THREADS;
const uint begin = metal::min(tid * per, complete);
const uint end = metal::min(begin + per, complete);
uint greater = 0;
uint equal = 0;
for (uint i = begin; i < end; ++i) {
  const uint key = score_order_key(row_scores[i]);
  greater += key > prefix;
  equal += key == prefix;
}
threadgroup uint greater_sums[THREADS / 32];
const uint simd_greater = simd_sum(greater);
uint eq_before = simd_prefix_exclusive_sum(equal);
if (lane == 31) {
  sums[simdgroup] = eq_before + equal;
}
if (lane == 0) {
  greater_sums[simdgroup] = simd_greater;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint remaining = uint(TOPK);
for (uint g = 0; g < THREADS / 32; ++g) {
  remaining -= greater_sums[g];
}
for (uint g = 0; g < simdgroup; ++g) {
  eq_before += sums[g];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint ties = eq_before >= remaining ? 0 : metal::min(equal, remaining - eq_before);
const uint mine = greater + ties;
uint before = simd_prefix_exclusive_sum(mine);
if (lane == 31) {
  sums[simdgroup] = before + mine;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint g = 0; g < simdgroup; ++g) {
  before += sums[g];
}
uint taken_ties = 0;
for (uint i = begin; i < end; ++i) {
  const uint key = score_order_key(row_scores[i]);
  bool take = key > prefix;
  if (key == prefix && taken_ties < ties) {
    take = true;
    ++taken_ties;
  }
  if (take) {
    out[before++] = int(i);
  }
}
