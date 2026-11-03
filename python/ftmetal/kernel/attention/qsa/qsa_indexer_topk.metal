// One threadgroup per request, thread t holding the order-preserving uint keys of blocks
// [t * per, t * per + per) in registers. Four 8-bit radix passes find the topk-th largest key:
// each counts the candidates' digits with one atomic add per (simdgroup, distinct digit), a
// capped loop retiring the lanes that share the lowest pending lane's digit each round, and
// simdgroup 0 finds the digit by a prefix sum over the bins, highest first. The blocks above
// that key, then its ties lowest index first, are written in index order.

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
auto row_scores = scores + row * num_blocks;
const uint per = (complete + THREADS - 1) / THREADS;
const uint begin = metal::min(tid * per, complete);
const uint count = metal::min(begin + per, complete) - begin;
uint keys[PER_THREAD];
for (uint j = 0; j < uint(PER_THREAD); ++j) {
  const uint bits = j < count ? as_type<uint>(row_scores[begin + j]) : 0u;
  keys[j] = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
threadgroup atomic_uint histogram[256];
threadgroup uint shared_state[2];
uint prefix = 0;
uint mask = 0;
uint remaining = TOPK;
for (int shift = 24; shift >= 0; shift -= 8) {
  for (uint b = tid; b < 256; b += THREADS) {
    atomic_store_explicit(&histogram[b], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < uint(PER_THREAD); ++j) {
    bool pending = j < count && (keys[j] & mask) == prefix;
    const uint bin = (keys[j] >> shift) & 255u;
    for (uint round = 0; round < 32; ++round) {
      const uint votes = uint(ulong(simd_ballot(pending)));
      if (votes == 0) {
        break;
      }
      const uint leader = ctz(votes);
      const uint leader_bin = simd_shuffle(bin, ushort(leader));
      const bool same = pending && bin == leader_bin;
      const uint same_votes = uint(ulong(simd_ballot(same)));
      if (lane == leader) {
        atomic_fetch_add_explicit(
            &histogram[leader_bin], popcount(same_votes), memory_order_relaxed);
      }
      pending = pending && !same;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0) {
    // Lane l holds bins 255 - 8 l .. 248 - 8 l, highest first.
    uint counts[8];
    uint total = 0;
    for (uint k = 0; k < 8; ++k) {
      counts[k] = atomic_load_explicit(&histogram[255u - 8u * lane - k], memory_order_relaxed);
      total += counts[k];
    }
    uint above = simd_prefix_exclusive_sum(total);
    if (above < remaining && remaining <= above + total) {
      uint bin = 255u - 8u * lane;
      for (uint k = 0; k < 8; ++k) {
        if (above + counts[k] >= remaining) {
          bin = 255u - 8u * lane - k;
          break;
        }
        above += counts[k];
      }
      shared_state[0] = bin;
      shared_state[1] = remaining - above;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  prefix |= shared_state[0] << shift;
  mask |= 255u << shift;
  remaining = shared_state[1];
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
// prefix is the topk-th largest key; the first remaining keys equal to it are taken.
uint greater = 0;
uint equal = 0;
for (uint j = 0; j < uint(PER_THREAD); ++j) {
  greater += j < count && keys[j] > prefix;
  equal += j < count && keys[j] == prefix;
}
threadgroup uint sums[THREADS / 32];
// Exclusive prefix sums over the threads, in index order.
uint equal_before = simd_prefix_exclusive_sum(equal);
if (lane == 31) {
  sums[simdgroup] = equal_before + equal;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint j = 0; j < simdgroup; ++j) {
  equal_before += sums[j];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint ties =
    equal_before >= remaining ? 0 : metal::min(equal, remaining - equal_before);
const uint mine = greater + ties;
uint before = simd_prefix_exclusive_sum(mine);
if (lane == 31) {
  sums[simdgroup] = before + mine;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint j = 0; j < simdgroup; ++j) {
  before += sums[j];
}
uint taken_ties = 0;
for (uint j = 0; j < uint(PER_THREAD); ++j) {
  bool take = j < count && keys[j] > prefix;
  if (j < count && keys[j] == prefix && taken_ties < ties) {
    take = true;
    ++taken_ties;
  }
  if (take) {
    out[before++] = int(begin + j);
  }
}
