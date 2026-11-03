// Histogram with threadgroup atomics, exclusive scan of the E counts (simdgroup prefix
// sums, then a scan of the simdgroup totals), then each pair claims its slot with an
// atomic cursor per expert.

const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint simdgroup = simdgroup_index_in_threadgroup;
const int num_pairs = topk_ids_shape[0] * K;
threadgroup atomic_int cursor[E];
threadgroup int simdgroup_offsets[TG / 32];

for (uint e = tid; e < E; e += TG) {
  atomic_store_explicit(&cursor[e], 0, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int i = tid; i < num_pairs; i += TG) {
  atomic_fetch_add_explicit(&cursor[topk_ids[i]], 1, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

const int count = tid < E ? atomic_load_explicit(&cursor[tid], memory_order_relaxed) : 0;
const int prefix = simd_prefix_exclusive_sum(count);
if (lane == 31) {
  simdgroup_offsets[simdgroup] = prefix + count;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simdgroup == 0) {
  const int total = lane < TG / 32 ? simdgroup_offsets[lane] : 0;
  const int scanned = simd_prefix_exclusive_sum(total);
  if (lane < TG / 32) {
    simdgroup_offsets[lane] = scanned;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const int offset = prefix + simdgroup_offsets[simdgroup];
if (tid < E) {
  expert_offsets[tid] = offset;
  atomic_store_explicit(&cursor[tid], offset, memory_order_relaxed);
}
if (tid == E - 1) {
  expert_offsets[E] = offset + count;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int i = tid; i < num_pairs; i += TG) {
  const int expert = topk_ids[i];
  const int position = atomic_fetch_add_explicit(&cursor[expert], 1, memory_order_relaxed);
  sorted_token_indices[position] = i / K;
  sorted_expert_ids[position] = expert;
  pair_positions[i] = position;
}
