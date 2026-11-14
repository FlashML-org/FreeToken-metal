// BLOCKS_PER_SIMDGROUP consecutive blocks of one request per simdgroup and step, the grid capped
// at SCORE_THREADGROUPS per request: a wide page table (a
// replayed step's) loops instead of launching a threadgroup per 32 blocks, most of them past the
// request's complete blocks. Each block's score: the sum over the index heads of relu(q . k) /
// sqrt(D) in fp32, -inf past the complete blocks.

const uint row = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
const uint num_blocks = uint(num_blocks_in);
const uint complete = uint(seq_lens[row]) / BLOCK_SIZE;
const uint stride = threadgroups_per_grid.x * SIMDGROUPS * BLOCKS_PER_SIMDGROUP;
constexpr uint PER_LANE = D / 32;
const uint width = page_table_shape[1];
float query[HEADS][PER_LANE];
bool has_query = false;
for (uint first_block =
         (threadgroup_position_in_grid.x * SIMDGROUPS + simdgroup_index_in_threadgroup) *
         BLOCKS_PER_SIMDGROUP;
     first_block < num_blocks;
     first_block += stride) {
  if (first_block >= complete) {
    if (lane < BLOCKS_PER_SIMDGROUP && first_block + lane < num_blocks) {
      scores[row * num_blocks + first_block + lane] = -INFINITY;
    }
    continue;
  }
  if (!has_query) {
    for (uint h = 0; h < HEADS; ++h) {
      for (uint i = 0; i < PER_LANE; ++i) {
        query[h][i] = static_cast<float>(queries[(row * HEADS + h) * D + lane + i * 32]);
      }
    }
    has_query = true;
  }
  float key[BLOCKS_PER_SIMDGROUP][PER_LANE];
  for (uint b = 0; b < BLOCKS_PER_SIMDGROUP; ++b) {
    // A block past the complete ones reads the last complete one's key (its score is -inf).
    const uint block = metal::min(first_block + b, complete - 1);
    const uint position = block * BLOCK_SIZE + BLOCK_SIZE - 1;
    const uint slot = uint(page_table[row * width + position / page_size]) * page_size +
                      position % page_size;
    for (uint i = 0; i < PER_LANE; ++i) {
      key[b][i] = static_cast<float>(index_keys[size_t(slot) * D + lane + i * 32]);
    }
  }
  for (uint b = 0; b < BLOCKS_PER_SIMDGROUP; ++b) {
    const uint block = first_block + b;
    if (block >= num_blocks) {
      break;
    }
    float total = 0.0f;
    for (uint h = 0; h < HEADS; ++h) {
      float dot = 0.0f;
      for (uint i = 0; i < PER_LANE; ++i) {
        dot += query[h][i] * key[b][i];
      }
      total += metal::max(simd_sum(dot), 0.0f);
    }
    if (lane == 0) {
      scores[row * num_blocks + block] = block < complete ? total * inv_sqrt_d : -INFINITY;
    }
  }
}
