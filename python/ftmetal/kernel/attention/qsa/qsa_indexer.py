"""QSA (Qwen Sparse Attention) decode block selection and index-key pooling.

A QSA layer keeps one per-token array index_keys [pool_tokens, index_head_dim]: the slot of
the last token of each block_size-token block holds that block's pooled, normed and roped
key (the only slots qsa_block_scores reads); every other slot holds its token's raw index
key (what qsa_window_mean averages). Page tables are page ids int32 [bs, width]: position p
of request b lives at slot page_table[b, p // page_size] * page_size + p % page_size. A
decode step scores each request's complete blocks against its index queries
(qsa_block_scores), keeps the topk best (qsa_topk) and lists the token slots of those blocks
and of its incomplete tail block for flash_decode (qsa_key_table). The kernels skip the
blocks past each request's own length, so a step replayed at the page table's full width
costs what the requests' lengths do.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel.custom import metal_kernel

SCORE_SIMDGROUPS = 8
# Each simdgroup scores this many consecutive blocks: their page-table and key loads issue
# together and the queries are read once.
BLOCKS_PER_SIMDGROUP = 4
# One threadgroup per request, the largest Metal allows.
TOPK_THREADS = 1024

# One simdgroup per BLOCKS_PER_SIMDGROUP consecutive blocks of one request: the sum over the
# index heads of relu(q . k) / sqrt(D) in fp32, -inf for a block past the complete ones.
SCORES_SOURCE = """
    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint first_block =
        (threadgroup_position_in_grid.x * SIMDGROUPS + simdgroup_index_in_threadgroup) *
        BLOCKS_PER_SIMDGROUP;
    const uint num_blocks = uint(num_blocks_in);
    if (first_block >= num_blocks) {
      return;
    }
    const uint complete = uint(seq_lens[row]) / BLOCK_SIZE;
    if (first_block >= complete) {
      if (lane < BLOCKS_PER_SIMDGROUP && first_block + lane < num_blocks) {
        scores[row * num_blocks + first_block + lane] = -INFINITY;
      }
      return;
    }
    constexpr uint PER_LANE = D / 32;
    float query[HEADS][PER_LANE];
    for (uint h = 0; h < HEADS; ++h) {
      for (uint i = 0; i < PER_LANE; ++i) {
        query[h][i] = static_cast<float>(queries[(row * HEADS + h) * D + lane + i * 32]);
      }
    }
    const uint width = page_table_shape[1];
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
"""

SCORES_KERNEL = metal_kernel(
    name="qsa_block_scores",
    input_names=[
        "queries",
        "index_keys",
        "page_table",
        "seq_lens",
        "page_size",
        "num_blocks_in",
        "inv_sqrt_d",
    ],
    output_names=["scores"],
    source=SCORES_SOURCE,
)

# One threadgroup per request, thread t holding the order-preserving uint keys of blocks
# [t * per, t * per + per) in registers. Four 8-bit radix passes find the topk-th largest key:
# each counts the candidates' digits with one atomic add per (simdgroup, distinct digit), a
# capped loop retiring the lanes that share the lowest pending lane's digit each round, and
# simdgroup 0 finds the digit by a prefix sum over the bins, highest first. The blocks above
# that key, then its ties lowest index first, are written in index order.
TOPK_SOURCE = """
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
"""

TOPK_KERNEL = metal_kernel(
    name="qsa_topk",
    input_names=["scores", "seq_lens", "num_blocks_in"],
    output_names=["selected"],
    source=TOPK_SOURCE,
)

# One thread per table entry: a request's selected blocks' token slots ascending, then its
# incomplete tail block's; entries past its count hold its last key's slot (never read).
TABLE_SOURCE = """
    const uint entry = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const uint topk = selected_shape[1];
    const uint width = (topk + 1) * BLOCK_SIZE;
    if (entry >= width) {
      return;
    }
    const uint seq_len = uint(seq_lens[row]);
    const uint complete = seq_len / BLOCK_SIZE;
    const uint num_selected = metal::min(complete, topk) * BLOCK_SIZE;
    uint position;
    if (entry < num_selected) {
      position = uint(selected[row * topk + entry / BLOCK_SIZE]) * BLOCK_SIZE + entry % BLOCK_SIZE;
    } else {
      const uint last = metal::max(seq_len, 1u) - 1;
      position = metal::min(complete * BLOCK_SIZE + entry - num_selected, last);
    }
    const uint pages = page_table_shape[1];
    key_slots[row * width + entry] =
        int(uint(page_table[row * pages + position / page_size]) * page_size +
            position % page_size);
    if (entry == 0) {
      counts[row] = int(num_selected + seq_len - complete * BLOCK_SIZE);
    }
"""

TABLE_KERNEL = metal_kernel(
    name="qsa_key_table",
    input_names=["selected", "page_table", "seq_lens", "page_size"],
    output_names=["key_slots", "counts"],
    source=TABLE_SOURCE,
)

# One thread per (token, column): the window's keys summed in order in fp32, times 1 / WINDOW,
# rounded to the keys' dtype.
WINDOW_MEAN_SOURCE = """
    const uint column = thread_position_in_grid.x;
    const uint token = thread_position_in_grid.y;
    if (column >= D) {
      return;
    }
    float total = 0.0f;
    for (uint i = 0; i < WINDOW; ++i) {
      const size_t slot = size_t(window_slots[token * WINDOW + i]);
      total += static_cast<float>(index_keys[slot * D + column]);
    }
    means[token * D + column] = static_cast<T>(total * (1.0f / float(WINDOW)));
"""

WINDOW_MEAN_KERNEL = metal_kernel(
    name="qsa_window_mean",
    input_names=["index_keys", "window_slots"],
    output_names=["means"],
    source=WINDOW_MEAN_SOURCE,
)


def qsa_block_scores(
    queries: mx.array,
    index_keys: mx.array,
    page_table: mx.array,
    seq_lens: mx.array,
    page_size: int,
    block_size: int,
    num_blocks: int,
) -> mx.array:
    """Block scores fp32 [bs, num_blocks] of each request's index queries [bs, heads, D].

    Block j of request b scores sum_h relu(queries[b, h] . k_j) / sqrt(D) in fp32 while it is
    complete (j < seq_lens[b] // block_size), k_j read from index_keys [pool_tokens, D] at the
    slot of the block's last token through page_table int32 [bs, width]; -inf past that.
    """
    bs, heads, head_dim = queries.shape
    assert head_dim % 32 == 0 and index_keys.shape[1] == head_dim
    blocks_per_threadgroup = SCORE_SIMDGROUPS * BLOCKS_PER_SIMDGROUP
    num_threadgroups = -(-num_blocks // blocks_per_threadgroup)
    (scores,) = SCORES_KERNEL(
        inputs=[queries, index_keys, page_table, seq_lens, page_size, num_blocks, head_dim**-0.5],
        template=[
            ("D", head_dim),
            ("HEADS", heads),
            ("BLOCK_SIZE", block_size),
            ("SIMDGROUPS", SCORE_SIMDGROUPS),
            ("BLOCKS_PER_SIMDGROUP", BLOCKS_PER_SIMDGROUP),
        ],
        grid=(num_threadgroups * SCORE_SIMDGROUPS * 32, bs, 1),
        threadgroup=(SCORE_SIMDGROUPS * 32, 1, 1),
        output_shapes=[(bs, num_blocks)],
        output_dtypes=[mx.float32],
    )
    return scores


def qsa_topk(scores: mx.array, seq_lens: mx.array, topk: int, block_size: int) -> mx.array:
    """The topk highest-scoring complete blocks int32 [bs, topk] of scores fp32 [bs, blocks].

    Ties go to the lower block index and the blocks come out ascending. A request with at
    most topk complete blocks (seq_lens[b] // block_size) gets all of them, then zeros.
    """
    assert topk >= 1
    bs, num_blocks = scores.shape
    (selected,) = TOPK_KERNEL(
        inputs=[scores, seq_lens, num_blocks],
        template=[
            ("TOPK", topk),
            ("BLOCK_SIZE", block_size),
            ("THREADS", TOPK_THREADS),
            ("PER_THREAD", max(1, -(-num_blocks // TOPK_THREADS))),
        ],
        grid=(TOPK_THREADS, bs, 1),
        threadgroup=(TOPK_THREADS, 1, 1),
        output_shapes=[(bs, topk)],
        output_dtypes=[mx.int32],
    )
    return selected


def qsa_key_table(
    selected: mx.array, page_table: mx.array, seq_lens: mx.array, page_size: int, block_size: int
) -> tuple[mx.array, mx.array]:
    """The token slots of the keys each request attends, as flash_decode takes them.

    selected int32 [bs, topk] is qsa_topk's output. Returns (key_slots int32 [bs, (topk + 1)
    * block_size], counts int32 [bs]): request b's first counts[b] entries are the slots, read
    through page_table int32 [bs, width], of its first min(complete, topk) selected blocks'
    tokens and then of its incomplete tail block's, ascending by position.
    """
    bs, topk = selected.shape
    width = (topk + 1) * block_size
    key_slots, counts = TABLE_KERNEL(
        inputs=[selected, page_table, seq_lens, page_size],
        template=[("BLOCK_SIZE", block_size)],
        grid=(width, bs, 1),
        threadgroup=(min(256, width), 1, 1),
        output_shapes=[(bs, width), (bs,)],
        output_dtypes=[mx.int32, mx.int32],
    )
    return key_slots, counts


def qsa_window_mean(index_keys: mx.array, window_slots: mx.array) -> mx.array:
    """mx.mean(index_keys[window_slots].astype(mx.float32), axis=1).astype(index_keys.dtype)
    for index_keys [pool_tokens, D] and window_slots int32 [T, window] -> [T, D], without
    the gathered [T, window, D] copy."""
    num_tokens, window = window_slots.shape
    head_dim = index_keys.shape[1]
    (means,) = WINDOW_MEAN_KERNEL(
        inputs=[index_keys, window_slots],
        template=[("T", index_keys.dtype), ("D", head_dim), ("WINDOW", window)],
        grid=(head_dim, num_tokens, 1),
        threadgroup=(min(head_dim, 256), 1, 1),
        output_shapes=[(num_tokens, head_dim)],
        output_dtypes=[index_keys.dtype],
    )
    return means


__all__ = ["qsa_block_scores", "qsa_key_table", "qsa_topk", "qsa_window_mean"]
