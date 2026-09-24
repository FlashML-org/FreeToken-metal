"""Paged flash-decode (fdecode v4): one query token per request over the paged K/V pool.

Each threadgroup streams one kv head's keys once for all G = Hq / Hkv query heads
(GQA packing); split-KV partitions fill the GPU at small batch; pass 1 writes
per-partition online-softmax partials and pass 2 merges them by log-sum-exp.
Divergences from the fp16 original: the dtype is a template argument (bf16, fp16,
fp32), pool rows are token slots page_id * page_size + offset, and the page-table
width, partition count and scale are runtime values, so a growing context never
recompiles.
"""

from __future__ import annotations

import math

import mlx.core as mx

SIMD_WIDTH = 32
SIMDGROUPS_PER_THREADGROUP = 4
# Apple GPUs cap threadgroup memory at 32 KiB; keep a margin for the compiler.
THREADGROUP_MEMORY_BYTES = 30 * 1024
# Partitioning tuned for fdecode v4 on the M4 Pro (20 GPU cores): about 1k keys per
# partition, at least 256, and at least two waves of threadgroups at small batch.
PARTITION_TOKENS = 1024
MIN_PARTITION_TOKENS = 256
MAX_PARTITIONS = 64
MIN_THREADGROUPS = 40
# From this head dim up 16-bit inputs accumulate in half: fp32 accumulators cost occupancy
# and ran 2x slower at D=256 on the M4 Pro.
HALF_ACCUMULATOR_HEAD_DIM = 256

# Grid (32 * NSG * P, HKV, B). Each simdgroup takes a contiguous slice of the partition's
# keys and keeps an online softmax per query head whose accumulator is the running
# weighted mean of V, so a half accumulator is bounded by max |v| and cannot overflow.
# The NSG slices merge through threadgroup memory and simdgroup 0 writes the partition's
# unnormalized [G, D] accumulator and its (max, sum) in the log2 domain.
PARTIALS_SOURCE = """
    constexpr int LPD = D / 32;
    constexpr int VW = LPD % 4 == 0 ? 4 : 2;
    constexpr int NV = LPD / VW;
    using InVec = vec<T, VW>;
    using FloatVec = vec<float, VW>;
    using AccVec = vec<AccT, VW>;

    const uint partition = threadgroup_position_in_grid.x;
    const uint kv_head = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const uint num_partitions = threadgroups_per_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint max_pages = page_table_shape[1];
    const size_t part_row = ((size_t)b * HKV + kv_head) * num_partitions + partition;

    const uint seq_len = seq_lens[b];
    const uint per_part = (seq_len + num_partitions - 1) / num_partitions;
    const uint t0 = partition * per_part;
    const uint t1 = min(seq_len, t0 + per_part);
    if (t0 >= t1) {
        if (sg == 0 && lane == 0) {
            for (int g = 0; g < G; ++g) {
                part_ml[(part_row * G + g) * 2] = -INFINITY;
                part_ml[(part_row * G + g) * 2 + 1] = 0.0f;
            }
        }
        return;
    }
    const uint per_sg = (t1 - t0 + NSG - 1) / NSG;
    const uint s0 = t0 + sg * per_sg;
    const uint s1 = min(t1, s0 + per_sg);

    threadgroup T q_tg[G * D];
    for (uint i = sg * 32 + lane; i < G * D; i += NSG * 32) {
        q_tg[i] = q[((size_t)b * HKV + kv_head) * G * D + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_g[G], l_g[G];
    AccVec acc[G][NV];
    for (int g = 0; g < G; ++g) {
        m_g[g] = -INFINITY;
        l_g[g] = 0.0f;
        for (int d = 0; d < NV; ++d) acc[g][d] = AccVec(0);
    }

    const uint lane_off = lane * LPD;
    for (uint t = s0; t < s1; ++t) {
        const uint page = page_table[(size_t)b * max_pages + t / BS];
        const size_t base = (((size_t)page * BS + t % BS) * HKV + kv_head) * D + lane_off;
        const device InVec* k_vec = (const device InVec*)(k_cache + base);
        FloatVec kv[NV];
        for (int d = 0; d < NV; ++d) kv[d] = FloatVec(k_vec[d]);
        float w[G];
        for (int g = 0; g < G; ++g) {
            const threadgroup InVec* q_vec = (const threadgroup InVec*)(q_tg + g * D + lane_off);
            FloatVec dot = kv[0] * FloatVec(q_vec[0]);
            for (int d = 1; d < NV; ++d) dot += kv[d] * FloatVec(q_vec[d]);
            float partial = 0.0f;
            for (int j = 0; j < VW; ++j) partial += dot[j];
            w[g] = simd_sum(partial) * scale_log2e;
        }
        const device InVec* v_vec = (const device InVec*)(v_cache + base);
        for (int d = 0; d < NV; ++d) kv[d] = FloatVec(v_vec[d]);
        for (int g = 0; g < G; ++g) {
            const float m_new = max(m_g[g], w[g]);
            const float corr = fast::exp2(m_g[g] - m_new);
            const float p = fast::exp2(w[g] - m_new);
            const float l_new = l_g[g] * corr + p;
            const float inv_l = fast::divide(1.0f, l_new);
            const AccT keep = (AccT)(l_g[g] * corr * inv_l);
            const AccT take = (AccT)(p * inv_l);
            for (int d = 0; d < NV; ++d)
                acc[g][d] = acc[g][d] * keep + take * AccVec(kv[d]);
            l_g[g] = l_new;
            m_g[g] = m_new;
        }
    }

    threadgroup float tg_ml[NSG * G * 2];
    threadgroup StageT tg_acc[NSG * G * D];
    for (int g = 0; g < G; ++g) {
        if (lane == 0) {
            tg_ml[(sg * G + g) * 2] = m_g[g];
            tg_ml[(sg * G + g) * 2 + 1] = l_g[g];
        }
        for (int d = 0; d < NV; ++d)
            for (int j = 0; j < VW; ++j)
                tg_acc[(sg * G + g) * D + lane_off + d * VW + j] = (StageT)acc[g][d][j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        for (int g = 0; g < G; ++g) {
            float m_max = -INFINITY;
            for (int s = 0; s < NSG; ++s) m_max = max(m_max, tg_ml[(s * G + g) * 2]);
            float l_sum = 0.0f, weight[NSG];
            for (int s = 0; s < NSG; ++s) {
                const float m_s = tg_ml[(s * G + g) * 2];
                weight[s] = (m_s == -INFINITY)
                    ? 0.0f : fast::exp2(m_s - m_max) * tg_ml[(s * G + g) * 2 + 1];
                l_sum += weight[s];
            }
            const size_t out_off = (part_row * G + g) * D + lane_off;
            for (int i = 0; i < LPD; ++i) {
                float o = 0.0f;
                for (int s = 0; s < NSG; ++s)
                    o += (float)tg_acc[(s * G + g) * D + lane_off + i] * weight[s];
                part_acc[out_off + i] = o;
            }
            if (lane == 0) {
                part_ml[(part_row * G + g) * 2] = m_max;
                part_ml[(part_row * G + g) * 2 + 1] = l_sum;
            }
        }
    }
"""

# Grid (32, HKV, B): one simdgroup per (request, kv head) merges the P partials per head.
MERGE_SOURCE = """
    constexpr int LPD = D / 32;
    const uint kv_head = threadgroup_position_in_grid.y;
    const uint b = threadgroup_position_in_grid.z;
    const uint lane = thread_index_in_simdgroup;
    const uint num_partitions = part_ml_shape[2];
    const uint lane_off = lane * LPD;
    const size_t head_row = (size_t)b * HKV + kv_head;

    for (int g = 0; g < G; ++g) {
        float m_max = -INFINITY;
        for (uint p = 0; p < num_partitions; ++p)
            m_max = max(m_max, part_ml[((head_row * num_partitions + p) * G + g) * 2]);
        float l_sum = 0.0f;
        float o[LPD];
        for (int i = 0; i < LPD; ++i) o[i] = 0.0f;
        for (uint p = 0; p < num_partitions; ++p) {
            const size_t ml_off = ((head_row * num_partitions + p) * G + g) * 2;
            const float m_p = part_ml[ml_off];
            if (m_p == -INFINITY) continue;
            const float weight = fast::exp2(m_p - m_max);
            l_sum += part_ml[ml_off + 1] * weight;
            const size_t acc_off = ((head_row * num_partitions + p) * G + g) * D + lane_off;
            for (int i = 0; i < LPD; ++i) o[i] += part_acc[acc_off + i] * weight;
        }
        const float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        const size_t out_off = (head_row * G + g) * D + lane_off;
        for (int i = 0; i < LPD; ++i) out[out_off + i] = static_cast<T>(o[i] * inv);
    }
"""

PARTIALS_KERNEL = mx.fast.metal_kernel(
    name="fdecode_partials",
    input_names=["q", "k_cache", "v_cache", "page_table", "seq_lens", "scale_log2e"],
    output_names=["part_acc", "part_ml"],
    source=PARTIALS_SOURCE,
)
MERGE_KERNEL = mx.fast.metal_kernel(
    name="fdecode_merge",
    input_names=["part_acc", "part_ml"],
    output_names=["out"],
    source=MERGE_SOURCE,
)


def flash_decode(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    seq_lens: mx.array,
    max_seqlen_k: int,
    page_size: int,
    scale: float,
) -> mx.array:
    """Attention of one query per request over its first seq_lens[b] paged keys.

    q [B, Hq, D]; k_cache / v_cache [num_pages * page_size, Hkv, D] in q's dtype;
    page_table [B, max_pages] int32 page ids; seq_lens [B] int32, each >= 1.
    max_seqlen_k bounds seq_lens and only sizes the partitioning. Returns [B, Hq, D].
    """
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    assert num_heads % num_kv_heads == 0
    assert head_dim % (2 * SIMD_WIDTH) == 0, "each lane needs at least a 2-wide vector"
    assert page_size & (page_size - 1) == 0, "page size must be a power of two"
    assert k_cache.dtype == v_cache.dtype == q.dtype
    assert page_table.dtype == seq_lens.dtype == mx.int32
    group = num_heads // num_kv_heads
    # 16-bit inputs stage the cross-simdgroup merge in half; fp32 inputs stay fp32.
    stage_dtype = mx.float32 if q.dtype == mx.float32 else mx.float16
    acc_dtype = stage_dtype if head_dim >= HALF_ACCUMULATOR_HEAD_DIM else mx.float32
    # Threadgroup memory: per simdgroup the staged [G, D] accumulator and (max, sum) pairs,
    # plus the [G, D] query block.
    simdgroup_bytes = group * (head_dim * stage_dtype.size + 8)
    query_bytes = group * head_dim * q.dtype.size
    num_simdgroups = SIMDGROUPS_PER_THREADGROUP
    while num_simdgroups > 1 and (
        num_simdgroups * simdgroup_bytes + query_bytes > THREADGROUP_MEMORY_BYTES
    ):
        num_simdgroups //= 2
    num_partitions = max(1, min(MAX_PARTITIONS, -(-max_seqlen_k // PARTITION_TOKENS)))
    while batch_size * num_kv_heads * num_partitions < MIN_THREADGROUPS and num_partitions < -(
        -max_seqlen_k // MIN_PARTITION_TOKENS
    ):
        num_partitions *= 2
    part_acc, part_ml = PARTIALS_KERNEL(
        inputs=[q, k_cache, v_cache, page_table, seq_lens, scale * math.log2(math.e)],
        template=[
            ("T", q.dtype),
            ("AccT", acc_dtype),
            ("StageT", stage_dtype),
            ("D", head_dim),
            ("G", group),
            ("HKV", num_kv_heads),
            ("BS", page_size),
            ("NSG", num_simdgroups),
        ],
        grid=(SIMD_WIDTH * num_simdgroups * num_partitions, num_kv_heads, batch_size),
        threadgroup=(SIMD_WIDTH * num_simdgroups, 1, 1),
        output_shapes=[
            (batch_size, num_kv_heads, num_partitions, group, head_dim),
            (batch_size, num_kv_heads, num_partitions, group, 2),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    (out,) = MERGE_KERNEL(
        inputs=[part_acc, part_ml],
        template=[("T", q.dtype), ("D", head_dim), ("G", group), ("HKV", num_kv_heads)],
        grid=(SIMD_WIDTH, num_kv_heads, batch_size),
        threadgroup=(SIMD_WIDTH, 1, 1),
        output_shapes=[(batch_size, num_heads, head_dim)],
        output_dtypes=[q.dtype],
    )
    return out


__all__ = ["flash_decode"]
