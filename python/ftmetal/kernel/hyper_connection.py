"""Fused hyper-connection kernels: the grouped (1 + w) RMS norm of the residual streams, with
the pending branch injected first; the sigmoid-weighted mean of the normed streams (fused with
its one-token up projection in kernel/hyper_up_mix.py); the (1 + w) norm of attention heads
with a partial NeoX rope.

Port of FreeToken kernel/triton/hc.py (grouped_gemma_rmsnorm, hc_combine_norm, hc_gate_mix) on
mx.fast.metal_kernel; intermediates stay in fp32 and round once, at the store. Divergences: the
norm reduces as mx.fast.rms_norm's single-row kernel does and scales by (1 + w) as one product,
so it equals the fp32 op chain bit for bit; the combine runs only fused into the norm; the head
norm can fuse mx.fast.rope's partial rotation.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from ftmetal.kernel.custom import metal_kernel

SIMD_WIDTH = 32
# mx.fast.rms_norm's single-row kernel: 4 values a thread, up to 4096 values a row.
NORM_READS_PER_THREAD = 4
MAX_NORM_SIZE = 4096
MIX_THREADGROUP = 256

NORM_HEADER = """
// MLX's sigmoid op, in fp32.
inline float precise_sigmoid(float x) {
  const float e = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? e : 1 - e;
}

// mx.fast.rms_norm's single-row reduction: 1 / rms of a row from each thread's sum of squares.
inline float row_inv_rms(
    float sum_squares,
    float size,
    float eps,
    threadgroup float* local_sums,
    threadgroup float* local_inv_mean,
    uint simd_lane,
    uint simd_group) {
  sum_squares = simd_sum(sum_squares);
  if (simd_group == 0) {
    local_sums[simd_lane] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) {
    local_sums[simd_group] = sum_squares;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    sum_squares = simd_sum(local_sums[simd_lane]);
    if (simd_lane == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(sum_squares / size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return local_inv_mean[0];
}
"""

# One threadgroup per (row, group) slice of SIZE values: with INJECT, the slice plus
# branch * 2 * sigmoid(logit / GROUPS) is stored first; the slice is normed and scaled by
# (1 + w); with ROTARY_DIM, its first ROTARY_DIM values are then rotated as mx.fast.rope does.
# The inject (and the mix below) turn fma contraction off: the op chain multiplies and adds in
# separate ops, while the norm's sum of squares and the rotation contract as MLX's kernels do.
NORM_SOURCE = """
    constexpr int READS = 4;
    const uint row = threadgroup_position_in_grid.y;
    const uint group = threadgroup_position_in_grid.x;
    const uint local_index = thread_position_in_threadgroup.x;
    const uint first = local_index * READS;
    const size_t base = (size_t(row) * GROUPS + group) * SIZE;
    threadgroup float local_sums[32];
    threadgroup float local_inv_mean[1];
    threadgroup T rotary_values[ROTARY_DIM > 0 ? ROTARY_DIM : 1];
    float gate = 0;
    if (INJECT) {
      const float logit = static_cast<float>(inject_logits[row * GROUPS + group]);
      gate = 2 * precise_sigmoid(logit / static_cast<float>(GROUPS));
    }
    float values[READS];
    float sum_squares = 0;
    for (int i = 0; i < READS; ++i) {
      const uint h = first + i;
      float value = 0;
      if (h < SIZE) {
        if (INJECT) {
          #pragma METAL fp contract(off)
          const T injected = static_cast<T>(
              static_cast<float>(hyper[base + h]) +
              static_cast<float>(branch[size_t(row) * SIZE + h]) * gate);
          hyper_out[base + h] = injected;
          value = static_cast<float>(injected);
        } else {
          value = static_cast<float>(hyper[base + h]);
        }
      }
      values[i] = value;
      sum_squares += value * value;
    }
    const float inv_rms = row_inv_rms(
        sum_squares, static_cast<float>(SIZE), eps, local_sums, local_inv_mean,
        thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
    for (int i = 0; i < READS; ++i) {
      const uint h = first + i;
      if (h < SIZE) {
        const float scale = 1 + static_cast<float>(weight[group * SIZE + h]);
        const T value = static_cast<T>(values[i] * inv_rms * scale);
        if (h < ROTARY_DIM) {
          rotary_values[h] = value;
        } else {
          normed[base + h] = value;
        }
      }
    }
    if (ROTARY_DIM > 0) {
      constexpr uint HALF = ROTARY_DIM / 2;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float position = static_cast<float>(positions[row / ROWS_PER_POSITION]);
      for (uint i = local_index; i < HALF; i += THREADS) {
        const float d = static_cast<float>(i) / static_cast<float>(HALF);
        const float theta = position * metal::exp2(-d * log2_base);
        const float cos_theta = metal::fast::cos(theta);
        const float sin_theta = metal::fast::sin(theta);
        const float x1 = static_cast<float>(rotary_values[i]);
        const float x2 = static_cast<float>(rotary_values[i + HALF]);
        normed[base + i] = static_cast<T>(x1 * cos_theta - x2 * sin_theta);
        normed[base + i + HALF] = static_cast<T>(x1 * sin_theta + x2 * cos_theta);
      }
    }
"""

NORM_KERNEL = metal_kernel(
    name="hyper_norm",
    input_names=["hyper", "branch", "inject_logits", "weight", "eps", "positions", "log2_base"],
    output_names=["normed", "hyper_out"],
    source=NORM_SOURCE,
    header=NORM_HEADER,
)

# One thread per (token, h): the mean over the streams of sigmoid(up) * normed.
MIX_SOURCE = """
    const uint element = thread_position_in_grid.x;
    const uint token = element / HIDDEN;
    const uint h = element % HIDDEN;
    float total = 0;
    for (int stream = 0; stream < HC_COUNT; ++stream) {
      #pragma METAL fp contract(off)
      const size_t index = (size_t(token) * HC_COUNT + stream) * HIDDEN + h;
      total += precise_sigmoid(static_cast<float>(up[index])) *
          static_cast<float>(normed[index]);
    }
    mixed[element] = static_cast<T>(total / static_cast<float>(HC_COUNT));
"""

MIX_KERNEL = metal_kernel(
    name="hyper_mix",
    input_names=["up", "normed"],
    output_names=["mixed"],
    source=MIX_SOURCE,
    header=NORM_HEADER,
)


def norm_threadgroup(size: int) -> int:
    """Threads per normed row, as mx.fast.rms_norm launches them: whole simdgroups of 4 values."""
    assert size <= MAX_NORM_SIZE, size
    reads = -(-size // NORM_READS_PER_THREAD)
    return -(-reads // SIMD_WIDTH) * SIMD_WIDTH


def hyper_norm(
    hyper: mx.array,
    weight: mx.array,
    eps: float,
    hc_count: int,
    branch: mx.array | None = None,
    inject_logits: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """(normed streams, streams) of hyper [T, hc_count * H], each H slice normed apart with its
    slice of weight [hc_count * H]. With branch [T, H] and the raw inject_logits [T, hc_count],
    stream s first becomes hyper_s + branch * 2 * sigmoid(logit_s / hc_count), rounded once."""
    assert (branch is None) == (inject_logits is None)
    num_tokens, width = hyper.shape
    hidden = width // hc_count
    assert hidden * hc_count == width and weight.size == width, (hyper.shape, weight.shape)
    threads = norm_threadgroup(hidden)
    is_injecting = branch is not None and inject_logits is not None
    # Inputs the kernel does not read stand in as hyper: the branch and logits without inject,
    # the positions and log2(base) without rotation.
    normed, hyper_out = NORM_KERNEL(
        inputs=[
            hyper,
            branch if branch is not None else hyper,
            inject_logits if inject_logits is not None else hyper,
            weight,
            eps,
            hyper,
            0.0,
        ],
        template=[
            ("T", hyper.dtype),
            ("SIZE", hidden),
            ("GROUPS", hc_count),
            ("THREADS", threads),
            ("INJECT", is_injecting),
            ("ROTARY_DIM", 0),
            ("ROWS_PER_POSITION", 1),
        ],
        grid=(hc_count * threads, num_tokens, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[hyper.shape, hyper.shape if is_injecting else (1,)],
        output_dtypes=[hyper.dtype, hyper.dtype],
    )
    return normed, hyper_out if is_injecting else hyper


def offset_rms_norm(
    x: mx.array, weight: mx.array, eps: float, group_size: int | None = None
) -> mx.array:
    """rmsnorm(x) * (1 + weight) in fp32 over x's last axis, which weight spans; with group_size,
    each group_size slice of it normed apart."""
    size = x.shape[-1]
    assert group_size is None or size % group_size == 0, (x.shape, group_size)
    groups = 1 if group_size is None else size // group_size
    normed, _ = hyper_norm(x.reshape(-1, size), weight, eps, groups)
    return normed.reshape(x.shape)


def supports_offset_rms_norm_rope(x: mx.array, rotary_dim: int) -> bool:
    """Heads x [T, heads, D] whose D one threadgroup of offset_rms_norm_rope holds."""
    return x.ndim == 3 and 0 < rotary_dim <= x.shape[-1] <= MAX_NORM_SIZE and rotary_dim % 2 == 0


def offset_rms_norm_rope(
    x: mx.array, weight: mx.array, eps: float, positions: mx.array, rotary_dim: int, base: float
) -> mx.array:
    """offset_rms_norm of each head of x [T, heads, D] (weight [D]), then mx.fast.rope's
    non-traditional rotation of its first rotary_dim dims at positions int32 [T], bit for bit."""
    assert supports_offset_rms_norm_rope(x, rotary_dim), (x.shape, rotary_dim)
    num_tokens, num_heads, head_dim = x.shape
    rows = x.reshape(num_tokens * num_heads, head_dim)
    threads = norm_threadgroup(head_dim)
    # mx.fast.rope keeps its base as float32 and passes log2 of it as float32.
    log2_base = math.log2(np.float32(base))
    normed, _ = NORM_KERNEL(
        inputs=[rows, rows, rows, weight, eps, positions, log2_base],
        template=[
            ("T", x.dtype),
            ("SIZE", head_dim),
            ("GROUPS", 1),
            ("THREADS", threads),
            ("INJECT", False),
            ("ROTARY_DIM", rotary_dim),
            ("ROWS_PER_POSITION", num_heads),
        ],
        grid=(threads, num_tokens * num_heads, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[rows.shape, (1,)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return normed.reshape(x.shape)


def hyper_mix(up: mx.array, normed: mx.array, hc_count: int) -> mx.array:
    """The mean over the streams of sigmoid(up) * normed in fp32, both [T, hc_count * H] ->
    [T, H]."""
    num_tokens, width = normed.shape
    hidden = width // hc_count
    work = num_tokens * hidden
    (mixed,) = MIX_KERNEL(
        inputs=[up, normed],
        template=[("T", normed.dtype), ("HIDDEN", hidden), ("HC_COUNT", hc_count)],
        grid=(work, 1, 1),
        threadgroup=(min(MIX_THREADGROUP, work), 1, 1),
        output_shapes=[(num_tokens, hidden)],
        output_dtypes=[normed.dtype],
    )
    return mixed


__all__ = [
    "MAX_NORM_SIZE",
    "NORM_HEADER",
    "SIMD_WIDTH",
    "hyper_mix",
    "hyper_norm",
    "norm_threadgroup",
    "offset_rms_norm",
    "offset_rms_norm_rope",
    "supports_offset_rms_norm_rope",
]
