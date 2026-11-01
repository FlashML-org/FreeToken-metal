"""The hyper-connection's one-token up projection fused with its mix: the 4-bit affine
GEMV of the activated down projection, then the sigmoid-weighted mean of the normed streams
(kernel/hyper_connection.py's hyper_mix), in one launch.

Port of FreeToken kernel/triton/hc.py's hc_gate_mix with its up GEMM folded in. Divergences:
the GEMV sums as mx.quantized_matmul's qmv does for one token, so the result equals the
GEMV then hyper_mix bit for bit.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel.custom import metal_kernel
from ftmetal.kernel.hyper_connection import NORM_HEADER, SIMD_WIDTH
from ftmetal.kernel.qmv_rows import QMV_FAST_INPUT_STEP, QUANTIZED_HEADER

UP_MIX_SIMDGROUPS_PER_THREADGROUP = 2
UP_MIX_BITS = 4
# mx.quantized_matmul runs one token through qmv_quad at these input sizes, qmv_fast at multiples
# of its input step and qmv otherwise; hyper_up_mix computes qmv's sums.
QMV_QUAD_INPUT_SIZES = (64, 128)

# One simdgroup per hidden index h of a token: the up projection's rows stream * HIDDEN + h,
# summed as MLX's qmv does for one token (4 bits, 8 values a lane, the last part-block through
# the safe loads) and stored in T, then MIX_SOURCE's mean. The last part-block's REMAINDER values
# fill REMAINDER_LANES lanes (input 320: 64 values, lanes 0-7); when SPLIT lane groups of that
# width divide the streams, group g computes its rows on the values of lanes
# 0..REMAINDER_LANES - 1 and shuffles them back there: the same products, added in the same order.
UP_MIX_SOURCE = """
    constexpr int pack_factor = get_pack_factor<4, 32>();
    constexpr int bytes_per_pack = get_bytes_per_pack<4, 32>();
    constexpr int VALUES = pack_factor;
    constexpr int BLOCK = VALUES * SIMD_SIZE;
    constexpr int scale_step_per_thread = GROUP_SIZE / VALUES;
    constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
    constexpr int in_vec_size_g = K / GROUP_SIZE;
    const uint lane = thread_index_in_simdgroup;
    const uint h = threadgroup_position_in_grid.x * SIMDGROUPS + simdgroup_index_in_threadgroup;
    const uint token = threadgroup_position_in_grid.y;
    if (h >= HIDDEN) {
      return;
    }
    const device uint8_t* ws[HC_COUNT];
    const device T* sl[HC_COUNT];
    const device T* bl[HC_COUNT];
    for (int stream = 0; stream < HC_COUNT; ++stream) {
      const uint out_row = stream * HIDDEN + h;
      ws[stream] = (const device uint8_t*)weight + out_row * in_vec_size_w + lane * bytes_per_pack;
      sl[stream] = scales + out_row * in_vec_size_g + lane / scale_step_per_thread;
      bl[stream] = biases + out_row * in_vec_size_g + lane / scale_step_per_thread;
    }
    const device T* xp = x + token * K + lane * VALUES;
    float x_thread[VALUES];
    float result[HC_COUNT] = {0};
    int k = 0;
    for (; k < K - BLOCK; k += BLOCK) {
      const float sum = load_vector<T, float, VALUES, 4>(xp, x_thread);
      for (int stream = 0; stream < HC_COUNT; ++stream) {
        result[stream] += qdot<float, VALUES, 4>(
            ws[stream], x_thread, sl[stream][0], bl[stream][0], sum);
        ws[stream] += BLOCK * bytes_per_pack / pack_factor;
        sl[stream] += BLOCK / GROUP_SIZE;
        bl[stream] += BLOCK / GROUP_SIZE;
      }
      xp += BLOCK;
    }
    constexpr int REMAINDER = K - (K - 1) / BLOCK * BLOCK;
    constexpr int REMAINDER_LANES = REMAINDER / VALUES;
    constexpr int SPLIT =
        REMAINDER % VALUES == 0 && SIMD_SIZE % REMAINDER_LANES == 0
        ? SIMD_SIZE / REMAINDER_LANES : 1;
    constexpr bool SPLIT_ROWS = SPLIT > 1 && HC_COUNT % SPLIT == 0;
    if (SPLIT_ROWS) {
      constexpr int GROUP_ROWS = SPLIT_ROWS ? HC_COUNT / SPLIT : 1;
      const int lane_group = lane / REMAINDER_LANES;
      const int lane_shift = lane_group * REMAINDER_LANES;
      const int group_shift = lane_shift / scale_step_per_thread;
      const float sum = load_vector_safe<T, float, VALUES, 4>(
          xp - lane_shift * VALUES, x_thread, VALUES);
      float part[GROUP_ROWS];
      for (int j = 0; j < GROUP_ROWS; ++j) {
        const int stream = lane_group * GROUP_ROWS + j;
        part[j] = qdot_safe<float, VALUES, 4>(
            ws[stream] - lane_shift * bytes_per_pack, x_thread, sl[stream][-group_shift],
            bl[stream][-group_shift], sum, VALUES);
      }
      for (int source = 0; source < SPLIT; ++source) {
        for (int j = 0; j < GROUP_ROWS; ++j) {
          const float value = simd_shuffle_down(part[j], ushort(source * REMAINDER_LANES));
          if (lane < uint(REMAINDER_LANES)) {
            result[source * GROUP_ROWS + j] += value;
          }
        }
      }
    } else {
      const int remaining = clamp(int(K - k - lane * VALUES), 0, VALUES);
      if (remaining > 0) {
        const float sum = load_vector_safe<T, float, VALUES, 4>(xp, x_thread, remaining);
        for (int stream = 0; stream < HC_COUNT; ++stream) {
          result[stream] += qdot_safe<float, VALUES, 4>(
              ws[stream], x_thread, sl[stream][0], bl[stream][0], sum, remaining);
        }
      }
    }
    for (int stream = 0; stream < HC_COUNT; ++stream) {
      result[stream] = simd_sum(result[stream]);
    }
    if (lane == 0) {
      float total = 0;
      for (int stream = 0; stream < HC_COUNT; ++stream) {
        #pragma METAL fp contract(off)
        const float up = static_cast<float>(static_cast<T>(result[stream]));
        const size_t index = (size_t(token) * HC_COUNT + stream) * HIDDEN + h;
        total += precise_sigmoid(up) * static_cast<float>(normed[index]);
      }
      mixed[size_t(token) * HIDDEN + h] = static_cast<T>(total / static_cast<float>(HC_COUNT));
    }
"""

UP_MIX_KERNEL = metal_kernel(
    name="hyper_up_mix",
    input_names=["x", "weight", "scales", "biases", "normed"],
    output_names=["mixed"],
    source=UP_MIX_SOURCE,
    header=QUANTIZED_HEADER + NORM_HEADER,
)


def supports_hyper_up_mix(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    normed: mx.array,
    hc_count: int,
    group_size: int,
    bits: int,
) -> bool:
    """Whether hyper_up_mix takes one token x [1, K] through the 4-bit affine up projection
    weight [hc_count * H, K / 8]: the shapes mx.quantized_matmul runs as qmv, scales in x's and
    normed's dtype."""
    input_size = x.shape[-1]
    return (
        bits == UP_MIX_BITS
        and x.ndim == 2
        and x.shape[0] == 1
        and normed.shape[0] == 1
        and input_size not in QMV_QUAD_INPUT_SIZES
        and input_size % QMV_FAST_INPUT_STEP[UP_MIX_BITS] != 0
        and weight.shape == (normed.shape[1], input_size * bits // 32)
        and normed.shape[1] % hc_count == 0
        and scales.shape == (normed.shape[1], input_size // group_size)
        and scales.dtype == x.dtype == normed.dtype
    )


def hyper_up_mix(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    normed: mx.array,
    hc_count: int,
    *,
    group_size: int,
    bits: int,
) -> mx.array:
    """hyper_mix(mx.quantized_matmul(x, weight, scales, biases, transpose=True), normed) for one
    token in one launch (supports_hyper_up_mix), bit for bit: [1, H]."""
    assert supports_hyper_up_mix(x, weight, scales, normed, hc_count, group_size, bits), (
        x.shape,
        weight.shape,
        bits,
    )
    num_tokens, input_size = x.shape
    hidden = normed.shape[1] // hc_count
    (mixed,) = UP_MIX_KERNEL(
        inputs=[x, weight, scales, biases, normed],
        template=[
            ("T", normed.dtype),
            ("K", input_size),
            ("HIDDEN", hidden),
            ("HC_COUNT", hc_count),
            ("GROUP_SIZE", group_size),
            ("SIMDGROUPS", UP_MIX_SIMDGROUPS_PER_THREADGROUP),
        ],
        grid=(hidden * SIMD_WIDTH, num_tokens, 1),
        threadgroup=(UP_MIX_SIMDGROUPS_PER_THREADGROUP * SIMD_WIDTH, 1, 1),
        output_shapes=[(num_tokens, hidden)],
        output_dtypes=[normed.dtype],
    )
    return mixed


__all__ = ["hyper_up_mix", "supports_hyper_up_mix"]
