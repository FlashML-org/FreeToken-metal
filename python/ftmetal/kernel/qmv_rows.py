"""qmv_rows: a one-token affine quantized GEMV with one output row per simdgroup.

No FreeToken donor. MLX's qmv_fast gives each simdgroup 4 output rows, so a projection with a
few hundred rows runs on a few dozen threadgroups and waits on memory latency. This is the same
code (quantized.h's qmv_fast_impl with its load_vector and qdot) with one row per simdgroup:
each row's per-lane sums and simd_sum are unchanged, so every output equals
mx.quantized_matmul's for one token bit for bit, on 4x the simdgroups. The hyper-connection
epilogue applies silu(y / hc_count) in fp32 to the down rows as it stores them.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel._ext import inline_mlx_headers
from ftmetal.kernel.custom import metal_kernel

SIMDGROUPS_PER_THREADGROUP = 2
# The input step of MLX's qmv_fast per bit width; mx.quantized_matmul runs one token as qmv_fast
# when the input size is a multiple of it and the output size of QMV_FAST_OUTPUT_TILE.
QMV_FAST_INPUT_STEP = {4: 512, 8: 256}
QMV_FAST_OUTPUT_TILE = 8


def mlx_quantized_header() -> str:
    """quantized.h and what it builds on, inlined; metal_kernel's preamble already holds utils.h."""
    included: set[str] = set()
    inline_mlx_headers('#include "mlx/backend/metal/kernels/utils.h"\n', included)
    return inline_mlx_headers(
        '#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"\n'
        '#include "mlx/backend/metal/kernels/quantized_utils.h"\n'
        '#include "mlx/backend/metal/kernels/quantized.h"\n',
        included,
    )


QUANTIZED_HEADER = mlx_quantized_header()

SOURCE = """
    constexpr int packs_per_thread = BITS == 2 ? 1 : 2;
    constexpr int pack_factor = get_pack_factor<BITS, 32>();
    constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * SIMD_SIZE;
    constexpr int scale_step_per_thread = GROUP_SIZE / values_per_thread;
    constexpr int in_vec_size_w = K * bytes_per_pack / pack_factor;
    constexpr int in_vec_size_g = K / GROUP_SIZE;
    const uint simd_lid = thread_index_in_simdgroup;
    const int out_row =
        threadgroup_position_in_grid.y * SIMDGROUPS + simdgroup_index_in_threadgroup;
    if (out_row >= N) {
      return;
    }
    typedef float U;
    thread U x_thread[values_per_thread];
    U result = 0;
    const device uint8_t* ws = (const device uint8_t*)weight + out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    const device T* sl = scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* bl = biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* xp = x + simd_lid * values_per_thread;
    for (int k = 0; k < K; k += block_size) {
      U sum = load_vector<T, U, values_per_thread, BITS>(xp, x_thread);
      U s = sl[0];
      U b = bl[0];
      result += qdot<U, values_per_thread, BITS>(ws, x_thread, s, b, sum);
      ws += block_size * bytes_per_pack / pack_factor;
      sl += block_size / GROUP_SIZE;
      bl += block_size / GROUP_SIZE;
      xp += block_size;
    }
    result = simd_sum(result);
    if (simd_lid == 0) {
      const T value = static_cast<T>(result);
      if (out_row < SILU_ROWS) {
        // silu of the stored value over hc_count in fp32, with MLX's sigmoid.
        const float scaled = static_cast<float>(value) / static_cast<float>(HC_COUNT);
        const float e = 1 / (1 + metal::precise::exp(metal::abs(scaled)));
        y[out_row] = static_cast<T>(scaled * ((scaled < 0) ? e : 1 - e));
      } else {
        y[out_row] = value;
      }
    }
"""

KERNEL = metal_kernel(
    name="qmv_rows",
    input_names=["weight", "scales", "biases", "x"],
    output_names=["y"],
    source=SOURCE,
    header=QUANTIZED_HEADER,
)


def supports_qmv_rows(
    x: mx.array, weight: mx.array, scales: mx.array, group_size: int, bits: int
) -> bool:
    """Whether qmv_rows takes x [1, K] through the affine weight [N, K * bits / 32] with scales
    in x's dtype: the shapes mx.quantized_matmul runs as qmv_fast for one token."""
    output_size, input_size = weight.shape[0], x.shape[-1]
    return (
        bits in QMV_FAST_INPUT_STEP
        and x.ndim == 2
        and x.shape[0] == 1
        and input_size % QMV_FAST_INPUT_STEP[bits] == 0
        and output_size % QMV_FAST_OUTPUT_TILE == 0
        and weight.shape[1] * 32 // bits == input_size
        and scales.shape == (output_size, input_size // group_size)
        and scales.dtype == x.dtype
    )


def qmv_rows(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    group_size: int,
    bits: int,
    hc_epilogue: tuple[int, int] | None = None,
) -> mx.array:
    """mx.quantized_matmul(x, weight, scales, biases, transpose=True) for x [1, K]
    (supports_qmv_rows), bit for bit: y [1, N]. hc_epilogue (lowrank, hc_count): output rows
    below lowrank become silu(y / hc_count) computed in fp32 from the stored y; the rest stay
    raw (the inject logits)."""
    assert supports_qmv_rows(x, weight, scales, group_size, bits), (x.shape, weight.shape, bits)
    output_size, input_size = weight.shape[0], x.shape[-1]
    silu_rows, hc_count = (0, 1) if hc_epilogue is None else hc_epilogue
    threadgroups = -(-output_size // SIMDGROUPS_PER_THREADGROUP)
    (y,) = KERNEL(
        inputs=[weight, scales, biases, x],
        template=[
            ("T", x.dtype),
            ("K", input_size),
            ("N", output_size),
            ("GROUP_SIZE", group_size),
            ("BITS", bits),
            ("SIMDGROUPS", SIMDGROUPS_PER_THREADGROUP),
            ("SILU_ROWS", silu_rows),
            ("HC_COUNT", hc_count),
        ],
        grid=(32, threadgroups * SIMDGROUPS_PER_THREADGROUP, 1),
        threadgroup=(32, SIMDGROUPS_PER_THREADGROUP, 1),
        output_shapes=[(1, output_size)],
        output_dtypes=[x.dtype],
    )
    return y


__all__ = ["QMV_FAST_INPUT_STEP", "QUANTIZED_HEADER", "qmv_rows", "supports_qmv_rows"]
