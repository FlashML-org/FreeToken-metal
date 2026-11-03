"""qmv_rows: a one-token affine quantized GEMV with one output row per simdgroup.

No FreeToken donor. MLX's qmv_fast gives each simdgroup 4 output rows, so a projection with a
few hundred rows runs on a few dozen threadgroups and waits on memory latency. This is the same
code (quantized.h's qmv_fast_impl with its load_vector and qdot) with one row per simdgroup:
each row's per-lane sums and simd_sum are unchanged, so every output equals
mx.quantized_matmul's for one token bit for bit, on 4x the simdgroups. The hyper-connection
epilogue applies silu(y / hc_count) in fp32 to the down rows as it stores them.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import metal_kernel_header, metal_source
from ftmetal.kernel.custom import metal_kernel

SIMDGROUPS_PER_THREADGROUP = 2
# The input step of MLX's qmv_fast per bit width; mx.quantized_matmul runs one token as qmv_fast
# when the input size is a multiple of it and the output size of QMV_FAST_OUTPUT_TILE.
QMV_FAST_INPUT_STEP = {4: 512, 8: 256}
QMV_FAST_OUTPUT_TILE = 8


QUANTIZED_HEADER = metal_kernel_header(Path(__file__).with_name("affine_qmv_rows.h"))

SOURCE = metal_source(Path(__file__).with_name("affine_qmv_rows.metal"))

KERNEL = metal_kernel(
    name="affine_qmv_rows",
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


__all__ = ["QMV_FAST_INPUT_STEP", "qmv_rows", "supports_qmv_rows"]
