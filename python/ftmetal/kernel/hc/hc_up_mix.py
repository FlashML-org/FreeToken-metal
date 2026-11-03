"""The hyper-connection's one-token up projection fused with its mix: the 4-bit affine
GEMV of the activated down projection, then the sigmoid-weighted mean of the normed streams
(kernel/hc/hc_mix.py), in one launch.

Port of FreeToken kernel/triton/hc.py's hc_gate_mix with its up GEMM folded in. Divergences:
the GEMV sums as mx.quantized_matmul's qmv does for one token, so the result equals the
GEMV then hc_mix bit for bit.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import metal_kernel_header, metal_source
from ftmetal.kernel.custom import metal_kernel
from ftmetal.kernel.gemm.affine.affine_qmv_rows import QMV_FAST_INPUT_STEP
from ftmetal.kernel.norm.offset_rms_norm import SIMD_WIDTH

UP_MIX_SIMDGROUPS_PER_THREADGROUP = 2
UP_MIX_BITS = 4
# mx.quantized_matmul runs one token through qmv_quad at these input sizes, qmv_fast at multiples
# of its input step and qmv otherwise; hc_up_mix computes qmv's sums.
QMV_QUAD_INPUT_SIZES = (64, 128)

UP_MIX_SOURCE = metal_source(Path(__file__).with_name("hc_up_mix.metal"))

UP_MIX_KERNEL = metal_kernel(
    name="hc_up_mix",
    input_names=["x", "weight", "scales", "biases", "normed"],
    output_names=["mixed"],
    source=UP_MIX_SOURCE,
    header=metal_kernel_header(Path(__file__).with_name("hc_up_mix.h")),
)


def supports_hc_up_mix(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    normed: mx.array,
    hc_count: int,
    group_size: int,
    bits: int,
) -> bool:
    """Whether hc_up_mix takes one token x [1, K] through the 4-bit affine up projection
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


def hc_up_mix(
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
    """hc_mix(mx.quantized_matmul(x, weight, scales, biases, transpose=True), normed) for one
    token in one launch (supports_hc_up_mix), bit for bit: [1, H]."""
    assert supports_hc_up_mix(x, weight, scales, normed, hc_count, group_size, bits), (
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


__all__ = ["hc_up_mix", "supports_hc_up_mix"]
