"""affine_gather_qmv: mx.gather_qmm's one-row affine matvec, skipping the pairs still pending.

No FreeToken donor. The offload decode's resident pass (moe_lookup_issue's slots): MLX's
affine_gather_qmv_fast and affine_gather_qmv (quantized.h) over each pair's slot, on the impl MLX's
gather_qmv picks for the shape (affine_slot_qmv.h), so every row equals mx.gather_qmm's (unsorted,
one row per pair) bit for bit, while a pair whose slot is negative, its expert absent, is left
unwritten for affine_swiglu_pending.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import metal_kernel_header, metal_source
from ftmetal.kernel.custom import metal_kernel
from ftmetal.kernel.moe.affine.affine_gather_qmv_pair_codes import (
    GATHER_QMV_FAST_INPUT_STEP,
    GROUP_SIZES,
    QMV_TILE_ROWS,
    SIMD_WIDTH,
    SIMDGROUPS_PER_THREADGROUP,
)

KERNEL = metal_kernel(
    name="affine_gather_qmv",
    input_names=["x", "w", "scales", "biases", "indices"],
    output_names=["y"],
    source=metal_source(Path(__file__).with_name("affine_gather_qmv.metal")),
    header=metal_kernel_header(Path(__file__).with_name("affine_slot_qmv.h")),
)


def affine_gather_qmv(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    indices: mx.array,
    *,
    group_size: int,
    bits: int,
) -> mx.array:
    """mx.gather_qmm(x, w, scales, biases, rhs_indices=indices, transpose=True) bit for bit at
    the pairs whose index is a slot; a negative index's rows are left unwritten.

    x [T, 1 or top_k, 1, K] in float32, float16 or bfloat16; indices int32 [T, top_k]; w [S, N,
    K * bits / 32] uint32; scales and biases [S, N, K / group_size] in x's dtype. Returns [T,
    top_k, 1, N] in x's dtype.
    """
    tokens, top_k = indices.shape
    num_slots, output_size, words = w.shape
    input_size = x.shape[-1]
    assert x.dtype in (mx.float32, mx.float16, mx.bfloat16), x.dtype
    assert x.ndim == 4 and x.shape[0] == tokens and x.shape[1] in (1, top_k) and x.shape[2] == 1
    assert indices.dtype == mx.int32, indices.dtype
    assert bits in GATHER_QMV_FAST_INPUT_STEP and group_size in GROUP_SIZES, (bits, group_size)
    assert input_size % group_size == 0 and words * 32 == input_size * bits, (w.shape, x.shape)
    assert w.dtype == mx.uint32, w.dtype
    group_shape = (num_slots, output_size, input_size // group_size)
    assert scales.shape == biases.shape == group_shape, (scales.shape, biases.shape)
    assert scales.dtype == biases.dtype == x.dtype, (scales.dtype, biases.dtype)
    is_fast = (
        output_size % QMV_TILE_ROWS == 0 and input_size % GATHER_QMV_FAST_INPUT_STEP[bits] == 0
    )
    (y,) = KERNEL(
        inputs=[x, w, scales, biases, indices],
        template=[
            ("T", x.dtype),
            ("GROUP_SIZE", group_size),
            ("BITS", bits),
            ("K", input_size),
            ("N", output_size),
            ("PAIRS_PER_INPUT", top_k // x.shape[1]),
            ("FAST", is_fast),
        ],
        grid=(
            SIMD_WIDTH,
            -(-output_size // QMV_TILE_ROWS) * SIMDGROUPS_PER_THREADGROUP,
            tokens * top_k,
        ),
        threadgroup=(SIMD_WIDTH, SIMDGROUPS_PER_THREADGROUP, 1),
        output_shapes=[(tokens, top_k, 1, output_size)],
        output_dtypes=[x.dtype],
    )
    return y


__all__ = ["affine_gather_qmv"]
