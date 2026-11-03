"""Expert-sorted gather matmuls, each expert's weights at its byte offsets.

MLX 0.32.3's rhs-sorted mx.gather_qmm schedules its row tiles by scanning every bank row
in each threadgroup and launches E - 1 extra threadgroups, so on a bank larger than the
layer its cost grows with the bank (5120 rows: 2.9x the resident 256 for Qwen3.6's gate_up
at B = 32768 on the M4 Pro, 8x for its down), and below B / E = 4 it runs one matvec per
row (3-4x from B = 8192); its rhs-sorted mx.gather_mm scans and launches the same way, at
any B. gather_qmm_sorted (affine or a float block mode) and gather_mm_sorted (unquantized)
run the same kernels with the tile scan over the layer's logical experts, offsets from
moe_align, and read expert e's weight, scales and biases at its byte offsets into raw
buffers: an offload cache's banks, whose pools and prefill buffers hold rows of several
layouts. Below four rows per logical expert MLX's per-row affine matvec beats either tile
kernel (1.3-1.9x at B 32 to 768) and does not depend on the bank. On a GPU without matrix
units these equal mx.gather_qmm and mx.gather_mm bit for bit wherever MLX takes its rhs
kernel; a NAX GPU (M5 class) gets MLX's matrix-unit kernels, which these do not carry.
Adapted from MLX v0.32.3 (affine/affine_gather_qmm_sorted.metal, fp/fp_gather_qmm_sorted.metal,
unquantized/gather_mm_sorted.metal, gather_sorted.h) and coupled to its internals: re-check them
on any MLX upgrade; the bit-exact tests are the guard.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx

from ftmetal.kernel._ext import _C

# MLX's quantization modes; the float block ones take one (bits, group size) each: mxfp4
# (4, 32), nvfp4 (4, 16), mxfp8 (8, 32).
QuantMode = Literal["affine", "mxfp4", "nvfp4", "mxfp8"]
# A float block mode's K is whole tiles of this many inputs.
FLOAT_INPUT_TILE = _C.QMM_TILE_K


def gather_qmm_sorted(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    expert_offsets: mx.array,
    expert_bytes: mx.array,
    *,
    n: int,
    group_size: int,
    bits: int,
    mode: QuantMode = "affine",
) -> mx.array:
    """x[b] @ dequantize(expert e's weights).T for the rows b of each expert e; returns
    [B, 1, n].

    x [B, 1, K] with its rows sorted by expert; w, scales and biases are row-contiguous
    buffers of any dtype holding expert e's [n, K * bits / 32] uint32 weight (transposed),
    [n, K / group_size] scales (x's dtype for affine, uint8 for a float block mode) and, affine
    only, biases in x's dtype at byte offsets expert_bytes[e] int64 [E, 3] (weight, scales,
    biases) or, a float block mode, [E, 2] (weight, scales); expert_offsets [E + 1] int32 with
    expert e's rows at [expert_offsets[e], expert_offsets[e + 1]) and expert_offsets[E] == B
    (moe_align's). A float block mode needs K a multiple of 32. The offsets are not checked.
    """
    return _C.gather_qmm_sorted(
        x, w, scales, biases, expert_offsets, expert_bytes, n, group_size, bits, mode
    )


def gather_mm_sorted(
    x: mx.array,
    w: mx.array,
    expert_offsets: mx.array,
    expert_bytes: mx.array,
    *,
    n: int,
) -> mx.array:
    """x[b] @ (expert e's weight).T for the rows b of each expert e; returns [B, 1, n].

    As gather_qmm_sorted, with w a row-contiguous buffer of any dtype holding expert e's
    [n, K] weight in x's dtype at byte offset expert_bytes[e] int64 [E, 1].
    """
    return _C.gather_mm_sorted(x, w, expert_offsets, expert_bytes, n)


__all__ = ["FLOAT_INPUT_TILE", "QuantMode", "gather_mm_sorted", "gather_qmm_sorted"]
