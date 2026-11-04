"""affine_gather_qmv_pair_codes: mx.gather_qmm's one-row affine matvec over pair-code rows.

No FreeToken donor. A pair-code bank row of an affine expert projection [N, K] holds, instead of
a scale and a bias per group, one code_bits-bit code (at most MAX_PAIR_CODE_BITS) per group into the
layer's pair table [P, 2] of (scale, bias) in x's dtype: code n * G + g for output row n and
group g of G = K / group_size, least significant bit first, the row zero padded to whole 64-byte
lines. This is MLX's
affine_gather_qmv_fast and affine_gather_qmv (quantized.h) with each group's scale and bias
looked up by its code, on the impl MLX's gather_qmv picks for the shape, so every output equals
mx.gather_qmm's (unsorted, one row per pair) on the expanded scales and biases bit for bit.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import _C, metal_kernel_header, metal_source
from ftmetal.kernel.custom import metal_kernel

SIMD_WIDTH = 32
# MLX gather_qmv's threadgroup: two simdgroups of four output rows each.
SIMDGROUPS_PER_THREADGROUP = 2
QMV_TILE_ROWS = 8
# MLX's qmv_fast_k_alignment per bit width: gather_qmv takes the fast kernel when K is a
# multiple of it and N of QMV_TILE_ROWS.
GATHER_QMV_FAST_INPUT_STEP = {2: 512, 3: 512, 4: 512, 5: 512, 6: 256, 8: 256}
GROUP_SIZES = (32, 64, 128)
MAX_PAIR_CODE_BITS = _C.MAX_PAIR_CODE_BITS
# Rows are whole cache lines: a CPU write to one slot's row shares no line with another's.
PAIR_CODE_ROW_ALIGNMENT = 64

SOURCE = metal_source(Path(__file__).with_name("affine_gather_qmv_pair_codes.metal"))

KERNEL = metal_kernel(
    name="affine_gather_qmv_pair_codes",
    input_names=["x", "w", "codes", "pairs", "indices"],
    output_names=["y"],
    source=SOURCE,
    header=metal_kernel_header(Path(__file__).with_name("affine_gather_qmv_pair_codes.h")),
)


def pair_code_row_bytes(num_codes: int, code_bits: int) -> int:
    """Bytes of a bank row of num_codes code_bits-bit codes: the bit stream in whole 64-byte
    lines."""
    line_bits = 8 * PAIR_CODE_ROW_ALIGNMENT
    return -(-num_codes * code_bits // line_bits) * PAIR_CODE_ROW_ALIGNMENT


def affine_gather_qmv_pair_codes(
    x: mx.array,
    w: mx.array,
    codes: mx.array,
    pairs: mx.array,
    indices: mx.array,
    *,
    group_size: int,
    bits: int,
    code_bits: int,
) -> mx.array:
    """mx.gather_qmm(x, w, scales, biases, rhs_indices=indices, transpose=True) bit for bit, the
    scales and biases [S, N, K / group_size] being pairs[code, 0] and pairs[code, 1].

    x [T, 1 or top_k, 1, K] in float32, float16 or bfloat16; indices [T, top_k] int32 or uint32
    slot ids; w [S, N, K * bits / 32] uint32; codes [S, pair_code_row_bytes(N * K / group_size,
    code_bits)] uint8; pairs [P, 2] in x's dtype. Returns [T, top_k, 1, N] in x's dtype.
    """
    tokens, top_k = indices.shape
    num_slots, output_size, words = w.shape
    input_size = x.shape[-1]
    assert x.dtype in (mx.float32, mx.float16, mx.bfloat16), x.dtype
    assert x.ndim == 4 and x.shape[0] == tokens and x.shape[1] in (1, top_k) and x.shape[2] == 1
    assert indices.dtype in (mx.int32, mx.uint32), indices.dtype
    assert bits in GATHER_QMV_FAST_INPUT_STEP and group_size in GROUP_SIZES, (bits, group_size)
    assert 1 <= code_bits <= MAX_PAIR_CODE_BITS, code_bits
    assert input_size % group_size == 0 and words * 32 == input_size * bits, (w.shape, x.shape)
    assert w.dtype == mx.uint32 and codes.dtype == mx.uint8, (w.dtype, codes.dtype)
    num_codes = output_size * input_size // group_size
    assert codes.shape == (num_slots, pair_code_row_bytes(num_codes, code_bits)), codes.shape
    assert pairs.ndim == 2 and pairs.shape[1] == 2 and pairs.dtype == x.dtype, pairs.shape
    is_fast = (
        output_size % QMV_TILE_ROWS == 0 and input_size % GATHER_QMV_FAST_INPUT_STEP[bits] == 0
    )
    (y,) = KERNEL(
        inputs=[x, w, codes, pairs, indices],
        template=[
            ("T", x.dtype),
            ("GROUP_SIZE", group_size),
            ("BITS", bits),
            ("CODE_BITS", code_bits),
            ("K", input_size),
            ("N", output_size),
            ("CODE_ROW_BYTES", codes.shape[1]),
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


__all__ = [
    "MAX_PAIR_CODE_BITS",
    "PAIR_CODE_ROW_ALIGNMENT",
    "affine_gather_qmv_pair_codes",
    "pair_code_row_bytes",
]
