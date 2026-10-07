"""Expert-sorted affine quantized gather matmul over an expert bank of any size.

MLX 0.32.3's rhs-sorted mx.gather_qmm schedules its row tiles by scanning every bank row
in each threadgroup and launches E - 1 extra threadgroups, so on a bank larger than the
layer its cost grows with the bank (5120 rows: 2.9x the resident 256 for Qwen3.6's gate_up
at B = 32768 on the M4 Pro, 8x for its down), and below B / E = 4 it runs one matvec per
row (3-4x from B = 8192). gather_qmm_sorted runs the same kernel with the tile scan over
the layer's logical experts, offsets from moe_align, and reads expert e's weights from bank
slot slot_of[e]: a resident layer passes arange(E), an offload cache its slots. Below four
rows per logical expert MLX's per-row matvec beats either tile kernel (1.3-1.9x at B 32 to
768) and does not depend on the bank. On a GPU without matrix units this equals
mx.gather_qmm bit for bit wherever MLX takes its rhs kernel; a NAX GPU (M5 class) gets
MLX's matrix-unit kernel from mx.gather_qmm, which this does not carry. Adapted from MLX
v0.32.3 (metal/gather_qmm_sorted.metal, csrc/gather_qmm_sorted.h) and coupled to its
internals: re-check it on any MLX upgrade; the bit-exact tests are the guard.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel._ext import _C


def gather_qmm_sorted(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    expert_offsets: mx.array,
    slot_of: mx.array,
    *,
    group_size: int,
    bits: int,
) -> mx.array:
    """x[b] @ dequantize(w[slot_of[e]]).T for the rows b of each expert e; returns [B, 1, N].

    x [B, 1, K] with its rows sorted by expert; affine w [S, N, K * bits / 32] uint32
    (transposed), scales and biases [S, N, K / group_size] in x's dtype; expert_offsets
    [E + 1] int32 with expert e's rows at [expert_offsets[e], expert_offsets[e + 1]) and
    expert_offsets[E] == B (moe_align's); slot_of [E] int32 slots in [0, S). Not checked.
    """
    return _C.gather_qmm_sorted(x, w, scales, biases, expert_offsets, slot_of, group_size, bits)


__all__ = ["gather_qmm_sorted"]
