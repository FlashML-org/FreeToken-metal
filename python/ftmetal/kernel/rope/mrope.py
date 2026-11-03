"""Fused 3-axis (t/h/w) rotary embedding: frequency slot i rotates the NeoX pair
(i, i + rotary_dim / 2) by the position of the axis the section table gives it; the
dimensions past rotary_dim pass through.

Port of FreeToken kernel/triton/rope.py (apply_mrope_with_cos_sin_cache_inplace), on
mx.fast.metal_kernel. Divergences: the angle is computed in the kernel from an inv_freq
table with precise cos and sin (no cos/sin cache), one tensor per call, out of place.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import metal_source
from ftmetal.kernel.custom import metal_kernel

MROPE_THREADGROUP = 256

MROPE_SOURCE = metal_source(Path(__file__).with_name("mrope.metal"))

MROPE_KERNEL = metal_kernel(
    name="mrope",
    input_names=["x", "positions", "inv_freq", "section"],
    output_names=["out"],
    source=MROPE_SOURCE,
)


def mrope(
    x: mx.array, positions: mx.array, inv_freq: mx.array, section: mx.array, rotary_dim: int
) -> mx.array:
    """x [tokens, heads, head_dim] rotated by int32 [3, tokens] positions; inv_freq and the
    int32 axis section are [rotary_dim / 2]."""
    num_tokens, num_heads, head_dim = x.shape
    work = num_tokens * num_heads * (head_dim - rotary_dim // 2)
    (out,) = MROPE_KERNEL(
        inputs=[x, positions, inv_freq, section],
        template=[("T", x.dtype), ("ROTARY_DIM", rotary_dim)],
        grid=(work, 1, 1),
        threadgroup=(min(MROPE_THREADGROUP, work), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


__all__ = ["mrope"]
