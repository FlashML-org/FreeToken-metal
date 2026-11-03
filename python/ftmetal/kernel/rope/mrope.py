"""Fused 3-axis (t/h/w) rotary embedding: frequency slot i rotates the NeoX pair
(i, i + rotary_dim / 2) by the position of the axis the section table gives it; the
dimensions past rotary_dim pass through.

Port of FreeToken kernel/triton/rope.py (apply_mrope_with_cos_sin_cache_inplace), on
mx.fast.metal_kernel. Divergences: the angle is computed in the kernel from an inv_freq
table with precise cos and sin (no cos/sin cache), one tensor per call, out of place.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel.custom import metal_kernel

MROPE_THREADGROUP = 256

MROPE_SOURCE = """
    const uint elem = thread_position_in_grid.x;
    const int num_tokens = x_shape[0];
    const int num_heads = x_shape[1];
    const int head_dim = x_shape[2];
    const int half_dim = ROTARY_DIM / 2;
    // Each (token, head) row has half_dim rotating pairs, then its pass-through dims.
    const int slots = head_dim - half_dim;
    if (elem >= uint(num_tokens * num_heads * slots)) {
      return;
    }
    const int slot = int(elem) % slots;
    const int row = int(elem) / slots;
    const int token = row / num_heads;
    const int base = row * head_dim;
    if (slot >= half_dim) {
      const int d = ROTARY_DIM + slot - half_dim;
      out[base + d] = x[base + d];
      return;
    }
    const int axis = section[slot];
    const float angle = static_cast<float>(positions[axis * num_tokens + token]) * inv_freq[slot];
    const float c = metal::precise::cos(angle);
    const float s = metal::precise::sin(angle);
    const float x1 = static_cast<float>(x[base + slot]);
    const float x2 = static_cast<float>(x[base + slot + half_dim]);
    out[base + slot] = static_cast<T>(x1 * c - x2 * s);
    out[base + slot + half_dim] = static_cast<T>(x2 * c + x1 * s);
"""

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
