"""Rotary embedding over per-token positions.

Port of FreeToken layers/rotary.py. Divergences: only the default (NeoX,
partial) rope is served; mx.fast.rope does the rotation with the batch's flat
positions as per-token offsets instead of a cos/sin cache.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.layers.base import BaseOP


class RotaryEmbedding(BaseOP):
    def __init__(self, head_dim: int, rotary_dim: int, base: float) -> None:
        super().__init__()
        assert 0 < rotary_dim <= head_dim and rotary_dim % 2 == 0
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.base = base

    def forward(self, positions: mx.array, x: mx.array) -> mx.array:
        """x is [num_tokens, num_heads, head_dim]; positions is int32 [num_tokens]."""
        num_tokens, num_heads, head_dim = x.shape
        rotated = mx.fast.rope(
            x.reshape(num_tokens, num_heads, 1, head_dim),
            self.rotary_dim,
            traditional=False,
            base=self.base,
            scale=1.0,
            offset=positions,
        )
        return rotated.reshape(num_tokens, num_heads, head_dim)


__all__ = ["RotaryEmbedding"]
