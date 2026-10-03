"""Rotary embedding over per-token positions.

Port of FreeToken layers/rotary.py. Divergences: only the default (NeoX, partial) and
proportional ropes are served; mx.fast.rope does the rotation with the batch's flat
positions as per-token offsets instead of a cos/sin cache.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.layers.base import BaseOP


class RotaryEmbedding(BaseOP):
    def __init__(
        self, head_dim: int, rotary_dim: int, base: float, proportional: bool = False
    ) -> None:
        """Default: the first rotary_dim dims rotate at frequencies spaced over rotary_dim.
        Proportional: all head_dim dims in NeoX pairs at frequencies spaced over head_dim,
        of which only the first rotary_dim / 2 pairs turn (the rest have infinite period)."""
        super().__init__()
        assert 0 < rotary_dim <= head_dim and rotary_dim % 2 == 0
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.base = base
        self.proportional = proportional
        # Underscored: a public array would become a loadable parameter.
        self._freqs: mx.array | None = None
        if proportional:
            exponents = mx.arange(0, rotary_dim, 2, dtype=mx.float32) / head_dim
            still = mx.full(((head_dim - rotary_dim) // 2,), mx.inf, dtype=mx.float32)
            self._freqs = mx.concatenate([base**exponents, still])
            # A constant of every step, not a kernel a captured step would record.
            mx.eval(self._freqs)

    def forward(self, positions: mx.array, x: mx.array) -> mx.array:
        """x is [num_tokens, num_heads, head_dim]; positions is int32 [num_tokens]."""
        num_tokens, num_heads, head_dim = x.shape
        x = x.reshape(num_tokens, num_heads, 1, head_dim)
        if self.proportional:
            rotated = mx.fast.rope(
                x,
                head_dim,
                traditional=False,
                base=None,
                scale=1.0,
                offset=positions,
                freqs=self._freqs,
            )
        else:
            rotated = mx.fast.rope(
                x,
                self.rotary_dim,
                traditional=False,
                base=self.base,
                scale=1.0,
                offset=positions,
            )
        return rotated.reshape(num_tokens, num_heads, head_dim)


__all__ = ["RotaryEmbedding"]
