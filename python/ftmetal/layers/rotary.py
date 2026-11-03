"""Rotary embedding over per-token positions, and the 3-axis (t/h/w) mrope.

Port of FreeToken layers/rotary.py. Divergences: only the default (NeoX, partial),
proportional and 3-axis ropes are served; mx.fast.rope does the 1-D rotation with the
batch's flat positions as per-token offsets, and kernel/rope/mrope.py the 3-axis one from an
inv_freq table, instead of a cos/sin cache; MRotaryEmbedding is its own op, not a
RotaryEmbedding subclass; build_section_table returns a numpy array.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
import numpy as np

from ftmetal.kernel.rope.mrope import mrope
from ftmetal.layers.base import BaseOP

# How the [t, h, w] mrope sections share the frequency slots (build_section_table).
MRopeLayout = Literal["contiguous", "interleaved", "interleaved_glm"]


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


def build_section_table(mrope_section: tuple[int, int, int], layout: MRopeLayout) -> np.ndarray:
    """Axis id (0=t, 1=h, 2=w) per frequency slot for a [t, h, w] section split.

    contiguous: T|H|W blocks. interleaved: H at slots 1,4,.. and W at 2,5,.. until each share
    is used, T takes the rest. interleaved_glm: round-robin over the three axes, skipping an
    axis once its share is used up.
    """
    st, sh, sw = mrope_section
    half = st + sh + sw
    section = np.zeros(half, dtype=np.int32)
    if layout == "contiguous":
        section[st : st + sh] = 1
        section[st + sh :] = 2
    elif layout == "interleaved":
        section[1 : 3 * sh : 3] = 1
        section[2 : 3 * sw : 3] = 2
        if int((section == 1).sum()) != sh or int((section == 2).sum()) != sw:
            raise ValueError(
                f"mrope_section {mrope_section} is not representable in the interleaved layout"
            )
    else:
        counts = [0, 0, 0]
        for i in range(half):
            axis = i % 3
            while counts[axis] >= mrope_section[axis]:
                axis = (axis + 1) % 3
            section[i] = axis
            counts[axis] += 1
    return section


class MRotaryEmbedding(BaseOP):
    """3-axis (t/h/w) rope on the NeoX half-rotation layout: frequency slot i turns by the
    position of the axis build_section_table gives it."""

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        base: float,
        mrope_section: tuple[int, int, int],
        layout: MRopeLayout,
    ) -> None:
        super().__init__()
        assert 0 < rotary_dim <= head_dim and rotary_dim % 2 == 0
        half = rotary_dim // 2
        assert sum(mrope_section) == half, (mrope_section, half)
        self.rotary_dim = rotary_dim
        # Underscored: a public array would become a loadable parameter. The frequencies of
        # the default rope, base ** (-2i / rotary_dim).
        self._inv_freq = mx.array((base ** (-np.arange(half) / half)).astype(np.float32))
        self._section = mx.array(build_section_table(mrope_section, layout))
        # Constants of every step, not kernels a captured step would record.
        mx.eval(self._inv_freq, self._section)

    def forward(self, positions: mx.array, x: mx.array) -> mx.array:
        """x is [num_tokens, num_heads, head_dim]; positions is int32 [3, num_tokens]."""
        return mrope(x, positions, self._inv_freq, self._section, self.rotary_dim)


__all__ = ["MRopeLayout", "MRotaryEmbedding", "RotaryEmbedding", "build_section_table"]
