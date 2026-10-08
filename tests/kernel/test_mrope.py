"""The 3-axis rope kernel: three equal rows are the 1-D rope mx.fast.rope applies, distinct
rows match a float64 numpy rotation, and the section tables of the three layouts.

Qwen3.6's geometry (head_dim 256, rotary_dim 64, theta 1e7, [11, 11, 10] interleaved) and
the tiny parity model's (64, 16, 1e4, [3, 3, 2]).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.layers.rotary import (
    MRopeLayout,
    MRotaryEmbedding,
    RotaryEmbedding,
    build_section_table,
)

# Max abs error against each reference, per dtype: mx.fast.rope takes a fast cos/sin of an
# angle it derives on its own; the numpy rotation is exact, so the kernel's float32 angle
# (about 2e-5 rad off at position 300) and the output rounding are all that differ.
TOLERANCES: dict[str, dict[mx.Dtype, float]] = {
    "fast_rope": {mx.float32: 1e-4, mx.bfloat16: 2e-2},
    "numpy": {mx.float32: 5e-5, mx.bfloat16: 2e-2},
}

# (head_dim, rotary_dim, base, mrope_section, layout, tokens, heads, dtype, max position)
MROPE_CASES: list[
    tuple[int, int, float, tuple[int, int, int], MRopeLayout, int, int, mx.Dtype, int]
] = [
    (256, 64, 1e7, (11, 11, 10), "interleaved", 37, 4, mx.float32, 300),
    (256, 64, 1e7, (11, 11, 10), "interleaved", 5, 2, mx.bfloat16, 300),
    (64, 16, 1e4, (3, 3, 2), "interleaved", 64, 2, mx.float32, 300),
    (128, 128, 1e6, (24, 20, 20), "contiguous", 9, 3, mx.float32, 300),
    # A whole head rotates: no pass-through dims.
    (64, 64, 1e4, (12, 10, 10), "interleaved_glm", 1, 1, mx.float32, 300),
]


def numpy_mrope(
    x: np.ndarray, positions: np.ndarray, base: float, section: np.ndarray, rotary_dim: int
) -> np.ndarray:
    """float64 rotation of x [tokens, heads, head_dim] by int [3, tokens] positions."""
    half = rotary_dim // 2
    inv_freq = base ** (-np.arange(half) / half)
    angle = positions[section].T.astype(np.float64) * inv_freq
    cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
    x1, x2, rest = x[..., :half], x[..., half:rotary_dim], x[..., rotary_dim:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], axis=-1)


def check_close(got: mx.array, expected: np.ndarray, tolerance: float) -> None:
    error = np.abs(np.array(got.astype(mx.float32)) - expected).max()
    assert error <= tolerance, f"max abs error {error} > {tolerance}"


@pytest.mark.parametrize(
    ("head_dim", "rotary_dim", "base", "section", "layout", "tokens", "heads", "dtype", "top"),
    MROPE_CASES,
)
def test_mrope_matches_its_references(
    head_dim: int,
    rotary_dim: int,
    base: float,
    section: tuple[int, int, int],
    layout: MRopeLayout,
    tokens: int,
    heads: int,
    dtype: mx.Dtype,
    top: int,
) -> None:
    rng = np.random.default_rng(tokens * heads)
    x = mx.array(rng.standard_normal((tokens, heads, head_dim)).astype(np.float32)).astype(dtype)
    host_x = np.array(x.astype(mx.float32), dtype=np.float64)
    rope = MRotaryEmbedding(head_dim, rotary_dim, base, section, layout)
    table = build_section_table(section, layout)

    sequence = rng.integers(0, top, tokens).astype(np.int32)
    equal_rows = np.ascontiguousarray(np.tile(sequence, (3, 1)))
    one_axis = RotaryEmbedding(head_dim, rotary_dim, base).forward(mx.array(sequence), x)
    check_close(
        rope.forward(mx.array(equal_rows), x),
        np.array(one_axis.astype(mx.float32)),
        TOLERANCES["fast_rope"][dtype],
    )
    distinct_rows = rng.integers(0, top, (3, tokens)).astype(np.int32)
    check_close(
        rope.forward(mx.array(distinct_rows), x),
        numpy_mrope(host_x, distinct_rows, base, table, rotary_dim),
        TOLERANCES["numpy"][dtype],
    )


# (mrope_section, layout, the axis of each frequency slot)
SECTION_CASES: list[tuple[tuple[int, int, int], MRopeLayout, list[int]]] = [
    ((3, 3, 2), "interleaved", [0, 1, 2, 0, 1, 2, 0, 1]),
    ((2, 1, 1), "contiguous", [0, 0, 1, 2]),
    ((1, 3, 2), "interleaved_glm", [0, 1, 2, 1, 1, 2]),
    # Qwen3.6: h and w interleave until their 11 and 10 slots are used, t takes the rest.
    (
        (11, 11, 10),
        "interleaved",
        [0, 1, 2] * 10 + [0, 1],
    ),
]


@pytest.mark.parametrize(("section", "layout", "axes"), SECTION_CASES)
def test_section_table_assigns_each_slot_its_axis(
    section: tuple[int, int, int], layout: MRopeLayout, axes: list[int]
) -> None:
    assert build_section_table(section, layout).tolist() == axes


def test_interleaved_layout_refuses_an_unrepresentable_split() -> None:
    with pytest.raises(ValueError, match="not representable"):
        build_section_table((1, 3, 3), "interleaved")
