"""The grouped (1 + w) RMS norm and its fusions against the fp32 op chains they fuse, bit for bit.

Each reference computes in fp32 with separate MLX ops and casts once: the (1 + w) norm as a
weightless mx.fast.rms_norm times (1 + w), the hyper-connection inject as hyper + branch * 2 *
sigmoid(logit / hc_count), the head norm then the model's RotaryEmbedding. Qwen3.8-Flash-Next's
shapes (hidden 2560, 4 streams; heads of 256 and index heads of 128, 64 rotary dims at base 1e7)
and small ones, bf16 and fp32.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.kernel.norm.offset_rms_norm import (
    hc_norm,
    offset_rms_norm,
    offset_rms_norm_rope,
    supports_offset_rms_norm_rope,
)
from ftmetal.layers.rotary import RotaryEmbedding

EPS = 1e-6
BF16 = mx.bfloat16
F32 = mx.float32


def check_equal(got: mx.array, expected: mx.array) -> None:
    assert got.shape == expected.shape and got.dtype == expected.dtype
    mismatches = (got != expected).sum().item()
    assert mismatches == 0, f"{mismatches} of {got.size} values differ"


def random(shape: tuple[int, ...], scale: float, dtype: mx.Dtype) -> mx.array:
    return (scale * mx.random.normal(shape)).astype(dtype)


def offset_norm_reference(x: mx.array, weight: mx.array, group_size: int) -> mx.array:
    groups = x.shape[-1] // group_size
    rows = x.astype(F32).reshape(-1, groups, group_size)
    scale = (1 + weight.astype(F32)).reshape(groups, group_size)
    return (mx.fast.rms_norm(rows, None, EPS) * scale).reshape(x.shape).astype(x.dtype)


def inject_reference(
    hyper: mx.array, branch: mx.array, logits: mx.array, hc_count: int
) -> mx.array:
    gates = 2 * mx.sigmoid(logits.astype(F32) / hc_count)
    streams = hyper.astype(F32).reshape(hyper.shape[0], hc_count, -1)
    streams = streams + branch.astype(F32)[:, None, :] * gates[:, :, None]
    return streams.reshape(hyper.shape).astype(hyper.dtype)


# (tokens, hc_count, hidden, dtype, inject, scale of the streams)
NORM_CASES = [
    (1, 4, 2560, BF16, True, 3.0),  # a decode token's streams, the pending branch injected
    (1, 4, 2560, F32, True, 0.01),
    (7, 4, 2560, BF16, False, 40.0),  # a prefill chunk
    (3, 4, 2560, F32, False, 1.0),
    (5, 2, 100, BF16, True, 1.0),  # 25 reading threads: part of a simdgroup
    (2, 3, 256, F32, True, 5.0),
]


@pytest.mark.parametrize(("tokens", "hc_count", "hidden", "dtype", "inject", "scale"), NORM_CASES)
def test_hc_norm(
    tokens: int, hc_count: int, hidden: int, dtype: mx.Dtype, inject: bool, scale: float
) -> None:
    mx.random.seed(tokens * hidden + hc_count)
    hyper = random((tokens, hc_count * hidden), scale, dtype)
    weight = random((hc_count * hidden,), 0.3, dtype)
    if inject:
        branch = random((tokens, hidden), scale, dtype)
        logits = random((tokens, hc_count), 4.0, dtype)
        normed, streams = hc_norm(hyper, weight, EPS, hc_count, branch, logits)
        check_equal(streams, inject_reference(hyper, branch, logits, hc_count))
    else:
        normed, streams = hc_norm(hyper, weight, EPS, hc_count)
        assert streams is hyper
    check_equal(normed, offset_norm_reference(streams, weight, hidden))


# (shape, group_size or None for the whole last axis, dtype)
OFFSET_NORM_CASES = [
    ((1, 24, 256), None, BF16),  # a decode token's query heads
    ((4, 2, 256), None, F32),
    ((2, 4 * 2560), 2560, BF16),  # the streams, grouped
    ((3, 5, 36), None, F32),
    ((2, 4096), None, BF16),  # the largest row: 1024 threads
]


@pytest.mark.parametrize(("shape", "group_size", "dtype"), OFFSET_NORM_CASES)
def test_offset_rms_norm(shape: tuple[int, ...], group_size: int | None, dtype: mx.Dtype) -> None:
    mx.random.seed(sum(shape))
    x = random(shape, 2.0, dtype)
    weight = random((shape[-1],), 0.3, dtype)
    expected = offset_norm_reference(x, weight, group_size or shape[-1])
    check_equal(offset_rms_norm(x, weight, EPS, group_size), expected)


# (tokens, heads, head_dim, rotary_dim, base, dtype)
NORM_ROPE_CASES = [
    (1, 24, 256, 64, 1e7, BF16),  # a decode token's queries
    (4, 2, 256, 64, 1e7, BF16),  # keys of 4 tokens
    (2, 4, 128, 64, 1e7, F32),  # index queries
    (3, 1, 128, 64, 1e7, BF16),  # index block keys
    (1, 2, 256, 256, 1e7, F32),  # a whole head rotates: more pairs than threads
    (5, 3, 64, 16, 1e4, F32),
]


@pytest.mark.parametrize(
    ("tokens", "heads", "head_dim", "rotary_dim", "base", "dtype"), NORM_ROPE_CASES
)
def test_offset_rms_norm_rope(
    tokens: int, heads: int, head_dim: int, rotary_dim: int, base: float, dtype: mx.Dtype
) -> None:
    mx.random.seed(tokens * heads + head_dim)
    x = random((tokens, heads, head_dim), 2.0, dtype)
    weight = random((head_dim,), 0.3, dtype)
    positions = mx.concatenate(
        [mx.array([0], dtype=mx.int32), mx.random.randint(1, 40000, (tokens - 1,))]
    ).astype(mx.int32)
    assert supports_offset_rms_norm_rope(x, rotary_dim)
    rope = RotaryEmbedding(head_dim, rotary_dim, base)
    expected = rope.forward(positions, offset_norm_reference(x, weight, head_dim))
    check_equal(offset_rms_norm_rope(x, weight, EPS, positions, rotary_dim, base), expected)


def test_offset_rms_norm_rope_refuses_other_shapes() -> None:
    assert not supports_offset_rms_norm_rope(mx.zeros((1, 2048), BF16), 64)
    assert not supports_offset_rms_norm_rope(mx.zeros((1, 2, 128), BF16), 63)
    assert not supports_offset_rms_norm_rope(mx.zeros((1, 2, 128), BF16), 256)
    assert not supports_offset_rms_norm_rope(mx.zeros((1, 2, 8192), BF16), 64)
