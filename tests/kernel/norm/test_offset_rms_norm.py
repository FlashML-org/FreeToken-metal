"""The hyper-connection kernels against the fp32 op chains they fuse, bit for bit.

Each reference computes in fp32 with separate MLX ops and casts once: the (1 + w) norm as a
weightless mx.fast.rms_norm times (1 + w), the inject as hyper + branch * 2 *
sigmoid(logit / hc_count), the mix as the mean over the streams of sigmoid(up) * normed, the up
mix on the one-token mx.quantized_matmul, the head norm then the model's RotaryEmbedding.
Qwen3.8-Flash-Next's shapes (hidden 2560, 4 streams, low rank 320; heads of 256 and index heads
of 128, 64 rotary dims at base 1e7) and small ones, bf16 and fp32.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.kernel.hyper_connection import (
    hyper_mix,
    hyper_norm,
    offset_rms_norm,
    offset_rms_norm_rope,
    supports_offset_rms_norm_rope,
)
from ftmetal.kernel.hyper_up_mix import hyper_up_mix, supports_hyper_up_mix
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


def mix_reference(up: mx.array, normed: mx.array, hc_count: int) -> mx.array:
    weighted = mx.sigmoid(up.astype(F32)) * normed.astype(F32)
    total = weighted.reshape(normed.shape[0], hc_count, -1).sum(axis=1)
    return (total / hc_count).astype(normed.dtype)


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
def test_hyper_norm(
    tokens: int, hc_count: int, hidden: int, dtype: mx.Dtype, inject: bool, scale: float
) -> None:
    mx.random.seed(tokens * hidden + hc_count)
    hyper = random((tokens, hc_count * hidden), scale, dtype)
    weight = random((hc_count * hidden,), 0.3, dtype)
    if inject:
        branch = random((tokens, hidden), scale, dtype)
        logits = random((tokens, hc_count), 4.0, dtype)
        normed, streams = hyper_norm(hyper, weight, EPS, hc_count, branch, logits)
        check_equal(streams, inject_reference(hyper, branch, logits, hc_count))
    else:
        normed, streams = hyper_norm(hyper, weight, EPS, hc_count)
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


# (tokens, hc_count, hidden, dtype)
MIX_CASES = [
    (1, 4, 2560, BF16),
    (6, 4, 2560, F32),
    (3, 2, 96, BF16),
    (2, 3, 100, F32),  # a mean over 3: the division rounds
]


@pytest.mark.parametrize(("tokens", "hc_count", "hidden", "dtype"), MIX_CASES)
def test_hyper_mix(tokens: int, hc_count: int, hidden: int, dtype: mx.Dtype) -> None:
    mx.random.seed(tokens + hidden)
    up = random((tokens, hc_count * hidden), 4.0, dtype)
    normed = random((tokens, hc_count * hidden), 1.0, dtype)
    check_equal(hyper_mix(up, normed, hc_count), mix_reference(up, normed, hc_count))


def quantized_up(
    hc_count: int, hidden: int, lowrank: int, group_size: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array, mx.array]:
    weight = random((hc_count * hidden, lowrank), 0.05, dtype)
    quantized, scales, biases = mx.quantize(weight, group_size=group_size, bits=4)
    return quantized, scales, biases


# (hidden, lowrank, hc_count, group_size, dtype)
UP_MIX_CASES = [
    # Input 320: one block, then 64 values on lanes 0-7, its 4 rows over 4 lane groups.
    (2560, 320, 4, 32, BF16),
    (2560, 320, 4, 32, F32),
    (256, 96, 4, 32, BF16),  # less than one block, on 12 lanes
    (128, 384, 4, 64, F32),  # 128 values on lanes 0-15, rows over 2 lane groups
    (64, 320, 2, 32, BF16),  # 2 rows: no split over 4 lane groups
]


@pytest.mark.parametrize(("hidden", "lowrank", "hc_count", "group_size", "dtype"), UP_MIX_CASES)
def test_hyper_up_mix(
    hidden: int, lowrank: int, hc_count: int, group_size: int, dtype: mx.Dtype
) -> None:
    mx.random.seed(hidden + lowrank)
    weight, scales, biases = quantized_up(hc_count, hidden, lowrank, group_size, dtype)
    x = random((1, lowrank), 1.0, dtype)
    normed = random((1, hc_count * hidden), 1.0, dtype)
    assert supports_hyper_up_mix(x, weight, scales, normed, hc_count, group_size, 4)
    up = mx.quantized_matmul(
        x, weight, scales, biases, transpose=True, group_size=group_size, bits=4
    )
    got = hyper_up_mix(x, weight, scales, biases, normed, hc_count, group_size=group_size, bits=4)
    check_equal(got, mix_reference(up, normed, hc_count))


def test_hyper_up_mix_refuses_other_projections() -> None:
    """Shapes mx.quantized_matmul runs on another kernel (qmv_quad at inputs 64 and 128,
    qmv_fast at multiples of 512), other bit widths, more tokens, mixed dtypes."""
    normed = mx.zeros((1, 4 * 256), BF16)
    for lowrank in (64, 128, 512):
        weight, scales, _ = quantized_up(4, 256, lowrank, 32, BF16)
        x = mx.zeros((1, lowrank), BF16)
        assert not supports_hyper_up_mix(x, weight, scales, normed, 4, 32, 4)
    weight, scales, _ = quantized_up(4, 256, 320, 32, BF16)
    x = mx.zeros((1, 320), BF16)
    assert supports_hyper_up_mix(x, weight, scales, normed, 4, 32, 4)
    assert not supports_hyper_up_mix(mx.zeros((2, 320), BF16), weight, scales, normed, 4, 32, 4)
    assert not supports_hyper_up_mix(mx.zeros((1, 320), F32), weight, scales, normed, 4, 32, 4)
    eight_bit, eight_bit_scales, _ = mx.quantize(mx.zeros((4 * 256, 320), BF16), 64, 8)
    assert not supports_hyper_up_mix(x, eight_bit, eight_bit_scales, normed, 4, 64, 8)
