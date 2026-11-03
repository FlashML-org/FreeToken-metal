"""The hyper-connection mix fused with its one-token up projection against the unfused pair,
bit for bit: mx.quantized_matmul for up, then the fp32 op chain of the mix (kernel/ref.py
hc_mix_ref). Qwen3.8-Flash-Next's shapes (hidden 2560, 4 streams, low rank 320) and small
ones, bf16 and fp32.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.kernel.hc.hc_up_mix import hc_up_mix, supports_hc_up_mix
from ftmetal.kernel.ref import hc_mix_ref

BF16 = mx.bfloat16
F32 = mx.float32


def check_equal(got: mx.array, expected: mx.array) -> None:
    assert got.shape == expected.shape and got.dtype == expected.dtype
    mismatches = (got != expected).sum().item()
    assert mismatches == 0, f"{mismatches} of {got.size} values differ"


def random(shape: tuple[int, ...], scale: float, dtype: mx.Dtype) -> mx.array:
    return (scale * mx.random.normal(shape)).astype(dtype)


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
def test_hc_up_mix(
    hidden: int, lowrank: int, hc_count: int, group_size: int, dtype: mx.Dtype
) -> None:
    mx.random.seed(hidden + lowrank)
    weight, scales, biases = quantized_up(hc_count, hidden, lowrank, group_size, dtype)
    x = random((1, lowrank), 1.0, dtype)
    normed = random((1, hc_count * hidden), 1.0, dtype)
    assert supports_hc_up_mix(x, weight, scales, normed, hc_count, group_size, 4)
    up = mx.quantized_matmul(
        x, weight, scales, biases, transpose=True, group_size=group_size, bits=4
    )
    got = hc_up_mix(x, weight, scales, biases, normed, hc_count, group_size=group_size, bits=4)
    check_equal(got, hc_mix_ref(up, normed, hc_count))


def test_hc_up_mix_refuses_other_projections() -> None:
    """Shapes mx.quantized_matmul runs on another kernel (qmv_quad at inputs 64 and 128,
    qmv_fast at multiples of 512), other bit widths, more tokens, mixed dtypes."""
    normed = mx.zeros((1, 4 * 256), BF16)
    for lowrank in (64, 128, 512):
        weight, scales, _ = quantized_up(4, 256, lowrank, 32, BF16)
        x = mx.zeros((1, lowrank), BF16)
        assert not supports_hc_up_mix(x, weight, scales, normed, 4, 32, 4)
    weight, scales, _ = quantized_up(4, 256, 320, 32, BF16)
    x = mx.zeros((1, 320), BF16)
    assert supports_hc_up_mix(x, weight, scales, normed, 4, 32, 4)
    assert not supports_hc_up_mix(mx.zeros((2, 320), BF16), weight, scales, normed, 4, 32, 4)
    assert not supports_hc_up_mix(mx.zeros((1, 320), F32), weight, scales, normed, 4, 32, 4)
    eight_bit, eight_bit_scales, _ = mx.quantize(mx.zeros((4 * 256, 320), BF16), 64, 8)
    assert not supports_hc_up_mix(x, eight_bit, eight_bit_scales, normed, 4, 64, 8)
