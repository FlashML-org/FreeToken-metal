"""The hyper-connection mix against the fp32 op chain, bit for bit: the mean over the streams
of sigmoid(up) * normed in fp32 with separate MLX ops, cast once (kernel/ref.py hc_mix_ref).
Qwen3.8-Flash-Next's shapes (hidden 2560, 4 streams) and small ones, bf16 and fp32.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.kernel.hc.hc_mix import hc_mix
from ftmetal.kernel.ref import hc_mix_ref

BF16 = mx.bfloat16
F32 = mx.float32


def check_equal(got: mx.array, expected: mx.array) -> None:
    assert got.shape == expected.shape and got.dtype == expected.dtype
    mismatches = (got != expected).sum().item()
    assert mismatches == 0, f"{mismatches} of {got.size} values differ"


def random(shape: tuple[int, ...], scale: float, dtype: mx.Dtype) -> mx.array:
    return (scale * mx.random.normal(shape)).astype(dtype)


# (tokens, hc_count, hidden, dtype)
MIX_CASES = [
    (1, 4, 2560, BF16),
    (6, 4, 2560, F32),
    (3, 2, 96, BF16),
    (2, 3, 100, F32),  # a mean over 3: the division rounds
]


@pytest.mark.parametrize(("tokens", "hc_count", "hidden", "dtype"), MIX_CASES)
def test_hc_mix(tokens: int, hc_count: int, hidden: int, dtype: mx.Dtype) -> None:
    mx.random.seed(tokens + hidden)
    up = random((tokens, hc_count * hidden), 4.0, dtype)
    normed = random((tokens, hc_count * hidden), 1.0, dtype)
    check_equal(hc_mix(up, normed, hc_count), hc_mix_ref(up, normed, hc_count))
