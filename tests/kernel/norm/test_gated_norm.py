"""gated_rms_norm against the fp32 op chain it fuses, bit for bit: mx.fast.rms_norm of the fp32
input with the fp32 weight, times sigmoid of the fp32 gate, cast once. The Gated DeltaNet
output norm's rows (value head dim 128) and small and wide ones, bf16 and fp32.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.kernel.norm.gated_norm import gated_rms_norm, supports_gated_rms_norm

EPS = 1e-6
BF16 = mx.bfloat16
F32 = mx.float32

# (tokens, heads, size, dtype, input scale)
CASES = [
    (1, 32, 128, BF16, 1.0),  # a decode token's value heads
    (4, 32, 128, F32, 5.0),
    (3, 2, 64, BF16, 0.01),  # 16 reading threads: half a simdgroup
    (1, 4, 36, F32, 40.0),
    (2, 2, 2560, BF16, 0.3),  # a row of 20 simdgroups
]


@pytest.mark.parametrize(("tokens", "heads", "size", "dtype", "scale"), CASES)
def test_gated_rms_norm(tokens: int, heads: int, size: int, dtype: mx.Dtype, scale: float) -> None:
    mx.random.seed(tokens * heads + size)
    x = (scale * mx.random.normal((tokens, heads, size))).astype(dtype)
    z = (3 * mx.random.normal((tokens, heads, size))).astype(dtype)
    weight = (1 + 0.2 * mx.random.normal((size,))).astype(dtype)
    assert supports_gated_rms_norm(x, z)
    normed = mx.fast.rms_norm(x.astype(F32), weight.astype(F32), EPS)
    expected = (normed * mx.sigmoid(z.astype(F32))).astype(dtype)
    got = gated_rms_norm(x, weight, z, EPS)
    assert got.shape == expected.shape and got.dtype == expected.dtype
    mismatches = (got != expected).sum().item()
    assert mismatches == 0, f"{mismatches} of {got.size} values differ"


def test_gated_rms_norm_refuses_other_shapes() -> None:
    x = mx.zeros((1, 2, 128), BF16)
    assert not supports_gated_rms_norm(x, mx.zeros((1, 2, 64), BF16))
    wide = mx.zeros((1, 8192), BF16)
    assert not supports_gated_rms_norm(wide, wide)
