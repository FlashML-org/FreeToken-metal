"""The fused residual add + RMSNorm against the ops path."""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.kernel.norm import add_rms_norm
from ftmetal.kernel.ref import add_rms_norm_ref


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("size", [64, 2048, 2560])
@pytest.mark.parametrize("num_rows", [1, 7])
def test_add_rms_norm_matches_ops_path(dtype: mx.Dtype, size: int, num_rows: int) -> None:
    mx.random.seed(size + num_rows)
    x = mx.random.normal((num_rows, size)).astype(dtype)
    r = (0.3 * mx.random.normal((num_rows, size))).astype(dtype)
    weight = (1.0 + 0.1 * mx.random.normal((size,))).astype(dtype)
    expected_h, expected_normed = add_rms_norm_ref(x, r, weight, 1e-6)
    h, normed = add_rms_norm(x, r, weight, 1e-6)
    mx.eval(h, normed, expected_h, expected_normed)
    assert mx.array_equal(h, expected_h)
    tolerance = 1e-2 if dtype == mx.bfloat16 else 1e-5
    assert mx.allclose(
        normed.astype(mx.float32),
        expected_normed.astype(mx.float32),
        atol=tolerance,
        rtol=tolerance,
    )
