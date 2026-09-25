"""The chunked gated delta rule against the per-token recurrence kernel."""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.kernel.gdn import gated_delta_recurrence
from ftmetal.kernel.gdn_chunk import gated_delta_chunked

SHAPES = {
    "tiny": dict(batch=2, num_k_heads=2, num_v_heads=4, head_dim=32),
    "qwen3_6": dict(batch=1, num_k_heads=16, num_v_heads=32, head_dim=128),
}


def make_inputs(
    seq_len: int, shape: str, dtype: mx.Dtype, seed: int
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Normalized keys and a head_dim**-0.5 query scale, as the GDN layer feeds the kernels."""
    mx.random.seed(seed)
    spec = SHAPES[shape]
    batch, num_k_heads = spec["batch"], spec["num_k_heads"]
    num_v_heads, head_dim = spec["num_v_heads"], spec["head_dim"]
    q = mx.random.normal((batch, seq_len, num_k_heads, head_dim)) * head_dim**-0.5
    k = mx.random.normal((batch, seq_len, num_k_heads, head_dim))
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    v = mx.random.normal((batch, seq_len, num_v_heads, head_dim))
    g = mx.exp(-mx.random.uniform(0.0, 2.0, (batch, seq_len, num_v_heads)))
    beta = mx.sigmoid(mx.random.normal((batch, seq_len, num_v_heads)))
    state = 0.3 * mx.random.normal((batch, num_v_heads, head_dim, head_dim))
    q, k, v, beta = (x.astype(dtype) for x in (q, k, v, beta))
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


def assert_matches_recurrence(
    inputs: tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array], tolerance: float
) -> None:
    expected_y, expected_state = gated_delta_recurrence(*inputs)
    y, state = gated_delta_chunked(*inputs)
    mx.eval(expected_y, expected_state, y, state)
    assert y.dtype == expected_y.dtype and y.shape == expected_y.shape
    assert mx.allclose(
        y.astype(mx.float32), expected_y.astype(mx.float32), atol=tolerance, rtol=tolerance
    )
    assert mx.allclose(state, expected_state, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("seq_len", [1, 63, 64, 65, 200, 2048])
def test_chunked_matches_recurrence(seq_len: int, shape: str, dtype: mx.Dtype) -> None:
    inputs = make_inputs(seq_len, shape, dtype, seed=seq_len)
    # bf16 inputs: both kernels run in fp32, y differs by its bf16 rounding.
    assert_matches_recurrence(inputs, tolerance=1e-4 if dtype == mx.float32 else 2e-2)


def test_chunked_matches_recurrence_through_a_zero_decay() -> None:
    q, k, v, g, beta, state = make_inputs(150, "tiny", mx.float32, seed=7)
    # A decay that underflowed to zero wipes the state at that step.
    g[:, 70] = 0.0
    g[:, 100:103] = 0.0
    assert_matches_recurrence((q, k, v, g, beta, state), tolerance=1e-4)
