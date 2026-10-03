"""The fused GDN kernels against their functional references: prefill inputs, decode step.

Two head shapes: tiny (D 32, one state column per lane) and Qwen3.6's (D 128, four).
Prefill chunk lengths cover a chunk shorter than the conv taps, one as long, one past a
simdgroup's 16 tokens and several simdgroups' worth; decode covers one request and three
on scattered slots, one of them at a position whose ring cells were never written.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.kernel.gdn import gdn_decode_step, gdn_prefill_inputs
from ftmetal.kernel.ref import gdn_decode_step_ref, gdn_prefill_inputs_ref

SHAPES = {
    "tiny": dict(num_k_heads=2, num_v_heads=4, key_dim=32, value_dim=32, conv_kernel=4),
    "qwen3_6": dict(num_k_heads=16, num_v_heads=32, key_dim=128, value_dim=128, conv_kernel=4),
}
DTYPES = {"bf16": mx.bfloat16, "fp32": mx.float32}


@pytest.mark.parametrize(
    ("shape", "dtype_name", "seq_len"),
    [
        ("tiny", "fp32", 1),
        ("tiny", "bf16", 3),
        ("tiny", "fp32", 17),
        ("qwen3_6", "bf16", 1),
        ("qwen3_6", "bf16", 17),
        ("qwen3_6", "fp32", 100),
    ],
)
def test_prefill_inputs_match_reference(shape: str, dtype_name: str, seq_len: int) -> None:
    dtype = DTYPES[dtype_name]
    mx.random.seed(seq_len)
    spec = SHAPES[shape]
    num_k_heads, num_v_heads = spec["num_k_heads"], spec["num_v_heads"]
    conv_kernel = spec["conv_kernel"]
    conv_dim = 2 * num_k_heads * spec["key_dim"] + num_v_heads * spec["value_dim"]
    # conv_in, a and b as the layer passes them: column views of the wider projections.
    projection = mx.random.normal((seq_len, conv_dim + 64)).astype(dtype)
    ba = mx.random.normal((seq_len, 2 * num_v_heads)).astype(dtype)
    inputs = (
        projection[:, :conv_dim],
        mx.random.normal((conv_kernel - 1, conv_dim)).astype(dtype),
        (0.5 * mx.random.normal((conv_dim, conv_kernel))).astype(dtype),
        ba[:, num_v_heads:],
        ba[:, :num_v_heads],
        mx.random.normal((num_v_heads,)).astype(dtype),
        mx.random.normal((num_v_heads,)).astype(dtype),
    )
    expected = gdn_prefill_inputs_ref(*inputs, num_k_heads)
    outputs = gdn_prefill_inputs(*inputs, num_k_heads)
    mx.eval(expected, outputs)

    tolerance = 2e-2 if dtype == mx.bfloat16 else 1e-5
    for output, reference in zip(outputs, expected, strict=True):
        assert output.dtype == reference.dtype and output.shape == reference.shape
        assert mx.allclose(
            output.astype(mx.float32),
            reference.astype(mx.float32),
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.parametrize(
    ("shape", "dtype_name", "batch"),
    [("tiny", "fp32", 1), ("tiny", "bf16", 3), ("qwen3_6", "bf16", 3), ("qwen3_6", "fp32", 1)],
)
def test_decode_step_matches_reference(shape: str, dtype_name: str, batch: int) -> None:
    dtype = DTYPES[dtype_name]
    mx.random.seed(batch)
    spec = SHAPES[shape]
    num_k_heads, num_v_heads = spec["num_k_heads"], spec["num_v_heads"]
    key_dim, value_dim, conv_kernel = spec["key_dim"], spec["value_dim"], spec["conv_kernel"]
    conv_dim = 2 * num_k_heads * key_dim + num_v_heads * value_dim
    num_slots = 6
    conv_pool = mx.random.normal((num_slots, conv_kernel, conv_dim)).astype(dtype)
    state_pool = 0.5 * mx.random.normal((num_slots, num_v_heads, value_dim, key_dim))
    conv_in = mx.random.normal((batch, conv_dim)).astype(dtype)
    conv_weight = (0.5 * mx.random.normal((conv_dim, conv_kernel))).astype(dtype)
    a = mx.random.normal((batch, num_v_heads)).astype(dtype)
    b = mx.random.normal((batch, num_v_heads)).astype(dtype)
    A_log = mx.random.normal((num_v_heads,)).astype(dtype)
    dt_bias = mx.random.normal((num_v_heads,)).astype(dtype)
    slots = mx.array([5, 1, 3][:batch], dtype=mx.int32)
    # Positions below the kernel width read ring cells that were never written.
    positions = mx.array([1, 7, 130][:batch], dtype=mx.int32)
    mx.eval(conv_pool, state_pool, conv_in, conv_weight, a, b, A_log, dt_bias)

    expected_conv, expected_state, expected_y = gdn_decode_step_ref(
        conv_pool, state_pool, conv_in, conv_weight, a, b, A_log, dt_bias, slots, positions
    )
    mx.eval(expected_conv, expected_state, expected_y)
    conv_pool, state_pool, y = gdn_decode_step(
        conv_pool, state_pool, conv_in, conv_weight, a, b, A_log, dt_bias, slots, positions
    )
    mx.eval(conv_pool, state_pool, y)

    tolerance = 2e-2 if dtype == mx.bfloat16 else 1e-4
    assert mx.allclose(
        y.astype(mx.float32), expected_y.astype(mx.float32), atol=tolerance, rtol=tolerance
    )
    assert mx.allclose(state_pool, expected_state, atol=tolerance, rtol=tolerance)
    assert mx.array_equal(conv_pool, expected_conv)


def test_decode_step_updates_the_pools_in_place() -> None:
    spec = SHAPES["tiny"]
    conv_dim = 2 * spec["num_k_heads"] * spec["key_dim"] + spec["num_v_heads"] * spec["value_dim"]
    conv_pool = mx.zeros((2, spec["conv_kernel"], conv_dim), dtype=mx.float32)
    state_pool = mx.zeros((2, spec["num_v_heads"], spec["value_dim"], spec["key_dim"]))
    mx.eval(conv_pool, state_pool)
    before = conv_pool
    ones = mx.ones((1, conv_dim))
    heads = mx.ones((1, spec["num_v_heads"]))
    conv_pool, state_pool, _ = gdn_decode_step(
        conv_pool,
        state_pool,
        ones,
        mx.ones((conv_dim, spec["conv_kernel"])),
        heads,
        heads,
        mx.zeros((spec["num_v_heads"],)),
        mx.zeros((spec["num_v_heads"],)),
        mx.array([1], dtype=mx.int32),
        mx.array([4], dtype=mx.int32),
    )
    mx.eval(conv_pool, state_pool)
    # The old binding sees the write: the primitive wrote through the shared buffer.
    assert mx.array_equal(before[1, 0], mx.ones((conv_dim,)))
    assert mx.array_equal(before[0], mx.zeros_like(before[0]))
