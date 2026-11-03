"""The fused GDN prefill inputs kernel (conv, silu, q/k norm, gating) against its functional
reference.

Two head shapes: tiny (D 32, one state column per lane) and Qwen3.6's (D 128, four). Chunk
lengths cover a chunk shorter than the conv taps, one as long, one past a simdgroup's 16
tokens and several simdgroups' worth.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.kernel.attention.gdn.gdn_prefill import gdn_prefill_inputs
from ftmetal.kernel.ref import gdn_prefill_inputs_ref

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
