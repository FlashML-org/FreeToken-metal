"""The fused router against the ops path: same experts, matching weights.

One simdgroup per row: 256 experts (eight per lane, Qwen3.6's top 8), 64 (two per lane)
and 4 (most lanes idle), each with and without renormalization and in both dtypes.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.core import host_floats
from ftmetal.kernel.ref import router_topk_ref
from ftmetal.kernel.router import router_topk


@pytest.mark.parametrize(
    ("num_experts", "top_k", "norm", "dtype", "num_rows"),
    [
        (256, 8, True, mx.bfloat16, 40),
        (256, 8, False, mx.float32, 5),
        (64, 1, True, mx.bfloat16, 5),
        (64, 1, False, mx.float32, 40),
        (4, 2, True, mx.float32, 1),
        (4, 2, False, mx.bfloat16, 40),
    ],
    ids=["E256-norm-bf16", "E256-fp32", "E64-norm-bf16", "E64-fp32", "E4-norm-fp32", "E4-bf16"],
)
def test_router_matches_ops_path(
    dtype: mx.Dtype, num_experts: int, top_k: int, num_rows: int, norm: bool
) -> None:
    mx.random.seed(num_rows * 7 + top_k)
    gates = (3.0 * mx.random.normal((num_rows, num_experts))).astype(dtype)
    expected_ids, expected_weights = router_topk_ref(gates, top_k, norm)
    ids, weights = router_topk(gates, top_k, norm)
    mx.eval(ids, weights, expected_ids, expected_weights)
    tolerance = 1e-2 if dtype == mx.bfloat16 else 1e-5
    for row in range(num_rows):
        # Ties (equal logits) may be broken differently: compare the chosen logits, not ids.
        chosen = sorted(host_floats(gates[row][ids[row]]))
        expected = sorted(host_floats(gates[row][expected_ids[row]]))
        assert chosen == expected
        got = sorted(host_floats(weights[row]))
        want = sorted(host_floats(expected_weights[row]))
        for a, b in zip(got, want):
            assert abs(a - b) <= tolerance * max(1.0, abs(b))
    assert ids.dtype == mx.int32 and weights.dtype == dtype
