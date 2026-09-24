"""The fused router against the ops path: same experts, matching weights."""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.core import host_floats
from ftmetal.kernel.ref import router_topk_ref
from ftmetal.kernel.router import router_topk


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("num_experts,top_k", [(256, 8), (4, 2), (64, 1)])
@pytest.mark.parametrize("num_rows", [1, 5, 40])
@pytest.mark.parametrize("norm", [True, False])
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
