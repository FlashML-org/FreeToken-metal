"""The fused router against the ops path: same experts, matching weights; and reading the first
experts of a wider row in place, the same ids and weights as of that row's copy, bit for bit.

One simdgroup per row: 512 experts (Qwen3.8-Flash-Next's top 10), 256 (eight per lane, Qwen3.6's
top 8), 128 (Gemma4's top 8), 64 (two per lane) and 4 (most lanes idle), with and without
renormalization and in both dtypes; and the first experts of wider rows, read in place.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.core import host_floats
from ftmetal.kernel.moe.moe_topk import moe_topk
from ftmetal.kernel.ref import moe_topk_ref


@pytest.mark.parametrize(
    ("num_experts", "top_k", "norm", "dtype", "num_rows"),
    [
        (512, 10, True, mx.bfloat16, 3),
        (128, 8, True, mx.bfloat16, 7),
        (256, 8, True, mx.bfloat16, 40),
        (256, 8, False, mx.float32, 5),
        (64, 1, True, mx.bfloat16, 5),
        (64, 1, False, mx.float32, 40),
        (4, 2, True, mx.float32, 1),
        (4, 2, False, mx.bfloat16, 40),
    ],
    ids=[
        "E512-norm-bf16",
        "E128-norm-bf16",
        "E256-norm-bf16",
        "E256-fp32",
        "E64-norm-bf16",
        "E64-fp32",
        "E4-norm-fp32",
        "E4-bf16",
    ],
)
def test_moe_topk_matches_ops_path(
    dtype: mx.Dtype, num_experts: int, top_k: int, num_rows: int, norm: bool
) -> None:
    mx.random.seed(num_rows * 7 + top_k)
    gates = (3.0 * mx.random.normal((num_rows, num_experts))).astype(dtype)
    expected_ids, expected_weights = moe_topk_ref(gates, top_k, norm)
    ids, weights = moe_topk(gates, top_k, norm)
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


@pytest.mark.parametrize(
    ("num_experts", "stride", "top_k", "dtype", "num_rows"),
    [
        (512, 520, 10, mx.bfloat16, 4),
        (256, 264, 8, mx.bfloat16, 6),
        (16, 24, 4, mx.float32, 7),
    ],
    ids=["E512-in-S520", "E256-in-S264", "E16-in-S24"],
)
def test_moe_topk_reads_a_wider_row_in_place(
    num_experts: int, stride: int, top_k: int, dtype: mx.Dtype, num_rows: int
) -> None:
    mx.random.seed(num_experts + stride + num_rows)
    wide = (3.0 * mx.random.normal((num_rows, stride))).astype(dtype)
    # Equal logits on two experts: the tie goes to the lower index either way.
    wide[:, 3] = wide[:, 1]
    expected_ids, expected_weights = moe_topk(mx.contiguous(wide[:, :num_experts]), top_k, True)
    ids, weights = moe_topk(wide, top_k, True, num_experts=num_experts)
    assert mx.array_equal(ids, expected_ids).item()
    bits = mx.uint16 if dtype == mx.bfloat16 else mx.uint32
    assert mx.array_equal(weights.view(bits), expected_weights.view(bits)).item()
