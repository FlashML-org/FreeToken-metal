"""The MoE alignment kernel against the argsort path: same offsets, a valid expert-sorted
permutation (ties inside an expert may be ordered differently), and the combine."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.core import host_ints
from ftmetal.kernel.moe_align import moe_align, moe_combine
from ftmetal.kernel.ref import moe_align_ref, moe_combine_ref


def routed_ids(num_tokens: int, top_k: int, num_experts: int, active_experts: int) -> mx.array:
    """Distinct experts per token drawn from the first active_experts; the rest stay empty."""
    rng = np.random.default_rng(num_tokens * 31 + num_experts)
    ids = [rng.choice(active_experts, top_k, replace=False) for _ in range(num_tokens)]
    return mx.array(np.array(ids, dtype=np.int32))


@pytest.mark.parametrize(
    "num_experts,top_k,active_experts",
    [(256, 8, 256), (256, 8, 40), (256, 8, 8), (4, 2, 4), (4, 2, 2)],
)
@pytest.mark.parametrize("num_tokens", [1, 2, 7, 64, 777, 2048])
def test_align_is_an_expert_sorted_permutation(
    num_experts: int, top_k: int, active_experts: int, num_tokens: int
) -> None:
    topk_ids = routed_ids(num_tokens, top_k, num_experts, active_experts)
    alignment = moe_align(topk_ids, num_experts)
    expected = moe_align_ref(topk_ids, num_experts)
    mx.eval(alignment, expected)
    num_pairs = num_tokens * top_k
    assert host_ints(alignment.expert_offsets) == host_ints(expected.expert_offsets)
    assert host_ints(alignment.sorted_expert_ids) == host_ints(expected.sorted_expert_ids)
    positions = alignment.pair_positions
    assert sorted(host_ints(positions)) == list(range(num_pairs))
    assert host_ints(alignment.sorted_expert_ids[positions]) == host_ints(topk_ids.flatten())
    token_of_pair = [pair // top_k for pair in range(num_pairs)]
    assert host_ints(alignment.sorted_token_indices[positions]) == token_of_pair


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("num_tokens", [1, 9, 300])
@pytest.mark.parametrize("num_experts,top_k,hidden_size", [(256, 8, 2048), (4, 2, 64)])
def test_combine_matches_gather_and_weighted_sum(
    dtype: mx.Dtype, num_tokens: int, num_experts: int, top_k: int, hidden_size: int
) -> None:
    mx.random.seed(num_tokens)
    topk_ids = routed_ids(num_tokens, top_k, num_experts, num_experts)
    pair_positions = moe_align(topk_ids, num_experts).pair_positions
    expert_out = mx.random.normal((num_tokens * top_k, 1, hidden_size)).astype(dtype)
    topk_weights = mx.softmax(mx.random.normal((num_tokens, top_k)), axis=-1).astype(dtype)
    out = moe_combine(expert_out, pair_positions, topk_weights)
    expected = moe_combine_ref(expert_out, pair_positions, topk_weights)
    tolerance = 1e-2 if dtype == mx.bfloat16 else 1e-5
    assert out.shape == (num_tokens, hidden_size) and out.dtype == dtype
    assert mx.allclose(out, expected, atol=tolerance, rtol=tolerance)
