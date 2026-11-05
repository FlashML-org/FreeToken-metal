"""QSA prefill selection: attention/qsa/metal.py's topk_select (qsa_topk) against
attention/qsa/common.py's select_query_blocks (MLX's sort), the same bool [Q, N] blocks.

Each query's topk of its complete blocks (query_end // block_size), ties to the lower block, on
scores with runs of relu's zeros and of equal steps (-0.0 tying +0.0, as MLX compares them);
queries with fewer complete blocks than topk, as many, more, and none; the real budget (512 of
2,048 blocks) and a small one.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.attention.qsa.common import select_query_blocks
from ftmetal.attention.qsa.metal import topk_select

BLOCK_SIZE = 4


@dataclass(frozen=True)
class Case:
    num_blocks: int
    topk: int
    pattern: str

    def __str__(self) -> str:
        return f"N{self.num_blocks}-k{self.topk}-{self.pattern}"


CASES = [
    Case(2048, 512, "relu"),
    Case(2048, 512, "steps"),
    Case(100, 16, "relu"),
    Case(100, 16, "distinct"),
]


@pytest.mark.parametrize("case", CASES, ids=str)
def test_topk_select_matches_the_sort(case: Case) -> None:
    rng = np.random.default_rng(case.num_blocks + case.topk)
    raw = rng.standard_normal((48, case.num_blocks)).astype(np.float32)
    if case.pattern == "relu":
        scores = np.maximum(raw, 0)
    elif case.pattern == "steps":
        scores = np.round(raw * 2) / 2
    else:
        scores = raw
    # Query ends from none complete through all of them, around the topk boundary.
    edges = [0, 3, case.topk * BLOCK_SIZE, case.topk * BLOCK_SIZE + 4, case.num_blocks * BLOCK_SIZE]
    ends = np.concatenate(
        [edges, rng.integers(1, case.num_blocks * BLOCK_SIZE + 1, 48 - len(edges))]
    )
    query_ends = mx.array(ends.astype(np.int32))
    got = topk_select(mx.array(scores), query_ends, BLOCK_SIZE, case.topk)
    expected = select_query_blocks(mx.array(scores), query_ends, BLOCK_SIZE, case.topk)
    complete = ends // BLOCK_SIZE
    sparse = complete > case.topk
    # Rows with at most topk complete blocks see all of them in visible_blocks; compare the rest.
    assert np.array_equal(np.array(got)[sparse], np.array(expected)[sparse])
    assert np.array(got)[sparse].sum(axis=1).tolist() == [case.topk] * int(sparse.sum())
