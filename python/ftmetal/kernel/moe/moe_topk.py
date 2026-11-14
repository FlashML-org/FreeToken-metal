"""Fused MoE router: softmax over the experts or each logit's sigmoid, top-k selection and
renormalization.

One simdgroup per token row on mx.fast.metal_kernel; replaces the softmax or sigmoid,
argpartition, take_along_axis, sum and divide dispatches of the ops path (kernel/ref.py). The
selection runs on per-lane registers: no
threadgroup memory or barrier on its serial path.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import metal_source
from ftmetal.kernel.custom import metal_kernel

SIMD_WIDTH = 32

SOURCE = metal_source(Path(__file__).with_name("moe_topk.metal"))

KERNEL = metal_kernel(
    name="moe_topk",
    input_names=["gates"],
    output_names=["ids", "scores"],
    source=SOURCE,
)


def moe_topk(
    gates: mx.array, top_k: int, norm_topk_prob: bool, sigmoid: bool = False
) -> tuple[mx.array, mx.array]:
    """gates [T, E] logits -> (expert ids int32 [T, top_k], weights [T, top_k] in gates' dtype);
    scored by each logit's fp32 sigmoid, else the row's fp32 softmax."""
    num_rows, num_experts = gates.shape
    assert 1 <= top_k <= num_experts, (top_k, num_experts)
    ids, scores = KERNEL(
        inputs=[gates],
        template=[
            ("InT", gates.dtype),
            ("E", num_experts),
            ("K", top_k),
            ("NORM", norm_topk_prob),
            ("SIGMOID", sigmoid),
        ],
        grid=(SIMD_WIDTH * num_rows, 1, 1),
        threadgroup=(SIMD_WIDTH, 1, 1),
        output_shapes=[(num_rows, top_k), (num_rows, top_k)],
        output_dtypes=[mx.int32, gates.dtype],
    )
    return ids, scores


__all__ = ["moe_topk"]
