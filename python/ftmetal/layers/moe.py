"""Resident routed experts over mx.gather_qmm / mx.gather_mm.

Port of FreeToken layers/moe.py (MoELayer). Divergences: the experts live in one
fused gate_up_proj and one down_proj bank per layer in bf16 or affine quantization;
routing is computed by the model and handed to forward. The offload subclass
arrives in round 3.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.layers.activation import silu_and_mul
from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.config import AffineScheme

# Below this many (token, expert) pairs the sort costs more than the scattered reads.
SORT_MIN_PAIRS = 64


class ExpertBank(BaseOP):
    """One [num_experts, output, input] projection shared by every expert of a layer."""

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        *,
        scheme: AffineScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.scheme = scheme
        if scheme is None:
            self.weight = mx.zeros((num_experts, output_size, input_size), dtype=dtype)
        else:
            packed = input_size * scheme.bits // 32
            groups = input_size // scheme.group_size
            self.weight = mx.zeros((num_experts, output_size, packed), dtype=mx.uint32)
            self.scales = mx.zeros((num_experts, output_size, groups), dtype=dtype)
            self.biases = mx.zeros((num_experts, output_size, groups), dtype=dtype)

    def forward(self, x: mx.array, indices: mx.array, sorted_indices: bool) -> mx.array:
        if self.scheme is None:
            return mx.gather_mm(  # pyright: ignore[reportCallIssue]  (stub requires lhs_indices)
                x, self.weight.swapaxes(-1, -2), rhs_indices=indices, sorted_indices=sorted_indices
            )
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=indices,
            transpose=True,
            group_size=self.scheme.group_size,
            bits=self.scheme.bits,
            mode="affine",
            sorted_indices=sorted_indices,
        )


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        scheme: AffineScheme | None = None,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = ExpertBank(
            num_experts, hidden_size, 2 * intermediate_size, scheme=scheme, dtype=dtype
        )
        self.down_proj = ExpertBank(
            num_experts, intermediate_size, hidden_size, scheme=scheme, dtype=dtype
        )

    def forward(
        self, hidden_states: mx.array, topk_weights: mx.array, topk_ids: mx.array
    ) -> mx.array:
        """Expert compute for a routing decision (the donor's routed_forward): hidden [T, H],
        weights / ids [T, top_k]; the model computes the routing."""
        x = mx.expand_dims(hidden_states, (-2, -3))
        do_sort = topk_ids.size >= SORT_MIN_PAIRS
        indices = topk_ids
        inverse_order = None
        if do_sort:
            flat = topk_ids.flatten()
            order = mx.argsort(flat)
            inverse_order = mx.argsort(order)
            x = x.flatten(0, -3)[order // self.top_k]
            indices = flat[order]
        gate_up = self.gate_up_proj.forward(x, indices, do_sort)
        y = self.down_proj.forward(silu_and_mul(gate_up), indices, do_sort)
        if inverse_order is not None:
            y = mx.unflatten(y[inverse_order], 0, topk_ids.shape)
        y = y.squeeze(-2)
        return (y * topk_weights[..., None]).sum(axis=-2)


__all__ = ["SORT_MIN_PAIRS", "ExpertBank", "MoELayer"]
