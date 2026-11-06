"""Routed experts: resident banks or an offload cache's, run by the layer's MoE method.

Port of FreeToken layers/moe.py (MoELayer, OffloadMoELayer). Divergences: routing is computed by
the model and handed to forward; the experts are one gate_up_proj bank (gate_proj and up_proj
banks under it when the checkpoint stores them in different schemes) and one down_proj bank per
layer, gated by silu or gelu_tanh, which the MoE method of layers/quantization/moe.py allocates
and runs. OffloadMoELayer serves its experts from the context's offload cache, whose layers
share one set of banks (gate and up apart in all layers once one keeps them apart), and holds
only its method's resident arrays. A decode or copy-miss forward gathers the slots its lookup
returns from its pool's rows, unsorted (below four rows per expert the per-row matvec is the
fast path); an overlap prefill sorts its pairs and reads each expert at its byte offsets in the
banks (the expert-sorted kernels), its prefill buffers and the order of its reads and layers
kept by the host, where FreeToken waits on CUDA events. A decode under split_lookup whose method
has the pending kernels issues its lookup instead (layers/quantization/moe_pending.py). A
forward given a shared expert (FreeToken's model computes it apart) adds its gated output, which
apply_pending fuses into its last kernel.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
from mlx.utils import tree_flatten

from ftmetal.context import get_global_ctx
from ftmetal.layers.activation import sigmoid_gated_add
from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.moe import (
    AffineMoEMethod,
    ExpertSchemes,
    MoEConfig,
    SharedExpertForward,
    moe_method,
)
from ftmetal.layers.quantization.moe_pending import apply_pending, supports_pending
from ftmetal.moe.offload_cache import OffloadMoeCache


def add_shared_expert(
    routed: mx.array, x: mx.array, shared_expert: SharedExpertForward | None
) -> mx.array:
    """The routed experts' output plus the shared expert's of x, gated."""
    if shared_expert is None:
        return routed
    else:
        shared, shared_gate = shared_expert(x)
        return sigmoid_gated_add(routed, shared, shared_gate)


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        schemes: ExpertSchemes,
        activation: Literal["silu", "gelu_tanh"] = "silu",
        dtype: mx.Dtype = mx.bfloat16,
        strategy: Literal["resident", "offload"] = "resident",
    ) -> None:
        """An offload layer holds its method's resident arrays alone: its offload cache holds
        the banks."""
        super().__init__()
        cfg = MoEConfig(
            num_experts=num_experts,
            top_k=top_k,
            hidden=hidden_size,
            intermediate=intermediate_size,
            schemes=schemes,
            activation=activation,
            dtype=dtype,
            strategy=strategy,
        )
        self.quant_method = moe_method(cfg)
        self.quant_method.create_weights(self)

    def forward(
        self,
        hidden_states: mx.array,
        topk_weights: mx.array,
        topk_ids: mx.array,
        shared_expert: SharedExpertForward | None = None,
    ) -> mx.array:
        """Expert compute for a routing decision (FreeToken's routed_forward): hidden [T, H],
        weights / ids [T, top_k]; the model computes the routing."""
        banks = dict(tree_flatten(self.parameters()))
        routed = self.quant_method.apply(banks, hidden_states, topk_weights, topk_ids)
        return add_shared_expert(routed, hidden_states, shared_expert)


class OffloadMoELayer(MoELayer):
    def __init__(
        self,
        moe_layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        schemes: ExpertSchemes,
        activation: Literal["silu", "gelu_tanh"] = "silu",
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        """moe_layer_id: the layer's index among the model's MoE layers, the cache's layer."""
        super().__init__(
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            schemes=schemes,
            activation=activation,
            dtype=dtype,
            strategy="offload",
        )
        self.moe_layer_id = moe_layer_id

    def forward(
        self,
        hidden_states: mx.array,
        topk_weights: mx.array,
        topk_ids: mx.array,
        shared_expert: SharedExpertForward | None = None,
    ) -> mx.array:
        cache = get_global_ctx().moe_offload_cache
        assert cache is not None, "an offload MoE layer runs on the engine's offload cache"
        resident = dict(tree_flatten(self.parameters()))
        method = self.quant_method
        if cache.mode == "overlap":
            return self.prefill_forward(
                cache, resident, hidden_states, topk_weights, topk_ids, shared_expert
            )
        elif (
            cache.mode == "decode"
            and cache.split_lookup
            and isinstance(method, AffineMoEMethod)
            and supports_pending(method, hidden_states, topk_weights)
        ):
            lookup = cache.issue_lookup(self.moe_layer_id, topk_ids)
            banks = {**cache.layer_banks(self.moe_layer_id), **resident}
            outputs = apply_pending(
                method, banks, hidden_states, topk_weights, lookup, shared_expert
            )
            cache.end_issued_lookup(outputs.mailbox)
            return outputs.out
        else:
            slots = cache.lookup_slots(self.moe_layer_id, topk_ids, hidden_states)
            banks = {**cache.layer_banks(self.moe_layer_id), **resident}
            routed = method.apply_slots(banks, hidden_states, topk_weights, topk_ids, slots)
            return add_shared_expert(routed, hidden_states, shared_expert)

    def prefill_forward(
        self,
        cache: OffloadMoeCache,
        resident: dict[str, mx.array],
        hidden_states: mx.array,
        topk_weights: mx.array,
        topk_ids: mx.array,
        shared_expert: SharedExpertForward | None,
    ) -> mx.array:
        """An overlap prefill layer: the layer's input, its shared expert and the next layer's
        predicted routing run while its prefill buffer is read; once its routing is on the host
        the unrouted reads are dropped and the next layer's start; its MoE runs as a command
        buffer of its own once its routed experts are in, a short prefill's experts in slots
        first, during the reads (OffloadMoeCache.resident_first)."""
        if self.moe_layer_id == 0:
            cache.begin_prefill_forward()
        # One evaluation signals once all its arrays are done: the routing alone first, so the
        # host has it before the prediction and the shared expert run.
        mx.async_eval(hidden_states, topk_weights, topk_ids)
        predicted = cache.predict_prefill_routing(self.moe_layer_id, hidden_states)
        if predicted is not None:
            mx.async_eval(predicted)
        shared = None if shared_expert is None else shared_expert(hidden_states)
        if shared is not None:
            mx.async_eval(*shared)
        cache.route_prefill_layer(self.moe_layer_id, topk_ids, predicted)
        banks = {**cache.banks, **cache.layer_pairs(self.moe_layer_id), **resident}
        method = self.quant_method
        first = cache.resident_first(self.moe_layer_id)
        if first is None:
            expert_bytes = cache.expert_bytes(self.moe_layer_id)
            out = method.apply_sorted(banks, hidden_states, topk_weights, topk_ids, expert_bytes)
        else:
            # The experts in slots run during the reads, the others once they are in.
            plan, ahead_bytes = first
            split = method.apply_sorted_first(
                banks,
                hidden_states,
                topk_ids,
                plan.order,
                plan.num_experts,
                plan.num_rows,
                ahead_bytes,
            )
            mx.async_eval(split.head)
            expert_bytes = cache.expert_bytes(self.moe_layer_id)
            out = method.apply_sorted_rest(banks, split, topk_weights, topk_ids, expert_bytes)
        mx.async_eval(out)
        cache.end_prefill_layer(self.moe_layer_id, out)
        return out if shared is None else sigmoid_gated_add(out, *shared)


__all__ = ["MoELayer", "OffloadMoELayer"]
