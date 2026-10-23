"""Routed experts: resident banks or an offload cache's, run by the layer's MoE method.

Port of FreeToken layers/moe.py (MoELayer, OffloadMoELayer). Divergences: routing is computed
by the model and handed to forward; the experts are one gate_up_proj bank (gate_proj and
up_proj banks under it when the checkpoint stores them in different schemes) and one down_proj
bank per layer, gated by silu or gelu_tanh, which the MoE method of
layers/quantization/moe.py allocates and runs.
OffloadMoELayer serves affine experts only, from the context's offload cache, whose layers
share one set of banks: gate and up apart in all layers once one keeps them apart. A decode
or copy-miss forward gathers the slots its lookup returns from its pool's rows, unsorted
(below four rows per expert the per-row matvec is the fast path); an overlap prefill sorts
its pairs and reads each expert at its byte offsets in the banks (gather_qmm_sorted), its
prefill buffers and the order of its reads and layers kept by the host, where FreeToken
waits on CUDA events.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
from mlx.utils import tree_flatten

from ftmetal.context import get_global_ctx
from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.moe import AffineMoEMethod, ExpertSchemes, MoEConfig, moe_method
from ftmetal.moe.offload_cache import OffloadMoeCache


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
        """An offload layer holds no banks: its offload cache does."""
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
        if strategy == "resident":
            self.quant_method.create_weights(self)

    def forward(
        self, hidden_states: mx.array, topk_weights: mx.array, topk_ids: mx.array
    ) -> mx.array:
        """Expert compute for a routing decision (FreeToken's routed_forward): hidden [T, H],
        weights / ids [T, top_k]; the model computes the routing."""
        banks = dict(tree_flatten(self.parameters()))
        return self.quant_method.apply(banks, hidden_states, topk_weights, topk_ids)


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
        self, hidden_states: mx.array, topk_weights: mx.array, topk_ids: mx.array
    ) -> mx.array:
        cache = get_global_ctx().moe_offload_cache
        assert cache is not None, "an offload MoE layer runs on the engine's offload cache"
        if cache.mode == "overlap":
            return self.prefill_forward(cache, hidden_states, topk_weights, topk_ids)
        else:
            slots = cache.lookup_slots(self.moe_layer_id, topk_ids, hidden_states)
            banks = cache.layer_banks(self.moe_layer_id)
            return self.quant_method.apply_unsorted(banks, hidden_states, topk_weights, slots)

    def prefill_forward(
        self,
        cache: OffloadMoeCache,
        hidden_states: mx.array,
        topk_weights: mx.array,
        topk_ids: mx.array,
    ) -> mx.array:
        """An overlap prefill layer: the layer's input runs while its prefill buffer is read,
        its MoE runs as a command buffer of its own once the buffer is in."""
        method = self.quant_method
        assert isinstance(method, AffineMoEMethod), "offload experts are affine quantized"
        if self.moe_layer_id == 0:
            cache.begin_prefill_forward()
        mx.async_eval(hidden_states, topk_weights, topk_ids)
        expert_bytes = cache.expert_bytes(self.moe_layer_id)
        out = method.apply_sorted(cache.banks, hidden_states, topk_weights, topk_ids, expert_bytes)
        mx.async_eval(out)
        cache.end_prefill_layer(self.moe_layer_id, out)
        return out


__all__ = ["MoELayer", "OffloadMoELayer"]
