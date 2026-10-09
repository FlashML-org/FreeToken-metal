"""Qwen3.5 sparse MoE block: softmax top-k router, routed experts, gated shared expert.

Port of FreeToken models/qwen3_5_moe/moe.py. Divergences: the router is the fused
kernel/router Metal kernel (fp32 softmax, top-k, renorm in one launch) and the
experts are the resident MoELayer, or OffloadMoELayer under the offload strategy (FreeToken's
make_moe_layer choice, made here).
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel.router import router_topk
from ftmetal.layers.activation import sigmoid_gated_add, silu_and_mul
from ftmetal.layers.base import BaseOP
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.moe import MoELayer, OffloadMoELayer
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.models.config import ModelConfig


class SharedExpert(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        gate_up_scheme: AffineScheme | None,
        down_scheme: AffineScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.gate_up_proj = LinearMerged(
            hidden_size, [intermediate_size, intermediate_size], scheme=gate_up_scheme, dtype=dtype
        )
        self.down_proj = Linear(intermediate_size, hidden_size, scheme=down_scheme, dtype=dtype)

    def forward(self, x: mx.array) -> mx.array:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5MoE(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        experts_gate_up_scheme: AffineScheme | None,
        experts_down_scheme: AffineScheme | None,
        gate_scheme: AffineScheme | None,
        shared_gate_up_scheme: AffineScheme | None,
        shared_down_scheme: AffineScheme | None,
        shared_gate_scheme: AffineScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        assert experts_gate_up_scheme == experts_down_scheme, "expert banks share one scheme"
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = Linear(config.hidden_size, config.num_experts, scheme=gate_scheme, dtype=dtype)
        self.experts: MoELayer
        if config.moe_strategy == "offload":
            # Every decoder layer routes experts, so the MoE layer index is the layer's.
            self.experts = OffloadMoELayer(
                layer_id,
                config.num_experts,
                config.num_experts_per_tok,
                config.hidden_size,
                config.moe_intermediate_size,
                scheme=experts_gate_up_scheme,
                dtype=dtype,
            )
        else:
            self.experts = MoELayer(
                config.num_experts,
                config.num_experts_per_tok,
                config.hidden_size,
                config.moe_intermediate_size,
                scheme=experts_gate_up_scheme,
                dtype=dtype,
            )
        self.shared_expert = SharedExpert(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            gate_up_scheme=shared_gate_up_scheme,
            down_scheme=shared_down_scheme,
            dtype=dtype,
        )
        self.shared_expert_gate = Linear(
            config.hidden_size, 1, scheme=shared_gate_scheme, dtype=dtype
        )

    def forward(self, hidden_states: mx.array) -> mx.array:
        topk_ids, topk_weights = router_topk(
            self.gate.forward(hidden_states), self.top_k, self.norm_topk_prob
        )
        routed = self.experts.forward(hidden_states, topk_weights, topk_ids)
        shared = self.shared_expert.forward(hidden_states)
        return sigmoid_gated_add(routed, shared, self.shared_expert_gate.forward(hidden_states))


__all__ = ["Qwen3_5MoE", "SharedExpert"]
