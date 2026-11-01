"""Sparse MoE block of the Qwen3.5 families: softmax top-k router, routed experts, a shared
expert gated by sigmoid(shared_expert_gate(x)).

Port of FreeToken models/qwen3_5_moe/moe.py, shared by Qwen3.5 / 3.6 and Qwen3.8's
qwen4_exp (FreeToken's Qwen4ExpMoE subclasses it unchanged). Divergences: the router is the fused
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
from ftmetal.layers.quantization.config import QuantScheme
from ftmetal.layers.quantization.moe import ExpertSchemes
from ftmetal.models.config import ModelConfig

# Checkpoint module paths of the fused gate_up projection's parts.
GATE_UP_PARTS = ("gate_proj", "up_proj")


class SharedExpert(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        gate_up_schemes: list[QuantScheme | None],
        down_scheme: QuantScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.gate_up_proj = LinearMerged(
            hidden_size,
            [intermediate_size, intermediate_size],
            schemes=gate_up_schemes,
            part_names=GATE_UP_PARTS,
            dtype=dtype,
        )
        self.down_proj = Linear(intermediate_size, hidden_size, scheme=down_scheme, dtype=dtype)

    def forward(self, x: mx.array) -> mx.array:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class SharedExpertMoE(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        experts_schemes: ExpertSchemes,
        gate_scheme: QuantScheme | None,
        shared_gate_up_schemes: list[QuantScheme | None],
        shared_down_scheme: QuantScheme | None,
        shared_gate_scheme: QuantScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
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
                schemes=experts_schemes,
                dtype=dtype,
            )
        else:
            self.experts = MoELayer(
                config.num_experts,
                config.num_experts_per_tok,
                config.hidden_size,
                config.moe_intermediate_size,
                schemes=experts_schemes,
                dtype=dtype,
            )
        self.shared_expert = SharedExpert(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            gate_up_schemes=shared_gate_up_schemes,
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


__all__ = ["GATE_UP_PARTS", "SharedExpert", "SharedExpertMoE"]
