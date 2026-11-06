"""Sparse MoE block of the Qwen3.5 families: softmax top-k router, routed experts, a shared
expert gated by sigmoid(shared_expert_gate(x)).

Port of FreeToken models/qwen3_5_moe/moe.py, shared by Qwen3.5 / 3.6 and Qwen3.8's
qwen4_exp (FreeToken's Qwen4ExpMoE subclasses it unchanged). Divergences: the router is the fused
kernel/moe/moe_topk Metal kernel (fp32 softmax, top-k, renorm in one launch) and the
experts are the resident MoELayer, or OffloadMoELayer under the offload strategy (FreeToken's
make_moe_layer choice, made here), whose forward adds the gated shared expert, so that an offload
decode runs it while its absent experts are read. Once loaded, an affine router gate and shared
expert gate of one scheme are one projection of their rows (fuse_gates): a decode token computes
both in one GEMV (the shared gate's one row alone would take a whole launch), the router reading
its experts' columns in place.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from ftmetal.kernel.moe.moe_topk import moe_topk
from ftmetal.layers.activation import silu_and_mul
from ftmetal.layers.base import BaseOP
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.moe import MoELayer, OffloadMoELayer
from ftmetal.layers.projection import fuse_rows, project, qmv_scheme, rows_forward
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
        return project(self.down_proj, silu_and_mul(project(self.gate_up_proj, x)))


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
        self.num_experts = config.num_experts
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

    def fuse_gates(self) -> None:
        """Replace the router gate and the shared expert gate, when both are affine in one scheme,
        by one projection of their rows (fuse_rows)."""
        fused = fuse_rows([self.gate, self.shared_expert_gate])
        if fused is not None:
            self.gates = fused
            self.pop("gate")
            self.pop("shared_expert_gate")

    def predict_routing(self, hidden_states: mx.array) -> mx.array:
        """The experts this block's router picks for hidden_states, int32 [T, top_k]: an offload
        prefill's prediction of this layer's routing from the layer before's MoE input."""
        if "gates" not in self:
            logits = project(self.gate, hidden_states)
        else:
            logits = rows_forward(self.gates, hidden_states, 0, self.num_experts)
        return moe_topk(logits, self.top_k, self.norm_topk_prob)[0]

    def forward(self, hidden_states: mx.array) -> mx.array:
        num_experts = self.num_experts
        shared_gate: mx.array | None = None
        if "gates" not in self:
            topk_ids, topk_weights = moe_topk(
                project(self.gate, hidden_states), self.top_k, self.norm_topk_prob
            )
        elif qmv_scheme(self.gates, hidden_states) is not None:
            logits = project(self.gates, hidden_states)
            topk_ids, topk_weights = moe_topk(
                logits, self.top_k, self.norm_topk_prob, num_experts=num_experts
            )
            shared_gate = logits[:, num_experts : num_experts + 1]
        else:
            router_logits = rows_forward(self.gates, hidden_states, 0, num_experts)
            topk_ids, topk_weights = moe_topk(router_logits, self.top_k, self.norm_topk_prob)
            shared_gate = rows_forward(self.gates, hidden_states, num_experts, num_experts + 1)

        def gated_shared_expert(x: mx.array) -> tuple[mx.array, mx.array]:
            """The shared expert's output and its gate logit, which the experts add gated."""
            gate = project(self.shared_expert_gate, x) if shared_gate is None else shared_gate
            return self.shared_expert.forward(x), gate

        return self.experts.forward(hidden_states, topk_weights, topk_ids, gated_shared_expert)


def offload_routers(model: nn.Module) -> dict[int, Callable[[mx.array], mx.array]]:
    """Each offloaded MoE layer's router (SharedExpertMoE.predict_routing), by its index among
    the model's MoE layers: OffloadMoeCache.set_routers' input."""
    return {
        block.experts.moe_layer_id: block.predict_routing
        for _, block in model.named_modules()
        if isinstance(block, SharedExpertMoE) and isinstance(block.experts, OffloadMoELayer)
    }


__all__ = ["GATE_UP_PARTS", "SharedExpert", "SharedExpertMoE", "offload_routers"]
