"""Gemma4 feed-forward parts: the dense / shared GeGLU MLP and the expert router.

Port of FreeToken models/gemma4/moe.py (Gemma4Router; the GatedMLP its Gemma4MLP and
Gemma4DenseMLP share). Divergences: the router's softmax, top-k and renormalization run
in the fused router_topk kernel (softmax over the top-k logits equals the full softmax
renormalized over the top k) and the per-expert scale is applied after it; the
feed-forward sandwich, its norms and the layer scalar are the decoder layer's, as in the
checkpoint's layout.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel.router import router_topk
from ftmetal.layers.activation import gelu_tanh_and_mul
from ftmetal.layers.base import BaseOP
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.models.config import ModelConfig
from ftmetal.models.gemma4.weight import GATE_UP_PARTS


class Gemma4GatedMLP(BaseOP):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        gate_up_schemes: list[AffineScheme | None],
        down_scheme: AffineScheme | None,
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
        return self.down_proj.forward(gelu_tanh_and_mul(self.gate_up_proj.forward(x)))


class Gemma4Router(BaseOP):
    """Unscaled RMSNorm times scale / sqrt(hidden), projection, top-k softmax, times the
    chosen experts' scales."""

    def __init__(
        self, config: ModelConfig, *, proj_scheme: AffineScheme | None, dtype: mx.Dtype
    ) -> None:
        super().__init__()
        self.eps = config.rms_norm_eps
        self.top_k = config.num_experts_per_tok
        self.root_size = config.hidden_size**-0.5
        self.proj = Linear(config.hidden_size, config.num_experts, scheme=proj_scheme, dtype=dtype)
        self.scale = mx.ones((config.hidden_size,), dtype=dtype)
        self.per_expert_scale = mx.ones((config.num_experts,), dtype=dtype)

    def forward(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """x [T, H] -> (expert ids int32 [T, top_k], weights [T, top_k])."""
        normed = mx.fast.rms_norm(x, self.scale * self.root_size, self.eps)
        topk_ids, topk_weights = router_topk(self.proj.forward(normed), self.top_k, True)
        return topk_ids, topk_weights * self.per_expert_scale[topk_ids]


__all__ = ["Gemma4GatedMLP", "Gemma4Router"]
