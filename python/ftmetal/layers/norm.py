"""RMSNorm variants on mx.fast.rms_norm.

Port of FreeToken layers/norm.py. Divergences: no fused add-residual variant (the
residual add is a separate elementwise op MLX fuses itself); GatedRMSNorm keeps the
FreeToken's norm-before-gate semantics, its gate silu or sigmoid (the sigmoid one in fp32 with
one rounding, as FreeToken's gated layernorm kernel, fused in kernel/norm/gated_norm.py).
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from ftmetal.kernel.norm.gated_norm import gated_rms_norm
from ftmetal.layers.base import BaseOP
from ftmetal.models.config import GatedDeltaOutputGate


class RMSNorm(BaseOP):
    def __init__(
        self, size: int, eps: float, dtype: mx.Dtype = mx.bfloat16, with_scale: bool = True
    ) -> None:
        """with_scale=False: no learned weight (Gemma4's v norm)."""
        super().__init__()
        self.eps = eps
        if with_scale:
            self.weight = mx.ones((size,), dtype=dtype)

    def forward(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight if "weight" in self else None, self.eps)


@partial(mx.compile, shapeless=True)
def silu_gate(x: mx.array, normed: mx.array, gate: mx.array) -> mx.array:
    gate = nn.silu(gate.astype(mx.float32))
    return (gate * normed.astype(mx.float32)).astype(x.dtype)


class GatedRMSNorm(BaseOP):
    """rmsnorm(x) * act(z), act silu or sigmoid: the GDN output norm."""

    def __init__(
        self,
        size: int,
        eps: float,
        activation: GatedDeltaOutputGate = "silu",
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = mx.ones((size,), dtype=dtype)

    def forward(self, x: mx.array, z: mx.array) -> mx.array:
        if self.activation == "silu":
            return silu_gate(x, mx.fast.rms_norm(x, self.weight, self.eps), z)
        else:
            return gated_rms_norm(x, self.weight, z, self.eps)


__all__ = ["GatedRMSNorm", "RMSNorm"]
