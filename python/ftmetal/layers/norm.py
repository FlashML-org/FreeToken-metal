"""RMSNorm variants on mx.fast.rms_norm.

Port of FreeToken layers/norm.py. Divergences: no fused add-residual variant (the
residual add is a separate elementwise op MLX fuses itself); GatedRMSNorm keeps the
donor's norm-before-gate silu semantics.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from ftmetal.layers.base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((size,), dtype=dtype)

    def forward(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


@partial(mx.compile, shapeless=True)
def silu_gate(x: mx.array, normed: mx.array, gate: mx.array) -> mx.array:
    gate = nn.silu(gate.astype(mx.float32))
    return (gate * normed.astype(mx.float32)).astype(x.dtype)


class GatedRMSNorm(BaseOP):
    """rmsnorm(x) * silu(z): the GDN output norm."""

    def __init__(self, size: int, eps: float, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((size,), dtype=dtype)

    def forward(self, x: mx.array, z: mx.array) -> mx.array:
        return silu_gate(x, mx.fast.rms_norm(x, self.weight, self.eps), z)


__all__ = ["GatedRMSNorm", "RMSNorm"]
