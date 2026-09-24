"""Gated activations and gated adds as compiled elementwise leaves.

Port of FreeToken layers/activation.py; only silu is served. The compiled leaf takes
the halves separately because a shapeless compile cannot split.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    """silu(gate) * up in fp32, cast back to the input dtype."""
    return (nn.silu(gate.astype(mx.float32)) * up.astype(mx.float32)).astype(gate.dtype)


def silu_and_mul(gate_up: mx.array) -> mx.array:
    """silu(gate) * up over the two halves of the last axis."""
    gate, up = mx.split(gate_up, 2, axis=-1)
    return silu_mul(gate, up)


@partial(mx.compile, shapeless=True)
def sigmoid_gate(x: mx.array, gate: mx.array) -> mx.array:
    """x * sigmoid(gate), the gate broadcast over the last axis."""
    return (x.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))).astype(x.dtype)


@partial(mx.compile, shapeless=True)
def sigmoid_gated_add(base: mx.array, x: mx.array, gate: mx.array) -> mx.array:
    """base + x * sigmoid(gate) in one launch."""
    out = base.astype(mx.float32) + x.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))
    return out.astype(base.dtype)


__all__ = ["sigmoid_gate", "sigmoid_gated_add", "silu_and_mul", "silu_mul"]
