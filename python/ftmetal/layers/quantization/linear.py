"""Linear methods: a Linear's weight arrays for its scheme, and its product.

Port of FreeToken layers/quantization/linear/ (LinearConfig, LinearMethod and its unquantized
method). Divergences: linear_method picks the method instead of the (kind, layer kind)
registry, and a method has no kernel table or finalize step, MLX's op being each kind's only
kernel; MLX affine and float block modes take the place of FreeToken's fp8 / fp4 / GGUF kinds;
apply reads the arrays
by role from a mapping (the layer itself) rather than the layer's attributes, and the layer adds
its bias after apply, MLX's quantized matmul taking none.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import mlx.core as mx

from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.config import AffineScheme, FpBlockScheme, QuantScheme


@partial(mx.compile, shapeless=True)
def scale_output(y: mx.array, output_scale: mx.array) -> mx.array:
    """y times its float32 output scale, broadcast, in y's dtype."""
    return (y * output_scale).astype(y.dtype)


@dataclass(frozen=True, kw_only=True)
class LinearConfig:
    """What a linear method sizes and computes by: the shape, the scheme and the float dtype."""

    in_features: int
    out_features: int
    scheme: QuantScheme | None
    dtype: mx.Dtype


class LinearMethod(ABC):
    def __init__(self, cfg: LinearConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def create_weights(self, layer: BaseOP) -> None:
        """Set the layer's weight arrays: placeholders the loader replaces."""

    @abstractmethod
    def apply(self, weights: Mapping[str, mx.array], x: mx.array) -> mx.array:
        """x @ W.T."""


class UnquantizedLinearMethod(LinearMethod):
    def create_weights(self, layer: BaseOP) -> None:
        cfg = self.cfg
        layer.weight = mx.zeros((cfg.out_features, cfg.in_features), dtype=cfg.dtype)

    def apply(self, weights: Mapping[str, mx.array], x: mx.array) -> mx.array:
        return x @ weights["weight"].T


class AffineLinearMethod(LinearMethod):
    def __init__(self, cfg: LinearConfig, scheme: AffineScheme) -> None:
        """scheme: cfg's, as its kind."""
        super().__init__(cfg)
        self.scheme = scheme

    def create_weights(self, layer: BaseOP) -> None:
        cfg = self.cfg
        for role, spec in self.scheme.layout(cfg.in_features, cfg.out_features, cfg.dtype).items():
            setattr(layer, role, mx.zeros(spec.shape, dtype=spec.dtype))

    def apply(self, weights: Mapping[str, mx.array], x: mx.array) -> mx.array:
        return mx.quantized_matmul(
            x,
            weights["weight"],
            weights["scales"],
            weights["biases"],
            transpose=True,
            group_size=self.scheme.group_size,
            bits=self.scheme.bits,
            mode="affine",
        )


class FpBlockLinearMethod(LinearMethod):
    """An output_scale is one float32 per output row, [out, 1]: the parts of a fused Linear
    each bring their own."""

    def __init__(self, cfg: LinearConfig, scheme: FpBlockScheme) -> None:
        """scheme: cfg's, as its kind."""
        super().__init__(cfg)
        self.scheme = scheme

    def create_weights(self, layer: BaseOP) -> None:
        cfg = self.cfg
        for role, spec in self.scheme.layout(cfg.in_features, cfg.out_features, cfg.dtype).items():
            setattr(layer, role, mx.zeros(spec.shape, dtype=spec.dtype))
        if self.scheme.has_output_scale:
            layer.output_scale = mx.zeros((cfg.out_features, 1), dtype=mx.float32)

    def apply(self, weights: Mapping[str, mx.array], x: mx.array) -> mx.array:
        y = mx.quantized_matmul(
            x,
            weights["weight"],
            weights["scales"],
            transpose=True,
            group_size=self.scheme.group_size,
            bits=self.scheme.bits,
            mode=self.scheme.mode,
        )
        if self.scheme.has_output_scale:
            return scale_output(y, weights["output_scale"].T)
        else:
            return y


def linear_method(cfg: LinearConfig) -> LinearMethod:
    if cfg.scheme is None:
        return UnquantizedLinearMethod(cfg)
    elif isinstance(cfg.scheme, AffineScheme):
        return AffineLinearMethod(cfg, cfg.scheme)
    else:
        return FpBlockLinearMethod(cfg, cfg.scheme)


__all__ = [
    "AffineLinearMethod",
    "FpBlockLinearMethod",
    "LinearConfig",
    "LinearMethod",
    "UnquantizedLinearMethod",
    "linear_method",
    "scale_output",
]
