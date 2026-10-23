"""QuantConfig: which scheme each module of an MLX checkpoint stores its weight in.

Port of FreeToken layers/quantization (QuantConfig / QuantScheme), reduced to the two
kinds this engine serves: bf16 and MLX affine quantization. A module is affine
quantized when the checkpoint holds its scales tensor; the per-module entries of the
quantization section override the default bits and group size. A scheme is one dataclass
per kind (the kind is its type) and QuantScheme names any of them. A scheme's layout is the
arrays it stores an [output, input] weight in, which the methods of
layers/quantization/{linear,embedding,moe}.py allocate and read; FreeToken's methods and
kernels size their arrays themselves, and a method still adds an array whose shape depends
on the layer kind. BankSpec is FreeToken's (layers/quantization/moe/base.py) with its
weight's scheme added and no resident (non-bank) arrays, and describes any layout's array,
not only an expert bank's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class AffineScheme:
    bits: int
    group_size: int

    def layout(self, input_size: int, output_size: int, dtype: mx.Dtype) -> dict[str, BankSpec]:
        """The packed uint32 weight with a scale and a bias per group of inputs, by role."""
        packed = input_size * self.bits // 32
        groups = input_size // self.group_size
        return {
            "weight": BankSpec((output_size, packed), mx.uint32, self),
            "scales": BankSpec((output_size, groups), dtype, self),
            "biases": BankSpec((output_size, groups), dtype, self),
        }


QuantScheme = AffineScheme


@dataclass(frozen=True)
class BankSpec:
    """One array of a weight layout; in an expert bank it is per expert and the allocator
    prepends the expert or slot axis. quant: the scheme of its weight (None: bf16)."""

    shape: tuple[int, ...]
    dtype: mx.Dtype
    quant: QuantScheme | None = None

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.size


def affine_scheme(entry: dict[str, object]) -> AffineScheme:
    bits, group_size = entry["bits"], entry["group_size"]
    assert isinstance(bits, int) and isinstance(group_size, int), "bits and group_size are ints"
    return AffineScheme(bits, group_size)


@dataclass(frozen=True)
class QuantConfig:
    default: QuantScheme | None
    overrides: dict[str, QuantScheme]
    quantized_modules: frozenset[str]

    @classmethod
    def from_checkpoint(
        cls, quantization: dict[str, object] | None, tensor_names: list[str]
    ) -> QuantConfig:
        """Read the checkpoint's quantization section; tensor_names are checkpoint keys."""
        quantized = frozenset(
            name.removesuffix(".scales") for name in tensor_names if name.endswith(".scales")
        )
        if quantization is None:
            return cls(default=None, overrides={}, quantized_modules=quantized)
        mode = quantization.get("mode", "affine")
        if mode != "affine":
            raise ValueError(f"unsupported quantization mode {mode!r}; only affine is served")
        default = affine_scheme(quantization)
        overrides: dict[str, QuantScheme] = {}
        for key, value in quantization.items():
            if isinstance(value, dict):
                overrides[key] = affine_scheme(value)
        return cls(default=default, overrides=overrides, quantized_modules=quantized)

    def scheme_for(self, checkpoint_name: str) -> QuantScheme | None:
        """Scheme of one module by its checkpoint name; None means bf16."""
        if checkpoint_name in self.overrides:
            return self.overrides[checkpoint_name]
        if checkpoint_name in self.quantized_modules:
            assert self.default is not None, f"{checkpoint_name} has scales but no default scheme"
            return self.default
        return None

    def schemes_for(self, checkpoint_names: list[str]) -> list[QuantScheme | None]:
        """The scheme of each part of a fused module; the parts fuse only when all agree."""
        return [self.scheme_for(name) for name in checkpoint_names]


__all__ = ["AffineScheme", "BankSpec", "QuantConfig", "QuantScheme"]
