"""QuantConfig: which scheme each module of a checkpoint stores its weight in.

Port of FreeToken layers/quantization (QuantConfig / QuantScheme), reduced to the kinds this
engine serves: bf16 and MLX's quantization modes, affine (AffineScheme) and the float block
modes mxfp4, nvfp4 and mxfp8 (FpBlockScheme). A module is quantized when the checkpoint holds
its scales tensor; the per-module entries of the quantization section override the default
and are affine unless they name a mode. A ModelOpt or compressed-tensors export, rewritten by
models/quant_export.py, brings the scheme of each module it rewrote instead (from_export). A
scheme is one dataclass per kind (the kind is its type) and QuantScheme names any of them. A
scheme's layout is the arrays it stores an [output, input] weight in, which the methods of
layers/quantization/{linear,embedding,moe}.py allocate and read; FreeToken's methods and
kernels size their arrays themselves, and a method still adds an array whose shape depends
on the layer kind. BankSpec is FreeToken's (layers/quantization/moe/base.py) with its
weight's scheme added, and describes any layout's array, not only an expert bank's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

FpBlockMode = Literal["mxfp4", "nvfp4", "mxfp8"]
# Bits per element and inputs per scale of each float block mode: the only ones MLX accepts.
FP_BLOCK_SHAPES: dict[FpBlockMode, tuple[int, int]] = {
    "mxfp4": (4, 32),
    "nvfp4": (4, 16),
    "mxfp8": (8, 32),
}


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


@dataclass(frozen=True)
class FpBlockScheme:
    """MLX's float modes: e2m1 or e4m3 elements with an 8-bit scale (e8m0, or e4m3 for nvfp4)
    per 1-D block of group_size inputs, and no biases. has_output_scale: its methods add an
    output_scale, a float32 scale after the matmul (an export's per-tensor weight scale, which
    MLX's matmuls do not take)."""

    mode: FpBlockMode
    has_output_scale: bool = False

    @property
    def bits(self) -> int:
        return FP_BLOCK_SHAPES[self.mode][0]

    @property
    def group_size(self) -> int:
        return FP_BLOCK_SHAPES[self.mode][1]

    def layout(self, input_size: int, output_size: int, dtype: mx.Dtype) -> dict[str, BankSpec]:
        """The packed uint32 weight with a uint8 scale per block of inputs, by role."""
        packed = input_size * self.bits // 32
        groups = input_size // self.group_size
        return {
            "weight": BankSpec((output_size, packed), mx.uint32, self),
            "scales": BankSpec((output_size, groups), mx.uint8, self),
        }


QuantScheme = AffineScheme | FpBlockScheme


@dataclass(frozen=True)
class BankSpec:
    """One array of a weight layout; in an expert bank it is per expert and the allocator
    prepends the expert or slot axis. quant: the scheme of its weight (None: bf16); resident:
    a per-expert array an offload layer keeps itself, indexed by expert, instead of a bank
    row (an export's output_scale)."""

    shape: tuple[int, ...]
    dtype: mx.Dtype
    quant: QuantScheme | None = None
    resident: bool = False

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.size


def quant_scheme(entry: dict[str, object]) -> QuantScheme:
    """One entry of an MLX quantization section: affine unless it names a mode."""
    bits, group_size = entry["bits"], entry["group_size"]
    assert isinstance(bits, int) and isinstance(group_size, int), "bits and group_size are ints"
    match entry.get("mode", "affine"):
        case "affine":
            return AffineScheme(bits, group_size)
        case "mxfp4" | "nvfp4" | "mxfp8" as mode:
            scheme = FpBlockScheme(mode)
            if (bits, group_size) != (scheme.bits, scheme.group_size):
                raise ValueError(
                    f"{mode} stores {scheme.bits}-bit elements in groups of {scheme.group_size}, "
                    f"not {bits}-bit in groups of {group_size}"
                )
            return scheme
        case mode:
            raise ValueError(f"unsupported quantization mode {mode!r}")


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
        default = quant_scheme(quantization)
        overrides: dict[str, QuantScheme] = {}
        for key, value in quantization.items():
            if isinstance(value, dict):
                overrides[key] = quant_scheme(value)
        return cls(default=default, overrides=overrides, quantized_modules=quantized)

    @classmethod
    def from_export(cls, schemes: dict[str, QuantScheme]) -> QuantConfig:
        """The schemes of an export's rewritten modules, by checkpoint module."""
        return cls(default=None, overrides=dict(schemes), quantized_modules=frozenset(schemes))

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


__all__ = [
    "FP_BLOCK_SHAPES",
    "AffineScheme",
    "BankSpec",
    "FpBlockMode",
    "FpBlockScheme",
    "QuantConfig",
    "QuantScheme",
]
