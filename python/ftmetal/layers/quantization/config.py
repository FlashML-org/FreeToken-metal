"""QuantConfig: which scheme each module of an MLX checkpoint stores its weight in.

Port of FreeToken layers/quantization (QuantConfig / QuantScheme), reduced to the two
kinds this engine serves: bf16 and MLX affine quantization. A module is affine
quantized when the checkpoint holds its scales tensor; the per-module entries of the
quantization section override the default bits and group size.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AffineScheme:
    bits: int
    group_size: int


def affine_scheme(entry: dict[str, object]) -> AffineScheme:
    bits, group_size = entry["bits"], entry["group_size"]
    assert isinstance(bits, int) and isinstance(group_size, int), "bits and group_size are ints"
    return AffineScheme(bits, group_size)


@dataclass(frozen=True)
class QuantConfig:
    default: AffineScheme | None
    overrides: dict[str, AffineScheme]
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
        overrides: dict[str, AffineScheme] = {}
        for key, value in quantization.items():
            if isinstance(value, dict):
                overrides[key] = affine_scheme(value)
        return cls(default=default, overrides=overrides, quantized_modules=quantized)

    def scheme_for(self, checkpoint_name: str) -> AffineScheme | None:
        """Scheme of one module by its checkpoint name; None means bf16."""
        if checkpoint_name in self.overrides:
            return self.overrides[checkpoint_name]
        if checkpoint_name in self.quantized_modules:
            assert self.default is not None, f"{checkpoint_name} has scales but no default scheme"
            return self.default
        return None

    def scheme_for_fused(self, checkpoint_names: list[str]) -> AffineScheme | None:
        """One scheme for modules fused into a single weight; they must agree."""
        schemes = {self.scheme_for(name) for name in checkpoint_names}
        if len(schemes) != 1:
            raise ValueError(f"fused modules {checkpoint_names} mix schemes {schemes}")
        return schemes.pop()


__all__ = ["AffineScheme", "QuantConfig"]
