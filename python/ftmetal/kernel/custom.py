"""mx.fast.metal_kernel with its real call signature; the MLX stub types the result as object."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import mlx.core as mx

TemplateValue = mx.Dtype | int | bool


class MetalKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: Sequence[mx.array | int | float],
        template: Sequence[tuple[str, TemplateValue]] | None = None,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: Sequence[tuple[int, ...]],
        output_dtypes: Sequence[mx.Dtype],
        init_value: float | None = None,
        verbose: bool = False,
    ) -> list[mx.array]: ...


def metal_kernel(
    name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    source: str,
    header: str = "",
    ensure_row_contiguous: bool = True,
) -> MetalKernel:
    """A jit-compiled custom Metal kernel, callable with keyword arguments.

    With ensure_row_contiguous False the inputs are bound as they are and the source
    indexes them through their <name>_strides.
    """
    kernel: MetalKernel = mx.fast.metal_kernel(  # pyright: ignore[reportAssignmentType]  (stub says object)
        name=name,
        input_names=input_names,
        output_names=output_names,
        source=source,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
    )
    return kernel


__all__ = ["MetalKernel", "TemplateValue", "metal_kernel"]
