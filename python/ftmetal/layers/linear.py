"""Linear layers in bf16 or MLX affine quantization.

Port of FreeToken layers/linear.py. Divergences: no tensor parallelism, so
LinearReplicated / LinearRowParallel collapse into Linear and LinearColParallelMerged
into LinearMerged; the quant method is the scheme itself (bf16 matmul or
mx.quantized_matmul) instead of a kernel registry.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.config import AffineScheme


class Linear(BaseOP):
    """y = x @ W.T (+ bias). Weights are placeholders until the loader fills them."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool = False,
        *,
        scheme: AffineScheme | None = None,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.scheme = scheme
        if scheme is None:
            self.weight = mx.zeros((output_size, input_size), dtype=dtype)
        else:
            packed = input_size * scheme.bits // 32
            self.weight = mx.zeros((output_size, packed), dtype=mx.uint32)
            self.scales = mx.zeros((output_size, input_size // scheme.group_size), dtype=dtype)
            self.biases = mx.zeros((output_size, input_size // scheme.group_size), dtype=dtype)
        if has_bias:
            self.bias = mx.zeros((output_size,), dtype=dtype)

    def forward(self, x: mx.array) -> mx.array:
        if self.scheme is None:
            y = x @ self.weight.T
        else:
            y = mx.quantized_matmul(
                x,
                self.weight,
                self.scales,
                self.biases,
                transpose=True,
                group_size=self.scheme.group_size,
                bits=self.scheme.bits,
                mode="affine",
            )
        if "bias" in self:
            y = y + self.bias
        return y


class LinearMerged(Linear):
    """Several projections of one input fused along the output axis (qkv, gate|up)."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        has_bias: bool = False,
        *,
        scheme: AffineScheme | None = None,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__(input_size, sum(output_sizes), has_bias, scheme=scheme, dtype=dtype)
        self._output_sizes = tuple(output_sizes)

    @property
    def output_sizes(self) -> tuple[int, ...]:
        return self._output_sizes

    def forward_split(self, x: mx.array) -> list[mx.array]:
        """The fused projection cut back into its parts."""
        bounds = []
        offset = 0
        for size in self._output_sizes[:-1]:
            offset += size
            bounds.append(offset)
        return mx.split(self.forward(x), bounds, axis=-1)


__all__ = ["Linear", "LinearMerged"]
