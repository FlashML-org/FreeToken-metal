"""Linear layers whose weight arrays and product are their scheme's linear method's.

Port of FreeToken layers/linear.py. Divergences: no tensor parallelism, so
LinearReplicated / LinearRowParallel collapse into Linear and LinearColParallelMerged
into LinearMerged; the layer takes its scheme rather than the quant config and prefix it
would be looked up by; a LinearMerged whose parts the checkpoint quantizes differently runs
one Linear per part. Projections of one input that the checkpoint keeps apart fuse into one
LinearMerged of their rows once loaded (LinearMerged.fuse_rows): a token runs them as one
product, several tokens each part's rows as they ran apart. Which kernel a product runs is the
linear method's, by shape.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from ftmetal.kernel.gemm.affine.affine_qmv_rows import QMV_FAST_OUTPUT_TILE
from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.config import QuantScheme
from ftmetal.layers.quantization.linear import AffineLinearMethod, LinearConfig, linear_method

# A row-fused LinearMerged pads its rows to a multiple of this: qmv_rows' and MLX's qmv_fast
# output tile.
FUSED_ROW_TILE = QMV_FAST_OUTPUT_TILE


class Linear(BaseOP):
    """y = x @ W.T (+ bias). Weights are placeholders until the loader fills them."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool = False,
        *,
        scheme: QuantScheme | None = None,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        cfg = LinearConfig(
            in_features=input_size, out_features=output_size, scheme=scheme, dtype=dtype
        )
        self.quant_method = linear_method(cfg)
        self.quant_method.create_weights(self)
        if has_bias:
            self.bias = mx.zeros((output_size,), dtype=dtype)

    def forward(self, x: mx.array) -> mx.array:
        y = self.quant_method.apply(self, x)
        if "bias" in self:
            y = y + self.bias
        return y


class LinearMerged(Linear):
    """Several projections of one input fused along the output axis (qkv, gate|up): one
    weight when every part has the same scheme, else one Linear per part under the part's
    name (part_names: the fuse rule's parts, which the loader then leaves apart)."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        has_bias: bool = False,
        *,
        schemes: list[QuantScheme | None],
        part_names: tuple[str, ...],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        assert len(schemes) == len(output_sizes) == len(part_names), "one scheme per part"
        if len(set(schemes)) == 1:
            super().__init__(
                input_size, sum(output_sizes), has_bias, scheme=schemes[0], dtype=dtype
            )
            self.part_names: tuple[str, ...] = ()
        else:
            BaseOP.__init__(self)
            self.input_size = input_size
            self.output_size = sum(output_sizes)
            self.part_names = part_names
            for name, size, scheme in zip(part_names, output_sizes, schemes):
                setattr(self, name, Linear(input_size, size, has_bias, scheme=scheme, dtype=dtype))
        self._output_sizes = tuple(output_sizes)
        self.is_row_fused = False

    @classmethod
    def fuse_rows(cls, parts: Sequence[Linear]) -> LinearMerged | None:
        """One LinearMerged of the loaded parts' output rows in turn, then zero rows up to a
        multiple of FUSED_ROW_TILE; None unless every part is affine in one scheme, without a
        bias. A token runs the parts as one product (forward, then cut by forward_split);
        several run each part's rows alone, as they ran apart."""
        schemes = [
            part.quant_method.scheme
            for part in parts
            if isinstance(part.quant_method, AffineLinearMethod) and "bias" not in part
        ]
        if len(schemes) != len(parts) or len(set(schemes)) != 1:
            return None
        sizes = [part["weight"].shape[0] for part in parts]
        padding = -sum(sizes) % FUSED_ROW_TILE
        # Built as one more part, the zero rows, which forward_split leaves out.
        padded = [*sizes, padding] if padding else sizes
        fused = cls(
            parts[0].input_size,
            padded,
            schemes=[schemes[0]] * len(padded),
            part_names=tuple(f"rows{i}" for i in range(len(padded))),
            dtype=parts[0]["scales"].dtype,
        )
        for name in ("weight", "scales", "biases"):
            arrays = [part[name] for part in parts]
            pad = mx.zeros((padding, arrays[0].shape[1]), arrays[0].dtype)
            fused[name] = mx.concatenate([*arrays, pad])
        fused._output_sizes = tuple(sizes)
        fused.is_row_fused = True
        mx.eval(fused.parameters())
        return fused

    @property
    def output_sizes(self) -> tuple[int, ...]:
        return self._output_sizes

    def forward(self, x: mx.array) -> mx.array:
        if self.part_names:
            return mx.concatenate(self.forward_split(x), axis=-1)
        return super().forward(x)

    def forward_split(self, x: mx.array) -> list[mx.array]:
        """The fused projection cut back into its parts."""
        if self.part_names:
            return [getattr(self, name).forward(x) for name in self.part_names]
        bounds = [0]
        for size in self._output_sizes:
            bounds.append(bounds[-1] + size)
        parts = list(zip(bounds, bounds[1:]))
        if self.is_row_fused and x.size // x.shape[-1] > 1:
            # Several tokens: each part's rows alone, as they ran apart.
            method = self.quant_method
            assert isinstance(method, AffineLinearMethod)
            return [method.apply_rows(self, x, start, stop) for start, stop in parts]
        else:
            y = self.forward(x)
            return [y[..., start:stop] for start, stop in parts]


__all__ = ["Linear", "LinearMerged"]
