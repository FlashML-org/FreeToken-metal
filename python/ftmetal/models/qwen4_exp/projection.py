"""One-token projections on kernel/qmv_rows: a decode step's GEMVs on more simdgroups.

A CommandGraph replays a decode step's dispatches back to back, so a GEMV with few
threadgroups per output row waits on memory latency; qmv_rows computes mx.quantized_matmul's
sums with one output row per simdgroup. Other shapes and schemes, and biased projections, run
Linear.forward.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ftmetal.kernel.qmv_rows import qmv_rows, supports_qmv_rows
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.layers.quantization.linear import AffineLinearMethod


def qmv_scheme(linear: Linear, x: mx.array) -> AffineScheme | None:
    """linear's affine scheme when qmv_rows takes x through it, else None."""
    method = linear.quant_method
    if not isinstance(method, AffineLinearMethod) or "bias" in linear:
        return None
    elif supports_qmv_rows(
        x, linear["weight"], linear["scales"], method.scheme.group_size, method.scheme.bits
    ):
        return method.scheme
    else:
        return None


def project(linear: Linear, x: mx.array, hc_epilogue: tuple[int, int] | None = None) -> mx.array:
    """linear.forward(x), bit for bit; hc_epilogue as qmv_rows' (a supported one-token x)."""
    if isinstance(linear, LinearMerged) and linear.part_names:
        return mx.concatenate(project_split(linear, x), axis=-1)
    else:
        scheme = qmv_scheme(linear, x)
        if scheme is None:
            assert hc_epilogue is None, "the epilogue runs in qmv_rows only"
            return linear.forward(x)
        else:
            return qmv_rows(
                x,
                linear["weight"],
                linear["scales"],
                linear["biases"],
                group_size=scheme.group_size,
                bits=scheme.bits,
                hc_epilogue=hc_epilogue,
            )


def project_split(merged: LinearMerged, x: mx.array) -> list[mx.array]:
    """merged.forward_split(x), bit for bit."""
    if merged.part_names:
        return [project(getattr(merged, name), x) for name in merged.part_names]
    else:
        bounds = tuple(np.cumsum(merged.output_sizes[:-1]).tolist())
        return project(merged, x).split(bounds, axis=-1)


__all__ = ["project", "project_split", "qmv_scheme"]
