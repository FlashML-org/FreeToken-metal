"""One-token projections on kernel/gemm/affine/affine_qmv_rows: a decode step's GEMVs on more
simdgroups.

A CommandGraph replays a decode step's dispatches back to back, so a GEMV with few
threadgroups per output row waits on memory latency; qmv_rows computes mx.quantized_matmul's
sums with one output row per simdgroup. Other shapes and schemes, and biased projections, run
Linear.forward. Projections that read the same input fuse into one of their rows once loaded
(fuse_rows), so a decode token computes them in one GEMV; several tokens run each part's rows
(rows_forward), as they did apart.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from ftmetal.kernel.gemm.affine.affine_qmv_rows import (
    QMV_FAST_OUTPUT_TILE,
    qmv_rows,
    supports_qmv_rows,
)
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


def fuse_rows(linears: Sequence[Linear]) -> Linear | None:
    """One affine linear of the linears' output rows in turn, then zero rows up to a whole
    qmv_rows tile; None unless every one is affine in one scheme, without a bias."""
    methods = [linear.quant_method for linear in linears]
    affine = [method for method in methods if isinstance(method, AffineLinearMethod)]
    if len(affine) != len(methods) or any("bias" in linear for linear in linears):
        return None
    first = affine[0]
    if any(method.scheme != first.scheme for method in affine):
        return None
    used = sum(linear["weight"].shape[0] for linear in linears)
    rows = -(-used // QMV_FAST_OUTPUT_TILE) * QMV_FAST_OUTPUT_TILE
    dtype = linears[0]["scales"].dtype
    fused = Linear(linears[0].input_size, rows, scheme=first.scheme, dtype=dtype)
    for name in ("weight", "scales", "biases"):
        parts = [linear[name] for linear in linears]
        pad = mx.zeros((rows - used, parts[0].shape[1]), parts[0].dtype)
        fused[name] = mx.concatenate([*parts, pad])
    mx.eval(fused.parameters())
    return fused


def rows_forward(linear: Linear, x: mx.array, start: int, stop: int) -> mx.array:
    """x through the output rows start..stop of an affine linear."""
    method = linear.quant_method
    assert isinstance(method, AffineLinearMethod)
    return mx.quantized_matmul(
        x,
        linear["weight"][start:stop],
        linear["scales"][start:stop],
        linear["biases"][start:stop],
        transpose=True,
        group_size=method.scheme.group_size,
        bits=method.scheme.bits,
        mode="affine",
    )


__all__ = ["fuse_rows", "project", "project_split", "qmv_scheme", "rows_forward"]
