"""gated_rms_norm: the Gated DeltaNet output norm rmsnorm(x) * w * sigmoid(z) in one launch.

Port of FreeToken kernel/fla/layernorm_gated.py (rms_norm_gated with norm_before_gate and the
sigmoid gate) on mx.fast.metal_kernel; fp32 throughout, rounded once at the store. Divergences:
RMS norm only, no bias; the row reduces as mx.fast.rms_norm's single-row kernel does, so the
output equals the fp32 op chain bit for bit.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import KERNEL_DIR, metal_kernel_header, metal_source
from ftmetal.kernel.custom import metal_kernel
from ftmetal.kernel.norm.offset_rms_norm import MAX_NORM_SIZE, norm_threadgroup

SOURCE = metal_source(Path(__file__).with_name("gated_norm.metal"))

KERNEL = metal_kernel(
    name="gated_norm",
    input_names=["x", "weight", "z", "eps"],
    output_names=["out"],
    source=SOURCE,
    header=metal_kernel_header(KERNEL_DIR / "common" / "norm.h"),
)


def supports_gated_rms_norm(x: mx.array, z: mx.array) -> bool:
    """Rows one threadgroup holds, z of x's shape."""
    return x.shape[-1] <= MAX_NORM_SIZE and z.shape == x.shape


def gated_rms_norm(x: mx.array, weight: mx.array, z: mx.array, eps: float) -> mx.array:
    """rmsnorm(x) * weight * sigmoid(z) over x's last axis in fp32, cast to x's dtype once
    (supports_gated_rms_norm)."""
    assert supports_gated_rms_norm(x, z), (x.shape, z.shape)
    size = x.shape[-1]
    rows = x.size // size
    threads = norm_threadgroup(size)
    (out,) = KERNEL(
        inputs=[x, weight, z, eps],
        template=[("T", x.dtype), ("SIZE", size)],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


__all__ = ["gated_rms_norm", "supports_gated_rms_norm"]
