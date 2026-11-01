"""gated_rms_norm: the Gated DeltaNet output norm rmsnorm(x) * w * sigmoid(z) in one launch.

Port of FreeToken kernel/fla/layernorm_gated.py (rms_norm_gated with norm_before_gate and the
sigmoid gate) on mx.fast.metal_kernel; fp32 throughout, rounded once at the store. Divergences:
RMS norm only, no bias; the row reduces as mx.fast.rms_norm's single-row kernel does, so the
output equals the fp32 op chain bit for bit.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel.custom import metal_kernel
from ftmetal.kernel.hyper_connection import MAX_NORM_SIZE, NORM_HEADER, norm_threadgroup

SOURCE = """
    constexpr int READS = 4;
    const uint row = threadgroup_position_in_grid.x;
    const uint first = thread_position_in_threadgroup.x * READS;
    const size_t base = size_t(row) * SIZE;
    threadgroup float local_sums[32];
    threadgroup float local_inv_mean[1];
    float values[READS];
    float sum_squares = 0;
    for (int i = 0; i < READS; ++i) {
      const uint h = first + i;
      const float value = h < SIZE ? static_cast<float>(x[base + h]) : 0;
      values[i] = value;
      sum_squares += value * value;
    }
    const float inv_rms = row_inv_rms(
        sum_squares, static_cast<float>(SIZE), eps, local_sums, local_inv_mean,
        thread_index_in_simdgroup, simdgroup_index_in_threadgroup);
    for (int i = 0; i < READS; ++i) {
      const uint h = first + i;
      if (h < SIZE) {
        const float normed = static_cast<float>(weight[h]) * (values[i] * inv_rms);
        out[base + h] = static_cast<T>(normed * precise_sigmoid(static_cast<float>(z[base + h])));
      }
    }
"""

KERNEL = metal_kernel(
    name="gated_rms_norm",
    input_names=["x", "weight", "z", "eps"],
    output_names=["out"],
    source=SOURCE,
    header=NORM_HEADER,
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
