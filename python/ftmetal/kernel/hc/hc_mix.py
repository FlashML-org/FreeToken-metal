"""The hyper-connection's mix: the sigmoid-weighted mean of the normed streams (fused with its
one-token up projection in kernel/hc/hc_up_mix.py).

Port of FreeToken kernel/triton/hc.py (hc_gate_mix) on mx.fast.metal_kernel; the products and
the mean stay in fp32 and round once, at the store.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import KERNEL_DIR, metal_kernel_header, metal_source
from ftmetal.kernel.custom import metal_kernel

MIX_THREADGROUP = 256

MIX_KERNEL = metal_kernel(
    name="hc_mix",
    input_names=["up", "normed"],
    output_names=["mixed"],
    source=metal_source(Path(__file__).with_name("hc_mix.metal")),
    header=metal_kernel_header(KERNEL_DIR / "common" / "norm.h"),
)


def hc_mix(up: mx.array, normed: mx.array, hc_count: int) -> mx.array:
    """The mean over the streams of sigmoid(up) * normed in fp32, both [T, hc_count * H] ->
    [T, H]."""
    num_tokens, width = normed.shape
    hidden = width // hc_count
    work = num_tokens * hidden
    (mixed,) = MIX_KERNEL(
        inputs=[up, normed],
        template=[("T", normed.dtype), ("HIDDEN", hidden), ("HC_COUNT", hc_count)],
        grid=(work, 1, 1),
        threadgroup=(min(MIX_THREADGROUP, work), 1, 1),
        output_shapes=[(num_tokens, hidden)],
        output_dtypes=[normed.dtype],
    )
    return mixed


__all__ = ["hc_mix"]
