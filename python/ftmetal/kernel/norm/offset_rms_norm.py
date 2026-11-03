"""The grouped (1 + w) RMS norm: each group of a row normed apart and scaled by (1 + w), fused
with the hyper-connection's inject of the pending branch into the residual streams (hc_norm) or
with a partial NeoX rope of attention heads (offset_rms_norm_rope).

Port of FreeToken kernel/triton/hc.py (grouped_gemma_rmsnorm, hc_combine_norm) on
mx.fast.metal_kernel; intermediates stay in fp32 and round once, at the store. Divergences: the
norm reduces as mx.fast.rms_norm's single-row kernel does and scales by (1 + w) as one product,
so it equals the fp32 op chain bit for bit; the combine runs only fused into the norm; the head
norm can fuse mx.fast.rope's partial rotation.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from ftmetal.kernel._ext import KERNEL_DIR, metal_kernel_header, metal_source
from ftmetal.kernel.custom import metal_kernel

SIMD_WIDTH = 32
# mx.fast.rms_norm's single-row kernel: 4 values a thread, up to 4096 values a row.
NORM_READS_PER_THREAD = 4
MAX_NORM_SIZE = 4096

NORM_KERNEL = metal_kernel(
    name="offset_rms_norm",
    input_names=["hyper", "branch", "inject_logits", "weight", "eps", "positions", "log2_base"],
    output_names=["normed", "hyper_out"],
    source=metal_source(Path(__file__).with_name("offset_rms_norm.metal")),
    header=metal_kernel_header(KERNEL_DIR / "common" / "norm.h"),
)


def norm_threadgroup(size: int) -> int:
    """Threads per normed row, as mx.fast.rms_norm launches them: whole simdgroups of 4 values."""
    assert size <= MAX_NORM_SIZE, size
    reads = -(-size // NORM_READS_PER_THREAD)
    return -(-reads // SIMD_WIDTH) * SIMD_WIDTH


def hc_norm(
    hyper: mx.array,
    weight: mx.array,
    eps: float,
    hc_count: int,
    branch: mx.array | None = None,
    inject_logits: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """(normed streams, streams) of hyper [T, hc_count * H], each H slice normed apart with its
    slice of weight [hc_count * H]. With branch [T, H] and the raw inject_logits [T, hc_count],
    stream s first becomes hyper_s + branch * 2 * sigmoid(logit_s / hc_count), rounded once."""
    assert (branch is None) == (inject_logits is None)
    num_tokens, width = hyper.shape
    hidden = width // hc_count
    assert hidden * hc_count == width and weight.size == width, (hyper.shape, weight.shape)
    threads = norm_threadgroup(hidden)
    is_injecting = branch is not None and inject_logits is not None
    # Inputs the kernel does not read stand in as hyper: the branch and logits without inject,
    # the positions and log2(base) without rotation.
    normed, hyper_out = NORM_KERNEL(
        inputs=[
            hyper,
            branch if branch is not None else hyper,
            inject_logits if inject_logits is not None else hyper,
            weight,
            eps,
            hyper,
            0.0,
        ],
        template=[
            ("T", hyper.dtype),
            ("SIZE", hidden),
            ("GROUPS", hc_count),
            ("THREADS", threads),
            ("INJECT", is_injecting),
            ("ROTARY_DIM", 0),
            ("ROWS_PER_POSITION", 1),
        ],
        grid=(hc_count * threads, num_tokens, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[hyper.shape, hyper.shape if is_injecting else (1,)],
        output_dtypes=[hyper.dtype, hyper.dtype],
    )
    return normed, hyper_out if is_injecting else hyper


def offset_rms_norm(
    x: mx.array, weight: mx.array, eps: float, group_size: int | None = None
) -> mx.array:
    """rmsnorm(x) * (1 + weight) in fp32 over x's last axis, which weight spans; with group_size,
    each group_size slice of it normed apart."""
    size = x.shape[-1]
    assert group_size is None or size % group_size == 0, (x.shape, group_size)
    groups = 1 if group_size is None else size // group_size
    normed, _ = hc_norm(x.reshape(-1, size), weight, eps, groups)
    return normed.reshape(x.shape)


def supports_offset_rms_norm_rope(x: mx.array, rotary_dim: int) -> bool:
    """Heads x [T, heads, D] whose D one threadgroup of offset_rms_norm_rope holds."""
    return x.ndim == 3 and 0 < rotary_dim <= x.shape[-1] <= MAX_NORM_SIZE and rotary_dim % 2 == 0


def offset_rms_norm_rope(
    x: mx.array, weight: mx.array, eps: float, positions: mx.array, rotary_dim: int, base: float
) -> mx.array:
    """offset_rms_norm of each head of x [T, heads, D] (weight [D]), then mx.fast.rope's
    non-traditional rotation of its first rotary_dim dims at positions int32 [T], bit for bit."""
    assert supports_offset_rms_norm_rope(x, rotary_dim), (x.shape, rotary_dim)
    num_tokens, num_heads, head_dim = x.shape
    rows = x.reshape(num_tokens * num_heads, head_dim)
    threads = norm_threadgroup(head_dim)
    # mx.fast.rope keeps its base as float32 and passes log2 of it as float32.
    log2_base = math.log2(np.float32(base))
    normed, _ = NORM_KERNEL(
        inputs=[rows, rows, rows, weight, eps, positions, log2_base],
        template=[
            ("T", x.dtype),
            ("SIZE", head_dim),
            ("GROUPS", 1),
            ("THREADS", threads),
            ("INJECT", False),
            ("ROTARY_DIM", rotary_dim),
            ("ROWS_PER_POSITION", num_heads),
        ],
        grid=(threads, num_tokens * num_heads, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[rows.shape, (1,)],
        output_dtypes=[x.dtype, x.dtype],
    )
    return normed.reshape(x.shape)


__all__ = [
    "MAX_NORM_SIZE",
    "SIMD_WIDTH",
    "hc_norm",
    "norm_threadgroup",
    "offset_rms_norm",
    "offset_rms_norm_rope",
    "supports_offset_rms_norm_rope",
]
