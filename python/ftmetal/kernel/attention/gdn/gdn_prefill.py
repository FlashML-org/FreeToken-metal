"""Gated delta rule prefill kernels: the recurrence inputs and the [B, T] recurrence.

The recurrence is adapted from mlx-lm's gated_delta kernel (Apple, MIT); it reads and
writes fresh state arrays and the GDN layer scatters the result into the pool. The
prefill inputs kernel fuses conv, silu, q/k norm and gating over one request's chunk
in the decode step's fp32 math (kernel/attention/gdn/gdn_decode.py).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import metal_source
from ftmetal.kernel.custom import MetalKernel, metal_kernel

# Threads per simdgroup; each lane owns key_head_dim / 32 state columns.
SIMD_WIDTH = 32
THREADGROUP_ROWS = 4


# The prefill inputs kernel: tokens per simdgroup (its conv window slides over them in
# registers; 8 and 16 measured equal at T = 2048 on the M4 Pro, 32 slower) and
# simdgroups per threadgroup.
PREFILL_TOKENS_PER_SIMDGROUP = 16
PREFILL_SIMDGROUPS = 4

PREFILL_INPUTS_SOURCE = metal_source(Path(__file__).with_name("gdn_prefill_inputs.metal"))

PREFILL_INPUTS_KERNEL = metal_kernel(
    name="gdn_prefill_inputs",
    input_names=["conv_in", "taps", "conv_weight", "a", "b", "A_log", "dt_bias", "T"],
    output_names=["q", "k", "v", "g", "beta"],
    source=PREFILL_INPUTS_SOURCE,
    ensure_row_contiguous=False,
)


def gdn_prefill_inputs(
    conv_in: mx.array,
    taps: mx.array,
    conv_weight: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    num_k_heads: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """One request's chunk of conv inputs -> the recurrence inputs q, k, v, g, beta.

    conv_in [T, conv_dim] and a, b [T, Hv] may be strided views of the projections;
    taps [K - 1, conv_dim] are the conv inputs just before the chunk, oldest first;
    conv_weight [conv_dim, K] and A_log, dt_bias [Hv] are row-contiguous. Returns q, k
    [1, T, Hk, D] and v [1, T, Hv, D] in conv_in's dtype, the per-step decay g and
    beta [1, T, Hv] in fp32.
    """
    seq_len, conv_dim = conv_in.shape
    num_v_heads = a.shape[1]
    kernel_size = conv_weight.shape[1]
    head_dim = conv_dim // (2 * num_k_heads + num_v_heads)
    assert head_dim * (2 * num_k_heads + num_v_heads) == conv_dim, "equal q, k, v head dims"
    assert head_dim % SIMD_WIDTH == 0 and taps.shape == (kernel_size - 1, conv_dim)
    num_rows = 2 * num_k_heads + num_v_heads
    assert num_rows % PREFILL_SIMDGROUPS == 0 and PREFILL_TOKENS_PER_SIMDGROUP <= SIMD_WIDTH
    tokens = PREFILL_TOKENS_PER_SIMDGROUP
    q, k, v, g, beta = PREFILL_INPUTS_KERNEL(
        inputs=[conv_in, taps, conv_weight, a, b, A_log, dt_bias, seq_len],
        template=[
            ("InT", conv_in.dtype),
            ("Hk", num_k_heads),
            ("Hv", num_v_heads),
            ("D", head_dim),
            ("K", kernel_size),
            ("TPS", tokens),
            ("NSG", PREFILL_SIMDGROUPS),
        ],
        grid=(SIMD_WIDTH * num_rows, -(-seq_len // tokens), 1),
        threadgroup=(SIMD_WIDTH * PREFILL_SIMDGROUPS, 1, 1),
        output_shapes=[
            (1, seq_len, num_k_heads, head_dim),
            (1, seq_len, num_k_heads, head_dim),
            (1, seq_len, num_v_heads, head_dim),
            (1, seq_len, num_v_heads),
            (1, seq_len, num_v_heads),
        ],
        output_dtypes=[conv_in.dtype, conv_in.dtype, conv_in.dtype, mx.float32, mx.float32],
    )
    return q, k, v, g, beta


def make_gated_delta_kernel(has_mask: bool) -> MetalKernel:
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    source = metal_source(Path(__file__).with_name("gdn_prefill_recurrence.metal"))
    source = source.replace("{{MASK}}", mask_source)
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")
    return metal_kernel(
        name="gdn_prefill_recurrence" + ("_mask" if has_mask else ""),
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


GATED_DELTA_KERNEL = make_gated_delta_kernel(has_mask=False)
GATED_DELTA_KERNEL_MASKED = make_gated_delta_kernel(has_mask=True)


def gated_delta_recurrence(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """q, k [B, T, Hk, Dk]; v [B, T, Hv, Dv]; g, beta [B, T, Hv]; state [B, Hv, Dv, Dk].

    Returns y [B, T, Hv, Dv] and the state after the last (unmasked) step.
    """
    batch_size, seq_len, num_k_heads, key_dim = k.shape
    num_v_heads, value_dim = v.shape[2:]
    assert key_dim % SIMD_WIDTH == 0, "key_head_dim must be a multiple of 32"
    kernel = GATED_DELTA_KERNEL if mask is None else GATED_DELTA_KERNEL_MASKED
    inputs = [q, k, v, g, beta, state, seq_len]
    if mask is not None:
        inputs.append(mask)
    y, state_out = kernel(
        inputs=inputs,
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", key_dim),
            ("Dv", value_dim),
            ("Hk", num_k_heads),
            ("Hv", num_v_heads),
        ],
        grid=(SIMD_WIDTH, value_dim, batch_size * num_v_heads),
        threadgroup=(SIMD_WIDTH, THREADGROUP_ROWS, 1),
        output_shapes=[(batch_size, seq_len, num_v_heads, value_dim), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
    return y, state_out


__all__ = ["gated_delta_recurrence", "gdn_prefill_inputs"]
