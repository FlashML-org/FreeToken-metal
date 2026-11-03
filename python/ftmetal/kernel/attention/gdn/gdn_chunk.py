"""Chunked gated delta rule for prefill: the WY form over chunks of CHUNK_SIZE tokens.

Port of FreeToken kernel/fla/chunk.py (chunk_gated_delta_rule_fwd). Divergences: two
launches instead of four. The intra-chunk pass (one k head for all its v heads) fuses
the gate cumsum, the beta * K K^T and Q K^T tiles and the blocked inverse of I + A,
and emits Tb = (I + A)^-1 diag(beta) instead of w and u; the state pass forms
v_new = Tb (V - exp(G) K S^T) itself and fuses the output (chunk_o) into the
chunk-sequential state update (chunk_delta_h), so neither w, u nor per-chunk states
are stored. Takes the per-step decay g (not its log) so it is a drop-in for
gated_delta_recurrence; everything past the inputs is fp32.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ftmetal.kernel._ext import metal_kernel_header, metal_source
from ftmetal.kernel.custom import metal_kernel

CHUNK_SIZE = 64
SIMD_WIDTH = 32
# Simdgroups per intra-chunk threadgroup (one chunk of one k head).
INTRA_SIMDGROUPS = 8
# Simdgroups per state-pass threadgroup; each owns 8 value columns of one head's state.
# Registers and threadgroup memory cap the resident simdgroups: 2 measured faster than 1
# or 4 (T = 2048 state pass 1.95 ms vs 1.98 / 2.33 ms on the M4 Pro).
STATE_SIMDGROUPS = 2

HEADER = metal_kernel_header(Path(__file__).with_name("gdn_chunk.h"))

INTRA_SOURCE = metal_source(Path(__file__).with_name("gdn_chunk_intra.metal"))

STATE_SOURCE = metal_source(Path(__file__).with_name("gdn_chunk_state.metal"))

INTRA_KERNEL = metal_kernel(
    name="gdn_chunk_intra",
    input_names=["q", "k", "v", "g", "beta", "T"],
    output_names=["tb", "aqk", "gcum", "q_tail", "k_tail", "v_tail"],
    source=INTRA_SOURCE,
    header=HEADER,
)

STATE_KERNEL = metal_kernel(
    name="gdn_chunk_state",
    input_names=["q", "k", "v", "q_tail", "k_tail", "v_tail", "tb", "aqk", "gcum", "state_in", "T"],
    output_names=["y", "state_out"],
    source=STATE_SOURCE,
    header=HEADER,
)


def gated_delta_chunked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array]:
    """Same contract as gated_delta_recurrence without a mask, for an fp32 state.

    q, k [B, T, Hk, Dk]; v [B, T, Hv, Dv]; g (per-step decay), beta [B, T, Hv];
    state [B, Hv, Dv, Dk]. Returns y [B, T, Hv, Dv] in q's dtype and the final state.
    """
    batch_size, seq_len, num_k_heads, key_dim = k.shape
    num_v_heads, value_dim = v.shape[2:]
    assert key_dim % SIMD_WIDTH == 0 and value_dim % SIMD_WIDTH == 0
    assert q.dtype == k.dtype == v.dtype and state.dtype == mx.float32
    num_chunks = -(-seq_len // CHUNK_SIZE)
    padded_len = num_chunks * CHUNK_SIZE
    template: list[tuple[str, mx.Dtype | int]] = [
        ("InT", q.dtype),
        ("Dk", key_dim),
        ("Dv", value_dim),
        ("Hk", num_k_heads),
        ("Hv", num_v_heads),
        ("C", CHUNK_SIZE),
    ]
    chunk_matrix = (batch_size, num_v_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE)
    tb, aqk, gcum, q_tail, k_tail, v_tail = INTRA_KERNEL(
        inputs=[q, k, v, g, beta, seq_len],
        template=[*template, ("NSG", INTRA_SIMDGROUPS)],
        grid=(num_chunks * SIMD_WIDTH * INTRA_SIMDGROUPS, num_k_heads, batch_size),
        threadgroup=(SIMD_WIDTH * INTRA_SIMDGROUPS, 1, 1),
        output_shapes=[
            chunk_matrix,
            chunk_matrix,
            (batch_size, num_v_heads, padded_len),
            (batch_size, num_k_heads, CHUNK_SIZE, key_dim),
            (batch_size, num_k_heads, CHUNK_SIZE, key_dim),
            (batch_size, num_v_heads, CHUNK_SIZE, value_dim),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32, q.dtype, q.dtype, q.dtype],
    )
    simdgroups = min(STATE_SIMDGROUPS, value_dim // 8)
    y, state_out = STATE_KERNEL(
        inputs=[q, k, v, q_tail, k_tail, v_tail, tb, aqk, gcum, state, seq_len],
        template=[*template, ("NSG", simdgroups)],
        grid=(SIMD_WIDTH * value_dim // 8, num_v_heads, batch_size),
        threadgroup=(SIMD_WIDTH * simdgroups, 1, 1),
        output_shapes=[(batch_size, seq_len, num_v_heads, value_dim), state.shape],
        output_dtypes=[q.dtype, mx.float32],
    )
    return y, state_out


__all__ = ["CHUNK_SIZE", "gated_delta_chunked"]
