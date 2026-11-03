"""The fused GDN decode step: conv, silu, q/k norm, gating and the gated delta recurrence for one
token per request in one C++ primitive (gdn_decode.cpp, gdn_decode.metal), updating the conv and
state pools in place.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel._ext import _C


def gdn_decode_step(
    conv_pool: mx.array,
    state_pool: mx.array,
    conv_in: mx.array,
    conv_weight: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    slots: mx.array,
    positions: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """One decode token per request, in place on the pools; returns (conv_pool, state_pool, y).

    conv_pool [S, K, conv_dim] is a ring of the last K conv inputs indexed by position
    mod K; state_pool [S, Hv, Dv, Dk] fp32; conv_in [B, conv_dim]; conv_weight
    [conv_dim, K]; a, b [B, Hv]; A_log, dt_bias [Hv]; slots, positions int32 [B].
    y is [B, Hv, Dv] in conv_in's dtype.
    """
    conv_pool, state_pool, y = _C.gdn_decode(
        conv_pool, state_pool, conv_in, conv_weight, a, b, A_log, dt_bias, slots, positions
    )
    return conv_pool, state_pool, y


__all__ = ["gdn_decode_step"]
