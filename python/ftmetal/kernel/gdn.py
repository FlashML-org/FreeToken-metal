"""Gated delta rule kernels: the [B, T] recurrence for prefill and the fused decode step.

The recurrence is adapted from mlx-lm's gated_delta kernel (Apple, MIT); it reads and
writes fresh state arrays and the GDN layer scatters the result into the pool. The
decode step (gdn_decode_step, a C++ primitive) fuses conv, gating, q/k norm and the
recurrence for one token per request and updates the pools in place.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from ftmetal.kernel._ext import _C

# Threads per simdgroup; each lane owns key_head_dim / 32 state columns.
SIMD_WIDTH = 32
THREADGROUP_ROWS = 4


@partial(mx.compile, shapeless=True)
def compute_gate_decay(A_log: mx.array, a: mx.array, dt_bias: mx.array) -> mx.array:
    """Per-step decay exp(-exp(A_log) * softplus(a + dt_bias)), fp32."""
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


def make_gated_delta_kernel(has_mask: bool):
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")
    return mx.fast.metal_kernel(
        name="gated_delta_step" + ("_mask" if has_mask else ""),
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
    conv_pool, state_pool, y = _C.gdn_step(
        conv_pool, state_pool, conv_in, conv_weight, a, b, A_log, dt_bias, slots, positions
    )
    return conv_pool, state_pool, y


__all__ = ["compute_gate_decay", "gated_delta_recurrence", "gdn_decode_step"]
