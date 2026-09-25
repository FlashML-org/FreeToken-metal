"""Gated delta rule kernels: prefill inputs, the [B, T] recurrence and the fused decode step.

The recurrence is adapted from mlx-lm's gated_delta kernel (Apple, MIT); it reads and
writes fresh state arrays and the GDN layer scatters the result into the pool. The
prefill inputs kernel fuses conv, silu, q/k norm and gating over one request's chunk
in the decode step's fp32 math. The decode step (gdn_decode_step, a C++ primitive)
fuses the same with the recurrence for one token per request and updates the pools
in place.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.kernel._ext import _C
from ftmetal.kernel.custom import MetalKernel, metal_kernel

# Threads per simdgroup; each lane owns key_head_dim / 32 state columns.
SIMD_WIDTH = 32
THREADGROUP_ROWS = 4


# The prefill inputs kernel: tokens per simdgroup (its conv window slides over them in
# registers; 8 and 16 measured equal at T = 2048 on the M4 Pro, 32 slower) and
# simdgroups per threadgroup.
PREFILL_TOKENS_PER_SIMDGROUP = 16
PREFILL_SIMDGROUPS = 4

# Grid: threadgroups (conv_dim / (D * NSG), ceil(T / TPS)); a simdgroup owns one head
# row of D channels (a q, k or v head, D / 32 per lane) for TPS tokens: causal conv
# over the taps and the chunk, silu, then the rms norm and scale of q and k heads. A v
# head's simdgroup also writes the decay and beta of its TPS tokens, one per lane.
# conv_in, taps, a and b are read through their strides (views of the projections).
PREFILL_INPUTS_SOURCE = """
    // The q/k norm epsilon of the decode step.
    constexpr float EPS = 1e-6f;
    constexpr int CPL = D / 32;
    const int row = threadgroup_position_in_grid.x * NSG + simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const int t_begin = threadgroup_position_in_grid.y * TPS;
    const int t_end = min(t_begin + TPS, T);
    const int c0 = row * D + lane * CPL;

    // window[j] holds the input at t - (K - 1) + j; positions before the chunk are taps.
    float weight[K][CPL];
    float window[K][CPL];
    for (int i = 0; i < CPL; ++i) {
      const int c = c0 + i;
      for (int j = 0; j < K; ++j) {
        weight[j][i] = static_cast<float>(conv_weight[c * K + j]);
      }
      for (int j = 0; j < K - 1; ++j) {
        const int t = t_begin - (K - 1) + j;
        window[j][i] = static_cast<float>(
            t < 0 ? taps[(K - 1 + t) * taps_strides[0] + c * taps_strides[1]]
                  : conv_in[t * conv_in_strides[0] + c * conv_in_strides[1]]);
      }
    }
    const float inv_scale = rsqrt(float(D));
    const float norm_scale = row < Hk ? inv_scale * inv_scale : inv_scale;
    for (int t = t_begin; t < t_end; ++t) {
      float value[CPL];
      float sum_sq = 0.0f;
      for (int i = 0; i < CPL; ++i) {
        window[K - 1][i] =
            static_cast<float>(conv_in[t * conv_in_strides[0] + (c0 + i) * conv_in_strides[1]]);
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
          acc += weight[j][i] * window[j][i];
        }
        value[i] = acc / (1.0f + exp(-acc));
        sum_sq += value[i] * value[i];
        for (int j = 0; j < K - 1; ++j) {
          window[j][i] = window[j + 1][i];
        }
      }
      if (row < 2 * Hk) {
        const float scale = rsqrt(simd_sum(sum_sq) / D + EPS) * norm_scale;
        device InT* dst = (row < Hk ? q : k) + ((size_t)t * Hk + row % Hk) * D + lane * CPL;
        for (int i = 0; i < CPL; ++i) {
          dst[i] = static_cast<InT>(value[i] * scale);
        }
      } else {
        device InT* dst = v + ((size_t)t * Hv + row - 2 * Hk) * D + lane * CPL;
        for (int i = 0; i < CPL; ++i) {
          dst[i] = static_cast<InT>(value[i]);
        }
      }
    }

    if (row >= 2 * Hk && t_begin + lane < t_end) {
      const int hv = row - 2 * Hk;
      const int t = t_begin + lane;
      const float x = static_cast<float>(a[t * a_strides[0] + hv * a_strides[1]])
                    + static_cast<float>(dt_bias[hv]);
      const float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
      g[(size_t)t * Hv + hv] = exp(-exp(static_cast<float>(A_log[hv])) * softplus);
      beta[(size_t)t * Hv + hv] =
          1.0f / (1.0f + exp(-static_cast<float>(b[t * b_strides[0] + hv * b_strides[1]])));
    }
"""

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
    return metal_kernel(
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


__all__ = ["gated_delta_recurrence", "gdn_decode_step", "gdn_prefill_inputs"]
