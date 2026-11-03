"""GDN prefill recurrence kernels head to head at the Qwen3.6 shapes (Hk 16, Hv 32, D 128).

ours chunked   = ftmetal kernel/attention/gdn/gdn_chunk.gated_delta_chunked (WY, C=64)
ours per-token = ftmetal kernel/attention/gdn/gdn_prefill.gated_delta_recurrence
                 (mlx-lm's kernel, tg y=4)
mlx-lm         = mlx_lm.models.gated_delta.gated_delta_kernel
fork           = FreeToken-shreyvishen kernel/metal/gdn.py gdn_step, ported: tg y=32,
                 state [Hv, Dk, Dv], log-decay input
splash         = splash runtime/metal/kernels/prefill/gdn.metal prefill_gdn_scan, ported:
                 16 state rows per 128-thread threadgroup, 16 columns per lane, 16-token
                 blocks staged in threadgroup memory with register prefetch
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel

from ftmetal.kernel.attention.gdn.gdn_chunk import gated_delta_chunked
from ftmetal.kernel.attention.gdn.gdn_prefill import gated_delta_recurrence
from ftmetal.kernel.custom import metal_kernel

HK, HV, D = 16, 32, 128
TOKENS_PER_RUN = 32768
MIN_CALLS = 4

FORK_SOURCE = """
  const uint n = thread_position_in_grid.z;
  const uint hv_idx = n % HV;
  const uint hk_idx = hv_idx / (HV / HK);
  const uint dk_lane = thread_position_in_threadgroup.x;
  const uint dv_idx = thread_position_in_grid.y;
  constexpr int N_PER_T = DK / 32;

  auto q_ = q + hk_idx * DK;
  auto k_ = k + hk_idx * DK;
  auto v_ = v + hv_idx * DV;
  auto y_ = y + hv_idx * DV;
  auto g_ = g;
  auto beta_ = beta;
  device const float* st_in = state_in + hv_idx * DK * DV + dv_idx;
  device float* st_out = state_out + hv_idx * DK * DV + dv_idx;

  float s[N_PER_T];
  for (int i = 0; i < N_PER_T; ++i) s[i] = st_in[(N_PER_T * dk_lane + i) * DV];

  for (uint t = 0; t < T; ++t) {
    float decay = exp(g_[hv_idx]);
    float kv_mem = 0.0f;
    for (int i = 0; i < N_PER_T; ++i) {
      uint s_idx = N_PER_T * dk_lane + i;
      s[i] *= decay;
      kv_mem += s[i] * static_cast<float>(k_[s_idx]);
    }
    kv_mem = simd_sum(kv_mem);
    float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_[hv_idx];
    float out = 0.0f;
    for (int i = 0; i < N_PER_T; ++i) {
      uint s_idx = N_PER_T * dk_lane + i;
      s[i] += static_cast<float>(k_[s_idx]) * delta;
      out += s[i] * static_cast<float>(q_[s_idx]);
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) y_[dv_idx] = static_cast<InT>(out);
    q_ += HK * DK; k_ += HK * DK; v_ += HV * DV; y_ += HV * DV; g_ += HV; beta_ += HV;
  }
  for (int i = 0; i < N_PER_T; ++i) st_out[(N_PER_T * dk_lane + i) * DV] = s[i];
"""

SPLASH_SOURCE = """
  constexpr uint Columns = 16;
  constexpr uint LanesPerRow = HD / Columns;
  constexpr uint Rows = 16;
  constexpr uint Threads = 128;
  constexpr uint Block = 16;
  constexpr uint HeadsPerKey = VH / KH;
  constexpr uint GroupsPerHead = HD / Rows;
  const uint tokens = T;
  threadgroup float keys[Block * HD];
  threadgroup float queries[Block * HD];
  threadgroup float values[Block * Rows];
  threadgroup float gates[2 * Block];
  const uint group = threadgroup_position_in_grid.x;
  const uint thread_index = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint simd_group = simdgroup_index_in_threadgroup;

  const uint value_head = group / GroupsPerHead;
  const uint key_head = value_head / HeadsPerKey;
  const uint row_base = (group % GroupsPerHead) * Rows;
  const uint row = simd_group * (32 / LanesPerRow) + lane / LanesPerRow;
  const uint column = (lane % LanesPerRow) * Columns;
  const ulong state_base = (ulong(value_head) * HD + row_base + row) * HD + column;

  float4 state[Columns / 4];
  for (uint c = 0; c < Columns / 4; ++c)
    state[c] = reinterpret_cast<device const float4*>(state_in + state_base)[c];

  float4 key_fetch[2][2] = {}, query_fetch[2][2] = {}, value_fetch[2] = {};
  float decay_fetch = 0.0f, beta_fetch = 0.0f;
  auto fetch = [&](uint start) {
    for (uint n = 0; n < 2; ++n) {
      const uint element = (n * Threads + thread_index) * 8;
      const uint token = start + element / HD;
      if (token < tokens) {
        const ulong base = (ulong(token) * KH + key_head) * HD + element % HD;
        for (uint h = 0; h < 2; ++h) {
          key_fetch[n][h] = float4(float(k[base + 4 * h]), float(k[base + 4 * h + 1]),
                                   float(k[base + 4 * h + 2]), float(k[base + 4 * h + 3]));
          query_fetch[n][h] = float4(float(q[base + 4 * h]), float(q[base + 4 * h + 1]),
                                     float(q[base + 4 * h + 2]), float(q[base + 4 * h + 3]));
        }
      }
    }
    if (thread_index < 32) {
      const uint element = thread_index * 8;
      const uint token = start + element / Rows;
      if (token < tokens) {
        const ulong base = (ulong(token) * VH + value_head) * HD + row_base + element % Rows;
        for (uint h = 0; h < 2; ++h)
          value_fetch[h] = float4(float(v[base + 4 * h]), float(v[base + 4 * h + 1]),
                                  float(v[base + 4 * h + 2]), float(v[base + 4 * h + 3]));
      }
    }
    if (thread_index < Block && start + thread_index < tokens) {
      const uint gate = (start + thread_index) * VH + value_head;
      decay_fetch = decay[gate];
      beta_fetch = beta[gate];
    }
  };
  auto stage = [&]() {
    for (uint n = 0; n < 2; ++n) {
      const uint element = (n * Threads + thread_index) * 8;
      for (uint h = 0; h < 2; ++h) {
        reinterpret_cast<threadgroup float4*>(keys + element)[h] = key_fetch[n][h];
        reinterpret_cast<threadgroup float4*>(queries + element)[h] = query_fetch[n][h];
      }
    }
    if (thread_index < 32) {
      for (uint h = 0; h < 2; ++h)
        reinterpret_cast<threadgroup float4*>(values + thread_index * 8)[h] = value_fetch[h];
    }
    if (thread_index < Block) {
      gates[thread_index] = decay_fetch;
      gates[Block + thread_index] = beta_fetch;
    }
  };

  fetch(0);
  stage();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint start = 0; start < tokens; start += Block) {
    const bool more = start + Block < tokens;
    if (more) fetch(start + Block);
    const uint count = min(Block, tokens - start);
    for (uint t = 0; t < count; ++t) {
      const float d = gates[t], b = gates[Block + t];
      const threadgroup float4* key =
          reinterpret_cast<const threadgroup float4*>(keys + t * HD + column);
      const threadgroup float4* query =
          reinterpret_cast<const threadgroup float4*>(queries + t * HD + column);
      float4 partial = 0.0f;
      for (uint c = 0; c < Columns / 4; ++c) {
        state[c] *= d;
        partial = fma(state[c], key[c], partial);
      }
      float memory = (partial.x + partial.y) + (partial.z + partial.w);
      for (uint s = 1; s < LanesPerRow; s <<= 1) memory += simd_shuffle_xor(memory, s);
      const float delta = (values[t * Rows + row] - memory) * b;
      partial = 0.0f;
      for (uint c = 0; c < Columns / 4; ++c) {
        state[c] = fma(key[c], delta, state[c]);
        partial = fma(state[c], query[c], partial);
      }
      float result = (partial.x + partial.y) + (partial.z + partial.w);
      for (uint s = 1; s < LanesPerRow; s <<= 1) result += simd_shuffle_xor(result, s);
      if (lane % LanesPerRow == 0)
        output[(ulong(start + t) * VH + value_head) * HD + row_base + row] =
            static_cast<InT>(result);
    }
    if (more) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      stage();
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint c = 0; c < Columns / 4; ++c)
    reinterpret_cast<device float4*>(state_out + state_base)[c] = state[c];
"""

fork_kernel = metal_kernel(
    name="fork_gdn_step",
    input_names=["q", "k", "v", "g", "beta", "state_in"],
    output_names=["y", "state_out"],
    source=FORK_SOURCE,
)
splash_kernel = metal_kernel(
    name="splash_gdn_scan",
    input_names=["q", "k", "v", "decay", "beta", "state_in"],
    output_names=["output", "state_out"],
    source=SPLASH_SOURCE,
)


def fork(q, k, v, g, beta, state):
    """state [1, Hv, Dv, Dk] in, same out; the kernel works on [Hv, Dk, Dv] and log-decay."""
    t = q.shape[1]
    st = mx.contiguous(state[0].transpose(0, 2, 1))
    y, st_out = fork_kernel(
        inputs=[q, k, v, mx.log(g), beta, st],
        template=[("InT", q.dtype), ("DK", D), ("DV", D), ("HK", HK), ("HV", HV), ("T", t)],
        grid=(32, D, HV),
        threadgroup=(32, 32, 1),
        output_shapes=[(t, HV, D), (HV, D, D)],
        output_dtypes=[q.dtype, mx.float32],
    )
    return y[None], st_out.transpose(0, 2, 1)[None]


def splash(q, k, v, g, beta, state):
    t = q.shape[1]
    y, st_out = splash_kernel(
        inputs=[q, k, v, g.astype(mx.float32), beta.astype(mx.float32), state[0]],
        template=[("InT", q.dtype), ("KH", HK), ("VH", HV), ("HD", D), ("T", t)],
        grid=(HV * D // 16 * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(t, HV, D), (HV, D, D)],
        output_dtypes=[q.dtype, mx.float32],
    )
    return y[None], st_out[None]


def mlxlm(q, k, v, g, beta, state):
    return gated_delta_kernel(q, k, v, g, beta, state)


RULES = {
    "ours chunked": gated_delta_chunked,
    "ours per-token": gated_delta_recurrence,
    "mlx-lm": mlxlm,
    "fork": fork,
    "splash": splash,
}


def make_inputs(t):
    mx.random.seed(t)
    q = mx.random.normal((1, t, HK, D)) * D**-0.5
    k = mx.random.normal((1, t, HK, D))
    k = k / mx.linalg.norm(k, axis=-1, keepdims=True)
    v = mx.random.normal((1, t, HV, D))
    g = mx.exp(-mx.random.uniform(0.0, 0.5, (1, t, HV)))
    beta = mx.sigmoid(mx.random.normal((1, t, HV)))
    q, k, v = (x.astype(mx.bfloat16) for x in (q, k, v))
    mx.eval(q, k, v, g, beta)
    return q, k, v, g, beta


def time_per_call(rule, inputs, calls, repeats):
    q, k, v, g, beta = inputs
    state = mx.zeros((1, HV, D, D), dtype=mx.float32)
    mx.eval(rule(q, k, v, g, beta, state))
    best = float("inf")
    for _ in range(repeats):
        outputs = []
        started = time.perf_counter()
        for _ in range(calls):
            y, state = rule(q, k, v, g + 0.0 * state[:, None, :, 0, 0], beta, state)
            outputs.append(y)
        mx.eval(outputs, state)
        best = min(best, (time.perf_counter() - started) / calls)
    return best


def check(t=200):
    q, k, v, g, beta = make_inputs(t)
    state = mx.random.normal((1, HV, D, D)) * 0.1
    ref_y, ref_s = gated_delta_recurrence(q, k, v, g, beta, state)
    mx.eval(ref_y, ref_s)
    for name, rule in RULES.items():
        y, s = rule(q, k, v, g, beta, state)
        mx.eval(y, s)
        dy = mx.abs(y.astype(mx.float32) - ref_y.astype(mx.float32)).max().item()
        ds = mx.abs(s - ref_s).max().item()
        print(f"check {name:15s} max|dy| {dy:.3e} max|ds| {ds:.3e}", flush=True)


def main():
    check()
    if "--check-only" in sys.argv:
        return
    lengths = (64, 512, 2048, 8192)
    print("T      " + "".join(f"{n:>16s}" for n in RULES))
    for t in lengths:
        inputs = make_inputs(t)
        calls = max(MIN_CALLS, TOKENS_PER_RUN // t)
        row = []
        for rule in RULES.values():
            row.append(time_per_call(rule, inputs, calls, 5) * 1e6)
        print(f"{t:<7d}" + "".join(f"{x:>13.0f} us" for x in row), flush=True)


if __name__ == "__main__":
    main()
