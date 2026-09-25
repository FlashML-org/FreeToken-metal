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

import mlx.core as mx

from ftmetal.kernel.custom import metal_kernel

CHUNK_SIZE = 64
SIMD_WIDTH = 32
# Simdgroups per intra-chunk threadgroup (one chunk of one k head).
INTRA_SIMDGROUPS = 8
# Simdgroups per state-pass threadgroup; each owns 8 value columns of one head's state.
STATE_SIMDGROUPS = 4

HEADER = """
using frag = simdgroup_matrix<float, 8, 8>;

// An 8x8 fp32 fragment from a row-major device matrix of any dtype; transpose loads
// the block's transpose.
template <typename T>
inline frag load_frag(const device T* src, int stride, bool transpose) {
  simdgroup_matrix<T, 8, 8> raw;
  simdgroup_load(raw, src, stride, ulong2(0, 0), transpose);
  frag out;
  out.thread_elements()[0] = static_cast<float>(raw.thread_elements()[0]);
  out.thread_elements()[1] = static_cast<float>(raw.thread_elements()[1]);
  return out;
}
"""

# Grid: threadgroups (num_chunks, Hk, B) of NSG simdgroups; one k head's raw K K^T and
# Q K^T lower tiles serve its Hv / Hk v heads. Per v head writes the cumulative log
# decay gcum, Aqk = tril(Q K^T * decay) and Tb = (I + A)^-1 diag(beta) with
# A = strict_tril(beta * K K^T * decay). Rows past T count as zero with beta = 0 and
# decay 1; the last chunk's q, k, v rows are also copied zero-padded into the tails so
# the state pass reads whole chunks. buf_a holds the K and Q slabs, then A, then its
# inverse in place: 8x8 diagonal blocks by substitution, then block rows
# T_ij = -T_ii sum_k A_ik T_kj.
INTRA_SOURCE = """
    constexpr int NT = 32 * NSG;
    constexpr int NB = C / 8;
    constexpr int NTILE = NB * (NB + 1) / 2;
    constexpr int SLAB = 32;
    constexpr int QK_JOBS = (2 * NTILE + NSG - 1) / NSG;
    constexpr int ROW_JOBS = (NB - 1 + NSG - 1) / NSG;
    constexpr int GROUP = Hv / Hk;
    const int chunk = threadgroup_position_in_grid.x;
    const int num_chunks = threadgroups_per_grid.x;
    const int hk = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int tid = thread_position_in_threadgroup.x;
    const int sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const short fm = ((lane / 4) & 4) + ((lane / 2) % 4);
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int t0 = chunk * C;
    const bool is_tail = t0 + C > T;
    const device InT* k_rows = k + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* q_rows = q + ((size_t)b * T * Hk + hk) * Dk;

    threadgroup float g_tg[GROUP * C];
    threadgroup float beta_tg[GROUP * C];
    threadgroup float buf_a[C * C];

    // Cumulative log decay of each v head over the chunk's rows.
    for (int idx = tid; idx < GROUP * C; idx += NT) {
      const int r = idx % C;
      const int hv = hk * GROUP + idx / C;
      const int t = t0 + r;
      float log_decay = 0.0f;
      float beta_t = 0.0f;
      if (t < T) {
        log_decay = log(max(static_cast<float>(g[((size_t)b * T + t) * Hv + hv]), FLT_MIN));
        beta_t = static_cast<float>(beta[((size_t)b * T + t) * Hv + hv]);
      }
      g_tg[idx] = log_decay;
      beta_tg[idx] = beta_t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < GROUP * 2) {
      threadgroup float* g_seg = g_tg + tid * 32;
      float sum = 0.0f;
      for (int r = 0; r < 32; ++r) {
        sum += g_seg[r];
        g_seg[r] = sum;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < GROUP * C; idx += NT) {
      if (idx % C >= 32) {
        g_tg[idx] += g_tg[idx - idx % C + 31];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < GROUP * C; idx += NT) {
      const size_t head = (size_t)b * Hv + hk * GROUP + idx / C;
      gcum[head * num_chunks * C + t0 + idx % C] = g_tg[idx];
    }

    // Raw K K^T and Q K^T lower tiles, over Dk in slabs staged through buf_a.
    frag acc[QK_JOBS];
    for (int jj = 0; jj < QK_JOBS; ++jj) {
      acc[jj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    threadgroup float* k_slab = buf_a;
    threadgroup float* q_slab = buf_a + C * SLAB;
    device InT* k_tail_head = k_tail + ((size_t)b * Hk + hk) * C * Dk;
    device InT* q_tail_head = q_tail + ((size_t)b * Hk + hk) * C * Dk;
    for (int s0 = 0; s0 < Dk; s0 += SLAB) {
      for (int idx = tid; idx < C * SLAB; idx += NT) {
        const int r = idx / SLAB;
        const int c = s0 + idx % SLAB;
        InT k_val = InT(0);
        InT q_val = InT(0);
        if (t0 + r < T) {
          k_val = k_rows[(size_t)(t0 + r) * Hk * Dk + c];
          q_val = q_rows[(size_t)(t0 + r) * Hk * Dk + c];
        }
        k_slab[idx] = static_cast<float>(k_val);
        q_slab[idx] = static_cast<float>(q_val);
        if (is_tail) {
          k_tail_head[r * Dk + c] = k_val;
          q_tail_head[r * Dk + c] = q_val;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int jj = 0; jj < QK_JOBS; ++jj) {
        const int job = sg + jj * NSG;
        if (job < 2 * NTILE) {
          const int tile = job % NTILE;
          int ti = 0;
          while ((ti + 1) * (ti + 2) / 2 <= tile) {
            ++ti;
          }
          const int tj = tile - ti * (ti + 1) / 2;
          const threadgroup float* left = job >= NTILE ? q_slab : k_slab;
          for (int kk = 0; kk < SLAB; kk += 8) {
            frag a_frag;
            frag b_frag;
            simdgroup_load(a_frag, left + 8 * ti * SLAB + kk, SLAB);
            simdgroup_load(b_frag, k_slab + 8 * tj * SLAB + kk, SLAB, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc[jj], a_frag, b_frag, acc[jj]);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int member = 0; member < GROUP; ++member) {
      const int hv = hk * GROUP + member;
      const size_t head = (size_t)b * Hv + hv;
      const threadgroup float* g_head = g_tg + member * C;
      const threadgroup float* beta_head = beta_tg + member * C;
      if (is_tail) {
        const device InT* v_rows = v + ((size_t)b * T * Hv + hv) * Dv;
        device InT* v_tail_head = v_tail + head * C * Dv;
        for (int idx = tid; idx < C * Dv; idx += NT) {
          const int t = t0 + idx / Dv;
          v_tail_head[idx] = t < T ? v_rows[(size_t)t * Hv * Dv + idx % Dv] : InT(0);
        }
      }

      device float* aqk_chunk = aqk + (head * num_chunks + chunk) * C * C;
      for (int jj = 0; jj < QK_JOBS; ++jj) {
        const int job = sg + jj * NSG;
        if (job < 2 * NTILE) {
          const int tile = job % NTILE;
          int ti = 0;
          while ((ti + 1) * (ti + 2) / 2 <= tile) {
            ++ti;
          }
          const int tj = tile - ti * (ti + 1) / 2;
          const int r = 8 * ti + fm;
          for (int e = 0; e < 2; ++e) {
            const int c = 8 * tj + fn + e;
            const float scaled = acc[jj].thread_elements()[e] * exp(g_head[r] - g_head[c]);
            if (job >= NTILE) {
              aqk_chunk[r * C + c] = r >= c ? scaled : 0.0f;
            } else {
              buf_a[r * C + c] = r > c ? beta_head[r] * scaled : 0.0f;
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Diagonal blocks: thread (block, column) substitutes down its column.
      float diag[8];
      const int diag_block = 8 * (tid / 8);
      const int diag_col = tid % 8;
      if (tid < C) {
        for (int r = 0; r < 8; ++r) {
          float x = r == diag_col ? 1.0f : 0.0f;
          for (int j = 0; j < r; ++j) {
            x -= buf_a[(diag_block + r) * C + diag_block + j] * diag[j];
          }
          diag[r] = x;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < C) {
        for (int r = 0; r < 8; ++r) {
          buf_a[(diag_block + r) * C + diag_block + diag_col] = diag[r];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (int i = 1; i < NB; ++i) {
        frag row_tiles[ROW_JOBS];
        for (int jj = 0; jj < ROW_JOBS; ++jj) {
          const int j = sg + jj * NSG;
          if (j < i) {
            frag sum = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int kb = j; kb < i; ++kb) {
              frag a_frag;
              frag t_frag;
              simdgroup_load(a_frag, buf_a + 8 * i * C + 8 * kb, C);
              simdgroup_load(t_frag, buf_a + 8 * kb * C + 8 * j, C);
              simdgroup_multiply_accumulate(sum, a_frag, t_frag, sum);
            }
            frag t_diag;
            simdgroup_load(t_diag, buf_a + 8 * i * C + 8 * i, C);
            simdgroup_multiply(row_tiles[jj], t_diag, sum);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int jj = 0; jj < ROW_JOBS; ++jj) {
          const int j = sg + jj * NSG;
          if (j < i) {
            for (int e = 0; e < 2; ++e) {
              row_tiles[jj].thread_elements()[e] = -row_tiles[jj].thread_elements()[e];
            }
            simdgroup_store(row_tiles[jj], buf_a + 8 * i * C + 8 * j, C);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      device float* tb_chunk = tb + (head * num_chunks + chunk) * C * C;
      for (int idx = tid; idx < C * C; idx += NT) {
        const int c = idx % C;
        if (c / 8 <= idx / (8 * C)) {
          tb_chunk[idx] = buf_a[idx] * beta_head[c];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# Grid: threadgroups (Dv / (8 * NSG), Hv, B) of NSG simdgroups; a simdgroup keeps
# 8 columns of the state transposed, S^T [Dk, 8], as Dk / 8 fragments in registers,
# and walks the chunks in order: u = V - exp(G) K S^T, v_new = Tb u,
# o = exp(G) Q S^T + Aqk v_new, S^T = exp(G_last) S^T + K^T (exp(G_last - G) v_new).
# v_new and the partial o live in threadgroup memory: registers go to the state, and
# fewer registers mean more resident simdgroups.
STATE_SOURCE = """
    constexpr int NK = Dk / 8;
    constexpr int NR = C / 8;
    const int hv = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int hk = hv / (Hv / Hk);
    const int sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const short fm = ((lane / 4) & 4) + ((lane / 2) % 4);
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int dv0 = (threadgroup_position_in_grid.x * NSG + sg) * 8;
    const int num_chunks = (T + C - 1) / C;
    const size_t head = (size_t)b * Hv + hv;
    const device float* g_head = gcum + head * num_chunks * C;

    threadgroup float scratch[NSG * 2 * C * 8];
    threadgroup float* vn_tg = scratch + sg * 2 * C * 8;
    threadgroup float* o_tg = vn_tg + C * 8;

    frag st[NK];
    for (int kk = 0; kk < NK; ++kk) {
      simdgroup_load(st[kk], state_in + (head * Dv + dv0) * Dk + 8 * kk, Dk, ulong2(0, 0), true);
    }

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
      const int t0 = chunk * C;
      const bool is_tail = t0 + C > T;
      const int qk_stride = is_tail ? Dk : Hk * Dk;
      const int v_stride = is_tail ? Dv : Hv * Dv;
      const device InT* q_chunk = is_tail ? q_tail + ((size_t)b * Hk + hk) * C * Dk
                                          : q + (((size_t)b * T + t0) * Hk + hk) * Dk;
      const device InT* k_chunk = is_tail ? k_tail + ((size_t)b * Hk + hk) * C * Dk
                                          : k + (((size_t)b * T + t0) * Hk + hk) * Dk;
      const device InT* v_chunk = (is_tail ? v_tail + head * C * Dv
                                           : v + (((size_t)b * T + t0) * Hv + hv) * Dv) + dv0;
      const device float* tb_chunk = tb + (head * num_chunks + chunk) * C * C;
      const device float* aqk_chunk = aqk + (head * num_chunks + chunk) * C * C;

      for (int i = 0; i < NR; ++i) {
        const float row_decay = exp(g_head[t0 + 8 * i + fm]);
        frag ks = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        frag qs = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int kk = 0; kk < NK; ++kk) {
          const frag k_frag = load_frag(k_chunk + 8 * i * qk_stride + 8 * kk, qk_stride, false);
          const frag q_frag = load_frag(q_chunk + 8 * i * qk_stride + 8 * kk, qk_stride, false);
          simdgroup_multiply_accumulate(ks, k_frag, st[kk], ks);
          simdgroup_multiply_accumulate(qs, q_frag, st[kk], qs);
        }
        frag u_frag = load_frag(v_chunk + 8 * i * v_stride, v_stride, false);
        for (int e = 0; e < 2; ++e) {
          u_frag.thread_elements()[e] -= row_decay * ks.thread_elements()[e];
          qs.thread_elements()[e] *= row_decay;
        }
        simdgroup_store(u_frag, vn_tg + 64 * i, 8);
        simdgroup_store(qs, o_tg + 64 * i, 8);
      }
      // v_new = Tb u in place, bottom up: row block i reads u of blocks j <= i.
      for (int i = NR - 1; i >= 0; --i) {
        frag acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (int j = 0; j <= i; ++j) {
          frag t_frag;
          frag u_frag;
          simdgroup_load(t_frag, tb_chunk + 8 * i * C + 8 * j, C);
          simdgroup_load(u_frag, vn_tg + 64 * j, 8);
          simdgroup_multiply_accumulate(acc, t_frag, u_frag, acc);
        }
        simdgroup_store(acc, vn_tg + 64 * i, 8);
      }

      for (int i = 0; i < NR; ++i) {
        frag acc;
        simdgroup_load(acc, o_tg + 64 * i, 8);
        for (int j = 0; j <= i; ++j) {
          frag a_frag;
          frag vn_frag;
          simdgroup_load(a_frag, aqk_chunk + 8 * i * C + 8 * j, C);
          simdgroup_load(vn_frag, vn_tg + 64 * j, 8);
          simdgroup_multiply_accumulate(acc, a_frag, vn_frag, acc);
        }
        const int t = t0 + 8 * i + fm;
        if (t < T) {
          device InT* dst = y + (((size_t)b * T + t) * Hv + hv) * Dv + dv0 + fn;
          *(device vec<InT, 2>*)dst = vec<InT, 2>(
              static_cast<InT>(acc.thread_elements()[0]),
              static_cast<InT>(acc.thread_elements()[1]));
        }
      }

      const float g_last = g_head[t0 + C - 1];
      for (int i = 0; i < NR; ++i) {
        const float row_decay = exp(g_last - g_head[t0 + 8 * i + fm]);
        for (int e = 0; e < 2; ++e) {
          vn_tg[64 * i + 8 * fm + fn + e] *= row_decay;
        }
      }
      const float chunk_decay = exp(g_last);
      // Two state fragments per pass: independent MMA chains, measured faster than one
      // (and than four, which costs registers).
      for (int kk = 0; kk < NK; kk += 2) {
        for (int e = 0; e < 2; ++e) {
          st[kk].thread_elements()[e] *= chunk_decay;
          st[kk + 1].thread_elements()[e] *= chunk_decay;
        }
        for (int j = 0; j < NR; ++j) {
          const frag k_frag = load_frag(k_chunk + 8 * j * qk_stride + 8 * kk, qk_stride, true);
          const frag k_next = load_frag(k_chunk + 8 * j * qk_stride + 8 * kk + 8, qk_stride, true);
          frag vn_frag;
          simdgroup_load(vn_frag, vn_tg + 64 * j, 8);
          simdgroup_multiply_accumulate(st[kk], k_frag, vn_frag, st[kk]);
          simdgroup_multiply_accumulate(st[kk + 1], k_next, vn_frag, st[kk + 1]);
        }
      }
    }

    for (int kk = 0; kk < NK; ++kk) {
      simdgroup_store(st[kk], state_out + (head * Dv + dv0) * Dk + 8 * kk, Dk, ulong2(0, 0), true);
    }
"""

INTRA_KERNEL = metal_kernel(
    name="gated_delta_chunk_intra",
    input_names=["q", "k", "v", "g", "beta", "T"],
    output_names=["tb", "aqk", "gcum", "q_tail", "k_tail", "v_tail"],
    source=INTRA_SOURCE,
    header=HEADER,
)

STATE_KERNEL = metal_kernel(
    name="gated_delta_chunk_state",
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
