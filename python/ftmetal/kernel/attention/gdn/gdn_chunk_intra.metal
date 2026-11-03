// Grid: threadgroups (num_chunks, Hk, B) of NSG simdgroups; one k head's raw K K^T and
// Q K^T lower tiles serve its Hv / Hk v heads. Per v head writes the cumulative log
// decay gcum, Aqk = tril(Q K^T * decay) and Tb = (I + A)^-1 diag(beta) with
// A = strict_tril(beta * K K^T * decay). Rows past T count as zero with beta = 0 and
// decay 1; the last chunk's q, k, v rows are also copied zero-padded into the tails so
// the state pass reads whole chunks. buf_a holds the K and Q slabs, then A, then its
// inverse in place: 8x8 diagonal blocks by substitution, then block rows
// T_ij = -T_ii sum_k A_ik T_kj.

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
