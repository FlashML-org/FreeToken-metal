// Grid: threadgroups (Dv / (8 * NSG), Hv, B) of NSG simdgroups; a simdgroup keeps
// 8 columns of the state transposed, S^T [Dk, 8], as Dk / 8 fragments in registers,
// and walks the chunks in order: u = V - exp(G) K S^T, v_new = Tb u,
// o = exp(G) Q S^T + Aqk v_new, S^T = exp(G_last) S^T + K^T (exp(G_last - G) v_new).
// v_new lives in threadgroup memory and o is formed row block by row block: registers
// go to the state and the less of both a simdgroup holds, the more of the grid is
// resident at once (all of it takes 2 waves at the Qwen3.6 shapes otherwise).

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

threadgroup float scratch[NSG * C * 8];
threadgroup float* vn_tg = scratch + sg * C * 8;

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
    for (int kk = 0; kk < NK; ++kk) {
      const frag k_frag = load_frag(k_chunk + 8 * i * qk_stride + 8 * kk, qk_stride, false);
      simdgroup_multiply_accumulate(ks, k_frag, st[kk], ks);
    }
    frag u_frag = load_frag(v_chunk + 8 * i * v_stride, v_stride, false);
    for (int e = 0; e < 2; ++e) {
      u_frag.thread_elements()[e] -= row_decay * ks.thread_elements()[e];
    }
    simdgroup_store(u_frag, vn_tg + 64 * i, 8);
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
    const float row_decay = exp(g_head[t0 + 8 * i + fm]);
    frag acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int kk = 0; kk < NK; ++kk) {
      const frag q_frag = load_frag(q_chunk + 8 * i * qk_stride + 8 * kk, qk_stride, false);
      simdgroup_multiply_accumulate(acc, q_frag, st[kk], acc);
    }
    for (int e = 0; e < 2; ++e) {
      acc.thread_elements()[e] *= row_decay;
    }
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
