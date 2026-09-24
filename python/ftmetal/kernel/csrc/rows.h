// In-place row writes into preallocated pools (KV pages, GDN state slots,
// the token pool). Every op returns the pool array itself: the output aliases
// the input buffer, so MLX orders readers after the write while the pool is
// never copied. Callers rebind, pool = op(pool, ...).
#pragma once

#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ftmetal {

// Register the Metal source of the row kernels; compiled on first use.
void init_library(const std::string& source);

// pool[dst_ids[i]] = rows[i]
mx::array scatter_rows(
    const mx::array& pool,
    const mx::array& rows,
    const mx::array& dst_ids,
    mx::StreamOrDevice s = {});

// pool[dst_ids[i]] = pool[src_ids[i]]
mx::array copy_rows(
    const mx::array& pool,
    const mx::array& src_ids,
    const mx::array& dst_ids,
    mx::StreamOrDevice s = {});

// pool[ids[i]] = 0
mx::array zero_rows(
    const mx::array& pool,
    const mx::array& ids,
    mx::StreamOrDevice s = {});

// k_cache[out_loc[i]] = k[i]; v_cache[out_loc[i]] = v[i]; returns (k_cache, v_cache)
std::vector<mx::array> kv_write(
    const mx::array& k_cache,
    const mx::array& v_cache,
    const mx::array& k,
    const mx::array& v,
    const mx::array& out_loc,
    mx::StreamOrDevice s = {});

enum class RowsKind { Scatter, Copy, Zero, KVWrite };

class InPlaceRows : public mx::Primitive {
 public:
  InPlaceRows(mx::Stream stream, RowsKind kind)
      : mx::Primitive(stream), kind_(kind) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override;

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  void encode(
      const mx::array& pool_out,
      const mx::array* rows,
      const mx::array* src_ids,
      const mx::array& dst_ids);

  RowsKind kind_;
};

} // namespace ftmetal
