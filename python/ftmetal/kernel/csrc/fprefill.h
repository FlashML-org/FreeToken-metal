// Paged flash-attention prefill as one primitive: the tile pass, or the split-KV
// partials pass and the merge pass with the partials in a temporary (see
// metal/fprefill.metal).
#pragma once

#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ftmetal {

// Register the Metal source template of the three kernels; compiled per configuration.
void init_fprefill_library(const std::string& source);

// q [num_tokens, Hq, D]; k_cache / v_cache [num_pages * page_size, Hkv, D] in q's dtype;
// page_table [B, max_pages] int32; tiles [num_tiles, 4] int32 rows (request, first query
// row, query count, position of the first query), each tile at most
// 8 * num_row_groups / G queries, together covering every query row once. The caller
// picks the threadgroup's 8-row groups, the simdgroups per row group (each takes
// D / head_split columns), the key block size, whether K and V share one staging buffer
// and the split-KV partition count. A sliding_window W > 0 limits a query at position p
// to keys (p - W, p]; positions count from the page table row's first key.
// Returns [num_tokens, Hq, D].
mx::array flash_prefill(
    const mx::array& q,
    const mx::array& k_cache,
    const mx::array& v_cache,
    const mx::array& page_table,
    const mx::array& tiles,
    int page_size,
    int num_row_groups,
    int head_split,
    int block_k,
    bool kv_shared,
    int num_partitions,
    int sliding_window,
    float scale,
    mx::StreamOrDevice s = {});

struct FlashPrefillConfig {
  int head_dim;
  int group;
  int num_kv_heads;
  int page_size;
  int num_row_groups;
  int head_split;
  int block_k;
  bool kv_shared;
  int num_partitions;
  int sliding_window;
  float scale;

  bool operator==(const FlashPrefillConfig&) const = default;
};

class FlashPrefill : public mx::Primitive {
 public:
  FlashPrefill(mx::Stream stream, FlashPrefillConfig config)
      : mx::Primitive(stream), config_(config) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;

  const char* name() const override {
    return "FlashPrefill";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  FlashPrefillConfig config_;
};

} // namespace ftmetal
