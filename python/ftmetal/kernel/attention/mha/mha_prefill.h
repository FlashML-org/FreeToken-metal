// Paged flash-attention prefill as one primitive: the tile pass, or the split-KV
// partials pass and the merge pass with the partials in a temporary (see
// mha_prefill.metal).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ftmetal {

// Register the Metal source template of the three kernels; compiled per configuration.
void init_mha_prefill_library(const std::string& source);

// q [num_tokens, Hq, D]; k_cache / v_cache [num_pages * page_size, Hkv, D] in q's dtype;
// page_table [B, max_pages] int32; tiles [num_tiles, 4] int32 rows (request, first query
// row, query count, position of the first query), each tile at most
// 8 * num_row_groups / G queries, together covering every query row once. The caller
// picks the threadgroup's 8-row groups, the simdgroups per row group (each takes
// D / head_split columns), the key block size, whether K and V share one staging buffer,
// the column passes a key block is staged in (num_passes > 1 shares the buffer), the
// split-KV partition count and whether the GEMMs run as MetalPerformancePrimitives
// matmul2d (mpp, macOS 26) or on simdgroup matrices. A sliding_window W > 0 limits a query
// at position p to keys (p - W, p]; positions count from the page table row's first key.
// block_ends [num_tokens] int32, when given, lets each query also see the keys before its
// own entry (the end of its image span in row positions, 0 for text), the window still
// applying. sinks [Hq] in q's dtype, when given, are each query head's learned logit in the
// softmax's denominator, unscaled. Returns [num_tokens, Hq, D].
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
    int num_passes,
    int num_partitions,
    int sliding_window,
    bool mpp,
    float scale,
    const std::optional<mx::array>& block_ends = std::nullopt,
    const std::optional<mx::array>& sinks = std::nullopt,
    mx::StreamOrDevice s = {});

// flash_prefill's QSA mode, without window or block ends: each tile streams only the keys of
// its listed qsa_block_size-key QSA blocks, qsa_blocks [num_tiles, list_width] int32 block
// ids ascending (the first qsa_counts [num_tiles] int32 of them real), and a query sees a
// listed key when bit b % 32 of word b / 32 of its row of qsa_bits [num_tokens, bits_width]
// uint32 is set for the key's block b and the key is not past it.
mx::array flash_prefill_qsa(
    const mx::array& q,
    const mx::array& k_cache,
    const mx::array& v_cache,
    const mx::array& page_table,
    const mx::array& tiles,
    const mx::array& qsa_blocks,
    const mx::array& qsa_counts,
    const mx::array& qsa_bits,
    int qsa_block_size,
    int page_size,
    int num_row_groups,
    int head_split,
    int block_k,
    bool kv_shared,
    int num_passes,
    int num_partitions,
    bool mpp,
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
  int num_passes;
  int num_partitions;
  int sliding_window;
  bool mpp;
  float scale;
  bool has_block_ends;
  // 0: every key up to the query is a candidate; else flash_prefill_qsa's block size.
  int qsa_block_size;
  bool has_sinks;

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
