// Paged flash-decode (fdecode v4) as one primitive: the partials pass and the merge
// pass are encoded together, with the partials in a temporary (see metal/fdecode.metal).
#pragma once

#include <string>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ftmetal {

// Register the Metal source template of the two kernels; compiled per configuration.
void init_fdecode_library(const std::string& source);

// q [B, Hq, D]; k_cache / v_cache [num_pages * page_size, Hkv, D] in q's dtype;
// page_table [B, max_pages] int32; seq_lens [B] int32. The caller picks the grid's
// split-KV partition count, the partition rule (see PartitionRule) that gives each
// request its share of the grid, and simdgroups per threadgroup. Returns [B, Hq, D].
mx::array flash_decode(
    const mx::array& q,
    const mx::array& k_cache,
    const mx::array& v_cache,
    const mx::array& page_table,
    const mx::array& seq_lens,
    int page_size,
    int num_partitions,
    int partition_tokens,
    int min_partition_tokens,
    int max_partitions,
    int min_threadgroups,
    int num_simdgroups,
    float scale,
    mx::StreamOrDevice s = {});

// The caller's partition count for a max_seqlen_k; the kernels apply it on the GPU to
// each request's own length, capped at the grid's partition count.
struct PartitionRule {
  int partition_tokens;
  int min_partition_tokens;
  int max_partitions;
  int min_threadgroups;

  bool operator==(const PartitionRule&) const = default;
};

struct FlashDecodeConfig {
  int head_dim;
  int group;
  int num_kv_heads;
  int page_size;
  int num_partitions;
  PartitionRule rule;
  int num_simdgroups;
  float scale;
};

class FlashDecode : public mx::Primitive {
 public:
  FlashDecode(mx::Stream stream, FlashDecodeConfig config)
      : mx::Primitive(stream), config_(config) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;

  const char* name() const override {
    return "FlashDecode";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  FlashDecodeConfig config_;
};

} // namespace ftmetal
