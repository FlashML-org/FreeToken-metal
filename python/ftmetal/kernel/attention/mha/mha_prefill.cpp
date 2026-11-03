#include "attention/mha/mha_prefill.h"

#include <cmath>
#include <stdexcept>

#include "common/metal_primitive.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace ftmetal {

namespace {

std::string mha_prefill_source;
const int SIMD_WIDTH = 32;
const int FRAGMENT_SIZE = 8;
const int MAX_THREADGROUP_THREADS = 1024;
// matmul2d's inner dimension (Q K^T: the head-dim slice, P V: the key block) and the
// P V output slices are multiples of 16.
const int MPP_GRANULE = 16;

} // namespace

void init_mha_prefill_library(const std::string& source) {
  mha_prefill_source = source;
}

namespace {

// Checks the arguments both modes share and builds the primitive; qsa_lists holds
// flash_prefill_qsa's (blocks, counts, bits) when qsa_block_size > 0.
mx::array make_flash_prefill(
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
    const std::optional<mx::array>& block_ends,
    int qsa_block_size,
    const std::vector<mx::array>& qsa_lists,
    mx::StreamOrDevice s) {
  if (q.ndim() != 3 || k_cache.ndim() != 3 || v_cache.ndim() != 3 || page_table.ndim() != 2 ||
      tiles.ndim() != 2 || tiles.shape(1) != 4) {
    throw std::invalid_argument(
        "ftmetal mha_prefill: q [T, Hq, D], k/v cache [slots, Hkv, D], page_table [B, P], "
        "tiles [num_tiles, 4]");
  }
  int num_heads = q.shape(1);
  int head_dim = q.shape(2);
  int num_kv_heads = k_cache.shape(1);
  metal_type(q.dtype(), "mha_prefill");
  if (k_cache.dtype() != q.dtype() || v_cache.dtype() != q.dtype()) {
    throw std::invalid_argument("ftmetal mha_prefill: k/v cache must be in q's dtype");
  }
  if (k_cache.shape() != v_cache.shape() || k_cache.shape(2) != head_dim) {
    throw std::invalid_argument("ftmetal mha_prefill: k/v cache shape mismatch");
  }
  if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    throw std::invalid_argument("ftmetal mha_prefill: Hq must be a multiple of Hkv");
  }
  if (head_dim % SIMD_WIDTH != 0) {
    throw std::invalid_argument("ftmetal mha_prefill: head_dim must be a multiple of 32");
  }
  if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
    throw std::invalid_argument("ftmetal mha_prefill: page size must be a power of two");
  }
  if (page_table.dtype() != mx::int32 || tiles.dtype() != mx::int32) {
    throw std::invalid_argument("ftmetal mha_prefill: page_table and tiles must be int32");
  }
  int group = num_heads / num_kv_heads;
  if (num_row_groups < 1 || head_split < 1 ||
      num_row_groups * head_split * SIMD_WIDTH > MAX_THREADGROUP_THREADS ||
      FRAGMENT_SIZE * num_row_groups < group) {
    throw std::invalid_argument(
        "ftmetal mha_prefill: a threadgroup's 8 * num_row_groups rows must hold one token's heads");
  }
  int granule = mpp ? MPP_GRANULE : FRAGMENT_SIZE;
  if (head_dim % (granule * head_split) != 0) {
    throw std::invalid_argument(
        "ftmetal mha_prefill: head_split must cut D into 8-column slices (16 with mpp)");
  }
  if (block_k < granule || block_k % granule != 0) {
    throw std::invalid_argument("ftmetal mha_prefill: block_k must be a multiple of 8 (16 with mpp)");
  }
  if (num_passes < 1 || head_split % num_passes != 0 || (num_passes > 1 && !kv_shared)) {
    throw std::invalid_argument(
        "ftmetal mha_prefill: num_passes must divide head_split, and share the K/V buffer");
  }
  if (num_partitions < 1) {
    throw std::invalid_argument("ftmetal mha_prefill: num_partitions must be >= 1");
  }
  if (sliding_window < 0) {
    throw std::invalid_argument("ftmetal mha_prefill: sliding_window must be >= 0 (0: none)");
  }
  if (block_ends && (block_ends->ndim() != 1 || block_ends->shape(0) != q.shape(0) ||
                     block_ends->dtype() != mx::int32)) {
    throw std::invalid_argument("ftmetal mha_prefill: block_ends must be int32 [num_tokens]");
  }
  auto stream = mx::to_stream(s);
  FlashPrefillConfig config{
      head_dim,
      group,
      num_kv_heads,
      page_size,
      num_row_groups,
      head_split,
      block_k,
      kv_shared,
      num_passes,
      num_partitions,
      sliding_window,
      mpp,
      scale,
      block_ends.has_value(),
      qsa_block_size};
  auto inputs = paged_attention_inputs(q, k_cache, v_cache, page_table, tiles, stream);
  if (block_ends) {
    inputs.push_back(mx::contiguous(*block_ends, false, stream));
  }
  for (const auto& list : qsa_lists) {
    inputs.push_back(mx::contiguous(list, false, stream));
  }
  return mx::array(
      q.shape(), q.dtype(), std::make_shared<FlashPrefill>(stream, config), std::move(inputs));
}

} // namespace

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
    const std::optional<mx::array>& block_ends,
    mx::StreamOrDevice s) {
  return make_flash_prefill(
      q, k_cache, v_cache, page_table, tiles, page_size, num_row_groups, head_split, block_k,
      kv_shared, num_passes, num_partitions, sliding_window, mpp, scale, block_ends, 0, {}, s);
}

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
    mx::StreamOrDevice s) {
  if (qsa_block_size < 1 || qsa_blocks.ndim() != 2 || qsa_blocks.shape(0) != tiles.shape(0) ||
      qsa_counts.ndim() != 1 || qsa_counts.shape(0) != tiles.shape(0) || qsa_bits.ndim() != 2 ||
      qsa_bits.shape(0) != q.shape(0) || qsa_blocks.dtype() != mx::int32 ||
      qsa_counts.dtype() != mx::int32 || qsa_bits.dtype() != mx::uint32) {
    throw std::invalid_argument(
        "ftmetal mha_prefill qsa: blocks [num_tiles, W] int32, counts [num_tiles] int32, bits "
        "[num_tokens, B] uint32, block size >= 1");
  }
  return make_flash_prefill(
      q, k_cache, v_cache, page_table, tiles, page_size, num_row_groups, head_split, block_k,
      kv_shared, num_passes, num_partitions, 0, mpp, scale, std::nullopt, qsa_block_size,
      {qsa_blocks, qsa_counts, qsa_bits}, s);
}

bool FlashPrefill::is_equivalent(const mx::Primitive& other) const {
  auto* o = dynamic_cast<const FlashPrefill*>(&other);
  return o != nullptr && o->config_ == config_;
}

void FlashPrefill::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("ftmetal mha_prefill: GPU only");
}

void FlashPrefill::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  check_pools_row_contiguous(inputs, "mha_prefill");
  const auto& q = inputs[0];
  auto& out = outputs[0];
  // A threadgroup reads only its own tile's query rows, all before it writes them.
  if (q.is_donatable() && q.flags().row_contiguous && q.size() == out.size()) {
    out.copy_shared_buffer(q);
  } else {
    out.set_data(mx::allocator::malloc(out.nbytes()));
  }
  int num_tiles = inputs[4].shape(0);
  if (num_tiles == 0) {
    return;
  }
  const auto& c = config_;
  std::string type = metal_type(out.dtype(), "mha_prefill");
  std::string library_name = "ftmetal_mha_prefill_" + type + "_" + std::to_string(c.head_dim) +
      "_" + std::to_string(c.group) + "_" + std::to_string(c.num_kv_heads) + "_" +
      std::to_string(c.page_size) + "_" + std::to_string(c.num_row_groups) + "_" +
      std::to_string(c.head_split) + "_" + std::to_string(c.block_k) + "_" +
      std::to_string(c.kv_shared) + "_" + std::to_string(c.num_passes) + "_" +
      std::to_string(c.sliding_window) + (c.mpp ? "_mpp" : "_simdgroup") +
      (c.has_block_ends ? "_blocks" : "") +
      (c.qsa_block_size > 0 ? "_qsa" + std::to_string(c.qsa_block_size) : "");

  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto* library = d.get_library(library_name, [&]() {
    std::string source = mha_prefill_source;
    replace_all(source, "{{T}}", type);
    replace_all(source, "{{D}}", std::to_string(c.head_dim));
    replace_all(source, "{{G}}", std::to_string(c.group));
    replace_all(source, "{{HKV}}", std::to_string(c.num_kv_heads));
    replace_all(source, "{{BS}}", std::to_string(c.page_size));
    replace_all(source, "{{NSG}}", std::to_string(c.num_row_groups * c.head_split));
    replace_all(source, "{{DS}}", std::to_string(c.head_split));
    replace_all(source, "{{BK}}", std::to_string(c.block_k));
    replace_all(source, "{{W}}", std::to_string(c.sliding_window));
    replace_all(source, "{{KV_SHARED}}", c.kv_shared ? "true" : "false");
    replace_all(source, "{{NPASS}}", std::to_string(c.num_passes));
    replace_all(source, "{{MMA}}", c.mpp ? "MMA_MPP" : "MMA_SIMDGROUP");
    replace_all(source, "{{BLOCKS}}", c.has_block_ends ? "1" : "0");
    replace_all(source, "{{QSA}}", std::to_string(c.qsa_block_size));
    return source;
  });

  auto& encoder = mx::metal::get_command_encoder(s);
  int max_pages = inputs[3].shape(1);
  float scale_log2e = c.scale * static_cast<float>(M_LOG2E);
  int rows = FRAGMENT_SIZE * c.num_row_groups;
  MTL::Size tile_group(SIMD_WIDTH * c.num_row_groups * c.head_split, 1, 1);
  // The inputs past the shared five: the block ends, or the QSA lists and their widths.
  auto set_mask_inputs = [&]() {
    if (c.has_block_ends) {
      encoder.set_input_array(inputs[5], 9);
    } else if (c.qsa_block_size > 0) {
      encoder.set_input_array(inputs[5], 10);
      encoder.set_input_array(inputs[6], 11);
      encoder.set_input_array(inputs[7], 12);
      encoder.set_bytes(static_cast<int>(inputs[5].shape(1)), 13);
      encoder.set_bytes(static_cast<int>(inputs[7].shape(1)), 14);
    }
  };
  if (c.num_partitions == 1) {
    encoder.set_compute_pipeline_state(d.get_kernel("mha_prefill", library));
    for (int i = 0; i < 5; ++i) {
      encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 5);
    encoder.set_bytes(max_pages, 7);
    encoder.set_bytes(scale_log2e, 8);
    set_mask_inputs();
    encoder.dispatch_threadgroups(MTL::Size(num_tiles, c.num_kv_heads, 1), tile_group);
    return;
  }

  auto [part_acc, part_ml] =
      split_partials({num_tiles, c.num_kv_heads, c.num_partitions, rows}, c.head_dim, encoder);

  encoder.set_compute_pipeline_state(d.get_kernel("mha_prefill_partials", library));
  for (int i = 0; i < 5; ++i) {
    encoder.set_input_array(inputs[i], i);
  }
  encoder.set_output_array(part_acc, 5);
  encoder.set_output_array(part_ml, 6);
  encoder.set_bytes(max_pages, 7);
  encoder.set_bytes(scale_log2e, 8);
  set_mask_inputs();
  encoder.dispatch_threadgroups(
      MTL::Size(num_tiles, c.num_kv_heads, c.num_partitions), tile_group);

  encoder.set_compute_pipeline_state(d.get_kernel("mha_prefill_merge", library));
  encoder.set_input_array(part_acc, 0);
  encoder.set_input_array(part_ml, 1);
  encoder.set_input_array(inputs[4], 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(c.num_partitions, 4);
  encoder.dispatch_threadgroups(
      MTL::Size(rows, c.num_kv_heads, num_tiles), MTL::Size(SIMD_WIDTH, 1, 1));
}

} // namespace ftmetal
