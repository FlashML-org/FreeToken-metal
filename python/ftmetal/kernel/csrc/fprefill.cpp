#include "fprefill.h"

#include <cmath>
#include <stdexcept>

#include "metal_primitive.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace ftmetal {

namespace {

std::string fprefill_source;
const int SIMD_WIDTH = 32;
const int FRAGMENT_SIZE = 8;
const int MAX_THREADGROUP_THREADS = 1024;

} // namespace

void init_fprefill_library(const std::string& source) {
  fprefill_source = source;
}

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
    mx::StreamOrDevice s) {
  if (q.ndim() != 3 || k_cache.ndim() != 3 || v_cache.ndim() != 3 || page_table.ndim() != 2 ||
      tiles.ndim() != 2 || tiles.shape(1) != 4) {
    throw std::invalid_argument(
        "ftmetal fprefill: q [T, Hq, D], k/v cache [slots, Hkv, D], page_table [B, P], "
        "tiles [num_tiles, 4]");
  }
  int num_heads = q.shape(1);
  int head_dim = q.shape(2);
  int num_kv_heads = k_cache.shape(1);
  metal_type(q.dtype(), "fprefill");
  if (k_cache.dtype() != q.dtype() || v_cache.dtype() != q.dtype()) {
    throw std::invalid_argument("ftmetal fprefill: k/v cache must be in q's dtype");
  }
  if (k_cache.shape() != v_cache.shape() || k_cache.shape(2) != head_dim) {
    throw std::invalid_argument("ftmetal fprefill: k/v cache shape mismatch");
  }
  if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    throw std::invalid_argument("ftmetal fprefill: Hq must be a multiple of Hkv");
  }
  if (head_dim % SIMD_WIDTH != 0) {
    throw std::invalid_argument("ftmetal fprefill: head_dim must be a multiple of 32");
  }
  if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
    throw std::invalid_argument("ftmetal fprefill: page size must be a power of two");
  }
  if (page_table.dtype() != mx::int32 || tiles.dtype() != mx::int32) {
    throw std::invalid_argument("ftmetal fprefill: page_table and tiles must be int32");
  }
  int group = num_heads / num_kv_heads;
  if (num_row_groups < 1 || head_split < 1 ||
      num_row_groups * head_split * SIMD_WIDTH > MAX_THREADGROUP_THREADS ||
      FRAGMENT_SIZE * num_row_groups < group) {
    throw std::invalid_argument(
        "ftmetal fprefill: a threadgroup's 8 * num_row_groups rows must hold one token's heads");
  }
  if (head_dim % (FRAGMENT_SIZE * head_split) != 0) {
    throw std::invalid_argument("ftmetal fprefill: head_split must cut D into 8-column slices");
  }
  if (block_k < FRAGMENT_SIZE || block_k % FRAGMENT_SIZE != 0) {
    throw std::invalid_argument("ftmetal fprefill: block_k must be a multiple of 8");
  }
  if (num_partitions < 1) {
    throw std::invalid_argument("ftmetal fprefill: num_partitions must be >= 1");
  }
  if (sliding_window < 0) {
    throw std::invalid_argument("ftmetal fprefill: sliding_window must be >= 0 (0: none)");
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
      num_partitions,
      sliding_window,
      scale};
  return mx::array(
      q.shape(),
      q.dtype(),
      std::make_shared<FlashPrefill>(stream, config),
      paged_attention_inputs(q, k_cache, v_cache, page_table, tiles, stream));
}

bool FlashPrefill::is_equivalent(const mx::Primitive& other) const {
  auto* o = dynamic_cast<const FlashPrefill*>(&other);
  return o != nullptr && o->config_ == config_;
}

void FlashPrefill::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("ftmetal fprefill: GPU only");
}

void FlashPrefill::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  check_pools_row_contiguous(inputs, "fprefill");
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
  std::string type = metal_type(out.dtype(), "fprefill");
  std::string library_name = "ftmetal_fprefill_" + type + "_" + std::to_string(c.head_dim) +
      "_" + std::to_string(c.group) + "_" + std::to_string(c.num_kv_heads) + "_" +
      std::to_string(c.page_size) + "_" + std::to_string(c.num_row_groups) + "_" +
      std::to_string(c.head_split) + "_" + std::to_string(c.block_k) + "_" +
      std::to_string(c.kv_shared) + "_" + std::to_string(c.sliding_window);

  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto* library = d.get_library(library_name, [&]() {
    std::string source = fprefill_source;
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
    return source;
  });

  auto& encoder = mx::metal::get_command_encoder(s);
  int max_pages = inputs[3].shape(1);
  float scale_log2e = c.scale * static_cast<float>(M_LOG2E);
  int rows = FRAGMENT_SIZE * c.num_row_groups;
  MTL::Size tile_group(SIMD_WIDTH * c.num_row_groups * c.head_split, 1, 1);
  if (c.num_partitions == 1) {
    encoder.set_compute_pipeline_state(d.get_kernel("fprefill", library));
    for (int i = 0; i < 5; ++i) {
      encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 5);
    encoder.set_bytes(max_pages, 7);
    encoder.set_bytes(scale_log2e, 8);
    encoder.dispatch_threadgroups(MTL::Size(num_tiles, c.num_kv_heads, 1), tile_group);
    return;
  }

  auto [part_acc, part_ml] =
      split_partials({num_tiles, c.num_kv_heads, c.num_partitions, rows}, c.head_dim, encoder);

  encoder.set_compute_pipeline_state(d.get_kernel("fprefill_partials", library));
  for (int i = 0; i < 5; ++i) {
    encoder.set_input_array(inputs[i], i);
  }
  encoder.set_output_array(part_acc, 5);
  encoder.set_output_array(part_ml, 6);
  encoder.set_bytes(max_pages, 7);
  encoder.set_bytes(scale_log2e, 8);
  encoder.dispatch_threadgroups(
      MTL::Size(num_tiles, c.num_kv_heads, c.num_partitions), tile_group);

  encoder.set_compute_pipeline_state(d.get_kernel("fprefill_merge", library));
  encoder.set_input_array(part_acc, 0);
  encoder.set_input_array(part_ml, 1);
  encoder.set_input_array(inputs[4], 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(c.num_partitions, 4);
  encoder.dispatch_threadgroups(
      MTL::Size(rows, c.num_kv_heads, num_tiles), MTL::Size(SIMD_WIDTH, 1, 1));
}

} // namespace ftmetal
