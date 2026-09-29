#include "fdecode.h"

#include <cmath>
#include <stdexcept>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace ftmetal {

namespace {

std::string fdecode_source;
const int SIMD_WIDTH = 32;
// From this head dim up 16-bit inputs accumulate in half: fp32 accumulators cost
// occupancy and ran 2x slower at D=256 on the M4 Pro.
const int HALF_ACCUMULATOR_HEAD_DIM = 256;

std::string metal_type(mx::Dtype dtype) {
  switch (dtype) {
    case mx::float16:
      return "half";
    case mx::bfloat16:
      return "bfloat";
    case mx::float32:
      return "float";
    default:
      throw std::invalid_argument(
          "ftmetal fdecode: unsupported dtype " + mx::type_to_name(dtype));
  }
}

void replace_all(std::string& text, const std::string& token, const std::string& value) {
  for (size_t pos = text.find(token); pos != std::string::npos; pos = text.find(token, pos)) {
    text.replace(pos, token.size(), value);
    pos += value.size();
  }
}

} // namespace

void init_fdecode_library(const std::string& source) {
  fdecode_source = source;
}

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
    mx::StreamOrDevice s) {
  if (q.ndim() != 3 || k_cache.ndim() != 3 || v_cache.ndim() != 3 || page_table.ndim() != 2 ||
      seq_lens.ndim() != 1) {
    throw std::invalid_argument(
        "ftmetal fdecode: q [B, Hq, D], k/v cache [slots, Hkv, D], page_table [B, P], seq_lens [B]");
  }
  int batch = q.shape(0);
  int num_heads = q.shape(1);
  int head_dim = q.shape(2);
  int num_kv_heads = k_cache.shape(1);
  metal_type(q.dtype());
  if (k_cache.dtype() != q.dtype() || v_cache.dtype() != q.dtype()) {
    throw std::invalid_argument("ftmetal fdecode: k/v cache must be in q's dtype");
  }
  if (k_cache.shape() != v_cache.shape() || k_cache.shape(2) != head_dim) {
    throw std::invalid_argument("ftmetal fdecode: k/v cache shape mismatch");
  }
  if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    throw std::invalid_argument("ftmetal fdecode: Hq must be a multiple of Hkv");
  }
  if (head_dim % (2 * SIMD_WIDTH) != 0) {
    throw std::invalid_argument("ftmetal fdecode: head_dim must be a multiple of 64");
  }
  if (page_size <= 0 || (page_size & (page_size - 1)) != 0) {
    throw std::invalid_argument("ftmetal fdecode: page size must be a power of two");
  }
  if (page_table.dtype() != mx::int32 || seq_lens.dtype() != mx::int32) {
    throw std::invalid_argument("ftmetal fdecode: page_table and seq_lens must be int32");
  }
  if (page_table.shape(0) != batch || seq_lens.shape(0) != batch) {
    throw std::invalid_argument("ftmetal fdecode: page_table and seq_lens must have B rows");
  }
  if (num_partitions < 1 || num_simdgroups < 1) {
    throw std::invalid_argument("ftmetal fdecode: partitions and simdgroups must be >= 1");
  }
  if (partition_tokens < 1 || min_partition_tokens < 1 || max_partitions < 1) {
    throw std::invalid_argument("ftmetal fdecode: partition rule values must be >= 1");
  }
  auto stream = mx::to_stream(s);
  FlashDecodeConfig config{
      head_dim,
      num_heads / num_kv_heads,
      num_kv_heads,
      page_size,
      num_partitions,
      {partition_tokens, min_partition_tokens, max_partitions, min_threadgroups},
      num_simdgroups,
      scale};
  std::vector<mx::array> inputs = {q, k_cache, v_cache, page_table, seq_lens};
  for (auto& input : inputs) {
    input = mx::contiguous(input, false, stream);
  }
  return mx::array(
      q.shape(), q.dtype(), std::make_shared<FlashDecode>(stream, config), std::move(inputs));
}

bool FlashDecode::is_equivalent(const mx::Primitive& other) const {
  auto* o = dynamic_cast<const FlashDecode*>(&other);
  return o != nullptr && o->config_.head_dim == config_.head_dim &&
      o->config_.group == config_.group && o->config_.num_kv_heads == config_.num_kv_heads &&
      o->config_.page_size == config_.page_size &&
      o->config_.num_partitions == config_.num_partitions && o->config_.rule == config_.rule &&
      o->config_.num_simdgroups == config_.num_simdgroups && o->config_.scale == config_.scale;
}

void FlashDecode::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("ftmetal fdecode: GPU only");
}

void FlashDecode::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  int batch = inputs[0].shape(0);
  if (batch == 0) {
    return;
  }
  const auto& c = config_;
  auto dtype = out.dtype();
  // 16-bit inputs stage the cross-simdgroup merge in half; fp32 inputs stay fp32.
  std::string type = metal_type(dtype);
  std::string stage_type = dtype == mx::float32 ? "float" : "half";
  std::string acc_type = c.head_dim >= HALF_ACCUMULATOR_HEAD_DIM ? stage_type : "float";
  std::string library_name = "ftmetal_fdecode_" + type + "_" + acc_type + "_" +
      std::to_string(c.head_dim) + "_" + std::to_string(c.group) + "_" +
      std::to_string(c.num_kv_heads) + "_" + std::to_string(c.page_size) + "_" +
      std::to_string(c.num_simdgroups) + "_" + std::to_string(c.rule.partition_tokens) + "_" +
      std::to_string(c.rule.min_partition_tokens) + "_" + std::to_string(c.rule.max_partitions) +
      "_" + std::to_string(c.rule.min_threadgroups);

  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto* library = d.get_library(library_name, [&]() {
    std::string source = fdecode_source;
    replace_all(source, "{{T}}", type);
    replace_all(source, "{{ACC_T}}", acc_type);
    replace_all(source, "{{STAGE_T}}", stage_type);
    replace_all(source, "{{D}}", std::to_string(c.head_dim));
    replace_all(source, "{{G}}", std::to_string(c.group));
    replace_all(source, "{{HKV}}", std::to_string(c.num_kv_heads));
    replace_all(source, "{{BS}}", std::to_string(c.page_size));
    replace_all(source, "{{NSG}}", std::to_string(c.num_simdgroups));
    replace_all(source, "{{PARTITION_TOKENS}}", std::to_string(c.rule.partition_tokens));
    replace_all(source, "{{MIN_PARTITION_TOKENS}}", std::to_string(c.rule.min_partition_tokens));
    replace_all(source, "{{MAX_PARTITIONS}}", std::to_string(c.rule.max_partitions));
    replace_all(source, "{{MIN_THREADGROUPS}}", std::to_string(c.rule.min_threadgroups));
    return source;
  });

  // Per-partition unnormalized accumulators and (max, sum) pairs live only in this step.
  mx::array part_acc(
      {batch, c.num_kv_heads, c.num_partitions, c.group, c.head_dim}, mx::float32, nullptr, {});
  mx::array part_ml({batch, c.num_kv_heads, c.num_partitions, c.group, 2}, mx::float32, nullptr, {});
  part_acc.set_data(mx::allocator::malloc(part_acc.nbytes()));
  part_ml.set_data(mx::allocator::malloc(part_ml.nbytes()));
  auto& encoder = mx::metal::get_command_encoder(s);
  encoder.add_temporary(part_acc);
  encoder.add_temporary(part_ml);

  int max_pages = inputs[3].shape(1);
  float scale_log2e = c.scale * static_cast<float>(M_LOG2E);
  encoder.set_compute_pipeline_state(d.get_kernel("fdecode_partials", library));
  for (int i = 0; i < 5; ++i) {
    encoder.set_input_array(inputs[i], i);
  }
  encoder.set_output_array(part_acc, 5);
  encoder.set_output_array(part_ml, 6);
  encoder.set_bytes(max_pages, 7);
  encoder.set_bytes(scale_log2e, 8);
  encoder.dispatch_threadgroups(
      MTL::Size(c.num_partitions, c.num_kv_heads, batch),
      MTL::Size(SIMD_WIDTH * c.num_simdgroups, 1, 1));

  encoder.set_compute_pipeline_state(d.get_kernel("fdecode_merge", library));
  encoder.set_input_array(part_acc, 0);
  encoder.set_input_array(part_ml, 1);
  encoder.set_input_array(inputs[4], 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(c.num_partitions, 4);
  encoder.dispatch_threadgroups(
      MTL::Size(c.group, c.num_kv_heads, batch), MTL::Size(SIMD_WIDTH, 1, 1));
}

} // namespace ftmetal
