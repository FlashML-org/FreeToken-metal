#include "moe/moe_lookup.h"

#include <stdexcept>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace ftmetal {

namespace {

std::string moe_lookup_source;
const char* LIBRARY_NAME = "ftmetal_moe_lookup";

void check_array(
    const mx::array& array,
    mx::Dtype dtype,
    const mx::Shape& shape,
    const char* what,
    const char* expected) {
  if (array.dtype() != dtype || array.shape() != shape) {
    throw std::invalid_argument(std::string("ftmetal moe_lookup: ") + what + " must be " + expected);
  }
}

// moe_lookup or, is_issue, moe_lookup_issue.
std::vector<mx::array> make_lookup(
    const mx::array& topk_ids,
    const mx::array& moe_step,
    const mx::array& slot_of,
    const mx::array& mailbox,
    const mx::array& last_use,
    const mx::array& hit_stats,
    int layer_id,
    int slot_base,
    MoeLookupKind kind,
    bool is_issue,
    mx::StreamOrDevice s) {
  if (topk_ids.dtype() != mx::int32 || topk_ids.ndim() != 2 || topk_ids.size() == 0) {
    throw std::invalid_argument("ftmetal moe_lookup: topk_ids must be int32 [T, top_k], T >= 1");
  }
  check_array(moe_step, mx::int32, {2}, "moe_step", "int32 [2] (real rows, step)");
  if (hit_stats.dtype() != mx::uint32 || hit_stats.ndim() != 2 || hit_stats.shape(0) < 1 ||
      hit_stats.shape(1) != 2) {
    throw std::invalid_argument("ftmetal moe_lookup: hit_stats must be uint32 [num_layers, 2]");
  }
  int num_layers = hit_stats.shape(0);
  if (slot_of.dtype() != mx::int32 || slot_of.ndim() != 1 || slot_of.size() % num_layers != 0) {
    throw std::invalid_argument(
        "ftmetal moe_lookup: slot_of must be int32 [num_layers * num_experts]");
  }
  int num_experts = static_cast<int>(slot_of.size()) / num_layers;
  if (num_experts < 1 || num_experts > MAX_LOOKUP_EXPERTS) {
    throw std::invalid_argument(
        "ftmetal moe_lookup: " + std::to_string(num_experts) + " experts per layer; at most " +
        std::to_string(MAX_LOOKUP_EXPERTS) + " fit the mailbox");
  }
  check_array(mailbox, mx::uint32, {MAILBOX_WORDS}, "mailbox", "uint32 [1024]");
  if (last_use.dtype() != mx::uint32 || last_use.ndim() != 1 || last_use.size() == 0) {
    throw std::invalid_argument("ftmetal moe_lookup: last_use must be uint32 [num_slots]");
  }
  if (layer_id < 0 || layer_id >= num_layers) {
    throw std::out_of_range(
        "ftmetal moe_lookup: layer " + std::to_string(layer_id) + " is not in [0, " +
        std::to_string(num_layers) + ")");
  }
  if (kind != MoeLookupKind::Decode && kind != MoeLookupKind::PrefillCopyMiss &&
      kind != MoeLookupKind::Probe) {
    throw std::invalid_argument("ftmetal moe_lookup: unknown lookup kind");
  }
  auto stream = mx::to_stream(s);
  mx::Shape slot_map_or_issued_shape = is_issue ? mx::Shape{ISSUED_WORDS} : mx::Shape{num_experts};
  // The shared tables stay as they are (see kvcache/pool_rows.h); eval_gpu checks they are row contiguous.
  return mx::array::make_arrays(
      {topk_ids.shape(),
       slot_map_or_issued_shape,
       mailbox.shape(),
       last_use.shape(),
       hit_stats.shape()},
      {mx::int32, mx::int32, mx::uint32, mx::uint32, mx::uint32},
      std::make_shared<MoeLookup>(
          stream, MoeLookupConfig{layer_id, num_experts, kind, slot_base, is_issue}),
      {mx::contiguous(topk_ids, false, stream),
       mx::contiguous(moe_step, false, stream),
       slot_of,
       mailbox,
       last_use,
       hit_stats});
}

} // namespace

void init_moe_lookup_library(const std::string& source) {
  moe_lookup_source = source;
}

std::vector<mx::array> moe_lookup(
    const mx::array& topk_ids,
    const mx::array& moe_step,
    const mx::array& slot_of,
    const mx::array& mailbox,
    const mx::array& last_use,
    const mx::array& hit_stats,
    int layer_id,
    int slot_base,
    MoeLookupKind kind,
    mx::StreamOrDevice s) {
  return make_lookup(
      topk_ids, moe_step, slot_of, mailbox, last_use, hit_stats, layer_id, slot_base, kind, false,
      s);
}

std::vector<mx::array> moe_lookup_issue(
    const mx::array& topk_ids,
    const mx::array& moe_step,
    const mx::array& slot_of,
    const mx::array& mailbox,
    const mx::array& last_use,
    const mx::array& hit_stats,
    int layer_id,
    int slot_base,
    mx::StreamOrDevice s) {
  return make_lookup(
      topk_ids,
      moe_step,
      slot_of,
      mailbox,
      last_use,
      hit_stats,
      layer_id,
      slot_base,
      MoeLookupKind::Decode,
      true,
      s);
}

bool MoeLookup::is_equivalent(const mx::Primitive& other) const {
  auto* lookup = dynamic_cast<const MoeLookup*>(&other);
  return lookup != nullptr && lookup->config_ == config_;
}

void MoeLookup::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("ftmetal moe_lookup: GPU only");
}

void MoeLookup::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  // inputs: topk_ids, moe_step, slot_of, mailbox, last_use, hit_stats.
  for (int i = 2; i < 6; ++i) {
    if (!inputs[i].flags().row_contiguous) {
      throw std::runtime_error(
          "ftmetal moe_lookup: slot_of, mailbox, last_use and hit_stats must be row contiguous");
    }
  }
  auto& slots = outputs[0];
  auto& slot_map_or_issued = outputs[1];
  slots.set_data(mx::allocator::malloc(slots.nbytes()));
  slot_map_or_issued.set_data(mx::allocator::malloc(slot_map_or_issued.nbytes()));
  outputs[2].copy_shared_buffer(inputs[3]);
  outputs[3].copy_shared_buffer(inputs[4]);
  outputs[4].copy_shared_buffer(inputs[5]);

  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto* library = d.get_library(LIBRARY_NAME, [&]() { return moe_lookup_source; });
  auto* kernel = d.get_kernel(config_.is_issue ? "moe_lookup_issue" : "moe_lookup", library);

  auto& encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  for (int i = 0; i < 3; ++i) {
    encoder.set_input_array(inputs[i], i);
  }
  for (int i = 2; i < 5; ++i) {
    encoder.set_output_array(outputs[i], i + 1);
  }
  encoder.set_output_array(slots, 6);
  encoder.set_output_array(slot_map_or_issued, 7);
  LookupParams params{
      config_.layer_id,
      config_.num_experts,
      inputs[0].shape(0),
      inputs[0].shape(1),
      static_cast<uint32_t>(config_.kind),
      config_.slot_base};
  encoder.set_bytes(params, 8);
  encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(MAX_LOOKUP_EXPERTS, 1, 1));
}

} // namespace ftmetal
