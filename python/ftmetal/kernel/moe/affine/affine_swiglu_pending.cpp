#include "moe/affine/affine_swiglu_pending.h"

#include <stdexcept>

#include "common/mailbox.h"
#include "common/metal_primitive.h"
#include "moe/affine/affine_pair_codes_shared.h"
#include "moe/affine/affine_swiglu_pending_shared.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace ftmetal {

namespace {

std::string affine_swiglu_pending_source;
const int SIMD_WIDTH = 32;
const std::string OP_NAME = "affine_swiglu_pending";

void check(bool condition, const std::string& expected) {
  if (!condition) {
    throw std::invalid_argument("ftmetal " + OP_NAME + ": " + expected);
  }
}

} // namespace

void init_affine_swiglu_pending_library(const std::string& source) {
  affine_swiglu_pending_source = source;
}

std::vector<mx::array> affine_swiglu_pending(
    const mx::array& x,
    const mx::array& w,
    const mx::array& scales_or_codes,
    const mx::array& biases_or_pairs,
    const mx::array& gate_up_resident,
    const mx::array& slots,
    const mx::array& issued,
    const mx::array& slot_of,
    const mx::array& mailbox,
    const mx::array& after,
    int layer_id,
    int slot_base,
    int num_experts,
    int group_size,
    int bits,
    int code_bits,
    bool is_fast,
    mx::StreamOrDevice s) {
  check_affine_quantization(OP_NAME, group_size, bits);
  check(
      code_bits >= 0 && code_bits <= MAX_PAIR_CODE_BITS,
      "code_bits must be 0 (plain scales) to " + std::to_string(MAX_PAIR_CODE_BITS));
  metal_type(x.dtype(), OP_NAME);
  check(x.ndim() == 4 && x.shape(1) == 1 && x.shape(2) == 1, "x must be [T, 1, 1, H]");
  int num_rows = x.shape(0);
  int hidden_size = x.shape(3);
  check(
      slots.dtype() == mx::int32 && slots.ndim() == 2 && slots.shape(0) == num_rows,
      "slots must be int32 [T, top_k]");
  int top_k = slots.shape(1);
  check(
      w.dtype() == mx::uint32 && w.ndim() == 3 && w.shape(2) * 32 == hidden_size * bits &&
          w.shape(1) % (2 * ACT_ROWS_PER_SIMDGROUP) == 0,
      "w must be uint32 [S, 2I, H * bits / 32], I a multiple of " +
          std::to_string(ACT_ROWS_PER_SIMDGROUP));
  int num_slots = w.shape(0);
  int intermediate_size = w.shape(1) / 2;
  check(hidden_size % group_size == 0, "H must be whole groups");
  if (code_bits == 0) {
    mx::Shape group_shape = {num_slots, 2 * intermediate_size, hidden_size / group_size};
    check(
        scales_or_codes.dtype() == x.dtype() && biases_or_pairs.dtype() == x.dtype() &&
            scales_or_codes.shape() == group_shape && biases_or_pairs.shape() == group_shape,
        "scales and biases must be [S, 2I, H / group_size] in x's dtype");
  } else {
    check(
        scales_or_codes.dtype() == mx::uint8 && scales_or_codes.ndim() == 2 &&
            scales_or_codes.shape(0) == num_slots,
        "codes must be uint8 [S, row bytes]");
    int64_t code_bits_total =
        int64_t(2 * intermediate_size) * (hidden_size / group_size) * code_bits;
    check(8 * int64_t(scales_or_codes.shape(1)) >= code_bits_total, "a code row is too short");
    check(
        biases_or_pairs.dtype() == x.dtype() && biases_or_pairs.ndim() == 2 &&
            biases_or_pairs.shape(1) == 2,
        "pairs must be [P, 2] in x's dtype");
  }
  check(
      gate_up_resident.dtype() == x.dtype() &&
          gate_up_resident.shape() == mx::Shape{num_rows, top_k, 1, 2 * intermediate_size},
      "gate_up_resident must be [T, top_k, 1, 2I] in x's dtype");
  check(
      issued.dtype() == mx::int32 && issued.shape() == mx::Shape{ISSUED_WORDS},
      "issued must be int32 [ISSUED_WORDS]");
  check(
      num_experts >= 1 && num_experts <= MAX_LOOKUP_EXPERTS && slot_of.dtype() == mx::int32 &&
          slot_of.ndim() == 1 && layer_id >= 0 &&
          int64_t(layer_id + 1) * num_experts <= slot_of.size(),
      "slot_of must be int32 [L * num_experts] holding layer_id");
  check(
      mailbox.dtype() == mx::uint32 && mailbox.shape() == mx::Shape{MAILBOX_WORDS},
      "mailbox must be uint32 [MAILBOX_WORDS]");
  auto stream = mx::to_stream(s);
  // The banks, slot_of and the mailbox stay as they are: the CPU writes them while the kernel
  // runs (see kvcache/pool_rows.h); eval_gpu checks they are row contiguous.
  return mx::array::make_arrays(
      {{num_rows, top_k, 1, intermediate_size}, {num_rows, top_k}, mailbox.shape()},
      {x.dtype(), mx::int32, mx::uint32},
      std::make_shared<AffineSwigluPending>(
          stream,
          AffineSwigluPendingConfig{
              layer_id, slot_base, num_experts, group_size, bits, code_bits, is_fast}),
      {mx::contiguous(x, false, stream),
       w,
       scales_or_codes,
       biases_or_pairs,
       mx::contiguous(gate_up_resident, false, stream),
       mx::contiguous(slots, false, stream),
       mx::contiguous(issued, false, stream),
       slot_of,
       mailbox,
       after});
}

bool AffineSwigluPending::is_equivalent(const mx::Primitive& other) const {
  auto* pending = dynamic_cast<const AffineSwigluPending*>(&other);
  return pending != nullptr && pending->config_ == config_;
}

void AffineSwigluPending::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("ftmetal affine_swiglu_pending: GPU only");
}

void AffineSwigluPending::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs: x, w, scale rows, bias rows, gate_up_resident, slots, issued, slot_of, mailbox and
  // after, which is not bound.
  for (int i : {1, 2, 3, 7, 8}) {
    if (!inputs[i].flags().row_contiguous) {
      throw std::runtime_error(
          "ftmetal affine_swiglu_pending: the banks, slot_of and the mailbox must be row "
          "contiguous");
    }
  }
  const auto& x = inputs[0];
  const auto& w = inputs[1];
  auto& act = outputs[0];
  auto& resolved_slots = outputs[1];
  act.set_data(mx::allocator::malloc(act.nbytes()));
  resolved_slots.set_data(mx::allocator::malloc(resolved_slots.nbytes()));
  // The waits write the mailbox's abandoned word; as an output it orders the next post after them.
  outputs[2].copy_shared_buffer(inputs[8]);
  const auto& c = config_;
  int num_rows = x.shape(0);
  int hidden_size = x.shape(3);
  int top_k = inputs[5].shape(1);
  int intermediate_size = w.shape(1) / 2;
  std::string type = metal_type(x.dtype(), OP_NAME);
  std::string library_name = "ftmetal_affine_swiglu_pending_" + type + "_gs_" +
      std::to_string(c.group_size) + "_b_" + std::to_string(c.bits) + "_c_" +
      std::to_string(c.code_bits) + (c.is_fast ? "_fast" : "");

  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto* library = d.get_library(library_name, [&]() {
    std::string templated = affine_swiglu_pending_source;
    replace_all(templated, "{{T}}", type);
    replace_all(templated, "{{GROUP_SIZE}}", std::to_string(c.group_size));
    replace_all(templated, "{{BITS}}", std::to_string(c.bits));
    replace_all(templated, "{{CODE_BITS}}", std::to_string(c.code_bits));
    replace_all(templated, "{{FAST}}", c.is_fast ? "true" : "false");
    return templated;
  });

  auto& encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(d.get_kernel("affine_swiglu_pending", library));
  for (int i = 0; i < 8; ++i) {
    encoder.set_input_array(inputs[i], i);
  }
  encoder.set_output_array(outputs[2], 8);
  encoder.set_output_array(act, 9);
  encoder.set_output_array(resolved_slots, 10);
  SwigluPendingParams params{
      hidden_size,
      intermediate_size,
      top_k,
      c.layer_id,
      c.num_experts,
      c.slot_base,
      c.code_bits == 0 ? 0 : static_cast<int>(inputs[2].shape(1))};
  encoder.set_bytes(params, 11);
  int act_rows_per_threadgroup = SWIGLU_PENDING_SIMDGROUPS * ACT_ROWS_PER_SIMDGROUP;
  encoder.dispatch_threadgroups(
      MTL::Size(
          1,
          (intermediate_size + act_rows_per_threadgroup - 1) / act_rows_per_threadgroup,
          num_rows * top_k),
      MTL::Size(SIMD_WIDTH, SWIGLU_PENDING_SIMDGROUPS, 1));
}

} // namespace ftmetal
