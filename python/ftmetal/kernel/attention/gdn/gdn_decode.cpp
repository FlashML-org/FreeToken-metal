#include "attention/gdn/gdn_decode.h"

#include <stdexcept>

#include "common/metal_primitive.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace ftmetal {

namespace {

std::string gdn_decode_source;
// Simdgroups per threadgroup and state rows per threadgroup tile.
const int NUM_SIMDGROUPS = 4;
const int ROWS_PER_THREADGROUP = 32;
const int SIMD_WIDTH = 32;

void check_shape(const mx::array& x, const mx::Shape& shape, mx::Dtype dtype, const char* what) {
  if (x.shape() != shape) {
    throw std::invalid_argument(std::string("ftmetal gdn_decode: bad shape for ") + what);
  }
  if (x.dtype() != dtype) {
    throw std::invalid_argument(std::string("ftmetal gdn_decode: bad dtype for ") + what);
  }
}

} // namespace

void init_gdn_decode_library(const std::string& source) {
  gdn_decode_source = source;
}

std::vector<mx::array> gdn_decode(
    const mx::array& conv_pool,
    const mx::array& state_pool,
    const mx::array& conv_in,
    const mx::array& conv_weight,
    const mx::array& a,
    const mx::array& b,
    const mx::array& A_log,
    const mx::array& dt_bias,
    const mx::array& slots,
    const mx::array& positions,
    mx::StreamOrDevice s) {
  if (conv_pool.ndim() != 3 || state_pool.ndim() != 4 || conv_in.ndim() != 2) {
    throw std::invalid_argument(
        "ftmetal gdn_decode: conv_pool [S, K, C], state_pool [S, HV, DV, DK], conv_in [B, C]");
  }
  if (state_pool.dtype() != mx::float32) {
    throw std::invalid_argument("ftmetal gdn_decode: state_pool must be float32");
  }
  auto dtype = conv_pool.dtype();
  metal_type(dtype, "gdn_decode");
  int num_slots = conv_pool.shape(0);
  int conv_kernel = conv_pool.shape(1);
  int conv_dim = conv_pool.shape(2);
  int num_v_heads = state_pool.shape(1);
  int value_dim = state_pool.shape(2);
  int key_dim = state_pool.shape(3);
  int batch = conv_in.shape(0);
  if (key_dim % SIMD_WIDTH != 0) {
    throw std::invalid_argument("ftmetal gdn_decode: key_head_dim must be a multiple of 32");
  }
  if (state_pool.shape(0) != num_slots) {
    throw std::invalid_argument("ftmetal gdn_decode: conv_pool and state_pool slot counts differ");
  }
  int rest = conv_dim - num_v_heads * value_dim;
  if (rest <= 0 || rest % (2 * key_dim) != 0 || num_v_heads % (rest / (2 * key_dim)) != 0) {
    throw std::invalid_argument("ftmetal gdn_decode: conv_dim must be 2 * HK * DK + HV * DV");
  }
  int num_k_heads = rest / (2 * key_dim);
  check_shape(conv_in, {batch, conv_dim}, dtype, "conv_in");
  check_shape(conv_weight, {conv_dim, conv_kernel}, dtype, "conv_weight");
  check_shape(a, {batch, num_v_heads}, dtype, "a");
  check_shape(b, {batch, num_v_heads}, dtype, "b");
  check_shape(A_log, {num_v_heads}, dtype, "A_log");
  check_shape(dt_bias, {num_v_heads}, dtype, "dt_bias");
  check_shape(slots, {batch}, mx::int32, "slots");
  check_shape(positions, {batch}, mx::int32, "positions");

  auto stream = mx::to_stream(s);
  GDNDecodeShape shape{num_k_heads, num_v_heads, key_dim, value_dim, conv_kernel};
  std::vector<mx::array> inputs = {
      conv_pool, state_pool, conv_in, conv_weight, a, b, A_log, dt_bias, slots, positions};
  // The pools stay as they are (see kvcache/pool_rows.h); eval_gpu checks they are row contiguous.
  for (size_t i = 2; i < inputs.size(); ++i) {
    inputs[i] = mx::contiguous(inputs[i], false, stream);
  }
  return mx::array::make_arrays(
      {conv_pool.shape(), state_pool.shape(), {batch, num_v_heads, value_dim}},
      {dtype, mx::float32, dtype},
      std::make_shared<GDNDecode>(stream, shape),
      inputs);
}

bool GDNDecode::is_equivalent(const mx::Primitive& other) const {
  auto* step = dynamic_cast<const GDNDecode*>(&other);
  return step != nullptr && step->shape_.num_k_heads == shape_.num_k_heads &&
      step->shape_.num_v_heads == shape_.num_v_heads && step->shape_.key_dim == shape_.key_dim &&
      step->shape_.value_dim == shape_.value_dim && step->shape_.conv_kernel == shape_.conv_kernel;
}

void GDNDecode::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("ftmetal gdn_decode: GPU only");
}

void GDNDecode::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  outputs[0].copy_shared_buffer(inputs[0]);
  outputs[1].copy_shared_buffer(inputs[1]);
  outputs[2].set_data(mx::allocator::malloc(outputs[2].nbytes()));
  if (!outputs[0].flags().row_contiguous || !outputs[1].flags().row_contiguous) {
    throw std::runtime_error("ftmetal gdn_decode: pools must be row contiguous");
  }
  int batch = inputs[2].shape(0);
  if (batch == 0) {
    return;
  }
  std::string type = metal_type(outputs[0].dtype(), "gdn_decode");
  std::string library_name = "ftmetal_gdn_decode_" + type + "_" + std::to_string(shape_.num_k_heads) +
      "_" + std::to_string(shape_.num_v_heads) + "_" + std::to_string(shape_.key_dim) + "_" +
      std::to_string(shape_.value_dim) + "_" + std::to_string(shape_.conv_kernel);

  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto* library = d.get_library(library_name, [&]() {
    std::string source = gdn_decode_source;
    replace_all(source, "{{T}}", type);
    replace_all(source, "{{HK}}", std::to_string(shape_.num_k_heads));
    replace_all(source, "{{HV}}", std::to_string(shape_.num_v_heads));
    replace_all(source, "{{DK}}", std::to_string(shape_.key_dim));
    replace_all(source, "{{DV}}", std::to_string(shape_.value_dim));
    replace_all(source, "{{KCONV}}", std::to_string(shape_.conv_kernel));
    replace_all(source, "{{NSG}}", std::to_string(NUM_SIMDGROUPS));
    replace_all(source, "{{ROWS_TG}}", std::to_string(ROWS_PER_THREADGROUP));
    return source;
  });
  auto* kernel = d.get_kernel("gdn_decode", library);

  auto& encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  auto conv_out = outputs[0];
  auto state_out = outputs[1];
  auto y = outputs[2];
  encoder.set_output_array(conv_out, 0);
  encoder.set_output_array(state_out, 1);
  encoder.set_output_array(y, 2);
  for (int i = 2; i < 10; ++i) {
    encoder.set_input_array(inputs[i], i + 1);
  }
  int tiles = (shape_.value_dim + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP;
  encoder.dispatch_threadgroups(
      MTL::Size(tiles, shape_.num_v_heads, batch), MTL::Size(SIMD_WIDTH * NUM_SIMDGROUPS, 1, 1));
}

} // namespace ftmetal
