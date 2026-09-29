#include "rows.h"

#include <algorithm>
#include <stdexcept>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace ftmetal {

namespace {

std::string rows_source;
const char* LIBRARY_NAME = "ftmetal_rows";
const int VECTOR_WIDTH = 4;

std::string metal_type(mx::Dtype dtype) {
  switch (dtype) {
    case mx::float16:
      return "half";
    case mx::bfloat16:
      return "bfloat";
    case mx::float32:
      return "float";
    case mx::int32:
      return "int";
    case mx::uint32:
      return "uint";
    default:
      throw std::invalid_argument(
          "ftmetal rows: unsupported pool dtype " + mx::type_to_name(dtype));
  }
}

void check_ids(const mx::array& ids, const char* what) {
  if (ids.dtype() != mx::int32) {
    throw std::invalid_argument(std::string("ftmetal rows: ") + what + " must be int32");
  }
  if (ids.ndim() != 1) {
    throw std::invalid_argument(std::string("ftmetal rows: ") + what + " must be 1-D");
  }
}

void check_rows(const mx::array& pool, const mx::array& rows, const mx::array& ids) {
  if (pool.ndim() < 2) {
    throw std::invalid_argument("ftmetal rows: pool must be [num_rows, ...]");
  }
  if (rows.dtype() != pool.dtype()) {
    throw std::invalid_argument("ftmetal rows: rows and pool dtypes differ");
  }
  if (rows.ndim() != pool.ndim() || rows.shape(0) != ids.shape(0)) {
    throw std::invalid_argument("ftmetal rows: rows must be [len(ids), *pool.shape[1:]]");
  }
  for (int i = 1; i < pool.ndim(); ++i) {
    if (rows.shape(i) != pool.shape(i)) {
      throw std::invalid_argument("ftmetal rows: rows must be [len(ids), *pool.shape[1:]]");
    }
  }
}

} // namespace

void init_library(const std::string& source) {
  rows_source = source;
}

mx::array scatter_rows(
    const mx::array& pool,
    const mx::array& rows,
    const mx::array& dst_ids,
    mx::StreamOrDevice s) {
  check_ids(dst_ids, "dst_ids");
  check_rows(pool, rows, dst_ids);
  auto stream = mx::to_stream(s);
  return mx::array(
      pool.shape(),
      pool.dtype(),
      std::make_shared<InPlaceRows>(stream, RowsKind::Scatter),
      {pool,
       mx::contiguous(rows, false, stream),
       mx::contiguous(dst_ids, false, stream)});
}

mx::array copy_rows(
    const mx::array& pool,
    const mx::array& src_ids,
    const mx::array& dst_ids,
    mx::StreamOrDevice s) {
  check_ids(src_ids, "src_ids");
  check_ids(dst_ids, "dst_ids");
  if (pool.ndim() < 2) {
    throw std::invalid_argument("ftmetal rows: pool must be [num_rows, ...]");
  }
  if (src_ids.shape(0) != dst_ids.shape(0)) {
    throw std::invalid_argument("ftmetal rows: src_ids and dst_ids differ in length");
  }
  auto stream = mx::to_stream(s);
  return mx::array(
      pool.shape(),
      pool.dtype(),
      std::make_shared<InPlaceRows>(stream, RowsKind::Copy),
      {pool,
       mx::contiguous(src_ids, false, stream),
       mx::contiguous(dst_ids, false, stream)});
}

mx::array zero_rows(const mx::array& pool, const mx::array& ids, mx::StreamOrDevice s) {
  check_ids(ids, "ids");
  if (pool.ndim() < 2) {
    throw std::invalid_argument("ftmetal rows: pool must be [num_rows, ...]");
  }
  auto stream = mx::to_stream(s);
  return mx::array(
      pool.shape(),
      pool.dtype(),
      std::make_shared<InPlaceRows>(stream, RowsKind::Zero),
      {pool, mx::contiguous(ids, false, stream)});
}

std::vector<mx::array> kv_write(
    const mx::array& k_cache,
    const mx::array& v_cache,
    const mx::array& k,
    const mx::array& v,
    const mx::array& out_loc,
    mx::StreamOrDevice s) {
  check_ids(out_loc, "out_loc");
  check_rows(k_cache, k, out_loc);
  check_rows(v_cache, v, out_loc);
  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {k_cache.shape(), v_cache.shape()},
      {k_cache.dtype(), v_cache.dtype()},
      std::make_shared<InPlaceRows>(stream, RowsKind::KVWrite),
      {k_cache,
       v_cache,
       mx::contiguous(k, false, stream),
       mx::contiguous(v, false, stream),
       mx::contiguous(out_loc, false, stream)});
}

const char* InPlaceRows::name() const {
  switch (kind_) {
    case RowsKind::Scatter:
      return "ScatterRows";
    case RowsKind::Copy:
      return "CopyRows";
    case RowsKind::Zero:
      return "ZeroRows";
    case RowsKind::KVWrite:
      return "KVWrite";
  }
  return "InPlaceRows";
}

bool InPlaceRows::is_equivalent(const mx::Primitive& other) const {
  auto* rows = dynamic_cast<const InPlaceRows*>(&other);
  return rows != nullptr && rows->kind_ == kind_;
}

void InPlaceRows::eval_cpu(
    const std::vector<mx::array>&,
    std::vector<mx::array>&) {
  throw std::runtime_error("ftmetal rows: GPU only");
}

void InPlaceRows::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  switch (kind_) {
    case RowsKind::Scatter:
      outputs[0].copy_shared_buffer(inputs[0]);
      encode(outputs[0], &inputs[1], nullptr, inputs[2]);
      break;
    case RowsKind::Copy:
      outputs[0].copy_shared_buffer(inputs[0]);
      encode(outputs[0], nullptr, &inputs[1], inputs[2]);
      break;
    case RowsKind::Zero:
      outputs[0].copy_shared_buffer(inputs[0]);
      encode(outputs[0], nullptr, nullptr, inputs[1]);
      break;
    case RowsKind::KVWrite:
      outputs[0].copy_shared_buffer(inputs[0]);
      outputs[1].copy_shared_buffer(inputs[1]);
      encode(outputs[0], &inputs[2], nullptr, inputs[4]);
      encode(outputs[1], &inputs[3], nullptr, inputs[4]);
      break;
  }
}

void InPlaceRows::encode(
    const mx::array& pool_out,
    const mx::array* rows,
    const mx::array* src_ids,
    const mx::array& dst_ids) {
  int num_rows = static_cast<int>(dst_ids.size());
  if (num_rows == 0) {
    return;
  }
  if (!pool_out.flags().row_contiguous) {
    throw std::runtime_error("ftmetal rows: pool must be row contiguous");
  }
  int row_elems = static_cast<int>(pool_out.size() / pool_out.shape(0));
  bool vectorized = row_elems % VECTOR_WIDTH == 0;
  int lanes = vectorized ? row_elems / VECTOR_WIDTH : row_elems;

  const char* base = rows != nullptr ? "scatter_rows" : (src_ids != nullptr ? "copy_rows" : "zero_rows");
  std::string kernel_name = std::string(base) + (vectorized ? "_vec_" : "_") + metal_type(pool_out.dtype());

  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto* library = d.get_library(LIBRARY_NAME, [&]() { return rows_source; });
  auto* kernel = d.get_kernel(kernel_name, library);

  auto& encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(kernel);
  // Same buffer index layout for every kernel; unused slots stay unbound.
  auto pool_mutable = pool_out;
  encoder.set_output_array(pool_mutable, 0);
  if (rows != nullptr) {
    encoder.set_input_array(*rows, 1);
  }
  if (src_ids != nullptr) {
    encoder.set_input_array(*src_ids, 2);
  }
  encoder.set_input_array(dst_ids, 3);
  encoder.set_bytes(lanes, 4);

  int group_x = std::min<int>(lanes, static_cast<int>(kernel->maxTotalThreadsPerThreadgroup()));
  encoder.dispatch_threads(MTL::Size(lanes, num_rows, 1), MTL::Size(group_x, 1, 1));
}

} // namespace ftmetal
