#include "gather_qmm_sorted.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "metal_primitive.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace ftmetal {

namespace {

std::string gather_qmm_sorted_source;
const int SIMD_WIDTH = 32;
// MLX's non-NAX rhs tiles: a BM x BN output tile per threadgroup of WM x WN simdgroups,
// K in BK steps. A NAX GPU (M5 class) gets MLX's matrix-unit kernel from mx.gather_qmm and
// not from here.
const int BM = 16;
const int BN = 32;
const int BK = 32;
const int WM = 1;
const int WN = 2;

} // namespace

void init_gather_qmm_sorted_library(const std::string& source) {
  gather_qmm_sorted_source = source;
}

mx::array gather_qmm_sorted(
    const mx::array& x,
    const mx::array& w,
    const mx::array& scales,
    const mx::array& biases,
    const mx::array& expert_offsets,
    const mx::array& expert_bytes,
    int n,
    int group_size,
    int bits,
    mx::StreamOrDevice s) {
  if (x.ndim() != 3 || x.shape(1) != 1) {
    throw std::invalid_argument("ftmetal gather_qmm_sorted: x must be [B, 1, K]");
  }
  int K = x.shape(2);
  metal_type(x.dtype(), "gather_qmm_sorted");
  bool is_instantiated = (group_size == 32 || group_size == 64 || group_size == 128) &&
      (bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8);
  if (!is_instantiated) {
    throw std::invalid_argument(
        "ftmetal gather_qmm_sorted: group size must be 32, 64 or 128 and bits 2-6 or 8, not (" +
        std::to_string(group_size) + ", " + std::to_string(bits) + ")");
  }
  if (n < 1 || K % group_size != 0 || K * bits % 32 != 0) {
    throw std::invalid_argument(
        "ftmetal gather_qmm_sorted: n must be positive and x's K a whole number of groups and "
        "packed words");
  }
  if (expert_offsets.dtype() != mx::int32 || expert_offsets.ndim() != 1 ||
      expert_bytes.dtype() != mx::int64 || expert_bytes.ndim() != 2 ||
      expert_bytes.shape(0) < 1 || expert_bytes.shape(1) != 3 ||
      expert_offsets.shape(0) != expert_bytes.shape(0) + 1) {
    throw std::invalid_argument(
        "ftmetal gather_qmm_sorted: expert_offsets must be [E + 1] int32 and expert_bytes "
        "[E, 3] int64");
  }
  auto stream = mx::to_stream(s);
  // The banks are taken as they are: mx::contiguous copies a slice of a larger buffer whole.
  return mx::array(
      {x.shape(0), 1, n},
      x.dtype(),
      std::make_shared<GatherQMMSorted>(stream, GatherQMMSortedConfig{group_size, bits}),
      {mx::contiguous(x, false, stream),
       w,
       scales,
       biases,
       mx::contiguous(expert_offsets, false, stream),
       mx::contiguous(expert_bytes, false, stream)});
}

bool GatherQMMSorted::is_equivalent(const mx::Primitive& other) const {
  auto* o = dynamic_cast<const GatherQMMSorted*>(&other);
  return o != nullptr && o->config_ == config_;
}

void GatherQMMSorted::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("ftmetal gather_qmm_sorted: GPU only");
}

void GatherQMMSorted::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs: x, w, scales, biases, expert_offsets, expert_bytes.
  for (int i = 1; i <= 3; ++i) {
    if (!inputs[i].flags().row_contiguous) {
      throw std::runtime_error(
          "ftmetal gather_qmm_sorted: w, scales and biases must be row contiguous");
    }
  }
  const auto& x = inputs[0];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  int M = x.shape(0);
  int N = out.shape(2);
  int K = x.shape(2);
  int num_experts = inputs[5].shape(0);
  if (M == 0) {
    return;
  }
  const auto& c = config_;
  std::string type = metal_type(x.dtype(), "gather_qmm_sorted");
  std::string library_name = "ftmetal_gather_qmm_sorted_" + type + "_gs_" +
      std::to_string(c.group_size) + "_b_" + std::to_string(c.bits);
  bool align_N = (N % BN) == 0;
  bool align_K = (K % BK) == 0;
  std::string hash_name = library_name + "_align_N_" + (align_N ? "t" : "n") + "_align_K_" +
      (align_K ? "t" : "n");
  mx::metal::MTLFCList constants = {
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto* library = d.get_library(library_name, [&]() {
    std::string source = gather_qmm_sorted_source;
    replace_all(source, "{{T}}", type);
    replace_all(source, "{{GROUP_SIZE}}", std::to_string(c.group_size));
    replace_all(source, "{{BITS}}", std::to_string(c.bits));
    replace_all(source, "{{BM}}", std::to_string(BM));
    replace_all(source, "{{BN}}", std::to_string(BN));
    replace_all(source, "{{BK}}", std::to_string(BK));
    replace_all(source, "{{WM}}", std::to_string(WM));
    replace_all(source, "{{WN}}", std::to_string(WN));
    return source;
  });

  auto& encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(
      d.get_kernel("gather_qmm_sorted", library, hash_name, constants));
  for (int i = 0; i < 5; ++i) {
    encoder.set_input_array(inputs[i], i);
  }
  encoder.set_output_array(out, 5);
  encoder.set_bytes(M, 6);
  encoder.set_bytes(N, 7);
  encoder.set_bytes(K, 8);
  // MLX runs gather_mm_offsets over the bank-row ids and passes E = bank rows, which makes
  // its scan and grid grow with the bank (2.8x at 5120 rows); the caller's logical-expert
  // offsets (inputs[4]) and count keep both at the resident layer's size.
  encoder.set_bytes(num_experts, 9);
  encoder.set_input_array(inputs[5], 10);
  // Every expert's tile count rounds up, so E experts take at most M / BM + E - 1 row tiles;
  // the threadgroups past the last tile return at once.
  encoder.dispatch_threadgroups(
      MTL::Size((N + BN - 1) / BN, std::min(M, (M + BM - 1) / BM + num_experts - 1), 1),
      MTL::Size(SIMD_WIDTH, WN, WM));
}

} // namespace ftmetal
