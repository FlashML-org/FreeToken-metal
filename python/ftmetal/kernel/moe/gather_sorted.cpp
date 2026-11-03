#include "moe/gather_sorted.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/metal_primitive.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace ftmetal {

namespace {

std::string gather_qmm_sorted_source;
std::string fp_gather_qmm_sorted_source;
std::string gather_mm_sorted_source;
const int SIMD_WIDTH = 32;

// MLX's non-NAX rhs tiles: a BM x BN output tile per threadgroup of WM x WN simdgroups, K in
// BK steps; gather_qmm_rhs's for both quantized kernels, gather_mm_rhs's for the unquantized
// one. A NAX GPU (M5 class) gets MLX's matrix-unit kernels from mx.gather_qmm and
// mx.gather_mm and not from here.
struct Tiles {
  int bm;
  int bn;
  int bk;
  int wm;
  int wn;
};
const Tiles QMM_TILES = {16, 32, QMM_TILE_K, 1, 2};
const Tiles MM_TILES = {16, 64, 16, 1, 2};

// The (bits, group size) of each float block mode: the only ones MLX instantiates.
const std::map<std::string, std::pair<int, int>> FP_BLOCK_SHAPES = {
    {"mxfp4", {4, 32}},
    {"nvfp4", {4, 16}},
    {"mxfp8", {8, 32}},
};

// What both ops take alike: x [B, 1, K] in a float dtype, a positive n, expert_offsets
// [E + 1] int32 and expert_bytes [E, num_buffers] int64.
void check_sorted_inputs(
    const std::string& op_name,
    const mx::array& x,
    int n,
    const mx::array& expert_offsets,
    const mx::array& expert_bytes,
    int num_buffers) {
  if (x.ndim() != 3 || x.shape(1) != 1) {
    throw std::invalid_argument("ftmetal " + op_name + ": x must be [B, 1, K]");
  }
  metal_type(x.dtype(), op_name);
  if (n < 1) {
    throw std::invalid_argument("ftmetal " + op_name + ": n must be positive");
  }
  if (expert_offsets.dtype() != mx::int32 || expert_offsets.ndim() != 1 ||
      expert_bytes.dtype() != mx::int64 || expert_bytes.ndim() != 2 ||
      expert_bytes.shape(0) < 1 || expert_bytes.shape(1) != num_buffers ||
      expert_offsets.shape(0) != expert_bytes.shape(0) + 1) {
    throw std::invalid_argument(
        "ftmetal " + op_name + ": expert_offsets must be [E + 1] int32 and expert_bytes [E, " +
        std::to_string(num_buffers) + "] int64");
  }
}

mx::array gather_sorted(
    const mx::array& x,
    const std::vector<mx::array>& buffers,
    const mx::array& expert_offsets,
    const mx::array& expert_bytes,
    int n,
    GatherSortedConfig config,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  // The banks are taken as they are: mx::contiguous copies a slice of a larger buffer whole.
  std::vector<mx::array> inputs = {mx::contiguous(x, false, stream)};
  inputs.insert(inputs.end(), buffers.begin(), buffers.end());
  inputs.push_back(mx::contiguous(expert_offsets, false, stream));
  inputs.push_back(mx::contiguous(expert_bytes, false, stream));
  return mx::array(
      {x.shape(0), 1, n},
      x.dtype(),
      std::make_shared<GatherSorted>(stream, config),
      std::move(inputs));
}

} // namespace

void init_gather_sorted_libraries(
    const std::string& affine_source,
    const std::string& fp_source,
    const std::string& unquantized_source) {
  gather_qmm_sorted_source = affine_source;
  fp_gather_qmm_sorted_source = fp_source;
  gather_mm_sorted_source = unquantized_source;
}

mx::array gather_qmm_sorted(
    const mx::array& x,
    const mx::array& w,
    const mx::array& scales,
    const std::optional<mx::array>& biases,
    const mx::array& expert_offsets,
    const mx::array& expert_bytes,
    int n,
    int group_size,
    int bits,
    const std::string& mode,
    mx::StreamOrDevice s) {
  auto fp_shape = FP_BLOCK_SHAPES.find(mode);
  GatherSortedKernel kernel;
  if (mode == "affine") {
    bool is_instantiated = (group_size == 32 || group_size == 64 || group_size == 128) &&
        (bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8);
    if (!is_instantiated) {
      throw std::invalid_argument(
          "ftmetal gather_qmm_sorted: affine group size must be 32, 64 or 128 and bits 2-6 "
          "or 8, not (" +
          std::to_string(group_size) + ", " + std::to_string(bits) + ")");
    }
    if (!biases) {
      throw std::invalid_argument("ftmetal gather_qmm_sorted: affine needs biases");
    }
    kernel = GatherSortedKernel::Affine;
  } else if (fp_shape != FP_BLOCK_SHAPES.end()) {
    auto [mode_bits, mode_group_size] = fp_shape->second;
    if (bits != mode_bits || group_size != mode_group_size) {
      throw std::invalid_argument(
          "ftmetal gather_qmm_sorted: " + mode + " takes bits " + std::to_string(mode_bits) +
          " and group size " + std::to_string(mode_group_size) + ", not (" +
          std::to_string(bits) + ", " + std::to_string(group_size) + ")");
    }
    if (biases) {
      throw std::invalid_argument("ftmetal gather_qmm_sorted: " + mode + " takes no biases");
    }
    kernel = GatherSortedKernel::Fp;
  } else {
    throw std::invalid_argument(
        "ftmetal gather_qmm_sorted: mode must be affine, mxfp4, nvfp4 or mxfp8, not " + mode);
  }
  std::vector<mx::array> buffers = {w, scales};
  if (biases) {
    buffers.push_back(*biases);
  }
  check_sorted_inputs(
      "gather_qmm_sorted", x, n, expert_offsets, expert_bytes, static_cast<int>(buffers.size()));
  int K = x.shape(2);
  if (K % group_size != 0 || K * bits % 32 != 0) {
    throw std::invalid_argument(
        "ftmetal gather_qmm_sorted: x's K must be a whole number of groups and packed words");
  }
  if (kernel == GatherSortedKernel::Fp && K % QMM_TILES.bk != 0) {
    throw std::invalid_argument(
        "ftmetal gather_qmm_sorted: " + mode + " needs K a multiple of " +
        std::to_string(QMM_TILES.bk) +
        ": the float tile reads its last partial K tile past the row, and an offload bank "
        "holds arbitrary bytes there");
  }
  return gather_sorted(
      x, buffers, expert_offsets, expert_bytes, n, {kernel, n, group_size, bits}, s);
}

mx::array gather_mm_sorted(
    const mx::array& x,
    const mx::array& w,
    const mx::array& expert_offsets,
    const mx::array& expert_bytes,
    int n,
    mx::StreamOrDevice s) {
  check_sorted_inputs("gather_mm_sorted", x, n, expert_offsets, expert_bytes, 1);
  return gather_sorted(
      x, {w}, expert_offsets, expert_bytes, n, {GatherSortedKernel::Unquantized, n, 0, 0}, s);
}

bool GatherSorted::is_equivalent(const mx::Primitive& other) const {
  auto* o = dynamic_cast<const GatherSorted*>(&other);
  return o != nullptr && o->config_ == config_;
}

void GatherSorted::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("ftmetal gather_sorted: GPU only");
}

void GatherSorted::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs: x, the weight buffers (weight, then scales and biases when quantized),
  // expert_offsets, expert_bytes.
  int num_buffers = static_cast<int>(inputs.size()) - 3;
  for (int i = 1; i <= num_buffers; ++i) {
    if (!inputs[i].flags().row_contiguous) {
      throw std::runtime_error(
          "ftmetal gather_sorted: the weight, scales and biases buffers must be row "
          "contiguous");
    }
  }
  const auto& x = inputs[0];
  const auto& expert_offsets = inputs[num_buffers + 1];
  const auto& expert_bytes = inputs[num_buffers + 2];
  auto& out = outputs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  int M = x.shape(0);
  int N = out.shape(2);
  int K = x.shape(2);
  int num_experts = expert_bytes.shape(0);
  if (M == 0) {
    return;
  }
  const auto& c = config_;
  std::string type = metal_type(x.dtype(), "gather_sorted");
  const std::string* source;
  std::string kernel_name;
  Tiles tiles;
  if (c.kernel == GatherSortedKernel::Affine) {
    source = &gather_qmm_sorted_source;
    kernel_name = "gather_qmm_sorted";
    tiles = QMM_TILES;
  } else if (c.kernel == GatherSortedKernel::Fp) {
    source = &fp_gather_qmm_sorted_source;
    kernel_name = "fp_gather_qmm_sorted";
    tiles = QMM_TILES;
  } else {
    source = &gather_mm_sorted_source;
    kernel_name = "gather_mm_sorted";
    tiles = MM_TILES;
  }
  std::string library_name = "ftmetal_" + kernel_name + "_" + type;
  if (c.kernel != GatherSortedKernel::Unquantized) {
    library_name += "_gs_" + std::to_string(c.group_size) + "_b_" + std::to_string(c.bits);
  }
  bool align_N = (N % tiles.bn) == 0;
  bool align_K = (K % tiles.bk) == 0;
  std::string hash_name = library_name + "_align_N_" + (align_N ? "t" : "n") + "_align_K_" +
      (align_K ? "t" : "n");
  mx::metal::MTLFCList constants = {
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto* library = d.get_library(library_name, [&]() {
    std::string templated = *source;
    replace_all(templated, "{{T}}", type);
    replace_all(templated, "{{GROUP_SIZE}}", std::to_string(c.group_size));
    replace_all(templated, "{{BITS}}", std::to_string(c.bits));
    replace_all(templated, "{{BM}}", std::to_string(tiles.bm));
    replace_all(templated, "{{BN}}", std::to_string(tiles.bn));
    replace_all(templated, "{{BK}}", std::to_string(tiles.bk));
    replace_all(templated, "{{WM}}", std::to_string(tiles.wm));
    replace_all(templated, "{{WN}}", std::to_string(tiles.wn));
    return templated;
  });

  auto& encoder = mx::metal::get_command_encoder(s);
  encoder.set_compute_pipeline_state(d.get_kernel(kernel_name, library, hash_name, constants));
  // gather_qmm_rhs's buffer order, also for the unquantized kernel; a float block kernel
  // leaves buffer 3 (MLX's biases or nvfp4 global scale) unbound.
  encoder.set_input_array(x, 0);
  for (int i = 1; i <= num_buffers; ++i) {
    encoder.set_input_array(inputs[i], i);
  }
  encoder.set_input_array(expert_offsets, 4);
  encoder.set_output_array(out, 5);
  encoder.set_bytes(M, 6);
  encoder.set_bytes(N, 7);
  encoder.set_bytes(K, 8);
  // MLX runs gather_mm_offsets over the bank-row ids and passes E = bank rows, which makes
  // its scan and grid grow with the bank (2.8x at 5120 rows); the caller's logical-expert
  // offsets (expert_offsets) and count keep both at the resident layer's size.
  encoder.set_bytes(num_experts, 9);
  encoder.set_input_array(expert_bytes, 10);
  // Every expert's tile count rounds up, so E experts take at most M / BM + E - 1 row tiles;
  // the threadgroups past the last tile return at once.
  encoder.dispatch_threadgroups(
      MTL::Size(
          (N + tiles.bn - 1) / tiles.bn,
          std::min(M, (M + tiles.bm - 1) / tiles.bm + num_experts - 1),
          1),
      MTL::Size(SIMD_WIDTH, tiles.wn, tiles.wm));
}

} // namespace ftmetal
