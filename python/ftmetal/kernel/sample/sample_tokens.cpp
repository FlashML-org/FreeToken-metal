#include "sample.h"

#include <stdexcept>

#include "metal_primitive.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/random.h"

namespace ftmetal {

namespace {

std::string sample_source;
// Shared with metal/sample.metal.
const int KIND_NONE = 0;
const int KIND_COUNT = 1;
const int KIND_MASS = 2;
const int NUM_BINS = 256;
const int ROUND_THREADS = 256;
const int FIND_THREADS = 1024;
const int MAX_GROUPS = 256;
const int STATE_SLOTS = 10;
const int STATE_WORDS = 8;
const int BITS_PER_ROUND = 8;

// One radix-select round: what it bins (counts for top-k, mass for top-p, nothing for the
// max-only first dispatch), the key bits below its bins, whether its bracket is open above
// (the first round of a search) and whether the draw reads its histograms.
struct RoundSpec {
  int kind;
  int shift;
  bool root;
  bool final;
};

const RoundSpec NO_ROUND{KIND_NONE, 0, false, false};

void check_rows(const mx::array& x, int rows, mx::Dtype dtype, const char* what) {
  if (x.ndim() != 1 || x.shape(0) != rows || x.dtype() != dtype) {
    throw std::invalid_argument(
        std::string("ftmetal sample: ") + what + " must be [B] " + mx::type_to_name(dtype));
  }
}

// The rounds before the draw: top-k's, then top-p's inside the top-k bracket. Mass needs the
// row max, so without top-k a max-only dispatch goes first, and with neither filter one
// open round over every key gives the draw its per-chunk mass.
std::vector<RoundSpec> round_chain(int key_bits, bool has_top_k, bool has_top_p) {
  std::vector<RoundSpec> chain;
  auto add_search = [&](int kind) {
    for (int shift = key_bits - BITS_PER_ROUND; shift >= 0; shift -= BITS_PER_ROUND) {
      chain.push_back({kind, shift, shift == key_bits - BITS_PER_ROUND, false});
    }
  };
  if (has_top_k) {
    add_search(KIND_COUNT);
  } else {
    chain.push_back(NO_ROUND);
  }
  if (has_top_p) {
    add_search(KIND_MASS);
  } else if (!has_top_k) {
    chain.push_back({KIND_MASS, key_bits - BITS_PER_ROUND, true, false});
  }
  chain.back().final = true;
  if (static_cast<int>(chain.size()) >= STATE_SLOTS) {
    throw std::logic_error("ftmetal sample: round chain exceeds the state slots");
  }
  return chain;
}

MTL::ComputePipelineState* round_pipeline(
    mx::metal::Device& device,
    MTL::Library* library,
    const std::string& library_name,
    const std::string& kernel,
    int dispatch_index,
    RoundSpec prev,
    RoundSpec cur,
    bool has_top_k,
    bool has_top_p) {
  mx::metal::MTLFCList constants = {
      {&dispatch_index, MTL::DataType::DataTypeInt, 0},
      {&prev.kind, MTL::DataType::DataTypeInt, 1},
      {&prev.shift, MTL::DataType::DataTypeInt, 2},
      {&prev.root, MTL::DataType::DataTypeBool, 3},
      {&cur.kind, MTL::DataType::DataTypeInt, 4},
      {&cur.shift, MTL::DataType::DataTypeInt, 5},
      {&cur.root, MTL::DataType::DataTypeBool, 6},
      {&cur.final, MTL::DataType::DataTypeBool, 7},
      {&has_top_k, MTL::DataType::DataTypeBool, 8},
      {&has_top_p, MTL::DataType::DataTypeBool, 9},
  };
  auto spec = [](RoundSpec round) {
    return std::to_string(round.kind) + "_" + std::to_string(round.shift) + "_" +
        std::to_string(round.root) + std::to_string(round.final);
  };
  std::string hash_name = library_name + "_" + kernel + "_" + std::to_string(dispatch_index) +
      "_" + spec(prev) + "_" + spec(cur) + "_" + std::to_string(has_top_k) +
      std::to_string(has_top_p);
  return device.get_kernel(kernel, library, hash_name, constants);
}

} // namespace

void init_sample_library(const std::string& source) {
  sample_source = source;
}

mx::array sample_tokens(
    const mx::array& logits,
    const mx::array& temperatures,
    const std::optional<mx::array>& top_k,
    const std::optional<mx::array>& top_p,
    int num_groups,
    const std::optional<mx::array>& random_bits,
    mx::StreamOrDevice s) {
  if (logits.ndim() != 2 || logits.shape(1) < 1) {
    throw std::invalid_argument("ftmetal sample: logits must be [B, V] with V >= 1");
  }
  metal_type(logits.dtype(), "sample");
  int rows = logits.shape(0);
  check_rows(temperatures, rows, mx::float32, "temperatures");
  if (random_bits) {
    check_rows(*random_bits, rows, mx::uint32, "random_bits");
  }
  if (top_k) {
    check_rows(*top_k, rows, mx::int32, "top_k");
  }
  if (top_p) {
    check_rows(*top_p, rows, mx::float32, "top_p");
  }
  if (num_groups < 1 || num_groups > MAX_GROUPS) {
    throw std::invalid_argument("ftmetal sample: num_groups must be in [1, 256]");
  }
  auto stream = mx::to_stream(s);
  std::vector<mx::array> inputs = {
      logits,
      temperatures,
      random_bits ? *random_bits : mx::random::bits({rows}, 4, std::nullopt, stream)};
  if (top_k) {
    inputs.push_back(*top_k);
  }
  if (top_p) {
    inputs.push_back(*top_p);
  }
  for (auto& input : inputs) {
    input = mx::contiguous(input, false, stream);
  }
  SampleConfig config{top_k.has_value(), top_p.has_value(), num_groups};
  return mx::array(
      {rows}, mx::int32, std::make_shared<SampleTokens>(stream, config), std::move(inputs));
}

bool SampleTokens::is_equivalent(const mx::Primitive& other) const {
  auto* o = dynamic_cast<const SampleTokens*>(&other);
  return o != nullptr && o->config_ == config_;
}

void SampleTokens::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("ftmetal sample: GPU only");
}

void SampleTokens::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  auto& tokens = outputs[0];
  tokens.set_data(mx::allocator::malloc(tokens.nbytes()));
  const auto& logits = inputs[0];
  int rows = logits.shape(0);
  if (rows == 0) {
    return;
  }
  const auto& c = config_;
  const auto& temperatures = inputs[1];
  const auto& random_bits = inputs[2];
  int next_input = 3;
  // An absent filter's slot is bound to the temperatures; its kernels never read it.
  const mx::array& top_k = c.has_top_k ? inputs[next_input++] : temperatures;
  const mx::array& top_p = c.has_top_p ? inputs[next_input++] : temperatures;
  uint32_t vocab = logits.shape(1);
  uint32_t groups = c.num_groups;
  uint32_t chunk = (vocab + groups - 1) / groups;
  int key_bits = logits.dtype() == mx::float32 ? 32 : 16;
  std::string type = metal_type(logits.dtype(), "sample");
  std::string library_name = "ftmetal_sample_" + type;

  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto* library = d.get_library(library_name, [&]() {
    std::string source = sample_source;
    replace_all(source, "{{T}}", type);
    replace_all(source, "{{KEY_BITS}}", std::to_string(key_bits));
    return source;
  });

  // Per-chunk maxima, per-row state slots, two parities of per-chunk counts and masses, and
  // two of per-chunk mass above the bracket (layout in metal/sample.metal).
  int64_t cells = static_cast<int64_t>(rows) * groups;
  int64_t words = cells + static_cast<int64_t>(rows) * STATE_SLOTS * STATE_WORDS +
      4 * cells * NUM_BINS + 2 * cells;
  mx::array workspace({static_cast<int>(words)}, mx::uint32, nullptr, {});
  workspace.set_data(mx::allocator::malloc(workspace.nbytes()));
  auto& encoder = mx::metal::get_command_encoder(s);
  encoder.add_temporary(workspace);

  auto bind_common = [&]() {
    encoder.set_input_array(logits, 0);
    encoder.set_input_array(temperatures, 1);
    encoder.set_input_array(top_k, 2);
    encoder.set_input_array(top_p, 3);
    encoder.set_output_array(workspace, 4);
    encoder.set_bytes(vocab, 5);
    encoder.set_bytes(chunk, 6);
    encoder.set_bytes(groups, 7);
  };
  auto chain = round_chain(key_bits, c.has_top_k, c.has_top_p);
  int num_rounds = static_cast<int>(chain.size());
  for (int i = 0; i < num_rounds; ++i) {
    encoder.set_compute_pipeline_state(round_pipeline(
        d,
        library,
        library_name,
        "sample_round",
        i,
        i > 0 ? chain[i - 1] : NO_ROUND,
        chain[i],
        c.has_top_k,
        c.has_top_p));
    bind_common();
    encoder.dispatch_threadgroups(MTL::Size(groups, rows, 1), MTL::Size(ROUND_THREADS, 1, 1));
  }
  encoder.set_compute_pipeline_state(round_pipeline(
      d,
      library,
      library_name,
      "sample_find",
      num_rounds,
      chain.back(),
      NO_ROUND,
      c.has_top_k,
      c.has_top_p));
  bind_common();
  encoder.set_input_array(random_bits, 8);
  encoder.set_output_array(tokens, 9);
  encoder.dispatch_threadgroups(MTL::Size(1, rows, 1), MTL::Size(FIND_THREADS, 1, 1));
}

} // namespace ftmetal
