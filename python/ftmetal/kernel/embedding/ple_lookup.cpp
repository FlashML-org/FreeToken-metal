#include "embedding/ple_lookup.h"

#include <stdexcept>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace ftmetal {

namespace {

std::string ple_lookup_source;
const char* LIBRARY_NAME = "ftmetal_ple_lookup";
// ple_wait's one threadgroup copies the rows out.
const int WAIT_THREADS = 256;

void check_mailbox(const mx::array& mailbox, const char* kernel) {
  if (mailbox.dtype() != mx::uint32 || mailbox.shape() != mx::Shape{PLE_MAILBOX_WORDS}) {
    throw std::invalid_argument(
        std::string("ftmetal ") + kernel + ": mailbox must be uint32 [" +
        std::to_string(PLE_MAILBOX_WORDS) + "]");
  }
}

} // namespace

void init_ple_lookup_library(const std::string& source) {
  ple_lookup_source = source;
}

std::vector<mx::array> ple_post(
    const mx::array& input_ids,
    const mx::array& slots,
    const mx::array& context,
    const mx::array& mailbox,
    mx::StreamOrDevice s) {
  if (input_ids.dtype() != mx::int32 || input_ids.ndim() != 1 || input_ids.size() < 1 ||
      input_ids.size() > MAX_PLE_ROWS) {
    throw std::invalid_argument(
        "ftmetal ple_post: input_ids must be int32 [num_rows], 1 <= num_rows <= " +
        std::to_string(MAX_PLE_ROWS));
  }
  if (slots.dtype() != mx::int32 || slots.shape() != input_ids.shape()) {
    throw std::invalid_argument("ftmetal ple_post: slots must be int32 [num_rows]");
  }
  if (context.dtype() != mx::int32 || context.ndim() != 2 || context.shape(1) != 2) {
    throw std::invalid_argument("ftmetal ple_post: context must be int32 [num_slots, 2]");
  }
  check_mailbox(mailbox, "ple_post");
  auto stream = mx::to_stream(s);
  int num_rows = static_cast<int>(input_ids.size());
  // The context pool and the mailbox stay as they are (see kvcache/pool_rows.h); eval_gpu checks
  // they are row contiguous.
  return mx::array::make_arrays(
      {context.shape(), input_ids.shape(), {1}},
      {mx::int32, mx::int32, mx::uint32},
      std::make_shared<PleLookup>(stream, PleKernel::Post, num_rows),
      {mx::contiguous(input_ids, false, stream),
       mx::contiguous(slots, false, stream),
       context,
       mailbox});
}

std::vector<mx::array> ple_wait(
    const mx::array& ticket,
    const mx::array& mailbox,
    const std::vector<mx::array>& staging,
    int num_rows,
    const std::vector<mx::array>& after,
    mx::StreamOrDevice s) {
  if (ticket.dtype() != mx::uint32 || ticket.shape() != mx::Shape{1}) {
    throw std::invalid_argument("ftmetal ple_wait: ticket must be ple_post's uint32 [1]");
  }
  check_mailbox(mailbox, "ple_wait");
  if (num_rows < 1 || num_rows > MAX_PLE_ROWS) {
    throw std::invalid_argument(
        "ftmetal ple_wait: num_rows must be in [1, " + std::to_string(MAX_PLE_ROWS) + "]");
  }
  if (staging.size() != 3) {
    throw std::invalid_argument(
        "ftmetal ple_wait: staging must be the weight, scales and biases buffers");
  }
  std::vector<mx::Shape> shapes;
  for (const mx::array& part : staging) {
    if (part.dtype() != mx::uint8 || part.ndim() != 3 || part.shape(0) != MAX_PLE_ROWS ||
        part.shape(1) != staging[0].shape(1)) {
      throw std::invalid_argument(
          "ftmetal ple_wait: staging parts must be uint8 [" + std::to_string(MAX_PLE_ROWS) +
          ", heads, row bytes] with one head count");
    }
    shapes.push_back({num_rows, part.shape(1), part.shape(2)});
  }
  auto stream = mx::to_stream(s);
  std::vector<mx::array> inputs{ticket, mailbox, staging[0], staging[1], staging[2]};
  inputs.insert(inputs.end(), after.begin(), after.end());
  return mx::array::make_arrays(
      std::move(shapes),
      {mx::uint8, mx::uint8, mx::uint8},
      std::make_shared<PleLookup>(stream, PleKernel::Wait, num_rows),
      std::move(inputs));
}

bool PleLookup::is_equivalent(const mx::Primitive& other) const {
  auto* lookup = dynamic_cast<const PleLookup*>(&other);
  return lookup != nullptr && lookup->kernel_ == kernel_ && lookup->num_rows_ == num_rows_;
}

void PleLookup::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("ftmetal ple lookup: GPU only");
}

void PleLookup::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto* library = d.get_library(LIBRARY_NAME, [&]() { return ple_lookup_source; });
  auto& encoder = mx::metal::get_command_encoder(s);
  if (kernel_ == PleKernel::Post) {
    // inputs: input_ids, slots, context, mailbox; outputs: context, input_ids copy, ticket.
    if (!inputs[2].flags().row_contiguous || !inputs[3].flags().row_contiguous) {
      throw std::runtime_error("ftmetal ple_post: context and mailbox must be row contiguous");
    }
    outputs[0].copy_shared_buffer(inputs[2]);
    outputs[1].set_data(mx::allocator::malloc(outputs[1].nbytes()));
    outputs[2].set_data(mx::allocator::malloc(outputs[2].nbytes()));
    encoder.set_compute_pipeline_state(d.get_kernel("ple_post", library));
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_output_array(outputs[0], 2);
    encoder.set_input_array(inputs[3], 3);
    encoder.set_output_array(outputs[1], 4);
    encoder.set_output_array(outputs[2], 5);
    PostParams params{num_rows_};
    encoder.set_bytes(params, 6);
    encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  } else {
    // inputs: ticket, mailbox, weight, scales and biases staging, after...; outputs: the
    // parts. The after arrays are not bound: they only order the encoding.
    for (int i = 1; i < 5; ++i) {
      if (!inputs[i].flags().row_contiguous) {
        throw std::runtime_error("ftmetal ple_wait: mailbox and staging must be row contiguous");
      }
    }
    WaitParams params{};
    for (int p = 0; p < 3; ++p) {
      outputs[p].set_data(mx::allocator::malloc(outputs[p].nbytes()));
      params.part_bytes[p] = static_cast<int>(outputs[p].nbytes());
    }
    encoder.set_compute_pipeline_state(d.get_kernel("ple_wait", library));
    for (int i = 0; i < 5; ++i) {
      encoder.set_input_array(inputs[i], i);
    }
    for (int p = 0; p < 3; ++p) {
      encoder.set_output_array(outputs[p], 5 + p);
    }
    encoder.set_bytes(params, 8);
    encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(WAIT_THREADS, 1, 1));
  }
}

} // namespace ftmetal
