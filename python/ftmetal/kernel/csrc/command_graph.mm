// CommandGraph recorder, capture classification and the GraphReplay primitive (see
// command_graph.h). The recorder replaces methods of the concrete classes behind a
// probe encoder and command buffer made on MLX's own device, so MTL_DEBUG_LAYER's
// wrapper classes are hooked the same way; every hook chains to the original and does
// nothing else while no session is open on the calling thread. The MLX internals read
// through member pointers (CommandEncoder::temporaries_ and encoder_,
// Device::library_kernels_ and kernel_mtx_) are pinned to the mlx 0.32.1 headers.

#include "command_graph.h"

#include <objc/runtime.h>

#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace ftmetal {

namespace {

// An explicit instantiation may name a private member; the friend it defines hands the
// member pointer out.
template <typename Tag, typename Tag::type Member>
struct PrivateMember {
  friend typename Tag::type private_member(Tag) {
    return Member;
  }
};

struct EncoderTemporaries {
  using type = std::vector<mx::array> mx::metal::CommandEncoder::*;
  friend type private_member(EncoderTemporaries);
};
template struct PrivateMember<EncoderTemporaries, &mx::metal::CommandEncoder::temporaries_>;

struct EncoderObject {
  using type = NS::SharedPtr<MTL::ComputeCommandEncoder> mx::metal::CommandEncoder::*;
  friend type private_member(EncoderObject);
};
template struct PrivateMember<EncoderObject, &mx::metal::CommandEncoder::encoder_>;

using KernelCache = std::unordered_map<
    MTL::Library*,
    std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>>>;

struct DeviceKernels {
  using type = KernelCache mx::metal::Device::*;
  friend type private_member(DeviceKernels);
};
template struct PrivateMember<DeviceKernels, &mx::metal::Device::library_kernels_>;

struct DeviceKernelMutex {
  using type = std::shared_mutex mx::metal::Device::*;
  friend type private_member(DeviceKernelMutex);
};
template struct PrivateMember<DeviceKernelMutex, &mx::metal::Device::kernel_mtx_>;

// Differences compare_graphs lists before it summarizes the rest.
const size_t MAX_REPORTED_DIFFERENCES = 40;
// Unclassified bindings end_capture lists before it summarizes the rest.
const size_t MAX_REPORTED_UNCLASSIFIED = 8;

struct IndexRange {
  NS::UInteger location;
  NS::UInteger length;
};

struct PendingBinding {
  uint32_t index;
  MTL::Buffer* buffer;
  int64_t offset;
};

struct TemporaryRecord {
  MTL::Buffer* buffer;
  int64_t nbytes;
  int64_t length;
};

struct Session {
  Session(bool recording, mx::Stream stream) : recording(recording), stream(stream) {}

  bool recording;
  mx::Stream stream;
  // Calls made from inside a hooked call (by the driver) are neither counted nor recorded.
  int depth = 0;
  CallCounts counts;

  std::shared_ptr<Graph> graph;
  std::string failure;
  size_t previous_cache_limit = 0;
  std::vector<mx::array> outputs;
  std::vector<mx::array> step_inputs;
  std::vector<mx::array> pools;
  std::vector<mx::array> externals;
  std::vector<mx::array> intermediates;
  std::vector<mx::array> leaves;
  std::vector<TemporaryRecord> temporaries;
  std::vector<mx::array> temporary_arrays;

  MTL::ComputePipelineState* pipeline = nullptr;
  bool sets_pipeline = false;
  bool pending_barrier = false;
  bool encoder_boundary = false;
  uint32_t encoder = 0;
  std::vector<PendingBinding> pending_bindings;
  std::vector<BytesArgument> pending_bytes;
  std::vector<ThreadgroupMemory> pending_memory;
};

thread_local Session* active_session = nullptr;

struct Nested {
  explicit Nested(Session* session) : session(session) {
    if (session != nullptr) {
      ++session->depth;
    }
  }
  ~Nested() {
    if (session != nullptr) {
      --session->depth;
    }
  }
  Session* session;
};

Session* recording_session() {
  Session* session = active_session;
  return session != nullptr && session->recording && session->depth == 0 ? session : nullptr;
}

Session* counting_session() {
  Session* session = active_session;
  return session != nullptr && session->depth == 0 ? session : nullptr;
}

void fail(Session& session, const std::string& message) {
  if (session.failure.empty()) {
    session.failure = message;
  }
}

void retain(Session& session, NS::Object* object) {
  if (object != nullptr && session.graph->retained_objects.insert(object).second) {
    object->retain();
  }
}

template <typename Record>
void drop_index(std::vector<Record>& records, NS::UInteger index) {
  std::erase_if(records, [index](const Record& record) { return record.index == index; });
}

void record_buffer(Session& session, id buffer, NS::UInteger offset, NS::UInteger index) {
  drop_index(session.pending_bindings, index);
  drop_index(session.pending_bytes, index);
  auto* metal_buffer = reinterpret_cast<MTL::Buffer*>(buffer);
  retain(session, metal_buffer);
  session.pending_bindings.push_back(
      {static_cast<uint32_t>(index), metal_buffer, static_cast<int64_t>(offset)});
}

void record_bytes(Session& session, const void* bytes, NS::UInteger length, NS::UInteger index) {
  drop_index(session.pending_bindings, index);
  drop_index(session.pending_bytes, index);
  auto& data = session.graph->bytes_data;
  auto offset = static_cast<uint32_t>(data.size());
  const auto* begin = static_cast<const uint8_t*>(bytes);
  data.insert(data.end(), begin, begin + length);
  session.pending_bytes.push_back(
      {static_cast<uint32_t>(index), offset, static_cast<uint32_t>(length)});
}

void record_dispatch(Session& session, GridSize grid, GridSize group, bool is_threads) {
  Graph& graph = *session.graph;
  if (session.pipeline == nullptr) {
    fail(session, "a dispatch without a compute pipeline state");
    return;
  }
  Dispatch dispatch{};
  dispatch.pipeline = session.pipeline;
  dispatch.sets_pipeline = session.sets_pipeline;
  dispatch.barrier_before =
      !graph.dispatches.empty() && (session.pending_barrier || session.encoder_boundary);
  dispatch.is_threads = is_threads;
  dispatch.encoder = session.encoder;
  dispatch.grid = grid;
  dispatch.group = group;
  dispatch.bindings_begin = static_cast<uint32_t>(graph.bindings.size());
  for (const auto& pending : session.pending_bindings) {
    graph.bindings.push_back(
        {pending.buffer, pending.offset, pending.index, BufferRole::Null, 0});
  }
  dispatch.bindings_end = static_cast<uint32_t>(graph.bindings.size());
  dispatch.bytes_begin = static_cast<uint32_t>(graph.bytes_arguments.size());
  graph.bytes_arguments.insert(
      graph.bytes_arguments.end(), session.pending_bytes.begin(), session.pending_bytes.end());
  dispatch.bytes_end = static_cast<uint32_t>(graph.bytes_arguments.size());
  dispatch.memory_begin = static_cast<uint32_t>(graph.threadgroup_memory.size());
  graph.threadgroup_memory.insert(
      graph.threadgroup_memory.end(),
      session.pending_memory.begin(),
      session.pending_memory.end());
  dispatch.memory_end = static_cast<uint32_t>(graph.threadgroup_memory.size());
  graph.dispatches.push_back(dispatch);
  session.pending_bindings.clear();
  session.pending_bytes.clear();
  session.pending_memory.clear();
  session.sets_pipeline = false;
  session.pending_barrier = false;
  session.encoder_boundary = false;
}

// MLX's end_encoding updates the fence, then moves temporaries_ into the completion
// handler, then ends the encoder: at updateFence the list is complete.
void record_temporaries(Session& session, id encoder) {
  mx::metal::CommandEncoder* command_encoder = nullptr;
  try {
    command_encoder = &mx::metal::get_command_encoder(session.stream);
  } catch (const std::runtime_error& error) {
    fail(session, std::string("updateFence: ") + error.what());
    return;
  }
  auto* mlx_encoder = (command_encoder->*private_member(EncoderObject{})).get();
  if (mlx_encoder != reinterpret_cast<MTL::ComputeCommandEncoder*>(encoder)) {
    fail(session, "updateFence: on an encoder that is not MLX's encoder of the capture stream");
    return;
  }
  for (const auto& temporary : command_encoder->*private_member(EncoderTemporaries{})) {
    auto* buffer = static_cast<MTL::Buffer*>(const_cast<void*>(temporary.buffer().ptr()));
    retain(session, buffer);
    session.temporaries.push_back(
        {buffer,
         static_cast<int64_t>(temporary.nbytes()),
         buffer == nullptr ? 0 : static_cast<int64_t>(buffer->length())});
    session.temporary_arrays.push_back(temporary);
  }
}

using PipelineMethod = void (*)(id, SEL, id);
using BufferMethod = void (*)(id, SEL, id, NS::UInteger, NS::UInteger);
using BytesMethod = void (*)(id, SEL, const void*, NS::UInteger, NS::UInteger);
using MemoryMethod = void (*)(id, SEL, NS::UInteger, NS::UInteger);
using DispatchMethod = void (*)(id, SEL, GridSize, GridSize);
using BarrierMethod = void (*)(id, SEL, NS::UInteger);
using FenceMethod = void (*)(id, SEL, id);
using PlainMethod = void (*)(id, SEL);
using EventMethod = void (*)(id, SEL, id, uint64_t);
using HandlerMethod = void (*)(id, SEL, id);
using EncoderMethod = id (*)(id, SEL, NS::UInteger);

PipelineMethod original_set_pipeline = nullptr;
BufferMethod original_set_buffer = nullptr;
BytesMethod original_set_bytes = nullptr;
MemoryMethod original_set_memory = nullptr;
DispatchMethod original_dispatch_threadgroups = nullptr;
DispatchMethod original_dispatch_threads = nullptr;
BarrierMethod original_barrier = nullptr;
FenceMethod original_update_fence = nullptr;
FenceMethod original_wait_fence = nullptr;
PlainMethod original_end_encoding = nullptr;
PlainMethod original_commit = nullptr;
EventMethod original_signal_event = nullptr;
HandlerMethod original_add_completed_handler = nullptr;
EncoderMethod original_compute_encoder = nullptr;

void hook_set_pipeline(id self, SEL command, id pipeline) {
  if (Session* session = recording_session()) {
    auto* metal_pipeline = reinterpret_cast<MTL::ComputePipelineState*>(pipeline);
    retain(*session, metal_pipeline);
    session->pipeline = metal_pipeline;
    session->sets_pipeline = true;
  }
  Nested nested(active_session);
  original_set_pipeline(self, command, pipeline);
}

void hook_set_buffer(id self, SEL command, id buffer, NS::UInteger offset, NS::UInteger index) {
  if (Session* session = recording_session()) {
    record_buffer(*session, buffer, offset, index);
  }
  Nested nested(active_session);
  original_set_buffer(self, command, buffer, offset, index);
}

void hook_set_bytes(
    id self,
    SEL command,
    const void* bytes,
    NS::UInteger length,
    NS::UInteger index) {
  if (Session* session = recording_session()) {
    record_bytes(*session, bytes, length, index);
  }
  Nested nested(active_session);
  original_set_bytes(self, command, bytes, length, index);
}

void hook_set_memory(id self, SEL command, NS::UInteger length, NS::UInteger index) {
  if (Session* session = recording_session()) {
    drop_index(session->pending_memory, index);
    session->pending_memory.push_back({static_cast<uint32_t>(index), length});
  }
  Nested nested(active_session);
  original_set_memory(self, command, length, index);
}

void hook_dispatch_threadgroups(id self, SEL command, GridSize grid, GridSize group) {
  if (Session* session = counting_session()) {
    ++session->counts.dispatches;
    if (session->recording) {
      record_dispatch(*session, grid, group, false);
    }
  }
  Nested nested(active_session);
  original_dispatch_threadgroups(self, command, grid, group);
}

void hook_dispatch_threads(id self, SEL command, GridSize grid, GridSize group) {
  if (Session* session = counting_session()) {
    ++session->counts.dispatches;
    if (session->recording) {
      record_dispatch(*session, grid, group, true);
    }
  }
  Nested nested(active_session);
  original_dispatch_threads(self, command, grid, group);
}

void hook_barrier(id self, SEL command, NS::UInteger scope) {
  if (Session* session = counting_session()) {
    ++session->counts.barriers;
    session->pending_barrier = session->recording;
  }
  Nested nested(active_session);
  original_barrier(self, command, scope);
}

void hook_update_fence(id self, SEL command, id fence) {
  if (Session* session = counting_session()) {
    ++session->counts.fence_updates;
    if (session->recording) {
      record_temporaries(*session, self);
    }
  }
  Nested nested(active_session);
  original_update_fence(self, command, fence);
}

void hook_wait_fence(id self, SEL command, id fence) {
  if (Session* session = counting_session()) {
    ++session->counts.fence_waits;
  }
  Nested nested(active_session);
  original_wait_fence(self, command, fence);
}

void hook_end_encoding(id self, SEL command) {
  if (Session* session = recording_session()) {
    session->encoder_boundary = true;
    ++session->encoder;
  }
  Nested nested(active_session);
  original_end_encoding(self, command);
}

void hook_commit(id self, SEL command) {
  if (Session* session = counting_session()) {
    ++session->counts.commits;
  }
  Nested nested(active_session);
  original_commit(self, command);
}

void hook_signal_event(id self, SEL command, id event, uint64_t value) {
  if (Session* session = counting_session()) {
    ++session->counts.signal_events;
  }
  Nested nested(active_session);
  original_signal_event(self, command, event, value);
}

void hook_add_completed_handler(id self, SEL command, id handler) {
  if (Session* session = counting_session()) {
    ++session->counts.completion_handlers;
  }
  Nested nested(active_session);
  original_add_completed_handler(self, command, handler);
}

id hook_compute_encoder(id self, SEL command, NS::UInteger dispatch_type) {
  if (Session* session = counting_session()) {
    ++session->counts.encoders;
    if (session->recording &&
        dispatch_type != static_cast<NS::UInteger>(MTL::DispatchTypeConcurrent)) {
      fail(*session, "a serial compute encoder (replay encodes into a concurrent one)");
    }
  }
  Nested nested(active_session);
  return original_compute_encoder(self, command, dispatch_type);
}

// Calls replay cannot rebuild: seen while recording, they fail the capture.
const int NUM_FORBIDDEN = 19;
IMP forbidden_originals[NUM_FORBIDDEN] = {};

template <int Slot, typename Result, typename... Args>
Result hook_forbidden(id self, SEL command, Args... args) {
  if (Session* session = recording_session()) {
    fail(
        *session,
        std::string("-[") + class_getName(object_getClass(self)) + " " + sel_getName(command) +
            "] is not replayable");
  }
  Nested nested(active_session);
  return reinterpret_cast<Result (*)(id, SEL, Args...)>(forbidden_originals[Slot])(
      self, command, args...);
}

// Replace cls's implementation of selector (adding an override when it is inherited);
// returns the original, or nullptr when cls does not respond to selector.
IMP install(Class cls, SEL selector, IMP hook) {
  Method method = class_getInstanceMethod(cls, selector);
  if (method == nullptr) {
    return nullptr;
  }
  IMP original = method_getImplementation(method);
  if (!class_addMethod(cls, selector, hook, method_getTypeEncoding(method))) {
    original = method_setImplementation(method, hook);
  }
  return original;
}

template <typename Function>
Function install_required(Class cls, SEL selector, Function hook) {
  IMP original = install(cls, selector, reinterpret_cast<IMP>(hook));
  if (original == nullptr) {
    throw std::runtime_error(
        std::string("CommandGraph: ") + class_getName(cls) + " does not implement " +
        sel_getName(selector));
  }
  return reinterpret_cast<Function>(original);
}

template <int Slot, typename Result, typename... Args>
void install_forbidden(Class cls, SEL selector) {
  static_assert(Slot < NUM_FORBIDDEN);
  forbidden_originals[Slot] =
      install(cls, selector, reinterpret_cast<IMP>(&hook_forbidden<Slot, Result, Args...>));
}

void install_hooks() {
  static std::once_flag once;
  std::call_once(once, [] {
    auto* pool = NS::AutoreleasePool::alloc()->init();
    auto* device = mx::metal::device(mx::Device::gpu).mtl_device();
    auto* queue = device->newCommandQueue();
    auto* command_buffer = queue->commandBufferWithUnretainedReferences();
    auto* encoder = command_buffer->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    Class encoder_class = object_getClass(reinterpret_cast<id>(encoder));
    Class command_buffer_class = object_getClass(reinterpret_cast<id>(command_buffer));
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    queue->release();
    pool->release();

    Class e = encoder_class;
    original_set_pipeline =
        install_required(e, @selector(setComputePipelineState:), &hook_set_pipeline);
    original_set_buffer =
        install_required(e, @selector(setBuffer:offset:atIndex:), &hook_set_buffer);
    original_set_bytes = install_required(e, @selector(setBytes:length:atIndex:), &hook_set_bytes);
    original_set_memory = install_required(
        e, @selector(setThreadgroupMemoryLength:atIndex:), &hook_set_memory);
    original_dispatch_threadgroups = install_required(
        e, @selector(dispatchThreadgroups:threadsPerThreadgroup:), &hook_dispatch_threadgroups);
    original_dispatch_threads = install_required(
        e, @selector(dispatchThreads:threadsPerThreadgroup:), &hook_dispatch_threads);
    original_barrier = install_required(e, @selector(memoryBarrierWithScope:), &hook_barrier);
    original_update_fence = install_required(e, @selector(updateFence:), &hook_update_fence);
    original_wait_fence = install_required(e, @selector(waitForFence:), &hook_wait_fence);
    original_end_encoding = install_required(e, @selector(endEncoding), &hook_end_encoding);

    install_forbidden<0, void, const id*, const NS::UInteger*, IndexRange>(
        e, @selector(setBuffers:offsets:withRange:));
    install_forbidden<1, void, NS::UInteger, NS::UInteger>(e, @selector(setBufferOffset:atIndex:));
    install_forbidden<2, void, id, NS::UInteger, NS::UInteger, NS::UInteger>(
        e, @selector(setBuffer:offset:attributeStride:atIndex:));
    install_forbidden<3, void, const void*, NS::UInteger, NS::UInteger, NS::UInteger>(
        e, @selector(setBytes:length:attributeStride:atIndex:));
    install_forbidden<4, void, id, NS::UInteger>(e, @selector(useResource:usage:));
    install_forbidden<5, void, const id*, NS::UInteger, NS::UInteger>(
        e, @selector(useResources:count:usage:));
    install_forbidden<6, void, id>(e, @selector(useHeap:));
    install_forbidden<7, void, const id*, NS::UInteger>(e, @selector(useHeaps:count:));
    install_forbidden<8, void, const id*, NS::UInteger>(
        e, @selector(memoryBarrierWithResources:count:));
    install_forbidden<9, void, id, IndexRange>(e, @selector(executeCommandsInBuffer:withRange:));
    install_forbidden<10, void, id, id, NS::UInteger>(
        e, @selector(executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:));
    install_forbidden<11, void, id, NS::UInteger, GridSize>(
        e, @selector(dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:));
    install_forbidden<12, void, id, NS::UInteger>(e, @selector(setTexture:atIndex:));
    install_forbidden<13, void, id, NS::UInteger>(e, @selector(setSamplerState:atIndex:));

    Class c = command_buffer_class;
    original_commit = install_required(c, @selector(commit), &hook_commit);
    original_signal_event =
        install_required(c, @selector(encodeSignalEvent:value:), &hook_signal_event);
    original_add_completed_handler =
        install_required(c, @selector(addCompletedHandler:), &hook_add_completed_handler);
    original_compute_encoder = install_required(
        c, @selector(computeCommandEncoderWithDispatchType:), &hook_compute_encoder);
    install_forbidden<14, id>(c, @selector(blitCommandEncoder));
    install_forbidden<15, id, id>(c, @selector(blitCommandEncoderWithDescriptor:));
    install_forbidden<16, void, id, uint64_t>(c, @selector(encodeWaitForEvent:value:));
    install_forbidden<17, id>(c, @selector(computeCommandEncoder));
    install_forbidden<18, id, id>(c, @selector(computeCommandEncoderWithDescriptor:));
  });
}

std::unordered_map<const void*, std::string> pipeline_names() {
  auto& device = mx::metal::device(mx::Device::gpu);
  std::shared_lock lock(device.*private_member(DeviceKernelMutex{}));
  std::unordered_map<const void*, std::string> names;
  for (const auto& [library, kernels] : device.*private_member(DeviceKernels{})) {
    for (const auto& [name, pipeline] : kernels) {
      names.emplace(pipeline.get(), name);
    }
  }
  return names;
}

const char* role_name(BufferRole role) {
  switch (role) {
    case BufferRole::StepInput:
      return "step input";
    case BufferRole::Pool:
      return "pool";
    case BufferRole::External:
      return "external";
    case BufferRole::Logits:
      return "logits";
    case BufferRole::Constant:
      return "constant";
    case BufferRole::Intermediate:
      return "intermediate";
    case BufferRole::Temporary:
      return "temporary";
    case BufferRole::Null:
      return "null";
  }
  return "?";
}

bool is_parameter(BufferRole role) {
  return role == BufferRole::StepInput || role == BufferRole::Pool || role == BufferRole::Logits;
}

// Temporaries are made without a primitive and stay unscheduled: test the data, not the status.
bool has_data(const mx::array& array) {
  return array.data_shared_ptr() != nullptr;
}

int64_t extent_bytes(const mx::array& array) {
  return static_cast<int64_t>(array.data_size() * array.itemsize());
}

void check_evaluated(const std::vector<mx::array>& arrays, const char* what) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (arrays[i].status() == mx::array::Status::unscheduled) {
      throw std::invalid_argument(
          std::string("begin_capture: ") + what + " " + std::to_string(i) +
          " is not evaluated; eval it before the capture");
    }
  }
}

struct BufferClass {
  BufferRole role;
  uint32_t parameter;
  int64_t base_offset;
  int64_t extent;
};

// Sort every binding into its role (see BufferRole) and fill the graph's specs, names,
// constant contents and stats; throws on a binding replay could not rebuild.
void classify(Session& session) {
  Graph& graph = *session.graph;
  std::unordered_map<const void*, BufferClass> classes;
  auto add = [&classes](const mx::array& array, BufferRole role, uint32_t parameter) {
    const void* buffer = has_data(array) ? array.buffer().ptr() : nullptr;
    return buffer != nullptr &&
        classes.emplace(buffer, BufferClass{role, parameter, array.offset(), extent_bytes(array)})
            .second;
  };
  for (size_t k = 0; k < session.step_inputs.size(); ++k) {
    if (!add(session.step_inputs[k], BufferRole::StepInput, k)) {
      throw std::invalid_argument(
          "end_capture: step input " + std::to_string(k) +
          " shares its buffer with another step input (or has none)");
    }
  }
  for (size_t k = 0; k < session.pools.size(); ++k) {
    if (!add(session.pools[k], BufferRole::Pool, k)) {
      throw std::invalid_argument(
          "end_capture: pool " + std::to_string(k) +
          " shares its buffer with a step input or another pool (or has none)");
    }
  }
  for (const auto& external : session.externals) {
    add(external, BufferRole::External, 0);
  }
  const auto& logits = session.outputs[0];
  if (!logits.flags().row_contiguous || logits.data_size() != logits.size()) {
    throw std::invalid_argument("end_capture: the logits must be row contiguous");
  }
  if (!add(logits, BufferRole::Logits, 0)) {
    throw std::invalid_argument("end_capture: the logits alias a step input, pool or external");
  }
  for (const auto& leaf : session.leaves) {
    add(leaf, BufferRole::Constant, 0);
  }
  for (const auto& intermediate : session.intermediates) {
    add(intermediate, BufferRole::Intermediate, 0);
  }
  for (const auto& temporary : session.temporary_arrays) {
    add(temporary, BufferRole::Temporary, 0);
  }

  // A fresh logits buffer replaces the capture's, so no other array may live in it.
  const void* logits_buffer = logits.buffer().ptr();
  auto check_inside_logits = [&](const mx::array& array) {
    if (!has_data(array) || array.buffer().ptr() != logits_buffer) {
      return;
    }
    if (array.offset() < logits.offset() ||
        array.offset() + extent_bytes(array) > logits.offset() + extent_bytes(logits)) {
      throw std::invalid_argument("end_capture: the logits must own their buffer");
    }
  };
  for (const auto& array : session.intermediates) {
    check_inside_logits(array);
  }
  for (const auto& array : session.temporary_arrays) {
    check_inside_logits(array);
  }
  for (size_t k = 0; k < session.pools.size(); ++k) {
    if (session.outputs[1 + k].buffer().ptr() != session.pools[k].buffer().ptr()) {
      throw std::invalid_argument(
          "end_capture: updated pool " + std::to_string(k) + " does not alias pool " +
          std::to_string(k) + " (the step must update it in place)");
    }
  }

  auto names = pipeline_names();
  graph.dispatch_names.reserve(graph.dispatches.size());
  for (const auto& dispatch : graph.dispatches) {
    auto it = names.find(dispatch.pipeline);
    graph.dispatch_names.push_back(it == names.end() ? "?" : it->second);
  }

  const int num_roles = static_cast<int>(BufferRole::Null) + 1;
  std::vector<std::unordered_map<const void*, uint32_t>> ordinals(num_roles);
  std::vector<std::string> unclassified;
  size_t num_unclassified = 0;
  for (size_t d = 0; d < graph.dispatches.size(); ++d) {
    const auto& dispatch = graph.dispatches[d];
    for (uint32_t i = dispatch.bindings_begin; i < dispatch.bindings_end; ++i) {
      auto& binding = graph.bindings[i];
      if (binding.buffer == nullptr) {
        binding.role = BufferRole::Null;
        continue;
      }
      auto it = classes.find(binding.buffer);
      if (it == classes.end()) {
        if (++num_unclassified <= MAX_REPORTED_UNCLASSIFIED) {
          unclassified.push_back(
              "dispatch " + std::to_string(d) + " (" + graph.dispatch_names[d] + ") slot " +
              std::to_string(binding.index) + ": buffer of " +
              std::to_string(binding.buffer->length()) + " bytes at offset " +
              std::to_string(binding.offset));
        }
        continue;
      }
      const auto& buffer_class = it->second;
      binding.role = buffer_class.role;
      auto& role_ordinals = ordinals[static_cast<int>(binding.role)];
      auto ordinal =
          role_ordinals.emplace(binding.buffer, static_cast<uint32_t>(role_ordinals.size()))
              .first->second;
      if (is_parameter(binding.role)) {
        binding.id = buffer_class.parameter;
        binding.offset -= buffer_class.base_offset;
        if (binding.offset < 0 ||
            (buffer_class.extent > 0 && binding.offset >= buffer_class.extent)) {
          throw std::invalid_argument(
              "end_capture: dispatch " + std::to_string(d) + " (" + graph.dispatch_names[d] +
              ") binds its " + role_name(binding.role) + " buffer outside the array");
        }
      } else {
        binding.id = ordinal;
      }
    }
  }
  if (num_unclassified > 0) {
    std::string message = "end_capture: " + std::to_string(num_unclassified) +
        " bindings are neither whitelisted, graph arrays nor MLX temporaries:";
    for (const auto& line : unclassified) {
      message += "\n  " + line;
    }
    throw std::runtime_error(message);
  }

  std::unordered_map<const void*, std::vector<const mx::array*>> leaves_by_buffer;
  for (const auto& leaf : session.leaves) {
    leaves_by_buffer[leaf.buffer().ptr()].push_back(&leaf);
  }
  const auto& constant_ordinals = ordinals[static_cast<int>(BufferRole::Constant)];
  graph.constant_contents.resize(constant_ordinals.size());
  for (const auto& [buffer, ordinal] : constant_ordinals) {
    auto& contents = graph.constant_contents[ordinal];
    for (const auto* leaf : leaves_by_buffer[buffer]) {
      const auto* data = leaf->data<uint8_t>();
      contents.insert(contents.end(), data, data + extent_bytes(*leaf));
    }
  }

  for (const auto& input : session.step_inputs) {
    graph.step_inputs.push_back({input.shape(), input.dtype()});
  }
  for (const auto& pool : session.pools) {
    graph.pools.push_back({pool.shape(), pool.dtype()});
  }
  graph.logits = {logits.shape(), logits.dtype()};

  GraphStats& stats = graph.stats;
  stats.num_dispatches = static_cast<int64_t>(graph.dispatches.size());
  std::unordered_set<uint32_t> encoders;
  for (const auto& dispatch : graph.dispatches) {
    encoders.insert(dispatch.encoder);
    stats.num_barriers += dispatch.barrier_before ? 1 : 0;
  }
  stats.num_encoders = static_cast<int64_t>(encoders.size());
  stats.num_bindings = static_cast<int64_t>(graph.bindings.size());
  stats.num_bytes_arguments = static_cast<int64_t>(graph.bytes_arguments.size());
  auto distinct = [&ordinals](BufferRole role) {
    return static_cast<int64_t>(ordinals[static_cast<int>(role)].size());
  };
  stats.num_step_input_buffers = distinct(BufferRole::StepInput);
  stats.num_pool_buffers = distinct(BufferRole::Pool);
  stats.num_external_buffers = distinct(BufferRole::External);
  stats.num_constant_buffers = distinct(BufferRole::Constant);
  stats.num_intermediate_buffers = distinct(BufferRole::Intermediate);
  stats.num_temporary_buffers = distinct(BufferRole::Temporary);
  for (const auto& temporary : session.temporaries) {
    stats.temporaries.emplace_back(temporary.nbytes, temporary.length);
  }
  std::unordered_set<const void*> pinned;
  auto pin = [&](const mx::array& array) {
    if (!has_data(array)) {
      return;
    }
    const void* buffer = array.buffer().ptr();
    auto it = classes.find(buffer);
    bool owned = it != classes.end() &&
        (it->second.role == BufferRole::Logits || it->second.role == BufferRole::Constant ||
         it->second.role == BufferRole::Intermediate ||
         it->second.role == BufferRole::Temporary);
    if (owned && pinned.insert(buffer).second) {
      stats.pinned_bytes += static_cast<int64_t>(static_cast<const MTL::Buffer*>(buffer)->length());
    }
  };
  for (const auto& array : session.intermediates) {
    pin(array);
  }
  for (const auto& array : session.leaves) {
    pin(array);
  }
  for (const auto& array : session.temporary_arrays) {
    pin(array);
  }
  stats.num_held_arrays =
      static_cast<int64_t>(session.intermediates.size() + session.leaves.size());
  stats.calls = session.counts;

  auto& held = graph.held_arrays;
  held.reserve(
      session.intermediates.size() + session.leaves.size() + session.temporary_arrays.size());
  for (auto* arrays : {&session.intermediates, &session.leaves, &session.temporary_arrays}) {
    held.insert(held.end(), arrays->begin(), arrays->end());
  }
}

std::string grid_text(const GridSize& size) {
  return "(" + std::to_string(size.width) + ", " + std::to_string(size.height) + ", " +
      std::to_string(size.depth) + ")";
}

bool same_grid(const GridSize& a, const GridSize& b) {
  return a.width == b.width && a.height == b.height && a.depth == b.depth;
}

} // namespace

Graph::~Graph() {
  for (void* object : retained_objects) {
    static_cast<NS::Object*>(object)->release();
  }
}

void begin_capture(
    const std::vector<mx::array>& outputs,
    const std::vector<mx::array>& step_inputs,
    const std::vector<mx::array>& pools,
    const std::vector<mx::array>& externals) {
  if (active_session != nullptr) {
    throw std::runtime_error("begin_capture: a capture or count is already open on this thread");
  }
  if (outputs.size() != pools.size() + 1) {
    throw std::invalid_argument(
        "begin_capture: outputs must be the logits followed by one updated array per pool");
  }
  if (outputs[0].status() != mx::array::Status::unscheduled) {
    throw std::invalid_argument("begin_capture: the logits are already evaluated");
  }
  check_evaluated(step_inputs, "step input");
  check_evaluated(pools, "pool");
  check_evaluated(externals, "external");
  for (size_t k = 0; k < step_inputs.size(); ++k) {
    if (!step_inputs[k].flags().row_contiguous) {
      throw std::invalid_argument(
          "begin_capture: step input " + std::to_string(k) + " is not row contiguous");
    }
  }
  auto stream = outputs[0].primitive().stream();
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument("begin_capture: the logits are not computed on the GPU");
  }
  install_hooks();

  auto session = std::make_unique<Session>(true, stream);
  session->graph = std::make_shared<Graph>();
  session->outputs = outputs;
  session->step_inputs = step_inputs;
  session->pools = pools;
  session->externals = externals;
  // Hold every array of the lazy graph: no donation, and nothing it binds is freed or
  // handed to another array while the graph lives. Arrays with data are leaves.
  std::unordered_set<std::uintptr_t> visited;
  std::vector<mx::array> stack(outputs.begin(), outputs.end());
  while (!stack.empty()) {
    mx::array array = std::move(stack.back());
    stack.pop_back();
    if (!visited.insert(array.id()).second) {
      continue;
    }
    if (array.status() != mx::array::Status::unscheduled) {
      session->leaves.push_back(std::move(array));
      continue;
    }
    if (array.primitive().device() != mx::Device::gpu ||
        array.primitive().stream() != stream) {
      throw std::invalid_argument(
          std::string("begin_capture: ") + array.primitive().name() +
          " runs off the capture's GPU stream");
    }
    for (const auto& sibling : array.siblings()) {
      stack.push_back(sibling);
    }
    for (const auto& input : array.inputs()) {
      stack.push_back(input);
    }
    session->intermediates.push_back(std::move(array));
  }
  session->previous_cache_limit = mx::set_cache_limit(0);
  active_session = session.release();
}

std::shared_ptr<Graph> end_capture() {
  if (active_session == nullptr || !active_session->recording) {
    throw std::runtime_error("end_capture: no capture is open on this thread");
  }
  std::unique_ptr<Session> session(active_session);
  active_session = nullptr;
  mx::set_cache_limit(session->previous_cache_limit);
  mx::synchronize(session->stream);
  if (!session->failure.empty()) {
    throw std::runtime_error("CommandGraph capture failed: " + session->failure);
  }
  for (const auto& output : session->outputs) {
    if (output.status() == mx::array::Status::unscheduled) {
      throw std::runtime_error("end_capture: the outputs were not evaluated during the capture");
    }
  }
  if (session->graph->dispatches.empty()) {
    throw std::runtime_error("end_capture: the capture recorded no dispatch");
  }
  classify(*session);
  return session->graph;
}

std::vector<mx::array> graph_replay(
    std::shared_ptr<Graph> graph,
    const std::vector<mx::array>& step_inputs,
    const std::vector<mx::array>& pools,
    mx::StreamOrDevice s) {
  if (step_inputs.size() != graph->step_inputs.size() || pools.size() != graph->pools.size()) {
    throw std::invalid_argument(
        "graph_replay: the graph takes " + std::to_string(graph->step_inputs.size()) +
        " step inputs and " + std::to_string(graph->pools.size()) + " pools");
  }
  auto check = [](const mx::array& array, const ArraySpec& spec, const char* what, size_t k) {
    if (array.shape() != spec.shape || array.dtype() != spec.dtype) {
      throw std::invalid_argument(
          std::string("graph_replay: ") + what + " " + std::to_string(k) +
          " differs in shape or dtype from the capture");
    }
  };
  auto stream = mx::to_stream(s);
  std::vector<mx::array> inputs;
  inputs.reserve(step_inputs.size() + pools.size());
  for (size_t k = 0; k < step_inputs.size(); ++k) {
    check(step_inputs[k], graph->step_inputs[k], "step input", k);
    inputs.push_back(mx::contiguous(step_inputs[k], false, stream));
  }
  std::vector<mx::Shape> shapes = {graph->logits.shape};
  std::vector<mx::Dtype> dtypes = {graph->logits.dtype};
  for (size_t k = 0; k < pools.size(); ++k) {
    check(pools[k], graph->pools[k], "pool", k);
    inputs.push_back(pools[k]);
    shapes.push_back(pools[k].shape());
    dtypes.push_back(pools[k].dtype());
  }
  return mx::array::make_arrays(
      std::move(shapes), dtypes, std::make_shared<GraphReplay>(stream, std::move(graph)), inputs);
}

void GraphReplay::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("GraphReplay: GPU only");
}

void GraphReplay::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  const Graph& graph = *graph_;
  size_t num_step_inputs = graph.step_inputs.size();
  size_t num_pools = graph.pools.size();
  auto& logits = outputs[0];
  logits.set_data(mx::allocator::malloc(logits.nbytes()));
  for (size_t k = 0; k < num_pools; ++k) {
    const auto& pool = inputs[num_step_inputs + k];
    if (!pool.flags().row_contiguous) {
      throw std::runtime_error("GraphReplay: pool " + std::to_string(k) + " is not row contiguous");
    }
    outputs[1 + k].copy_shared_buffer(pool);
  }
  // Step inputs, then pools, then the logits: the buffer and offset each parameter binds.
  std::vector<std::pair<const MTL::Buffer*, int64_t>> parameters;
  parameters.reserve(inputs.size() + 1);
  for (const auto& input : inputs) {
    parameters.emplace_back(static_cast<const MTL::Buffer*>(input.buffer().ptr()), input.offset());
  }
  parameters.emplace_back(static_cast<const MTL::Buffer*>(logits.buffer().ptr()), logits.offset());
  size_t pool_base = num_step_inputs;
  size_t logits_index = num_step_inputs + num_pools;

  auto& encoder = mx::metal::get_command_encoder(stream());
  // Orders this replay after everything already in the encoder, a previous replay included.
  encoder.barrier();
  for (const auto& input : inputs) {
    encoder.set_input_array(input, 0);
  }
  for (const auto& output : outputs) {
    encoder.register_output_array(output);
  }
  for (const auto& dispatch : graph.dispatches) {
    if (dispatch.barrier_before) {
      encoder.barrier();
    }
    if (dispatch.sets_pipeline) {
      encoder.set_compute_pipeline_state(dispatch.pipeline);
    }
    for (uint32_t i = dispatch.bindings_begin; i < dispatch.bindings_end; ++i) {
      const auto& binding = graph.bindings[i];
      switch (binding.role) {
        case BufferRole::StepInput: {
          const auto& [buffer, offset] = parameters[binding.id];
          encoder.set_buffer(buffer, binding.index, offset + binding.offset);
          break;
        }
        case BufferRole::Pool: {
          const auto& [buffer, offset] = parameters[pool_base + binding.id];
          encoder.set_buffer(buffer, binding.index, offset + binding.offset);
          break;
        }
        case BufferRole::Logits: {
          const auto& [buffer, offset] = parameters[logits_index];
          encoder.set_buffer(buffer, binding.index, offset + binding.offset);
          break;
        }
        case BufferRole::Null:
          break;
        default:
          encoder.set_buffer(binding.buffer, binding.index, binding.offset);
      }
    }
    for (uint32_t i = dispatch.bytes_begin; i < dispatch.bytes_end; ++i) {
      const auto& argument = graph.bytes_arguments[i];
      encoder.set_bytes(
          graph.bytes_data.data() + argument.offset,
          static_cast<int>(argument.length),
          static_cast<int>(argument.index));
    }
    for (uint32_t i = dispatch.memory_begin; i < dispatch.memory_end; ++i) {
      const auto& memory = graph.threadgroup_memory[i];
      encoder.set_threadgroup_memory_length(memory.length, static_cast<int>(memory.index));
    }
    MTL::Size grid(dispatch.grid.width, dispatch.grid.height, dispatch.grid.depth);
    MTL::Size group(dispatch.group.width, dispatch.group.height, dispatch.group.depth);
    if (dispatch.is_threads) {
      encoder.dispatch_threads(grid, group);
    } else {
      encoder.dispatch_threadgroups(grid, group);
    }
  }
  // Command buffers do not retain what they bind: the graph (its pinned buffers and
  // pipelines) lives until this one completes.
  MTL::CommandBufferHandler release_graph = [graph = graph_](MTL::CommandBuffer*) mutable {
    graph.reset();
  };
  encoder.get_command_buffer()->addCompletedHandler(release_graph);
}

std::vector<std::string> compare_graphs(const Graph& first, const Graph& second) {
  std::vector<std::string> differences;
  size_t num_differences = 0;
  auto report = [&](std::string message) {
    if (++num_differences <= MAX_REPORTED_DIFFERENCES) {
      differences.push_back(std::move(message));
    }
  };
  if (first.dispatches.size() != second.dispatches.size()) {
    report(
        "dispatch count " + std::to_string(first.dispatches.size()) + " vs " +
        std::to_string(second.dispatches.size()));
    return differences;
  }
  if (first.step_inputs.size() != second.step_inputs.size() ||
      first.pools.size() != second.pools.size()) {
    report("step input or pool count differs");
  }
  std::vector<std::string> constant_sites(first.constant_contents.size());
  for (size_t d = 0; d < first.dispatches.size(); ++d) {
    const auto& a = first.dispatches[d];
    const auto& b = second.dispatches[d];
    std::string where = "dispatch " + std::to_string(d) + " (" + first.dispatch_names[d] + ")";
    if (a.pipeline != b.pipeline) {
      report(where + ": pipeline differs, " + second.dispatch_names[d] + " in the second");
    }
    if (a.is_threads != b.is_threads || !same_grid(a.grid, b.grid) ||
        !same_grid(a.group, b.group)) {
      report(
          where + ": grid " + grid_text(a.grid) + " x " + grid_text(a.group) + " vs " +
          grid_text(b.grid) + " x " + grid_text(b.group));
    }
    if (a.bindings_end - a.bindings_begin != b.bindings_end - b.bindings_begin) {
      report(where + ": binding count differs");
    } else {
      for (uint32_t i = 0; i < a.bindings_end - a.bindings_begin; ++i) {
        const auto& x = first.bindings[a.bindings_begin + i];
        const auto& y = second.bindings[b.bindings_begin + i];
        std::string slot = where + " slot " + std::to_string(x.index);
        if (x.index != y.index || x.role != y.role || x.id != y.id || x.offset != y.offset ||
            (x.role == BufferRole::External && x.buffer != y.buffer)) {
          report(
              slot + ": " + role_name(x.role) + " #" + std::to_string(x.id) + " +" +
              std::to_string(x.offset) + " vs " + role_name(y.role) + " #" +
              std::to_string(y.id) + " +" + std::to_string(y.offset) + " at slot " +
              std::to_string(y.index));
        } else if (x.role == BufferRole::Constant && constant_sites[x.id].empty()) {
          constant_sites[x.id] = slot;
        }
      }
    }
    if (a.bytes_end - a.bytes_begin != b.bytes_end - b.bytes_begin) {
      report(where + ": setBytes count differs");
    } else {
      for (uint32_t i = 0; i < a.bytes_end - a.bytes_begin; ++i) {
        const auto& x = first.bytes_arguments[a.bytes_begin + i];
        const auto& y = second.bytes_arguments[b.bytes_begin + i];
        if (x.index != y.index || x.length != y.length ||
            std::memcmp(
                first.bytes_data.data() + x.offset,
                second.bytes_data.data() + y.offset,
                x.length) != 0) {
          report(
              where + " slot " + std::to_string(x.index) +
              ": setBytes payload differs (a per-step value passed by value)");
        }
      }
    }
    if (a.memory_end - a.memory_begin != b.memory_end - b.memory_begin) {
      report(where + ": threadgroup memory count differs");
    } else {
      for (uint32_t i = 0; i < a.memory_end - a.memory_begin; ++i) {
        const auto& x = first.threadgroup_memory[a.memory_begin + i];
        const auto& y = second.threadgroup_memory[b.memory_begin + i];
        if (x.index != y.index || x.length != y.length) {
          report(where + ": threadgroup memory differs");
        }
      }
    }
  }
  if (first.constant_contents.size() != second.constant_contents.size()) {
    report("constant leaf count differs");
  } else {
    for (size_t k = 0; k < first.constant_contents.size(); ++k) {
      if (first.constant_contents[k] != second.constant_contents[k]) {
        report(
            "constant leaf #" + std::to_string(k) + " (first bound at " + constant_sites[k] +
            ") differs between the captures: an undeclared step input");
      }
    }
  }
  if (num_differences > MAX_REPORTED_DIFFERENCES) {
    differences.push_back(
        "... and " + std::to_string(num_differences - MAX_REPORTED_DIFFERENCES) + " more");
  }
  return differences;
}

void begin_count() {
  if (active_session != nullptr) {
    throw std::runtime_error("begin_count: a capture or count is already open on this thread");
  }
  install_hooks();
  active_session = new Session(false, mx::default_stream(mx::Device::gpu));
}

CallCounts end_count() {
  if (active_session == nullptr || active_session->recording) {
    throw std::runtime_error("end_count: no count is open on this thread");
  }
  std::unique_ptr<Session> session(active_session);
  active_session = nullptr;
  return session->counts;
}

void wait_until_available(const std::vector<mx::array>& arrays) {
  for (const auto& array : arrays) {
    if (array.status() == mx::array::Status::unscheduled) {
      throw std::invalid_argument("wait_until_available: an array is not scheduled; eval it first");
    }
    while (!array.is_available()) {
    }
  }
}

} // namespace ftmetal
