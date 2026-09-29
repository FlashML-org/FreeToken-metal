// CommandGraph: record the Metal dispatches of one mx.eval and replay them as one
// primitive. The recorder replaces methods of the driver's compute encoder and command
// buffer classes and records while a thread-local session is open; GraphReplay encodes
// the recorded dispatches into MLX's encoder, rebinding the step inputs, the pools and
// the logits each step. Every other buffer the graph binds (intermediate arrays,
// constant leaves, MLX temporaries) is pinned for the graph's lifetime.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "mlx/array.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace MTL {
class Buffer;
class ComputePipelineState;
} // namespace MTL

namespace ftmetal {

// What a bound buffer is, in classification order: the caller's whitelist first (step
// inputs, pools, other externals), then the logits, the graph's constant leaves and
// intermediates, then MLX temporaries. Null is a nil binding.
enum class BufferRole : uint8_t {
  StepInput,
  Pool,
  External,
  Logits,
  Constant,
  Intermediate,
  Temporary,
  Null,
};

struct GridSize {
  uint64_t width;
  uint64_t height;
  uint64_t depth;
};

struct Binding {
  // The capture-time buffer; bound as is unless the role is a replay parameter.
  MTL::Buffer* buffer;
  // Absolute for pinned buffers; relative to the array's own offset for step inputs,
  // pools and the logits.
  int64_t offset;
  uint32_t index;
  BufferRole role;
  // The parameter index for step inputs and pools; for other roles the buffer's
  // first-appearance ordinal among the bindings of that role.
  uint32_t id;
};

struct BytesArgument {
  uint32_t index;
  uint32_t offset;
  uint32_t length;
};

struct ThreadgroupMemory {
  uint32_t index;
  uint64_t length;
};

struct Dispatch {
  // The pipeline in effect; replay sets it only when the recording did.
  MTL::ComputePipelineState* pipeline;
  bool sets_pipeline;
  // A recorded memory barrier or an encoder boundary precedes this dispatch.
  bool barrier_before;
  bool is_threads;
  uint32_t encoder;
  GridSize grid;
  GridSize group;
  uint32_t bindings_begin;
  uint32_t bindings_end;
  uint32_t bytes_begin;
  uint32_t bytes_end;
  uint32_t memory_begin;
  uint32_t memory_end;
};

struct ArraySpec {
  mx::Shape shape;
  mx::Dtype dtype;
};

struct CallCounts {
  int64_t dispatches = 0;
  int64_t encoders = 0;
  int64_t barriers = 0;
  int64_t commits = 0;
  int64_t signal_events = 0;
  int64_t completion_handlers = 0;
  int64_t fence_updates = 0;
  int64_t fence_waits = 0;
};

struct GraphStats {
  int64_t num_dispatches = 0;
  int64_t num_encoders = 0;
  // Barriers replay encodes, encoder boundaries included (the leading one is not).
  int64_t num_barriers = 0;
  int64_t num_bindings = 0;
  int64_t num_bytes_arguments = 0;
  // Distinct buffers bound, per role.
  int64_t num_step_input_buffers = 0;
  int64_t num_pool_buffers = 0;
  int64_t num_external_buffers = 0;
  int64_t num_constant_buffers = 0;
  int64_t num_intermediate_buffers = 0;
  int64_t num_temporary_buffers = 0;
  // Arrays the graph holds: intermediates (siblings included) and leaves.
  int64_t num_held_arrays = 0;
  // (nbytes, MTLBuffer length) of every temporary MLX registered during the capture.
  std::vector<std::tuple<int64_t, int64_t>> temporaries;
  // MTLBuffer lengths of the distinct buffers only the graph keeps alive.
  int64_t pinned_bytes = 0;
  // Encoder and command-buffer calls seen during the capture.
  CallCounts calls;
};

class Graph {
 public:
  Graph() = default;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  ~Graph();

  std::vector<Dispatch> dispatches;
  std::vector<Binding> bindings;
  std::vector<BytesArgument> bytes_arguments;
  std::vector<uint8_t> bytes_data;
  std::vector<ThreadgroupMemory> threadgroup_memory;

  std::vector<ArraySpec> step_inputs;
  std::vector<ArraySpec> pools;
  ArraySpec logits{{}, mx::float32};

  // Kernel name of each dispatch's pipeline ("?" when MLX's cache does not know it).
  std::vector<std::string> dispatch_names;
  // Contents of each constant leaf buffer by its id, for the double-capture check.
  std::vector<std::vector<uint8_t>> constant_contents;
  GraphStats stats;

  // Everything a replay binds without declaring it as a primitive input.
  std::vector<mx::array> held_arrays;
  std::unordered_set<void*> retained_objects;
};

// Open a capture on this thread. outputs are the step's logits followed by the updated
// pools, in the order of pools, all still unevaluated; step_inputs, pools and externals
// are evaluated. The caller then evals outputs and calls end_capture.
void begin_capture(
    const std::vector<mx::array>& outputs,
    const std::vector<mx::array>& step_inputs,
    const std::vector<mx::array>& pools,
    const std::vector<mx::array>& externals);

// Close this thread's capture; throws when it recorded something replay cannot rebuild.
std::shared_ptr<Graph> end_capture();

// One replay of graph on new step inputs and the current pools; returns the logits and
// the pools (aliasing the pools passed in).
std::vector<mx::array> graph_replay(
    std::shared_ptr<Graph> graph,
    const std::vector<mx::array>& step_inputs,
    const std::vector<mx::array>& pools,
    mx::StreamOrDevice s = {});

// Differences that forbid replaying one capture in place of the other: the dispatch
// sequence, bindings, setBytes payloads and constant leaf contents.
std::vector<std::string> compare_graphs(const Graph& first, const Graph& second);

// Count encoder and command-buffer calls on this thread without recording them.
void begin_count();
CallCounts end_count();

// Busy-wait until every array is available (no sleep, so clocks stay up while timing).
void wait_until_available(const std::vector<mx::array>& arrays);

class GraphReplay : public mx::Primitive {
 public:
  GraphReplay(mx::Stream stream, std::shared_ptr<Graph> graph)
      : mx::Primitive(stream), graph_(std::move(graph)) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs)
      override;

  const char* name() const override {
    return "GraphReplay";
  }

 private:
  std::shared_ptr<Graph> graph_;
};

} // namespace ftmetal
