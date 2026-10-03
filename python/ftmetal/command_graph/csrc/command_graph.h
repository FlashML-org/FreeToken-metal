// CommandGraph: record the Metal dispatches of one mx.eval and replay them as one
// primitive. The recorder replaces methods of the driver's compute encoder and command
// buffer classes and records while a thread-local session is open; GraphReplay encodes
// the recorded dispatches into MLX's encoder, rebinding the step inputs, the pools and
// the logits each step. The other buffers a capture binds are its constant leaves,
// pinned for the graph's lifetime, and its blocks (intermediate arrays and MLX
// temporaries): pinned too until place_blocks moves them into an arena that
// bind_arena supplies, a temporary the caller leaves out staying pinned.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
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
  // The block of an intermediate or temporary buffer, else -1.
  int32_t block;
  // Where replay binds a placed block's binding in the arena, else -1.
  int64_t arena_offset;
};

// One distinct intermediate or temporary buffer, from byte 0 to its length (views and
// slices bind it at offsets inside).
struct Block {
  MTL::Buffer* buffer;
  int64_t length;
  BufferRole role;
  // "<primitive> <dtype> <shape>" of the array that made an intermediate's buffer, or
  // "temporary of <kernel>" naming a temporary's first dispatch.
  std::string label;
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
  // MTLBuffer lengths of the distinct buffers only the graph keeps alive: at capture,
  // and now (after place_blocks, the constants and the temporaries left pinned).
  int64_t capture_pinned_bytes = 0;
  int64_t pinned_bytes = 0;
  // Capture-time step input buffers, released when the capture is classified.
  int64_t step_input_bytes = 0;
  int64_t num_pinned_temporaries = 0;
  // Bytes of arena the placed blocks reach.
  int64_t arena_bytes = 0;
  // Encoder and command-buffer calls seen during the capture.
  CallCounts calls;
};

// Shares ownership through enable_shared_from_this, so the Python object and C++ hold
// one control block: a replay's completion handler may drop the last reference without
// the GIL (nanobind's own shared_ptr would take it, and deadlock a GIL-holding wait).
class Graph : public std::enable_shared_from_this<Graph> {
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
  std::vector<Block> blocks;
  GraphStats stats;

  // What a replay binds without declaring it as a primitive input: the constant leaves,
  // and until place_blocks the intermediates and temporaries (then only those pinned).
  std::vector<mx::array> constants;
  std::vector<mx::array> intermediates;
  std::vector<mx::array> temporaries;
  // Pipelines, and the buffers the recorder saw until classification and place_blocks
  // release them (a pinned temporary's stays).
  std::unordered_set<void*> retained_objects;

  // Set by place_blocks; replay then needs bind_arena's arena, a declared input.
  bool is_placed = false;
  std::optional<mx::array> arena;
  // Blocks poison_blocks filled, with their contents for restore_blocks.
  std::vector<std::pair<int32_t, std::vector<uint8_t>>> poisoned;
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

// Drop this thread's capture, if any, without building a graph (after a failed eval).
void abort_capture();

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

// The distinct blocks each dispatch binds, in dispatch order.
std::vector<std::vector<int32_t>> dispatch_blocks(const Graph& graph);

// Fill the buffers of blocks with 0xFF once the GPU is idle, saving their contents.
void poison_blocks(Graph& graph, const std::vector<int32_t>& blocks);

// Write back what poison_blocks saved, once the GPU is idle.
void restore_blocks(Graph& graph);

// Bind block b at arena byte offsets[b], or keep it pinned where offsets[b] is -1 (a
// temporary only); releases every other block, and replay waits for bind_arena.
void place_blocks(Graph& graph, const std::vector<int64_t>& offsets);

// The arena of a placed graph: evaluated uint8 of at least stats.arena_bytes, which
// every replay declares as an input and output (graphs may share one).
void bind_arena(Graph& graph, const mx::array& arena);

// Fill the arena of a bound graph with 0xFF once the GPU is idle.
void poison_arena(const Graph& graph);

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
