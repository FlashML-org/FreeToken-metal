#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "command_graph.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_C, m) {
  m.doc() = "ftmetal CommandGraph recorder and GraphReplay primitive";

  nb::class_<ftmetal::CallCounts>(m, "CallCounts")
      .def_ro("dispatches", &ftmetal::CallCounts::dispatches)
      .def_ro("encoders", &ftmetal::CallCounts::encoders)
      .def_ro("barriers", &ftmetal::CallCounts::barriers)
      .def_ro("commits", &ftmetal::CallCounts::commits)
      .def_ro("signal_events", &ftmetal::CallCounts::signal_events)
      .def_ro("completion_handlers", &ftmetal::CallCounts::completion_handlers)
      .def_ro("fence_updates", &ftmetal::CallCounts::fence_updates)
      .def_ro("fence_waits", &ftmetal::CallCounts::fence_waits);

  nb::class_<ftmetal::GraphStats>(m, "GraphStats")
      .def_ro("num_dispatches", &ftmetal::GraphStats::num_dispatches)
      .def_ro("num_encoders", &ftmetal::GraphStats::num_encoders)
      .def_ro("num_barriers", &ftmetal::GraphStats::num_barriers)
      .def_ro("num_bindings", &ftmetal::GraphStats::num_bindings)
      .def_ro("num_bytes_arguments", &ftmetal::GraphStats::num_bytes_arguments)
      .def_ro("num_step_input_buffers", &ftmetal::GraphStats::num_step_input_buffers)
      .def_ro("num_pool_buffers", &ftmetal::GraphStats::num_pool_buffers)
      .def_ro("num_external_buffers", &ftmetal::GraphStats::num_external_buffers)
      .def_ro("num_constant_buffers", &ftmetal::GraphStats::num_constant_buffers)
      .def_ro("num_intermediate_buffers", &ftmetal::GraphStats::num_intermediate_buffers)
      .def_ro("num_temporary_buffers", &ftmetal::GraphStats::num_temporary_buffers)
      .def_ro("num_held_arrays", &ftmetal::GraphStats::num_held_arrays)
      .def_ro("temporaries", &ftmetal::GraphStats::temporaries)
      .def_ro("capture_pinned_bytes", &ftmetal::GraphStats::capture_pinned_bytes)
      .def_ro("pinned_bytes", &ftmetal::GraphStats::pinned_bytes)
      .def_ro("step_input_bytes", &ftmetal::GraphStats::step_input_bytes)
      .def_ro("num_pinned_temporaries", &ftmetal::GraphStats::num_pinned_temporaries)
      .def_ro("arena_bytes", &ftmetal::GraphStats::arena_bytes)
      .def_ro("calls", &ftmetal::GraphStats::calls);

  nb::class_<ftmetal::Graph>(m, "Graph")
      .def_ro("stats", &ftmetal::Graph::stats)
      .def_ro("dispatch_names", &ftmetal::Graph::dispatch_names)
      .def_prop_ro(
          "block_sizes",
          [](const ftmetal::Graph& graph) {
            std::vector<int64_t> sizes;
            for (const auto& block : graph.blocks) {
              sizes.push_back(block.length);
            }
            return sizes;
          })
      .def_prop_ro(
          "block_labels",
          [](const ftmetal::Graph& graph) {
            std::vector<std::string> labels;
            for (const auto& block : graph.blocks) {
              labels.push_back(block.label);
            }
            return labels;
          })
      .def_prop_ro(
          "temporary_blocks",
          [](const ftmetal::Graph& graph) {
            std::vector<int32_t> blocks;
            for (size_t b = 0; b < graph.blocks.size(); ++b) {
              if (graph.blocks[b].role == ftmetal::BufferRole::Temporary) {
                blocks.push_back(static_cast<int32_t>(b));
              }
            }
            return blocks;
          })
      .def_prop_ro("dispatch_blocks", &ftmetal::dispatch_blocks)
      .def_prop_ro(
          "dispatch_barriers",
          [](const ftmetal::Graph& graph) {
            std::vector<bool> barriers;
            for (const auto& dispatch : graph.dispatches) {
              barriers.push_back(dispatch.barrier_before);
            }
            return barriers;
          })
      .def_prop_ro("dispatch_encoders", [](const ftmetal::Graph& graph) {
        std::vector<uint32_t> encoders;
        for (const auto& dispatch : graph.dispatches) {
          encoders.push_back(dispatch.encoder);
        }
        return encoders;
      });

  m.def(
      "begin_capture",
      &ftmetal::begin_capture,
      "outputs"_a,
      "step_inputs"_a,
      "pools"_a,
      "externals"_a,
      "Open a capture: outputs are the unevaluated logits and updated pools; eval them next.");

  m.def("end_capture", &ftmetal::end_capture, "Close the capture and return its graph.");
  m.def("abort_capture", &ftmetal::abort_capture, "Drop the open capture, if any.");

  m.def(
      "graph_replay",
      &ftmetal::graph_replay,
      "graph"_a,
      "step_inputs"_a,
      "pools"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "One replay of graph; returns [logits, *pools], the pools aliasing the ones passed in.");

  m.def(
      "compare_graphs",
      &ftmetal::compare_graphs,
      "first"_a,
      "second"_a,
      "Differences that forbid replaying one capture in place of the other.");

  m.def(
      "poison_blocks",
      &ftmetal::poison_blocks,
      "graph"_a,
      "blocks"_a,
      "Fill the blocks' buffers with 0xFF, saving their contents for restore_blocks.");
  m.def("restore_blocks", &ftmetal::restore_blocks, "graph"_a, "Undo poison_blocks.");
  m.def(
      "place_blocks",
      &ftmetal::place_blocks,
      "graph"_a,
      "offsets"_a,
      "Bind block b at arena offset offsets[b] (-1: a pinned temporary); release the rest.");
  m.def(
      "bind_arena",
      &ftmetal::bind_arena,
      "graph"_a,
      "arena"_a,
      "The uint8 arena a placed graph replays in.");
  m.def("poison_arena", &ftmetal::poison_arena, "graph"_a, "Fill the graph's arena with 0xFF.");

  m.def("begin_count", &ftmetal::begin_count, "Count encoder calls on this thread.");
  m.def("end_count", &ftmetal::end_count, "Stop counting; returns the counts.");

  m.def(
      "wait_until_available",
      &ftmetal::wait_until_available,
      "arrays"_a,
      nb::call_guard<nb::gil_scoped_release>(),
      "Busy-wait until every array is available.");
}
