#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "command_graph.h"
#include "fdecode.h"
#include "gdn_step.h"
#include "rows.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_C, m) {
  m.doc() = "ftmetal in-place pool primitives and the CommandGraph recorder";

  m.def("init_library", &ftmetal::init_library, "source"_a);
  m.def("init_gdn_step_library", &ftmetal::init_gdn_step_library, "source"_a);
  m.def("init_fdecode_library", &ftmetal::init_fdecode_library, "source"_a);

  m.def(
      "scatter_rows",
      &ftmetal::scatter_rows,
      "pool"_a,
      "rows"_a,
      "dst_ids"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "pool[dst_ids[i]] = rows[i], in place; returns pool.");

  m.def(
      "copy_rows",
      &ftmetal::copy_rows,
      "pool"_a,
      "src_ids"_a,
      "dst_ids"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "pool[dst_ids[i]] = pool[src_ids[i]], in place; returns pool.");

  m.def(
      "zero_rows",
      &ftmetal::zero_rows,
      "pool"_a,
      "ids"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "pool[ids[i]] = 0, in place; returns pool.");

  m.def(
      "kv_write",
      &ftmetal::kv_write,
      "k_cache"_a,
      "v_cache"_a,
      "k"_a,
      "v"_a,
      "out_loc"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "k_cache[out_loc[i]] = k[i] and v_cache[out_loc[i]] = v[i], in place; returns (k_cache, v_cache).");

  m.def(
      "gdn_step",
      &ftmetal::gdn_step,
      "conv_pool"_a,
      "state_pool"_a,
      "conv_in"_a,
      "conv_weight"_a,
      "a"_a,
      "b"_a,
      "A_log"_a,
      "dt_bias"_a,
      "slots"_a,
      "positions"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "One GDN decode step per request in place on the pools; returns (conv_pool, state_pool, y).");

  m.def(
      "flash_decode",
      &ftmetal::flash_decode,
      "q"_a,
      "k_cache"_a,
      "v_cache"_a,
      "page_table"_a,
      "seq_lens"_a,
      "page_size"_a,
      "num_partitions"_a,
      "num_simdgroups"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Paged flash-decode of one query per request over its first seq_lens[b] keys; returns [B, Hq, D].");

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
      .def_ro("pinned_bytes", &ftmetal::GraphStats::pinned_bytes)
      .def_ro("calls", &ftmetal::GraphStats::calls);

  nb::class_<ftmetal::Graph>(m, "Graph")
      .def_ro("stats", &ftmetal::Graph::stats)
      .def_ro("dispatch_names", &ftmetal::Graph::dispatch_names);

  m.def(
      "begin_capture",
      &ftmetal::begin_capture,
      "outputs"_a,
      "step_inputs"_a,
      "pools"_a,
      "externals"_a,
      "Open a capture: outputs are the unevaluated logits and updated pools; eval them next.");

  m.def("end_capture", &ftmetal::end_capture, "Close the capture and return its graph.");

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

  m.def("begin_count", &ftmetal::begin_count, "Count encoder calls on this thread.");
  m.def("end_count", &ftmetal::end_count, "Stop counting; returns the counts.");

  m.def(
      "wait_until_available",
      &ftmetal::wait_until_available,
      "arrays"_a,
      nb::call_guard<nb::gil_scoped_release>(),
      "Busy-wait until every array is available.");
}
