#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "expert_lru.h"
#include "expert_service.h"
#include "fdecode.h"
#include "fprefill.h"
#include "gather_qmm_sorted.h"
#include "gdn_step.h"
#include "rows.h"
#include "sample.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_C, m) {
  m.doc() = "ftmetal in-place pool primitives";

  m.def("init_library", &ftmetal::init_library, "source"_a);
  m.def("init_gdn_step_library", &ftmetal::init_gdn_step_library, "source"_a);
  m.def("init_fdecode_library", &ftmetal::init_fdecode_library, "source"_a);
  m.def("init_fprefill_library", &ftmetal::init_fprefill_library, "source"_a);
  m.def("init_gather_qmm_sorted_library", &ftmetal::init_gather_qmm_sorted_library, "source"_a);
  m.def("init_sample_library", &ftmetal::init_sample_library, "source"_a);

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
      "partition_tokens"_a,
      "min_partition_tokens"_a,
      "max_partitions"_a,
      "min_threadgroups"_a,
      "num_head_groups"_a,
      "num_simdgroups"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Paged flash-decode of one query per request over its first seq_lens[b] keys; returns [B, Hq, D].");

  m.def(
      "flash_prefill",
      &ftmetal::flash_prefill,
      "q"_a,
      "k_cache"_a,
      "v_cache"_a,
      "page_table"_a,
      "tiles"_a,
      "page_size"_a,
      "num_row_groups"_a,
      "head_split"_a,
      "block_k"_a,
      "kv_shared"_a,
      "num_passes"_a,
      "num_partitions"_a,
      "sliding_window"_a,
      "mpp"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Paged causal flash-attention of each tile's queries over its request's keys; returns [T, Hq, D].");

  m.def(
      "gather_qmm_sorted",
      &ftmetal::gather_qmm_sorted,
      "x"_a,
      "w"_a,
      "scales"_a,
      "biases"_a,
      "expert_offsets"_a,
      "slot_of"_a,
      "group_size"_a,
      "bits"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "x[b] @ dequantize(w[slot_of[e]]).T for the rows b of each expert e; returns [B, 1, N].");

  nb::class_<ftmetal::SlotLru>(m, "SlotLru")
      .def(
          nb::init<int, int, int, int, mx::array, mx::array>(),
          "num_layers"_a,
          "num_experts"_a,
          "num_slots"_a,
          "window_slots"_a,
          "slot_of"_a,
          "last_use"_a,
          "Token-granular LRU over the slots; writes slot_of and last_use in place.")
      .def(
          "fill",
          &ftmetal::SlotLru::fill,
          "flat_ids"_a,
          "Empty the cache and place flat_ids from slot window_slots on; returns their slots.")
      .def(
          "resolve",
          [](ftmetal::SlotLru& lru, int layer_id, const std::vector<int32_t>& expert_ids,
             uint32_t step) {
            ftmetal::SlotAssignment assignment = lru.resolve(layer_id, expert_ids, step);
            return std::make_pair(std::move(assignment.slots), std::move(assignment.loads));
          },
          "layer_id"_a,
          "expert_ids"_a,
          "step"_a,
          "Slots of the layer's distinct experts at step; returns (slots, [(expert, slot)] to load).")
      .def(
          "invalidate_windows",
          &ftmetal::SlotLru::invalidate_windows,
          "Drop every expert held in a window slot.")
      .def(
          "prefill_slot_map",
          [](const ftmetal::SlotLru& lru, int layer_id) {
            ftmetal::SlotAssignment assignment = lru.prefill_slot_map(layer_id);
            return std::make_pair(std::move(assignment.slots), std::move(assignment.loads));
          },
          "layer_id"_a,
          "Every expert's slot, window layer_id % 2 for the absent; returns (slot_map, loads).");

  nb::class_<ftmetal::ExpertServiceStats>(m, "ExpertServiceStats")
      .def_ro("hits", &ftmetal::ExpertServiceStats::hits)
      .def_ro("misses", &ftmetal::ExpertServiceStats::misses)
      .def_ro("experts_read", &ftmetal::ExpertServiceStats::experts_read)
      .def_ro("bytes_read", &ftmetal::ExpertServiceStats::bytes_read)
      .def_ro("read_wait_ns", &ftmetal::ExpertServiceStats::read_wait_ns);

  using SegmentTable = nb::ndarray<const int64_t, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
  nb::class_<ftmetal::ExpertService>(m, "ExpertService")
      .def(
          "__init__",
          [](ftmetal::ExpertService* service,
             int num_layers,
             int num_experts,
             int num_slots,
             int window_slots,
             mx::array slot_of,
             mx::array last_use,
             std::vector<mx::array> banks,
             const std::vector<std::string>& files,
             SegmentTable segments,
             int io_threads) {
            if (segments.shape(0) != static_cast<size_t>(num_layers) || segments.shape(2) != 5) {
              throw std::invalid_argument("ExpertService: segments must be int64 [num_layers, n, 5]");
            }
            std::vector<ftmetal::ExpertSegment> table(segments.shape(0) * segments.shape(1));
            const int64_t* row = segments.data();
            for (ftmetal::ExpertSegment& segment : table) {
              segment = {row[0], row[1], row[2], row[3], row[4]};
              row += 5;
            }
            new (service) ftmetal::ExpertService(
                num_layers,
                num_experts,
                num_slots,
                window_slots,
                std::move(slot_of),
                std::move(last_use),
                std::move(banks),
                files,
                std::move(table),
                io_threads);
          },
          "num_layers"_a,
          "num_experts"_a,
          "num_slots"_a,
          "window_slots"_a,
          "slot_of"_a,
          "last_use"_a,
          "banks"_a,
          "files"_a,
          "segments"_a,
          "io_threads"_a,
          "The slot cache's LRU and IO threads over the banks and the checkpoint files.")
      .def(
          "fill_initial",
          &ftmetal::ExpertService::fill_initial,
          "flat_ids"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Empty the cache, place flat_ids from slot window_slots on and read them.")
      .def(
          "ensure_experts",
          &ftmetal::ExpertService::ensure_experts,
          "layer_id"_a,
          "expert_ids"_a,
          "step"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Slots of the layer's distinct experts at step, the misses read before it returns.")
      .def(
          "begin_prefill",
          &ftmetal::ExpertService::begin_prefill,
          nb::call_guard<nb::gil_scoped_release>(),
          "Drain the window reads and drop the experts the window slots hold.")
      .def(
          "prefetch_prefill_layer",
          &ftmetal::ExpertService::prefetch_prefill_layer,
          "layer_id"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Start reading the layer's absent experts into window layer_id % 2; returns its slot map.")
      .def(
          "wait_prefill_layer",
          &ftmetal::ExpertService::wait_prefill_layer,
          "layer_id"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Wait for the layer's window reads.")
      .def(
          "stats",
          &ftmetal::ExpertService::stats,
          nb::call_guard<nb::gil_scoped_release>(),
          "Per-layer hits, misses, experts and bytes read, read wait.");

  m.def(
      "sample_tokens",
      &ftmetal::sample_tokens,
      "logits"_a,
      "temperatures"_a,
      "top_k"_a.none(),
      "top_p"_a.none(),
      "num_groups"_a,
      "random_bits"_a.none() = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      "One token per row from softmax(logits / T) under top-k / top-p; returns int32 [B].");
}
