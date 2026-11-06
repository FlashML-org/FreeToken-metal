#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "attention/gdn/gdn_decode.h"
#include "attention/mha/mha_decode.h"
#include "attention/mha/mha_prefill.h"
#include "embedding/ple_lookup.h"
#include "kvcache/pool_rows.h"
#include "moe/affine/affine_swiglu_pending.h"
#include "moe/affine/affine_swiglu_pending_shared.h"
#include "moe/gather_sorted.h"
#include "moe/moe_lookup.h"
#include "sample/sample_tokens.h"
#include "service/expert/expert_lru.h"
#include "service/expert/expert_service.h"
#include "service/ple/ple_service.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_C, m) {
  m.doc() = "ftmetal in-place pool primitives";

  m.def("init_pool_rows_library", &ftmetal::init_pool_rows_library, "source"_a);
  m.def("init_gdn_decode_library", &ftmetal::init_gdn_decode_library, "source"_a);
  m.def("init_mha_decode_library", &ftmetal::init_mha_decode_library, "source"_a);
  m.def("init_mha_prefill_library", &ftmetal::init_mha_prefill_library, "source"_a);
  m.def(
      "init_gather_sorted_libraries",
      &ftmetal::init_gather_sorted_libraries,
      "affine_source"_a,
      "affine_pair_codes_source"_a,
      "fp_source"_a,
      "unquantized_source"_a);
  m.def("init_sample_tokens_library", &ftmetal::init_sample_tokens_library, "source"_a);
  m.def("init_moe_lookup_library", &ftmetal::init_moe_lookup_library, "source"_a);
  m.def(
      "init_affine_swiglu_pending_library",
      &ftmetal::init_affine_swiglu_pending_library,
      "source"_a);
  m.def("init_ple_lookup_library", &ftmetal::init_ple_lookup_library, "source"_a);

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
      "gdn_decode",
      &ftmetal::gdn_decode,
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
      "block_ends"_a.none() = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Paged causal flash-attention of each tile's queries over its request's keys; returns [T, Hq, D].");

  m.def(
      "flash_prefill_qsa",
      &ftmetal::flash_prefill_qsa,
      "q"_a,
      "k_cache"_a,
      "v_cache"_a,
      "page_table"_a,
      "tiles"_a,
      "qsa_blocks"_a,
      "qsa_counts"_a,
      "qsa_bits"_a,
      "qsa_block_size"_a,
      "page_size"_a,
      "num_row_groups"_a,
      "head_split"_a,
      "block_k"_a,
      "kv_shared"_a,
      "num_passes"_a,
      "num_partitions"_a,
      "mpp"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "flash_prefill over each tile's listed QSA blocks, masked per query by its block bits; returns [T, Hq, D].");

  m.def(
      "gather_qmm_sorted",
      &ftmetal::gather_qmm_sorted,
      "x"_a,
      "w"_a,
      "scales"_a,
      "biases"_a.none(),
      "expert_offsets"_a,
      "expert_bytes"_a,
      "n"_a,
      "group_size"_a,
      "bits"_a,
      "mode"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "x[b] @ dequantize(expert e's weights at its byte offsets).T for the rows b of each expert e; returns [B, 1, n].");

  m.def(
      "gather_qmm_sorted_pair_codes",
      &ftmetal::gather_qmm_sorted_pair_codes,
      "x"_a,
      "w"_a,
      "codes"_a,
      "pairs"_a,
      "expert_offsets"_a,
      "expert_bytes"_a,
      "n"_a,
      "group_size"_a,
      "bits"_a,
      "code_bits"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "gather_qmm_sorted for affine experts in pair-code rows, each group's scale and bias the pair table's at its code; returns [B, 1, n].");

  m.def(
      "gather_mm_sorted",
      &ftmetal::gather_mm_sorted,
      "x"_a,
      "w"_a,
      "expert_offsets"_a,
      "expert_bytes"_a,
      "n"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "x[b] @ (expert e's weight at its byte offset).T for the rows b of each expert e; returns [B, 1, n].");

  m.attr("MAILBOX_WORDS") = ftmetal::MAILBOX_WORDS;
  m.attr("EXPERT_STAGING_BYTES") = ftmetal::EXPERT_STAGING_BYTES;
  m.attr("FILL_FILE") = ftmetal::FILL_FILE;
  m.attr("QMM_TILE_K") = ftmetal::QMM_TILE_K;
  m.attr("MAX_PAIR_CODE_BITS") = ftmetal::MAX_PAIR_CODE_BITS;
  m.attr("ISSUED_WORDS") = ftmetal::ISSUED_WORDS;
  m.attr("PENDING_FALLBACK_SLOT") = ftmetal::PENDING_FALLBACK_SLOT;
  m.attr("MAILBOX_SPIN_LIMIT") = ftmetal::MAILBOX_SPIN_LIMIT;
  m.attr("ACT_ROWS_PER_SIMDGROUP") = ftmetal::ACT_ROWS_PER_SIMDGROUP;
  nb::enum_<ftmetal::MailboxWord>(m, "MailboxWord", nb::is_arithmetic())
      .value("STATE", ftmetal::MailboxWord::State)
      .value("SEQUENCE", ftmetal::MailboxWord::Sequence)
      .value("ACKNOWLEDGED", ftmetal::MailboxWord::Acknowledged)
      .value("SHUTDOWN", ftmetal::MailboxWord::Shutdown)
      .value("ABANDONED", ftmetal::MailboxWord::Abandoned)
      .value("SPIN_LIMIT", ftmetal::MailboxWord::SpinLimit)
      .value("KIND", ftmetal::MailboxWord::Kind)
      .value("LAYER", ftmetal::MailboxWord::Layer)
      .value("STEP", ftmetal::MailboxWord::Step)
      .value("NUM_ROWS", ftmetal::MailboxWord::NumRows)
      .value("NUM_IDS", ftmetal::MailboxWord::NumIds)
      .value("NUM_MISSES", ftmetal::MailboxWord::NumMisses)
      .value("IDS", ftmetal::MailboxWord::Ids)
      .value("NUM_LOADED", ftmetal::MailboxWord::NumLoaded)
      .value("NUM_EVICTED", ftmetal::MailboxWord::NumEvicted)
      .value("ERROR", ftmetal::MailboxWord::Error)
      .value("CLAIM_TIME", ftmetal::MailboxWord::ClaimTime)
      .value("DONE_TIME", ftmetal::MailboxWord::DoneTime);
  nb::enum_<ftmetal::MailboxState>(m, "MailboxState", nb::is_arithmetic())
      .value("IDLE", ftmetal::MailboxState::Idle)
      .value("POSTED", ftmetal::MailboxState::Posted)
      .value("CLAIMED", ftmetal::MailboxState::Claimed)
      .value("DONE", ftmetal::MailboxState::Done)
      .value("FAILED", ftmetal::MailboxState::Failed)
      .value("SHUTDOWN", ftmetal::MailboxState::Shutdown);
  nb::enum_<ftmetal::VictimMemory>(m, "VictimMemory")
      .value("PURGEABLE", ftmetal::VictimMemory::Purgeable)
      .value("ANONYMOUS", ftmetal::VictimMemory::Anonymous);

  nb::enum_<ftmetal::MailboxError>(m, "MailboxError", nb::is_arithmetic())
      .value("NONE", ftmetal::MailboxError::None)
      .value("IO", ftmetal::MailboxError::Io)
      .value("TIMEOUT", ftmetal::MailboxError::Timeout)
      .value("PROTOCOL", ftmetal::MailboxError::Protocol)
      .value("SHUTDOWN", ftmetal::MailboxError::Shutdown)
      .value("CAPACITY", ftmetal::MailboxError::Capacity);
  nb::enum_<ftmetal::IssuedWord>(m, "IssuedWord", nb::is_arithmetic())
      .value("SEQUENCE", ftmetal::IssuedWord::Sequence)
      .value("NUM_MISSES", ftmetal::IssuedWord::NumMisses);
  nb::enum_<ftmetal::MoeLookupKind>(m, "MoeLookupKind", nb::is_arithmetic())
      .value("DECODE", ftmetal::MoeLookupKind::Decode)
      .value("PREFILL_COPY_MISS", ftmetal::MoeLookupKind::PrefillCopyMiss)
      .value("PROBE", ftmetal::MoeLookupKind::Probe);

  m.def(
      "moe_lookup",
      &ftmetal::moe_lookup,
      "topk_ids"_a,
      "moe_step"_a,
      "slot_of"_a,
      "mailbox"_a,
      "last_use"_a,
      "hit_stats"_a,
      "layer_id"_a,
      "slot_base"_a,
      "kind"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Pool slots of one MoE layer's routed experts, the absent read by the CPU first; returns (slots, slot_map, mailbox, last_use, hit_stats).");

  m.def(
      "moe_lookup_issue",
      &ftmetal::moe_lookup_issue,
      "topk_ids"_a,
      "moe_step"_a,
      "slot_of"_a,
      "mailbox"_a,
      "last_use"_a,
      "hit_stats"_a,
      "layer_id"_a,
      "slot_base"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "moe_lookup for a decode without the wait: the absent experts posted, their pairs' slots negative; returns (slots, issued, mailbox, last_use, hit_stats).");

  m.def(
      "affine_swiglu_pending",
      &ftmetal::affine_swiglu_pending,
      "x"_a,
      "w"_a,
      "scales_or_codes"_a,
      "biases_or_pairs"_a,
      "gate_up_resident"_a,
      "slots"_a,
      "issued"_a,
      "slot_of"_a,
      "mailbox"_a,
      "after"_a,
      "layer_id"_a,
      "slot_base"_a,
      "num_experts"_a,
      "group_size"_a,
      "bits"_a,
      "code_bits"_a,
      "is_fast"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "silu_and_mul of a MoE layer's routed gate_up rows, the pending pairs' computed once the CPU answered moe_lookup_issue's request; returns (act, slots, mailbox).");

  nb::class_<ftmetal::SlotLru>(m, "SlotLru")
      .def(
          nb::init<
              int,
              int,
              int,
              const std::vector<int32_t>&,
              const std::vector<int32_t>&,
              const std::vector<int32_t>&,
              mx::array,
              mx::array,
              double>(),
          "num_layers"_a,
          "num_experts"_a,
          "buffer_size"_a,
          "layer_pools"_a,
          "pool_bounds"_a,
          "pool_volatile"_a,
          "slot_of"_a,
          "last_use"_a,
          "lrfu_half_life"_a = 0.0,
          "Per-pool token-granular LRUs (or LRFUs) over the slots; writes slot_of and last_use "
          "in place.")
      .def_prop_ro("num_slots", &ftmetal::SlotLru::num_slots)
      .def("reset_scores", &ftmetal::SlotLru::reset_scores, "Forget the LRFU use counts.")
      .def(
          "fill",
          &ftmetal::SlotLru::fill,
          "flat_ids"_a,
          "Empty the cache and place each of flat_ids in its layer's pool; returns their slots.")
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
          "invalidate_prefill_buffers",
          &ftmetal::SlotLru::invalidate_prefill_buffers,
          "Drop every expert held in a slot under the prefill buffers; returns them as [(flat id, slot)], the most recently used first.")
      .def(
          "prefill_slot_map",
          [](const ftmetal::SlotLru& lru, int layer_id) {
            ftmetal::SlotAssignment assignment = lru.prefill_slot_map(layer_id);
            return std::make_pair(std::move(assignment.slots), std::move(assignment.loads));
          },
          "layer_id"_a,
          "Every expert's slot, -1 - its buffer row if absent; returns (slot_map, loads).");

  nb::class_<ftmetal::ExpertServiceStats>(m, "ExpertServiceStats")
      .def_ro("hits", &ftmetal::ExpertServiceStats::hits)
      .def_ro("misses", &ftmetal::ExpertServiceStats::misses)
      .def_ro("requests", &ftmetal::ExpertServiceStats::requests)
      .def_ro("experts_read", &ftmetal::ExpertServiceStats::experts_read)
      .def_ro("bytes_read", &ftmetal::ExpertServiceStats::bytes_read)
      .def_ro("read_wait_ns", &ftmetal::ExpertServiceStats::read_wait_ns)
      .def_ro("serve_ns", &ftmetal::ExpertServiceStats::serve_ns)
      .def_ro("lookups", &ftmetal::ExpertServiceStats::lookups)
      .def_ro("lookup_hits", &ftmetal::ExpertServiceStats::lookup_hits)
      .def_ro("victim_hits", &ftmetal::ExpertServiceStats::victim_hits)
      .def_ro("victim_buffer_hits", &ftmetal::ExpertServiceStats::victim_buffer_hits)
      .def_ro("victim_stores", &ftmetal::ExpertServiceStats::victim_stores)
      .def_ro("victim_drops", &ftmetal::ExpertServiceStats::victim_drops)
      .def_ro("deferred_stores", &ftmetal::ExpertServiceStats::deferred_stores)
      .def_ro("victim_copy_ns", &ftmetal::ExpertServiceStats::victim_copy_ns)
      .def_ro("victim_entries", &ftmetal::ExpertServiceStats::victim_entries)
      .def_ro("victim_bytes", &ftmetal::ExpertServiceStats::victim_bytes)
      .def_ro("victim_locked_bytes", &ftmetal::ExpertServiceStats::victim_locked_bytes);

  using Int64Table3 = nb::ndarray<const int64_t, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
  using Int64Table4 = nb::ndarray<const int64_t, nb::ndim<4>, nb::c_contig, nb::device::cpu>;
  nb::class_<ftmetal::ExpertService>(m, "ExpertService")
      .def(
          "__init__",
          [](ftmetal::ExpertService* service,
             int num_layers,
             int num_experts,
             int buffer_size,
             const std::vector<int32_t>& layer_pools,
             const std::vector<int32_t>& pool_bounds,
             const std::vector<int32_t>& pool_volatile,
             mx::array slot_of,
             mx::array last_use,
             mx::array mailbox,
             mx::array hit_stats,
             std::vector<mx::array> banks,
             const std::vector<std::string>& files,
             Int64Table3 segments,
             Int64Table4 locations,
             Int64Table3 destinations,
             const std::vector<int64_t>& buffer_bytes,
             int64_t victim_cache_bytes,
             int io_threads,
             double stall_timeout_s,
             bool run_coordinator,
             ftmetal::VictimMemory victim_memory,
             std::vector<int32_t> pool_spares,
             double lrfu_half_life,
             int64_t ssd_keepalive_us) {
            if (segments.shape(0) != static_cast<size_t>(num_layers) || segments.shape(2) != 3) {
              throw std::invalid_argument("ExpertService: segments must be int64 [num_layers, n, 3]");
            }
            if (locations.shape(0) != segments.shape(0) || locations.shape(1) != segments.shape(1) ||
                locations.shape(2) != static_cast<size_t>(num_experts) || locations.shape(3) != 2) {
              throw std::invalid_argument(
                  "ExpertService: locations must be int64 [num_layers, n, num_experts, 2]");
            }
            if (destinations.shape(0) != static_cast<size_t>(num_layers) ||
                destinations.shape(1) != banks.size() || destinations.shape(2) != 2) {
              throw std::invalid_argument(
                  "ExpertService: destinations must be int64 [num_layers, num_banks, 2]");
            }
            std::vector<ftmetal::ExpertSegment> table(segments.shape(0) * segments.shape(1));
            const int64_t* row = segments.data();
            for (ftmetal::ExpertSegment& segment : table) {
              segment = {row[0], row[1], row[2]};
              row += 3;
            }
            std::vector<ftmetal::ExpertLocation> places(table.size() * num_experts);
            const int64_t* place = locations.data();
            for (ftmetal::ExpertLocation& location : places) {
              location = {place[0], place[1]};
              place += 2;
            }
            new (service) ftmetal::ExpertService(
                num_layers,
                num_experts,
                buffer_size,
                layer_pools,
                pool_bounds,
                pool_volatile,
                std::move(slot_of),
                std::move(last_use),
                std::move(mailbox),
                std::move(hit_stats),
                std::move(banks),
                files,
                std::move(table),
                std::move(places),
                std::vector<int64_t>(destinations.data(), destinations.data() + destinations.size()),
                buffer_bytes,
                victim_cache_bytes,
                io_threads,
                stall_timeout_s,
                run_coordinator,
                victim_memory,
                std::move(pool_spares),
                lrfu_half_life,
                ssd_keepalive_us);
          },
          "num_layers"_a,
          "num_experts"_a,
          "buffer_size"_a,
          "layer_pools"_a,
          "pool_bounds"_a,
          "pool_volatile"_a,
          "slot_of"_a,
          "last_use"_a,
          "mailbox"_a,
          "hit_stats"_a,
          "banks"_a,
          "files"_a,
          "segments"_a,
          "locations"_a,
          "destinations"_a,
          "buffer_bytes"_a,
          "victim_cache_bytes"_a,
          "io_threads"_a,
          "stall_timeout_s"_a,
          "run_coordinator"_a,
          "victim_memory"_a = ftmetal::VictimMemory::Purgeable,
          "pool_spares"_a = std::vector<int32_t>(),
          "lrfu_half_life"_a = 0.0,
          "ssd_keepalive_us"_a = 0,
          "The slot cache's LRU, IO threads, victim cache, lookup coordinator and watchdog.")
      .def(
          "reset_scores",
          &ftmetal::ExpertService::reset_scores,
          nb::call_guard<nb::gil_scoped_release>(),
          "Experiment (LRFU): forget the slots' use counts.")
      .def(
          "fill_initial",
          &ftmetal::ExpertService::fill_initial,
          "flat_ids"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Empty the cache, place each of flat_ids in its layer's pool and read them.")
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
          "Drain the prefill buffer reads and drop the experts the buffer slots hold, storing "
          "the most recently used in the victim cache when there is one.")
      .def(
          "prefetch_prefill_layer",
          &ftmetal::ExpertService::prefetch_prefill_layer,
          "layer_id"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Start reading the layer's absent experts into its prefill buffer; returns its slot map.")
      .def(
          "wait_prefill_layer",
          &ftmetal::ExpertService::wait_prefill_layer,
          "layer_id"_a,
          nb::call_guard<nb::gil_scoped_release>(),
          "Wait for the layer's prefill buffer reads.")
      .def(
          "stats",
          &ftmetal::ExpertService::stats,
          nb::call_guard<nb::gil_scoped_release>(),
          "Per-layer hits, misses, experts and bytes read, read wait, GPU lookups and hits, and the victim cache's counts.")
      .def(
          "arm",
          &ftmetal::ExpertService::arm,
          "Keep the coordinator spinning through the forward about to be submitted.")
      .def(
          "check_healthy",
          &ftmetal::ExpertService::check_healthy,
          "Raise the first failure: a read, a broken or unanswered request, the shutdown.")
      .def(
          "shutdown",
          &ftmetal::ExpertService::shutdown,
          nb::call_guard<nb::gil_scoped_release>(),
          "Answer a waiting lookup with SHUTDOWN, refuse later ones and stop the threads.")
      .def(
          "purge_victim_cache",
          &ftmetal::ExpertService::purge_victim_cache,
          nb::call_guard<nb::gil_scoped_release>(),
          "For tests: discard the victim cache's bytes as memory pressure would.")
      .def(
          "victim_cache_ids",
          &ftmetal::ExpertService::victim_cache_ids,
          nb::call_guard<nb::gil_scoped_release>(),
          "For tests: the flat ids the victim cache holds, sorted.");

  m.attr("PLE_MAILBOX_WORDS") = ftmetal::PLE_MAILBOX_WORDS;
  m.attr("MAX_PLE_ROWS") = ftmetal::MAX_PLE_ROWS;
  nb::enum_<ftmetal::PleMailboxWord>(m, "PleMailboxWord", nb::is_arithmetic())
      .value("STATE", ftmetal::PleMailboxWord::State)
      .value("SEQUENCE", ftmetal::PleMailboxWord::Sequence)
      .value("ACKNOWLEDGED", ftmetal::PleMailboxWord::Acknowledged)
      .value("SHUTDOWN", ftmetal::PleMailboxWord::Shutdown)
      .value("ABANDONED", ftmetal::PleMailboxWord::Abandoned)
      .value("SPIN_LIMIT", ftmetal::PleMailboxWord::SpinLimit)
      .value("NUM_ROWS", ftmetal::PleMailboxWord::NumRows)
      .value("ERROR", ftmetal::PleMailboxWord::Error)
      .value("ROWS", ftmetal::PleMailboxWord::Rows);

  using Int64Table2 = nb::ndarray<const int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
  using RowIds = nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
  using RowPart = nb::ndarray<uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
  nb::class_<ftmetal::PleRowStore>(m, "PleRowStore")
      .def(
          "__init__",
          [](ftmetal::PleRowStore* store,
             const std::vector<std::string>& files,
             Int64Table3 parts,
             std::vector<int64_t> shard_starts,
             std::vector<int64_t> row_bytes,
             int64_t cache_rows,
             int io_threads) {
            if (parts.shape(1) != 3 || parts.shape(2) != 3) {
              throw std::invalid_argument("PleRowStore: parts must be int64 [num_shards, 3, 3]");
            }
            new (store) ftmetal::PleRowStore(
                files,
                std::vector<int64_t>(parts.data(), parts.data() + parts.size()),
                std::move(shard_starts),
                std::move(row_bytes),
                cache_rows,
                io_threads);
          },
          "files"_a,
          "parts"_a,
          "shard_starts"_a,
          "row_bytes"_a,
          "cache_rows"_a,
          "io_threads"_a,
          "The PLE table's rows, read from the checkpoint files into a row cache.")
      .def(
          "gather",
          [](const ftmetal::PleRowStore& store,
             RowIds ids,
             RowPart weight,
             RowPart scales,
             RowPart biases) {
            RowPart parts[3] = {weight, scales, biases};
            uint8_t* data[3];
            for (int p = 0; p < 3; ++p) {
              if (parts[p].shape(0) != ids.shape(0) ||
                  static_cast<int64_t>(parts[p].shape(1)) != store.row_bytes(p)) {
                throw std::invalid_argument(
                    "PleRowStore.gather: each part must be uint8 [num_ids, its row bytes]");
              }
              data[p] = parts[p].data();
            }
            for (size_t i = 0; i < ids.shape(0); ++i) {
              if (ids(i) < 0 || ids(i) >= store.num_rows()) {
                throw std::out_of_range(
                    "PleRowStore.gather: row " + std::to_string(ids(i)) + " is not in [0, " +
                    std::to_string(store.num_rows()) + ")");
              }
            }
            nb::gil_scoped_release release;
            store.gather(ids.data(), static_cast<int64_t>(ids.shape(0)), data);
          },
          "ids"_a,
          "weight"_a,
          "scales"_a,
          "biases"_a,
          "Copy the rows ids int64 [n] into the parts uint8 [n, row bytes].")
      .def_prop_ro("num_rows", &ftmetal::PleRowStore::num_rows)
      .def_prop_ro(
          "row_bytes",
          [](const ftmetal::PleRowStore& store) {
            return std::vector<int64_t>{store.row_bytes(0), store.row_bytes(1), store.row_bytes(2)};
          })
      .def_prop_ro("hits", &ftmetal::PleRowStore::hits)
      .def_prop_ro("misses", &ftmetal::PleRowStore::misses)
      .def_prop_ro("bytes_read", &ftmetal::PleRowStore::bytes_read);

  nb::class_<ftmetal::PleService>(m, "PleService")
      .def(
          nb::init<
              mx::array,
              std::vector<mx::array>,
              const ftmetal::PleRowStore&,
              std::vector<int64_t>,
              std::vector<int64_t>,
              std::vector<int64_t>,
              int64_t,
              double,
              bool>(),
          "mailbox"_a,
          "staging"_a,
          "store"_a,
          "multipliers"_a,
          "head_sizes"_a,
          "head_offsets"_a,
          "eos_token_id"_a,
          "stall_timeout_s"_a,
          "run_coordinator"_a,
          nb::keep_alive<1, 4>(),
          "Answer ple_post's posts with the store's rows; the store lives as long as the service.")
      .def_prop_ro("num_heads", &ftmetal::PleService::num_heads)
      .def(
          "ngram_ids",
          [](const ftmetal::PleService& service, Int64Table2 triples) {
            if (triples.shape(1) != 3) {
              throw std::invalid_argument(
                  "PleService.ngram_ids: triples must be int64 [n, 3] (token, previous, before "
                  "previous)");
            }
            size_t num_triples = triples.shape(0);
            size_t heads = static_cast<size_t>(service.num_heads());
            auto* ids = new int64_t[num_triples * heads];
            nb::capsule owner(ids, [](void* pointer) noexcept {
              delete[] static_cast<int64_t*>(pointer);
            });
            service.ngram_ids(triples.data(), static_cast<int64_t>(num_triples), ids);
            return nb::ndarray<nb::numpy, int64_t, nb::ndim<2>>(ids, {num_triples, heads}, owner);
          },
          "triples"_a,
          "The table row of each head of each (token, previous, before previous) triple.")
      .def(
          "arm",
          &ftmetal::PleService::arm,
          "Keep the coordinator spinning through the step about to be submitted.")
      .def(
          "check_healthy",
          &ftmetal::PleService::check_healthy,
          "Raise the first failure: a read, a broken or unanswered post, the shutdown.")
      .def(
          "shutdown",
          &ftmetal::PleService::shutdown,
          nb::call_guard<nb::gil_scoped_release>(),
          "Answer a waiting ple_wait with SHUTDOWN, refuse later posts and stop the threads.");

  m.def(
      "ple_post",
      &ftmetal::ple_post,
      "input_ids"_a,
      "slots"_a,
      "context"_a,
      "mailbox"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      "Post a decode step's rows and advance their n-gram context; returns (context, input_ids copy, ticket).");

  m.def(
      "ple_wait",
      &ftmetal::ple_wait,
      "ticket"_a,
      "mailbox"_a,
      "staging"_a,
      "num_rows"_a,
      "after"_a = std::vector<mx::array>{},
      nb::kw_only(),
      "stream"_a = nb::none(),
      "The rows of ple_post's ticket once the CPU answered (zeros if it failed); returns (weight, scales, biases).");

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
