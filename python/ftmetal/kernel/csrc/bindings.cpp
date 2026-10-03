#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "fdecode.h"
#include "fprefill.h"
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
