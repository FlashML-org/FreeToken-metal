// PleLookup: a decode step's PLE (per-layer n-gram embedding) table rows, looked up inside the
// step. ple_post, at the step's start, writes each batch row's token and n-gram context (the two
// tokens before it) to the mailbox, posts it and advances the context pool; PleService's
// coordinator (service/ple/ple_service.h) hashes the n-grams, gathers the rows into the staging
// buffers and answers; ple_wait, encoded just before the PLE layer, waits for the answer
// (common/mailbox_wait.h) and copies the rows out with coherent loads. No host code reads the
// step's token ids. A post refused after a shutdown, a failed answer, a released wait and one that
// gave up give zero rows; the service records the first three and the GPU the last (the abandoned
// word), so check_healthy raises before the step's tokens are used. The mailbox and the staging
// buffers are written by the CPU behind MLX's back.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/mailbox.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace ftmetal {

// Register the Metal source of ple_post and ple_wait; compiled on first use.
void init_ple_lookup_library(const std::string& source);

// input_ids int32 [num_rows], 1 <= num_rows <= MAX_PLE_ROWS; slots int32 [num_rows] in
// [0, num_slots), the rows' context slots (padding rows may share one); context int32
// [num_slots, 2] (before previous, previous), updated in place; mailbox uint32
// [PLE_MAILBOX_WORDS]. Returns {context, a copy of input_ids, ticket uint32 [1]}: the copy
// orders whatever embeds the step's tokens after the post, and the ticket is the post's
// sequence number, 0 when a shutdown refused it.
std::vector<mx::array> ple_post(
    const mx::array& input_ids,
    const mx::array& slots,
    const mx::array& context,
    const mx::array& mailbox,
    mx::StreamOrDevice s = {});

// The rows of ple_post's ticket once the CPU answered: {weight, scales, biases}, the first
// num_rows batch rows of each staging part uint8 [MAX_PLE_ROWS, heads, row bytes], zeros
// when the post was refused, failed or released by a shutdown. Encoded after the arrays in
// after, which it does not read.
std::vector<mx::array> ple_wait(
    const mx::array& ticket,
    const mx::array& mailbox,
    const std::vector<mx::array>& staging,
    int num_rows,
    const std::vector<mx::array>& after = {},
    mx::StreamOrDevice s = {});

enum class PleKernel { Post, Wait };

class PleLookup : public mx::Primitive {
 public:
  PleLookup(mx::Stream stream, PleKernel kernel, int num_rows)
      : mx::Primitive(stream), kernel_(kernel), num_rows_(num_rows) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return kernel_ == PleKernel::Post ? "PlePost" : "PleWait";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

 private:
  PleKernel kernel_;
  int num_rows_;
};

} // namespace ftmetal
