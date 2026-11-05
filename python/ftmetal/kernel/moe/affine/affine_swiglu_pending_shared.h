// affine_swiglu_pending's threadgroup shape and parameter block, included by its C++ and Metal.
#pragma once

#ifdef __METAL_VERSION__
#define SWIGLU_PENDING_CONSTANT constant constexpr
#else
#define SWIGLU_PENDING_CONSTANT constexpr
namespace ftmetal {
#endif

// A threadgroup's simdgroups, each computing ACT_ROWS_PER_SIMDGROUP rows of one pair's act from
// as many gate and up rows: MLX gather_qmv's threadgroup of two simdgroups of four rows.
SWIGLU_PENDING_CONSTANT int SWIGLU_PENDING_SIMDGROUPS = 2;
SWIGLU_PENDING_CONSTANT int ACT_ROWS_PER_SIMDGROUP = 4;

struct SwigluPendingParams {
  // The gate_up projection's input size (x's row) and the act's rows, its output size's half.
  int hidden_size;
  int intermediate_size;
  int top_k;
  int layer_id;
  int num_experts;
  // The resolved slots are slot_of's less this: the first slot of the layer's pool.
  int slot_base;
  // A pair-code row's bytes; 0 over plain scales and biases.
  int code_row_bytes;
};

#ifndef __METAL_VERSION__
} // namespace ftmetal
#endif
#undef SWIGLU_PENDING_CONSTANT
