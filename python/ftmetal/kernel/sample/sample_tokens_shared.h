// What sample_tokens.cpp encodes the dispatch chain with and sample_tokens.metal computes
// with: the round kinds, the radix-select geometry, the threadgroup sizes and the workspace
// layout. One definition, included by C++ and Metal alike.
#pragma once

#ifdef __METAL_VERSION__
#define SAMPLE_CONSTANT constant constexpr
#else
#define SAMPLE_CONSTANT constexpr
namespace ftmetal {
#endif

// What a round bins: nothing (the max-only first dispatch), counts (top-k) or mass (top-p).
SAMPLE_CONSTANT int SAMPLE_KIND_NONE = 0;
SAMPLE_CONSTANT int SAMPLE_KIND_COUNT = 1;
SAMPLE_CONSTANT int SAMPLE_KIND_MASS = 2;
// Key bits a round selects, and the bins they make.
SAMPLE_CONSTANT int SAMPLE_BITS_PER_ROUND = 8;
SAMPLE_CONSTANT int SAMPLE_NUM_BINS = 256;
// Threads of a sample_round and of the sample_find threadgroup.
SAMPLE_CONSTANT int SAMPLE_ROUND_THREADS = 256;
SAMPLE_CONSTANT int SAMPLE_FIND_THREADS = 1024;
// The most chunks a row splits into.
SAMPLE_CONSTANT int SAMPLE_MAX_GROUPS = 256;
// Per-row state in the workspace: a slot of words per dispatch of the chain.
SAMPLE_CONSTANT int SAMPLE_STATE_SLOTS = 10;
SAMPLE_CONSTANT int SAMPLE_STATE_WORDS = 8;

#ifndef __METAL_VERSION__
} // namespace ftmetal
#endif
#undef SAMPLE_CONSTANT
