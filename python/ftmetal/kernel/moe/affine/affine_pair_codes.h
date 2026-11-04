// The pair-code rows of an affine expert bank: instead of a scale and a bias per group, a row
// holds one CODE_BITS-bit code per group into the layer's pair table [2^CODE_BITS, 2] of
// (scale, bias), CODE_BITS at most MAX_PAIR_CODE_BITS. Code i of a projection [N, K]
// (i = n * K / group_size + group) is bits CODE_BITS * i to CODE_BITS * i + CODE_BITS - 1 of the
// row, least significant first; a row is whole 64-byte lines, so the words read here never
// leave it.
#pragma once

#include "moe/affine/affine_pair_codes_shared.h"

// Code i of a row of pair codes; words is the row, 4-byte aligned.
template <int CODE_BITS>
inline uint pair_code(const device uint32_t* words, uint i) {
  static_assert(CODE_BITS >= 1 && CODE_BITS <= MAX_PAIR_CODE_BITS, "no pair code that wide");
  const uint bit = i * CODE_BITS;
  const uint word = bit >> 5;
  const uint shift = bit & 31;
  const uint low = words[word];
  // The next word when the code crosses into it, else the same word again, whose bits land
  // above the code's: both loads are always issued, without a divergent branch.
  const uint high = words[word + (shift > 32 - CODE_BITS ? 1u : 0u)];
  return ((low >> shift) | ((high << 1) << (31 - shift))) & ((1u << CODE_BITS) - 1);
}
