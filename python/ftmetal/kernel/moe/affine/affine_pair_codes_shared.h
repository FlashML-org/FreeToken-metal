// The widest pair code: a layer's pair table has 2^code_bits rows, and past 16 bits it outgrows
// what a table should cost (2^16 bf16 pairs are 256 KiB, read at random by every group). One
// definition, included by C++ and Metal alike.
#pragma once

#ifdef __METAL_VERSION__
#define PAIR_CODES_CONSTANT constant constexpr
#else
#define PAIR_CODES_CONSTANT constexpr
namespace ftmetal {
#endif

PAIR_CODES_CONSTANT int MAX_PAIR_CODE_BITS = 16;

#ifndef __METAL_VERSION__
} // namespace ftmetal
#endif
#undef PAIR_CODES_CONSTANT
