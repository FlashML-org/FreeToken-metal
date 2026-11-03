// The vector width of the _vec_ row kernels: pool_rows.metal instantiates them at it and
// pool_rows.cpp picks them for rows whose element count it divides. One definition, included
// by C++ and Metal alike.
#pragma once

#ifdef __METAL_VERSION__
#define POOL_ROWS_CONSTANT constant constexpr
#else
#define POOL_ROWS_CONSTANT constexpr
namespace ftmetal {
#endif

POOL_ROWS_CONSTANT int POOL_ROWS_VECTOR_WIDTH = 4;

#ifndef __METAL_VERSION__
} // namespace ftmetal
#endif
#undef POOL_ROWS_CONSTANT
