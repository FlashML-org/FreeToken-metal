"""Custom kernels: C++ primitives under csrc/, Metal sources under metal/, wrappers here."""

from ftmetal.kernel.inplace import copy_rows, kv_write, scatter_rows, zero_rows

__all__ = ["copy_rows", "kv_write", "scatter_rows", "zero_rows"]
