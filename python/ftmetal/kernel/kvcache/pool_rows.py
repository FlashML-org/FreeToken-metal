"""In-place row writes into pools. Every call returns the pool; callers rebind it."""

import mlx.core as mx

from ftmetal.kernel._ext import _C


def scatter_rows(pool: mx.array, rows: mx.array, dst_ids: mx.array) -> mx.array:
    """pool[dst_ids[i]] = rows[i]."""
    return _C.scatter_rows(pool, rows, dst_ids)


def copy_rows(pool: mx.array, src_ids: mx.array, dst_ids: mx.array) -> mx.array:
    """pool[dst_ids[i]] = pool[src_ids[i]]."""
    return _C.copy_rows(pool, src_ids, dst_ids)


def zero_rows(pool: mx.array, ids: mx.array) -> mx.array:
    """pool[ids[i]] = 0."""
    return _C.zero_rows(pool, ids)


def kv_write(
    k_cache: mx.array, v_cache: mx.array, k: mx.array, v: mx.array, out_loc: mx.array
) -> tuple[mx.array, mx.array]:
    """k_cache[out_loc[i]] = k[i] and v_cache[out_loc[i]] = v[i]; returns both caches."""
    k_cache, v_cache = _C.kv_write(k_cache, v_cache, k, v, out_loc)
    return k_cache, v_cache


__all__ = ["copy_rows", "kv_write", "scatter_rows", "zero_rows"]
