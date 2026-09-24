"""Pure-MLX references for the kernels; the tests compare every kernel against these."""

import mlx.core as mx


def scatter_rows_ref(pool: mx.array, rows: mx.array, dst_ids: mx.array) -> mx.array:
    out = mx.array(pool)
    out[dst_ids] = rows
    return out


def copy_rows_ref(pool: mx.array, src_ids: mx.array, dst_ids: mx.array) -> mx.array:
    out = mx.array(pool)
    out[dst_ids] = pool[src_ids]
    return out


def zero_rows_ref(pool: mx.array, ids: mx.array) -> mx.array:
    out = mx.array(pool)
    out[ids] = mx.zeros(out.shape[1:], dtype=out.dtype)
    return out


__all__ = ["copy_rows_ref", "scatter_rows_ref", "zero_rows_ref"]
