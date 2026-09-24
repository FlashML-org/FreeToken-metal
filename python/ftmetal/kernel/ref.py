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


def flash_decode_ref(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    seq_lens: mx.array,
    page_size: int,
    scale: float,
) -> mx.array:
    """fp32 attention of q [B, Hq, D] over each request's first seq_lens[b] paged keys."""
    group = q.shape[1] // k_cache.shape[1]
    offsets = mx.arange(page_size, dtype=mx.int32)
    outputs = []
    for i, seq_len in enumerate(seq_lens.tolist()):
        slots = (page_table[i][:, None] * page_size + offsets).reshape(-1)[:seq_len]
        keys = mx.repeat(k_cache[slots].astype(mx.float32), group, axis=1)
        values = mx.repeat(v_cache[slots].astype(mx.float32), group, axis=1)
        scores = mx.einsum("hd,lhd->hl", q[i].astype(mx.float32), keys) * scale
        outputs.append(mx.einsum("hl,lhd->hd", mx.softmax(scores, axis=-1), values))
    return mx.stack(outputs)


__all__ = ["copy_rows_ref", "flash_decode_ref", "scatter_rows_ref", "zero_rows_ref"]
