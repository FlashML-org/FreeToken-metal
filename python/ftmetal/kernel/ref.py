"""Pure-MLX references for the kernels; the tests compare every kernel against these."""

import mlx.core as mx
import mlx.nn as nn

from ftmetal.core import host_ints


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


def gdn_decode_step_ref(
    conv_pool: mx.array,
    state_pool: mx.array,
    conv_in: mx.array,
    conv_weight: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    slots: mx.array,
    positions: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Functional fp32 reference of gdn_decode_step with the same ring semantics."""
    num_v_heads, value_dim, key_dim = state_pool.shape[1:]
    conv_kernel, conv_dim = conv_pool.shape[1:]
    num_k_heads = (conv_dim - num_v_heads * value_dim) // (2 * key_dim)
    key_width = num_k_heads * key_dim
    conv_out_pool = conv_pool
    state_out_pool = state_pool
    ys = []
    for i, (slot, position) in enumerate(zip(host_ints(slots), host_ints(positions))):
        ring = conv_pool[slot].astype(mx.float32)
        current = conv_in[i].astype(mx.float32)
        taps = [
            ring[(position - (conv_kernel - 1) + j) % conv_kernel] for j in range(conv_kernel - 1)
        ]
        window = mx.stack([*taps, current])  # [K, conv_dim], oldest first
        conv = (window * conv_weight.astype(mx.float32).T).sum(axis=0)
        x = conv * mx.sigmoid(conv)
        q = x[:key_width].reshape(num_k_heads, key_dim)
        k = x[key_width : 2 * key_width].reshape(num_k_heads, key_dim)
        v = x[2 * key_width :].reshape(num_v_heads, value_dim)
        inv_scale = key_dim**-0.5
        q = (inv_scale**2) * q * mx.rsqrt((q * q).mean(axis=-1, keepdims=True) + 1e-6)
        k = inv_scale * k * mx.rsqrt((k * k).mean(axis=-1, keepdims=True) + 1e-6)
        gate = a[i].astype(mx.float32) + dt_bias.astype(mx.float32)
        decay = mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(gate))
        beta = mx.sigmoid(b[i].astype(mx.float32))
        group = num_v_heads // num_k_heads
        k_heads = mx.repeat(k, group, axis=0)  # [Hv, Dk]
        q_heads = mx.repeat(q, group, axis=0)
        state = state_pool[slot] * decay[:, None, None]  # [Hv, Dv, Dk]
        kv = (state * k_heads[:, None, :]).sum(axis=-1)  # [Hv, Dv]
        delta = (v - kv) * beta[:, None]
        state = state + delta[:, :, None] * k_heads[:, None, :]
        y = (state * q_heads[:, None, :]).sum(axis=-1)
        ys.append(y.astype(conv_in.dtype))
        state_out_pool = scatter_rows_ref(
            state_out_pool, state[None], mx.array([slot], dtype=mx.int32)
        )
        new_ring = conv_pool[slot]
        new_ring[position % conv_kernel] = conv_in[i]
        conv_out_pool = scatter_rows_ref(
            conv_out_pool, new_ring[None], mx.array([slot], dtype=mx.int32)
        )
    return conv_out_pool, state_out_pool, mx.stack(ys)


def router_topk_ref(gates: mx.array, top_k: int, norm_topk_prob: bool) -> tuple[mx.array, mx.array]:
    """The ops path of the router: fp32 softmax, argpartition, renormalized weights."""
    scores = mx.softmax(gates, axis=-1, precise=True)
    topk_ids = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    topk_weights = mx.take_along_axis(scores, topk_ids, axis=-1)
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    return topk_ids.astype(mx.int32), topk_weights


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
    for i, seq_len in enumerate(host_ints(seq_lens)):
        slots = (page_table[i][:, None] * page_size + offsets).reshape(-1)[:seq_len]
        keys = mx.repeat(k_cache[slots].astype(mx.float32), group, axis=1)
        values = mx.repeat(v_cache[slots].astype(mx.float32), group, axis=1)
        scores = mx.einsum("hd,lhd->hl", q[i].astype(mx.float32), keys) * scale
        outputs.append(mx.einsum("hl,lhd->hd", mx.softmax(scores, axis=-1), values))
    return mx.stack(outputs)


__all__ = [
    "copy_rows_ref",
    "flash_decode_ref",
    "gdn_decode_step_ref",
    "router_topk_ref",
    "scatter_rows_ref",
    "zero_rows_ref",
]
