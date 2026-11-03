"""Pure-MLX references that more than one test compares a kernel against; a reference only
one test uses lives in that test."""

import mlx.core as mx
import mlx.nn as nn

from ftmetal.core import host_ints
from ftmetal.kernel.moe.moe_align import ExpertAlignment

# Queries per SDPA call of flash_prefill_ref, which materializes [Hq, rows, keys] fp32 scores.
PREFILL_REF_QUERY_ROWS = 512


def mrope_ref(
    x: mx.array, positions: mx.array, inv_freq: mx.array, section: mx.array, rotary_dim: int
) -> mx.array:
    """kernel.rope.mrope in MLX ops: x [tokens, heads, head_dim], int32 [3, tokens] positions."""
    half = rotary_dim // 2
    angle = positions[section].T.astype(mx.float32) * inv_freq
    cos = mx.cos(angle)[:, None, :]
    sin = mx.sin(angle)[:, None, :]
    x1 = x[..., :half].astype(mx.float32)
    x2 = x[..., half:rotary_dim].astype(mx.float32)
    rest = x[..., rotary_dim:].astype(mx.float32)
    rotated = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], axis=-1)
    return rotated.astype(x.dtype)


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


def gdn_prefill_inputs_ref(
    conv_in: mx.array,
    taps: mx.array,
    conv_weight: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    num_k_heads: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """fp32 ops path of gdn_prefill_inputs: conv1d over taps + chunk, silu, norms, gates."""
    seq_len, conv_dim = conv_in.shape
    num_v_heads = a.shape[1]
    head_dim = conv_dim // (2 * num_k_heads + num_v_heads)
    key_width = num_k_heads * head_dim
    window = mx.concatenate([taps, conv_in]).astype(mx.float32)[None]
    weight = conv_weight.astype(mx.float32)[:, :, None]
    conv = mx.conv1d(window, weight, groups=conv_dim)[0]
    x = conv * mx.sigmoid(conv)
    q = x[:, :key_width].reshape(1, seq_len, num_k_heads, head_dim)
    k = x[:, key_width : 2 * key_width].reshape(1, seq_len, num_k_heads, head_dim)
    v = x[:, 2 * key_width :].reshape(1, seq_len, num_v_heads, head_dim)
    inv_scale = head_dim**-0.5
    q = (inv_scale**2) * q * mx.rsqrt((q * q).mean(axis=-1, keepdims=True) + 1e-6)
    k = inv_scale * k * mx.rsqrt((k * k).mean(axis=-1, keepdims=True) + 1e-6)
    gate = a.astype(mx.float32) + dt_bias.astype(mx.float32)
    decay = mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(gate))
    beta = mx.sigmoid(b.astype(mx.float32))
    dtype = conv_in.dtype
    return q.astype(dtype), k.astype(dtype), v.astype(dtype), decay[None], beta[None]


def moe_topk_ref(gates: mx.array, top_k: int, norm_topk_prob: bool) -> tuple[mx.array, mx.array]:
    """The ops path of the router: fp32 softmax, argpartition, renormalized weights."""
    scores = mx.softmax(gates, axis=-1, precise=True)
    topk_ids = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    topk_weights = mx.take_along_axis(scores, topk_ids, axis=-1)
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    return topk_ids.astype(mx.int32), topk_weights


def moe_align_ref(topk_ids: mx.array, num_experts: int) -> ExpertAlignment:
    """The ops path of the alignment: an argsort of the flat expert ids and its inverse."""
    flat = topk_ids.flatten()
    order = mx.argsort(flat)
    counts = mx.equal(flat[:, None], mx.arange(num_experts)).sum(axis=0)
    expert_offsets = mx.concatenate([mx.zeros((1,), dtype=mx.int32), mx.cumsum(counts)])
    return ExpertAlignment(
        expert_offsets=expert_offsets.astype(mx.int32),
        sorted_token_indices=(order // topk_ids.shape[1]).astype(mx.int32),
        sorted_expert_ids=flat[order].astype(mx.uint32),
        pair_positions=mx.argsort(order).astype(mx.int32),
    )


def moe_combine_ref(
    expert_out: mx.array, pair_positions: mx.array, topk_weights: mx.array
) -> mx.array:
    """fp32 gather of each token's expert rows and their routing-weighted sum."""
    num_tokens, top_k = topk_weights.shape
    rows = expert_out.reshape(num_tokens * top_k, -1)[pair_positions].astype(mx.float32)
    weighted = rows.reshape(num_tokens, top_k, -1) * topk_weights.astype(mx.float32)[..., None]
    return weighted.sum(axis=1).astype(expert_out.dtype)


def hc_mix_ref(up: mx.array, normed: mx.array, hc_count: int) -> mx.array:
    """kernel.hc.hc_mix in MLX ops: the mean over the streams of sigmoid(up) * normed in fp32,
    cast once."""
    weighted = mx.sigmoid(up.astype(mx.float32)) * normed.astype(mx.float32)
    total = weighted.reshape(normed.shape[0], hc_count, -1).sum(axis=1)
    return (total / hc_count).astype(normed.dtype)


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


def flash_prefill_ref(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    page_table: mx.array,
    extend_lens: list[int],
    cached_lens: list[int],
    page_size: int,
    scale: float,
    sliding_window: int | None = None,
    block_ends: list[int] | None = None,
) -> mx.array:
    """fp32 SDPA of each request's queries in q [T, Hq, D] over its paged keys, causal from
    its cached prefix, also seeing the keys before its block_ends entry when given, and,
    with a sliding window W, limited to keys past p - W; at most PREFILL_REF_QUERY_ROWS
    queries per call bound the scores."""
    offsets = mx.arange(page_size, dtype=mx.int32)
    outputs = []
    row = 0
    for i, (extend_len, cached_len) in enumerate(zip(extend_lens, cached_lens)):
        slots = (page_table[i][:, None] * page_size + offsets).reshape(-1)
        slots = slots[: cached_len + extend_len]
        keys = k_cache[slots].astype(mx.float32).transpose(1, 0, 2)[None]
        values = v_cache[slots].astype(mx.float32).transpose(1, 0, 2)[None]
        for start in range(0, extend_len, PREFILL_REF_QUERY_ROWS):
            stop = min(extend_len, start + PREFILL_REF_QUERY_ROWS)
            queries = q[row + start : row + stop].astype(mx.float32).transpose(1, 0, 2)[None]
            ends = [] if block_ends is None else block_ends[row + start : row + stop]
            visible = max([cached_len + stop, *ends])
            if sliding_window is None and block_ends is None:
                # The causal mask aligns the last query with the last key.
                mask: mx.array | str = "causal"
            else:
                positions = mx.arange(cached_len + start, cached_len + stop)[:, None]
                key_positions = mx.arange(visible)[None, :]
                causal = key_positions <= positions
                seen = (
                    causal
                    if block_ends is None
                    else causal | (key_positions < mx.array(ends)[:, None])
                )
                in_window = key_positions > positions - (sliding_window or visible)
                mask = seen & in_window
            out = mx.fast.scaled_dot_product_attention(
                queries, keys[:, :, :visible], values[:, :, :visible], scale=scale, mask=mask
            )
            outputs.append(out[0].transpose(1, 0, 2))
        row += extend_len
    return mx.concatenate(outputs)


__all__ = [
    "copy_rows_ref",
    "flash_decode_ref",
    "flash_prefill_ref",
    "gdn_decode_step_ref",
    "gdn_prefill_inputs_ref",
    "hc_mix_ref",
    "moe_align_ref",
    "moe_combine_ref",
    "moe_topk_ref",
    "scatter_rows_ref",
    "zero_rows_ref",
]
