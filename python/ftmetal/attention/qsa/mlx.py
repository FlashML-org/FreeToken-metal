"""QSA on mx.fast.scaled_dot_product_attention over gathered pages.

The parity oracle and bring-up backend, as attention/full/mlx.py: the index scores, the
top-k and the visible-key mask are plain MLX ops over each request's gathered block keys,
and the attention is one masked SDPA per decode batch or per prefill request.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.attention.full.common import FullAttnMetadata
from ftmetal.attention.full.mlx import FullAttnMLXBackend
from ftmetal.attention.qsa.common import QSAAttnBackend, block_scores, visible_blocks


class QSAMLXBackend(QSAAttnBackend, FullAttnMLXBackend):
    def visible_keys(
        self, index_queries: mx.array, slots: mx.array, query_ends: mx.array, layer_id: int
    ) -> mx.array:
        """bool [Q, key_len]: the keys at slots [key_len] each query ending at query_ends [Q]
        attends, scored against the complete blocks of those keys."""
        block_size = self.block_size
        key_len = slots.shape[0]
        num_complete = key_len // block_size
        block_keys = self.qsa_pool.index_cache(layer_id)[
            slots[block_size - 1 : num_complete * block_size : block_size]
        ]
        scores = block_scores(index_queries, block_keys)
        num_blocks = -(-key_len // block_size)
        blocks = visible_blocks(scores, query_ends, block_size, self.topk_blocks, num_blocks)
        tokens = mx.repeat(blocks, block_size, axis=1)[:, :key_len]
        causal = mx.arange(key_len, dtype=mx.int32)[None] < query_ends[:, None]
        return tokens & causal

    def sparse_decode(
        self, q: mx.array, index_queries: mx.array, layer_id: int, metadata: FullAttnMetadata
    ) -> mx.array:
        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)
        slots = self.token_slots(metadata)
        num_tokens, num_heads, head_dim = q.shape
        outputs = []
        for i, seq_len in enumerate(metadata.seq_lens_host):
            row_slots = slots[i, :seq_len]
            visible = self.visible_keys(
                index_queries[i : i + 1], row_slots, mx.array([seq_len]), layer_id
            )
            out = mx.fast.scaled_dot_product_attention(
                q[i : i + 1].transpose(1, 0, 2)[None],
                k_cache[row_slots].transpose(1, 0, 2)[None],
                v_cache[row_slots].transpose(1, 0, 2)[None],
                scale=self.scale,
                mask=visible[None, None],
            )
            outputs.append(out[0].transpose(1, 0, 2))
        out = mx.concatenate(outputs) if len(outputs) > 1 else outputs[0]
        return out.reshape(num_tokens, num_heads, head_dim)

    def sparse_prefill(
        self, q: mx.array, index_queries: mx.array, layer_id: int, metadata: FullAttnMetadata
    ) -> mx.array:
        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)
        slots = self.token_slots(metadata)
        outputs = []
        start = 0
        for i, extend_len in enumerate(metadata.extend_lens_host):
            seq_len = metadata.seq_lens_host[i]
            cached_len = metadata.cached_lens_host[i]
            row_slots = slots[i, :seq_len]
            query_ends = mx.arange(cached_len + 1, seq_len + 1, dtype=mx.int32)
            visible = self.visible_keys(
                index_queries[start : start + extend_len], row_slots, query_ends, layer_id
            )
            out = mx.fast.scaled_dot_product_attention(
                q[start : start + extend_len].transpose(1, 0, 2)[None],
                k_cache[row_slots].transpose(1, 0, 2)[None],
                v_cache[row_slots].transpose(1, 0, 2)[None],
                scale=self.scale,
                mask=visible[None, None],
            )
            outputs.append(out[0].transpose(1, 0, 2))
            start += extend_len
        return mx.concatenate(outputs) if len(outputs) > 1 else outputs[0]


__all__ = ["QSAMLXBackend"]
