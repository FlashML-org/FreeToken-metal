"""Full and sliding-window attention on mx.fast.scaled_dot_product_attention over
gathered pages.

The parity oracle and bring-up backend: decode gathers every request's pages into a
padded [size, max_seqlen_k] window and masks the tail; prefill runs one SDPA per
request over its cached prefix plus the fresh keys, causal or, for a window row (which
starts at its oldest visible key), banded to the window.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.attention.full.common import FullAttnBackend, FullAttnMetadata
from ftmetal.core import Batch


class FullAttnMLXBackend(FullAttnBackend):
    def token_slots(self, metadata: FullAttnMetadata) -> mx.array:
        """Page ids [size, max_pages] -> token slots [size, max_pages * page_size]."""
        offsets = mx.arange(self.table_page_size, dtype=mx.int32)
        slots = metadata.page_table[:, :, None] * self.table_page_size + offsets
        return slots.reshape(metadata.page_table.shape[0], -1)

    def attend(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        layer_id: int,
        batch: Batch,
        metadata: FullAttnMetadata,
    ) -> mx.array:
        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)
        num_tokens, num_heads, head_dim = q.shape
        if metadata.max_seqlen_q == 1:
            slots = self.token_slots(metadata)
            if num_tokens == 1:
                # One request: the exact window, no mask; bit-identical to a contiguous cache.
                slots = slots[:, : metadata.seq_lens_host[0]]
                mask = None
            else:
                window = slots.shape[1]
                in_range = mx.arange(window, dtype=mx.int32)[None, :] < metadata.seq_lens[:, None]
                mask = in_range[:, None, None, :]
            keys = k_cache[slots].transpose(0, 2, 1, 3)
            values = v_cache[slots].transpose(0, 2, 1, 3)
            out = mx.fast.scaled_dot_product_attention(
                q[:, :, None, :], keys, values, scale=self.scale, mask=mask
            )
            return out.reshape(num_tokens, num_heads, head_dim)
        outputs = []
        start = 0
        for i, extend_len in enumerate(metadata.extend_lens_host):
            cached_len = metadata.cached_lens_host[i]
            fresh_k = k[start : start + extend_len]
            fresh_v = v[start : start + extend_len]
            if cached_len > 0:
                prefix = self.token_slots(metadata)[i, :cached_len]
                keys = mx.concatenate([k_cache[prefix], fresh_k])
                values = mx.concatenate([v_cache[prefix], fresh_v])
            else:
                keys, values = fresh_k, fresh_v
            if self.sliding_window is None:
                visible: mx.array | str = "causal"
            else:
                positions = mx.arange(cached_len, cached_len + extend_len)[:, None]
                key_positions = mx.arange(cached_len + extend_len)[None, :]
                visible = (key_positions <= positions) & (
                    key_positions > positions - self.sliding_window
                )
            out = mx.fast.scaled_dot_product_attention(
                q[start : start + extend_len].transpose(1, 0, 2)[None],
                keys.transpose(1, 0, 2)[None],
                values.transpose(1, 0, 2)[None],
                scale=self.scale,
                mask=visible,
            )
            outputs.append(out[0].transpose(1, 0, 2))
            start += extend_len
        return mx.concatenate(outputs) if len(outputs) > 1 else outputs[0]


__all__ = ["FullAttnMLXBackend"]
