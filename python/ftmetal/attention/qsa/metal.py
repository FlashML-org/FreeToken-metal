"""QSA on the paged Metal kernels.

A decode step scores each request's complete blocks (qsa_block_scores), keeps the best
(qsa_topk), lists the token slots of those blocks and of its incomplete one
(qsa_key_table) and runs flash_decode over that list as a page-size-1 table, all at the
page table's width with no host value, so a CommandGraph replays it. A prefill whose
queries all fit the budget runs flash_prefill; otherwise each request's queries, in chunks,
select blocks with MLX ops and attend them with flash_prefill's block-sparse mode.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.attention.full.common import FullAttnMetadata
from ftmetal.attention.full.metal import FullAttnMetalBackend
from ftmetal.attention.qsa.common import QSAAttnBackend, block_scores, visible_blocks
from ftmetal.kernel.fdecode import flash_decode
from ftmetal.kernel.fprefill import flash_prefill_qsa
from ftmetal.kernel.qsa import qsa_block_scores, qsa_key_table, qsa_topk, qsa_window_mean

# Prefill queries selecting at once, at most: bounds their [queries, blocks] scores and masks.
QUERY_CHUNK = 256
# A chunk's selection holds about this many fp32 [queries, blocks] arrays at once (the scores,
# their sort, the tie ranks, the masks); the chunk shrinks so they stay in the workspace.
SELECTION_ARRAYS = 6
SELECTION_WORKSPACE_BYTES = 128 << 20


class QSAMetalBackend(QSAAttnBackend, FullAttnMetalBackend):
    def window_mean(self, index_cache: mx.array, window_slots: mx.array) -> mx.array:
        return qsa_window_mean(index_cache, window_slots)

    def sparse_decode(
        self, q: mx.array, index_queries: mx.array, layer_id: int, metadata: FullAttnMetadata
    ) -> mx.array:
        block_size = self.block_size
        page_size = self.table_page_size
        num_blocks = metadata.page_table.shape[1] * page_size // block_size
        if num_blocks == 0:
            # No request has a complete block yet: only its tail is listed.
            selected = mx.zeros((q.shape[0], 1), dtype=mx.int32)
        else:
            scores = qsa_block_scores(
                index_queries,
                self.qsa_pool.index_cache(layer_id),
                metadata.page_table,
                metadata.seq_lens,
                page_size,
                block_size,
                num_blocks,
            )
            # The full topk at any width: an eager decode's width grows with its context, and
            # a request with fewer complete blocks gets them all.
            selected = qsa_topk(scores, metadata.seq_lens, self.topk_blocks, block_size)
        key_slots, counts = qsa_key_table(
            selected, metadata.page_table, metadata.seq_lens, page_size, block_size
        )
        return flash_decode(
            q,
            self.kvcache.k_cache(layer_id),
            self.kvcache.v_cache(layer_id),
            key_slots,
            counts,
            key_slots.shape[1],
            1,
            self.scale,
        )

    def sparse_prefill(
        self, q: mx.array, index_queries: mx.array, layer_id: int, metadata: FullAttnMetadata
    ) -> mx.array:
        block_size = self.block_size
        page_size = self.table_page_size
        k_cache = self.kvcache.k_cache(layer_id)
        v_cache = self.kvcache.v_cache(layer_id)
        index_cache = self.qsa_pool.index_cache(layer_id)
        outputs = []
        start = 0
        for i, extend_len in enumerate(metadata.extend_lens_host):
            cached_len = metadata.cached_lens_host[i]
            num_complete = metadata.seq_lens_host[i] // block_size
            page_row = metadata.page_table[i : i + 1]
            # Each complete block's key, at the slot of its last token.
            ends = mx.arange(block_size - 1, num_complete * block_size, block_size)
            block_keys = index_cache[page_row[0, ends // page_size] * page_size + ends % page_size]
            chunk = SELECTION_WORKSPACE_BYTES // (SELECTION_ARRAYS * 4 * max(1, num_complete))
            chunk = max(1, min(QUERY_CHUNK, chunk))
            for row in range(0, extend_len, chunk):
                rows = min(chunk, extend_len - row)
                first = cached_len + row
                key_len = first + rows
                query_ends = mx.arange(first + 1, key_len + 1, dtype=mx.int32)
                queries = slice(start + row, start + row + rows)
                scores = block_scores(index_queries[queries], block_keys[: key_len // block_size])
                visible = visible_blocks(
                    scores, query_ends, block_size, self.topk_blocks, -(-key_len // block_size)
                )
                out = flash_prefill_qsa(
                    q[queries],
                    k_cache,
                    v_cache,
                    page_row,
                    page_size,
                    visible,
                    first,
                    block_size,
                    self.scale,
                )
                # A chunk's scores and masks are freed before the next chunk builds its own.
                mx.async_eval(out)
                outputs.append(out)
            start += extend_len
        return mx.concatenate(outputs) if len(outputs) > 1 else outputs[0]


__all__ = ["QSAMetalBackend"]
