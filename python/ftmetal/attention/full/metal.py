"""Full attention with the paged flash-decode Metal kernel for decode.

Decode attends straight over the pool: store_kv has already written this step's key,
and seq_lens counts it. Prefill stays on the MLX backend's SDPA path.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.attention.full.common import FullAttnMetadata
from ftmetal.attention.full.mlx import FullAttnMLXBackend
from ftmetal.core import Batch
from ftmetal.kernel.fdecode import flash_decode


class FullAttnMetalBackend(FullAttnMLXBackend):
    def attend(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        layer_id: int,
        batch: Batch,
        metadata: FullAttnMetadata,
    ) -> mx.array:
        if metadata.max_seqlen_q > 1:
            return super().attend(q, k, v, layer_id, batch, metadata)
        else:
            return flash_decode(
                q,
                self.kvcache.k_cache(layer_id),
                self.kvcache.v_cache(layer_id),
                metadata.page_table,
                metadata.seq_lens,
                metadata.max_seqlen_k,
                self.page_size,
                self.scale,
            )


__all__ = ["FullAttnMetalBackend"]
