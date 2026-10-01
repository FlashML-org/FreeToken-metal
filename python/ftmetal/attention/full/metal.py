"""Full attention on the paged Metal kernels: flash-prefill for prefill, flash-decode for decode.

Both attend straight over the pool through the page table: store_kv has already written
this step's keys, and seq_lens counts them. A prefill batch's flash-prefill tiles are
planned once in prepare_metadata and shared by every layer. FTMETAL_PREFILL_KERNEL=0
sends prefill back to the MLX backend's SDPA path.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ftmetal.attention.full.common import FullAttnMetadata
from ftmetal.attention.full.mlx import FullAttnMLXBackend
from ftmetal.attention.metadata import AttnType
from ftmetal.core import Batch
from ftmetal.env import read_env
from ftmetal.kernel.fdecode import flash_decode
from ftmetal.kernel.fprefill import flash_prefill, prefill_tiles
from ftmetal.kvcache.base import BasePool
from ftmetal.models.config import KVCacheGroupSpec, ModelConfig


class FullAttnMetalBackend(FullAttnMLXBackend):
    def __init__(
        self,
        config: ModelConfig,
        spec: KVCacheGroupSpec,
        kvcache: BasePool,
        page_table: np.ndarray,
        page_size: int,
    ) -> None:
        super().__init__(config, spec, kvcache, page_table, page_size)
        self.num_kv_heads = spec.num_kv_heads
        self.use_prefill_kernel = read_env("PREFILL_KERNEL") != "0"

    def prepare_metadata(self, batch: Batch) -> None:
        super().prepare_metadata(batch)
        metadata = batch.attn_metadata[AttnType.FULL]
        assert isinstance(metadata, FullAttnMetadata)
        is_kernel_prefill = metadata.max_seqlen_q > 1 and self.use_prefill_kernel
        metadata.prefill_tiles = (
            prefill_tiles(
                metadata.extend_lens_host,
                metadata.cached_lens_host,
                self.config.num_qo_heads,
                self.num_kv_heads,
            )
            if is_kernel_prefill
            else None
        )

    def attend(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        layer_id: int,
        batch: Batch,
        metadata: FullAttnMetadata,
    ) -> mx.array:
        if metadata.max_seqlen_q == 1:
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
        elif metadata.prefill_tiles is not None:
            return flash_prefill(
                q,
                self.kvcache.k_cache(layer_id),
                self.kvcache.v_cache(layer_id),
                metadata.page_table,
                metadata.prefill_tiles,
                self.page_size,
                self.scale,
            )
        else:
            return super().attend(q, k, v, layer_id, batch, metadata)


__all__ = ["FullAttnMetalBackend"]
