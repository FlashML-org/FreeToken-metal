"""Full and sliding-window attention on the paged Metal kernels: flash_decode for
decode, flash_prefill for prefill.

Both attend straight over the pool through the page table: store_kv has already written
this step's keys, and seq_lens counts them. A window row starts at the oldest key its
queries see, so decode reads the whole row and prefill masks within it. A prefill
batch's flash_prefill tiles are planned once in prepare_metadata and shared by every
layer of the type.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ftmetal.attention.full.common import FullAttnBackend, FullAttnMetadata
from ftmetal.core import Batch
from ftmetal.kernel.fdecode import flash_decode
from ftmetal.kernel.fprefill import PrefillTiles, flash_prefill, prefill_tiles
from ftmetal.kvcache.base import BasePool
from ftmetal.models.config import KVCacheGroupSpec, ModelConfig


@dataclass
class MetalAttnMetadata(FullAttnMetadata):
    # Prefill: the flash_prefill tiles, planned once for every layer; None for decode batches.
    prefill_tiles: PrefillTiles | None = None


class FullAttnMetalBackend(FullAttnBackend):
    metadata_cls = MetalAttnMetadata

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

    def prepare_metadata(self, batch: Batch) -> None:
        super().prepare_metadata(batch)
        metadata = batch.attn_metadata[self.attn_type]
        assert isinstance(metadata, MetalAttnMetadata)
        metadata.prefill_tiles = (
            prefill_tiles(
                metadata.extend_lens_host,
                metadata.cached_lens_host,
                self.config.num_qo_heads,
                self.num_kv_heads,
                self.sliding_window,
            )
            if metadata.max_seqlen_q > 1
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
                self.table_page_size,
                self.scale,
            )
        else:
            assert isinstance(metadata, MetalAttnMetadata) and metadata.prefill_tiles is not None
            return flash_prefill(
                q,
                self.kvcache.k_cache(layer_id),
                self.kvcache.v_cache(layer_id),
                metadata.page_table,
                metadata.prefill_tiles,
                self.table_page_size,
                self.scale,
                self.sliding_window,
                block_ends=metadata.block_ends,
            )


__all__ = ["FullAttnMetalBackend", "MetalAttnMetadata"]
