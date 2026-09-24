"""Full (causal GQA) attention over the paged K/V pool: shared metadata and dispatch.

Port of the shared part of FreeToken's full-attention backends (attention/fa.py's
prepare_metadata and forward). Divergences: the metadata is built from the host page
table into fresh mx.arrays each step, out_loc is computed here, and the kernel
implementation is the subclass's attend().
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ftmetal.attention.base import AttentionSpec, BaseAttnBackend
from ftmetal.attention.metadata import AttnType, BaseAttnMetadata
from ftmetal.core import Batch
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.models.config import ModelConfig


@dataclass
class FullAttnMetadata(BaseAttnMetadata):
    # Query index pointer over the flattened batch, int32 [size + 1].
    cu_seqlens_q: mx.array
    # Keys per request after this step's writes (device_len), int32 [size].
    seq_lens: mx.array
    # Page ids per request, int32 [size, max_pages]; unused entries hold the dummy page.
    page_table: mx.array
    max_seqlen_q: int
    max_seqlen_k: int
    seq_lens_host: list[int]
    extend_lens_host: list[int]
    cached_lens_host: list[int]

    def get_last_indices(self, bs: int) -> mx.array:
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FullAttnBackend(BaseAttnBackend):
    def __init__(
        self,
        config: ModelConfig,
        kvcache: MHAKVCache,
        page_table: np.ndarray,
        page_size: int,
    ) -> None:
        self.config = config
        self.kvcache = kvcache
        self.page_table_host = page_table
        self.page_size = page_size
        self.scale = config.head_dim**-0.5
        # The dummy page fills unused page-table entries so padded reads stay in bounds.
        self.dummy_page = kvcache.num_pages - 1

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.reqs
        seq_lens = [req.device_len for req in reqs]
        extend_lens = [req.extend_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seq_lens)
        max_seqlen_q = max(extend_lens)
        max_pages = -(-max_seqlen_k // self.page_size)
        page_table = np.full((len(reqs), max_pages), self.dummy_page, dtype=np.int32)
        out_loc = np.empty(sum(extend_lens), dtype=np.int32)
        offset = 0
        for i, req in enumerate(reqs):
            row = self.page_table_host[req.table_idx]
            pages = -(-req.device_len // self.page_size)
            page_table[i, :pages] = row[: pages * self.page_size : self.page_size] // self.page_size
            out_loc[offset : offset + req.extend_len] = row[req.cached_len : req.device_len]
            offset += req.extend_len
        cu = np.zeros(len(reqs) + 1, dtype=np.int32)
        np.cumsum(extend_lens, out=cu[1:])
        batch.out_loc = mx.array(out_loc)
        batch.attn_metadata[AttnType.FULL] = FullAttnMetadata(
            cu_seqlens_q=mx.array(cu),
            seq_lens=mx.array(np.asarray(seq_lens, dtype=np.int32)),
            page_table=mx.array(page_table),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seq_lens_host=seq_lens,
            extend_lens_host=extend_lens,
            cached_lens_host=cached_lens,
        )

    def forward(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> mx.array:
        """q [T, num_heads, D], k / v [T, num_kv_heads, D] -> [T, num_heads, D]."""
        if attn_spec is not None:
            raise ValueError("full attention takes no per-call AttentionSpec")
        metadata = batch.attn_metadata[AttnType.FULL]
        assert isinstance(metadata, FullAttnMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return self.attend(q, k, v, layer_id, batch, metadata)

    @abstractmethod
    def attend(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        layer_id: int,
        batch: Batch,
        metadata: FullAttnMetadata,
    ) -> mx.array:
        """Attention of q over the pool (which already holds k / v at out_loc)."""


__all__ = ["FullAttnBackend", "FullAttnMetadata"]
