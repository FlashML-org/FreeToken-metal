"""Full (causal GQA) attention over the paged K/V pool: shared metadata and dispatch.

Port of the shared part of FreeToken's full-attention backends (attention/fa.py's
prepare_metadata, capture hooks and forward). Divergences: the metadata is built from
the host page table into fresh mx.arrays each step, out_loc is computed here, and the
kernel implementation is the subclass's attend(). prepare_for_replay swaps a decode
batch's page table for its requests' whole page-table rows (the graph's width is the
row width) and max_seqlen_k for the row width, so the flash-decode partition count is
fixed per graph; entries past a request's keys are stale but in bounds and never read.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np

from ftmetal.attention.base import AttentionSpec, BaseAttnBackend
from ftmetal.attention.metadata import AttnType, BaseAttnMetadata
from ftmetal.core import Batch
from ftmetal.kvcache.base import BasePool
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.models.config import KVCacheGroupSpec, ModelConfig


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

    def step_inputs(self) -> list[mx.array]:
        return [self.cu_seqlens_q, self.seq_lens, self.page_table]


class FullAttnBackend(BaseAttnBackend):
    # The record prepare_metadata builds; a backend whose kernels plan more extends it.
    metadata_cls: type[FullAttnMetadata] = FullAttnMetadata

    def __init__(
        self,
        config: ModelConfig,
        spec: KVCacheGroupSpec,
        kvcache: BasePool,
        page_table: np.ndarray,
        page_size: int,
    ) -> None:
        assert isinstance(kvcache, MHAKVCache), "full attention reads the paged K/V pool"
        self.config = config
        self.kvcache = kvcache
        self.page_table_host = page_table
        self.page_size = page_size
        self.scale = spec.head_dim**-0.5
        # The dummy page fills unused page-table entries so padded reads stay in bounds.
        self.dummy_page = kvcache.num_pages - 1
        self.capture_max_seq_len = 0
        self.capture_bs: list[int] = []

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
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
        batch.attn_metadata[AttnType.FULL] = self.metadata_cls(
            cu_seqlens_q=mx.array(cu),
            seq_lens=mx.array(np.asarray(seq_lens, dtype=np.int32)),
            page_table=mx.array(page_table),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seq_lens_host=seq_lens,
            extend_lens_host=extend_lens,
            cached_lens_host=cached_lens,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        assert max_seq_len == self.page_table_host.shape[1], "a graph spans whole table rows"
        self.capture_max_seq_len = max_seq_len
        self.capture_bs = sorted(bs_list)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata = batch.attn_metadata[AttnType.FULL]
        assert isinstance(metadata, FullAttnMetadata)
        assert batch.is_decode and batch.padded_size in self.capture_bs
        table = self.page_table_host[[req.table_idx for req in batch.padded_reqs]]
        if self.page_size > 1:
            table = table[:, :: self.page_size] // self.page_size
        batch.attn_metadata[AttnType.FULL] = replace(
            metadata, page_table=mx.array(table), max_seqlen_k=self.capture_max_seq_len
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
