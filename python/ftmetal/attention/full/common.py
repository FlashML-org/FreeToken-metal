"""Full (causal GQA) and sliding-window attention over a paged K/V pool: shared metadata
and dispatch.

Port of the shared part of FreeToken's full-attention backends (attention/fa.py's
prepare_metadata, capture hooks and forward, and the swa_indices translation of
attention/triton.py). Divergences: the metadata is built from the host page table into
fresh mx.arrays each step, out_loc is computed here, and the kernel implementation is the
subclass's attend(); one backend instance serves one type, FULL over MHAKVCache or SWA
over SWAKVCache, whose rows hold only the keys a step's windows reach, translated to
window slots, so the kernels never see out-of-window keys (FreeToken translates the whole
row and masks). prepare_for_replay swaps a decode batch's page table for its graph's
fixed width (FULL: the requests' whole page-table rows; SWA: the window) and max_seqlen_k
for that width (a window decode row is built that wide), so the flash-decode partition
count is fixed per graph; entries past a request's keys are stale but in bounds and
never read.
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
from ftmetal.kvcache.swa_pool import SENTINEL_SLOT, SWAKVCache
from ftmetal.models.config import KVCacheGroupSpec, ModelConfig


@dataclass
class FullAttnMetadata(BaseAttnMetadata):
    """One step's paged-attention plan. FULL: request b's keys are its first seq_lens[b]
    tokens, page_table names its pages and cached_lens[b] tokens precede its queries.
    SWA: request b's row starts at its window start s_b = max(0, cached_len - W + 1), the
    oldest key any of its queries sees: page_table holds the window slots of tokens
    [s_b, device_len) (page size 1; a decode step's rows are W wide, its graph's width),
    seq_lens[b] = device_len - s_b (at most W for a decode step), cached_lens[b] =
    cached_len - s_b, and positions count from s_b."""

    # Query index pointer over the flattened batch, int32 [size + 1].
    cu_seqlens_q: mx.array
    # Keys per request after this step's writes, int32 [size].
    seq_lens: mx.array
    # Page ids (window slots for SWA) per request, int32 [size, width]; unused entries
    # hold the dummy page (the sentinel slot).
    page_table: mx.array
    max_seqlen_q: int
    max_seqlen_k: int
    seq_lens_host: list[int]
    extend_lens_host: list[int]
    cached_lens_host: list[int]
    # SWA: the window slots this step's keys are stored at; FULL stores at batch.out_loc.
    out_loc: mx.array | None = None

    def get_last_indices(self, bs: int) -> mx.array:
        return self.cu_seqlens_q[1 : 1 + bs] - 1

    def step_inputs(self) -> list[mx.array]:
        arrays = [self.cu_seqlens_q, self.seq_lens, self.page_table]
        return arrays if self.out_loc is None else [*arrays, self.out_loc]


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
        assert isinstance(kvcache, MHAKVCache), "full attention reads a paged K/V pool"
        assert spec.attn_type in (AttnType.FULL, AttnType.SWA)
        self.config = config
        self.attn_type = spec.attn_type
        self.sliding_window = spec.sliding_window
        self.kvcache = kvcache
        self.page_table_host = page_table
        self.page_size = page_size
        self.scale = (
            config.attn_sm_scale if config.attn_sm_scale is not None else spec.head_dim**-0.5
        )
        if spec.attn_type is AttnType.SWA:
            assert isinstance(kvcache, SWAKVCache) and self.sliding_window is not None
            self.window_pool: SWAKVCache | None = kvcache
            # The window pool is token-granular; padded entries read the sentinel slot.
            self.table_page_size = 1
            self.pad_entry = SENTINEL_SLOT
            # No request reaches past the page table's width.
            self.decode_width = min(self.sliding_window, page_table.shape[1])
        else:
            self.window_pool = None
            self.table_page_size = page_size
            # The dummy page fills unused page-table entries so padded reads stay in bounds.
            self.pad_entry = kvcache.num_pages - 1
            self.decode_width = page_table.shape[1]
        self.capture_max_seq_len = 0
        self.capture_bs: list[int] = []

    def prepare_metadata(self, batch: Batch) -> None:
        if self.window_pool is None:
            self.prepare_full_metadata(batch)
        else:
            self.prepare_window_metadata(batch, self.window_pool)

    def prepare_full_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        seq_lens = [req.device_len for req in reqs]
        extend_lens = [req.extend_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seq_lens)
        max_pages = -(-max_seqlen_k // self.page_size)
        page_table = np.full((len(reqs), max_pages), self.pad_entry, dtype=np.int32)
        out_loc = np.empty(sum(extend_lens), dtype=np.int32)
        offset = 0
        for i, req in enumerate(reqs):
            row = self.page_table_host[req.table_idx]
            pages = -(-req.device_len // self.page_size)
            page_table[i, :pages] = row[: pages * self.page_size : self.page_size] // self.page_size
            out_loc[offset : offset + req.extend_len] = row[req.cached_len : req.device_len]
            offset += req.extend_len
        batch.out_loc = mx.array(out_loc)
        batch.attn_metadata[AttnType.FULL] = self.new_metadata(
            page_table, seq_lens, extend_lens, cached_lens, None
        )

    def prepare_window_metadata(self, batch: Batch, pool: SWAKVCache) -> None:
        reqs = batch.padded_reqs
        window = pool.sliding_window
        starts = [max(0, req.cached_len - window + 1) for req in reqs]
        seq_lens = [req.device_len - start for req, start in zip(reqs, starts)]
        extend_lens = [req.extend_len for req in reqs]
        cached_lens = [req.cached_len - start for req, start in zip(reqs, starts)]
        width = self.decode_width if batch.is_decode else max(seq_lens)
        page_table = np.full((len(reqs), width), self.pad_entry, dtype=np.int32)
        out_loc = np.empty(sum(extend_lens), dtype=np.int32)
        offset = 0
        for i, (req, start) in enumerate(zip(reqs, starts)):
            slots = pool.translate(self.page_table_host[req.table_idx, start : req.device_len])
            page_table[i, : len(slots)] = slots
            out_loc[offset : offset + req.extend_len] = slots[cached_lens[i] :]
            offset += req.extend_len
        batch.attn_metadata[AttnType.SWA] = self.new_metadata(
            page_table, seq_lens, extend_lens, cached_lens, mx.array(out_loc)
        )

    def new_metadata(
        self,
        page_table: np.ndarray,
        seq_lens: list[int],
        extend_lens: list[int],
        cached_lens: list[int],
        out_loc: mx.array | None,
    ) -> FullAttnMetadata:
        cu = np.zeros(len(extend_lens) + 1, dtype=np.int32)
        np.cumsum(extend_lens, out=cu[1:])
        return self.metadata_cls(
            cu_seqlens_q=mx.array(cu),
            seq_lens=mx.array(np.asarray(seq_lens, dtype=np.int32)),
            page_table=mx.array(page_table),
            max_seqlen_q=max(extend_lens),
            max_seqlen_k=max(seq_lens),
            seq_lens_host=seq_lens,
            extend_lens_host=extend_lens,
            cached_lens_host=cached_lens,
            out_loc=out_loc,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        assert max_seq_len == self.page_table_host.shape[1], "a graph spans whole table rows"
        self.capture_max_seq_len = self.decode_width
        self.capture_bs = sorted(bs_list)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata = batch.attn_metadata[self.attn_type]
        assert isinstance(metadata, FullAttnMetadata)
        assert batch.is_decode and batch.padded_size in self.capture_bs
        if self.window_pool is None:
            table = self.page_table_host[[req.table_idx for req in batch.padded_reqs]]
            if self.page_size > 1:
                table = table[:, :: self.page_size] // self.page_size
            metadata = replace(metadata, page_table=mx.array(table))
        batch.attn_metadata[self.attn_type] = replace(
            metadata, max_seqlen_k=self.capture_max_seq_len
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
        metadata = batch.attn_metadata[self.attn_type]
        assert isinstance(metadata, FullAttnMetadata)
        out_loc = batch.out_loc if metadata.out_loc is None else metadata.out_loc
        self.kvcache.store_kv(k, v, out_loc, layer_id)
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
