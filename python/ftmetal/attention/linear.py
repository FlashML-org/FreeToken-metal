"""Linear (GDN) attention: metadata only; the layer computes on the state pool.

Port of FreeToken attention/linear.py (build_fla_metadata with its hybrid-radix track
metadata). A prefill request whose chunk holds a SNAPSHOT_STRIDE boundary strictly inside
it tracks the deepest one: the GDN layer writes the state there into the request's next
ping-pong slot, and the request records the boundary for its commit. The stride is the
chunked kernel's CHUNK_SIZE as in FreeToken, independent of the page size (which must
divide it): at page size 1 a page-based stride would put the snapshot at P-1, past the
P-2 where Qwen's next turn diverges after the client drops the reasoning, and the turn
would resume from the previous prefill chunk. Divergences: the builder is a backend with
the shared prepare_metadata entry; the track fields are a slot and a token count per
request (the layer splits its recurrence there) instead of chunk rows into the kernel's
per-chunk states; a decode batch's metadata already has its bucket's shapes, so a replay
keeps it; a capture batch's dummy rows run on scratch slots borrowed from the state pool
for the captures, one each and zeroed before every replay (on FreeToken's shared padding
slot they would race, and the arena checks compare the replays bit for bit).
"""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import numpy as np

from ftmetal.attention.base import BaseAttnBackend
from ftmetal.attention.metadata import AttnType, LinearAttnMetadata
from ftmetal.core import Batch
from ftmetal.kernel.gdn_chunk import CHUNK_SIZE
from ftmetal.kvcache.base import BasePool
from ftmetal.kvcache.linear_state_pool import LinearStatePool

# Snapshot boundaries sit cached_len + k * SNAPSHOT_STRIDE; the radix tree keys nodes by
# page, so the stride is a multiple of the page size and cached_len is page-aligned.
SNAPSHOT_STRIDE = CHUNK_SIZE


class LinearAttnBackend(BaseAttnBackend):
    def __init__(self, pool: BasePool, page_size: int) -> None:
        assert isinstance(pool, LinearStatePool), "linear attention runs on the GDN state pool"
        if SNAPSHOT_STRIDE % page_size != 0:
            raise ValueError(
                f"GDN snapshot stride {SNAPSHOT_STRIDE} must be a multiple of page_size {page_size}"
            )
        self.pool = pool
        self.capture_bs: list[int] = []
        self.capture_slots: list[int] = []

    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        self.capture_bs = sorted(bs_list)
        self.capture_slots = self.pool.alloc(max(bs_list))

    def prepare_for_capture(self, batch: Batch) -> None:
        self.prepare_for_replay(batch)
        metadata = batch.attn_metadata[AttnType.LINEAR]
        assert isinstance(metadata, LinearAttnMetadata)
        slots = self.capture_slots[: batch.padded_size]
        batch.attn_metadata[AttnType.LINEAR] = replace(
            metadata, cache_indices=mx.array(slots, dtype=mx.int32), slots_host=slots
        )

    def prepare_for_replay(self, batch: Batch) -> None:
        assert batch.is_decode and batch.padded_size in self.capture_bs

    def reset_capture_state(self) -> None:
        self.pool.clear_slots(self.capture_slots)

    def release_capture(self) -> None:
        self.pool.free(self.capture_slots)
        self.capture_slots = []

    def prepare_metadata(self, batch: Batch) -> None:
        reqs = batch.padded_reqs
        lens = [req.extend_len for req in reqs]
        slots: list[int] = []
        track_slots: list[int] = []
        track_seqlens: list[int] = []
        for req in reqs:
            assert req.linear_slot_idx is not None, "every request needs a GDN state slot"
            slots.append(req.linear_slot_idx)
            track_slot, track_seqlen = -1, 0
            strides = (req.extend_len - 1) // SNAPSHOT_STRIDE
            if batch.is_prefill and req.mamba_ping_pong is not None and strides > 0:
                track_seqlen = req.cached_len + strides * SNAPSHOT_STRIDE
                track_slot = req.mamba_ping_pong[req.mamba_next_track_idx]
                req.mamba_last_track_seqlen = track_seqlen
                req.mamba_next_track_idx = 1 - req.mamba_next_track_idx
            track_slots.append(track_slot)
            track_seqlens.append(track_seqlen)
        cu = np.zeros(len(lens) + 1, dtype=np.int32)
        np.cumsum(lens, out=cu[1:])
        batch.attn_metadata[AttnType.LINEAR] = LinearAttnMetadata(
            cu_seqlens=mx.array(cu),
            cache_indices=mx.array(np.asarray(slots, dtype=np.int32)),
            track_slots=mx.array(np.asarray(track_slots, dtype=np.int32)),
            seq_lens_host=lens,
            slots_host=slots,
            track_slots_host=track_slots,
            track_seqlens_host=track_seqlens,
        )


__all__ = ["LinearAttnBackend"]
