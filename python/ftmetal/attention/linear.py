"""Linear (GDN) attention: metadata only; the layer computes on the state pool.

Port of FreeToken attention/linear.py (build_fla_metadata with its hybrid-radix track
metadata). A prefill request whose chunk holds a SNAPSHOT_CHUNK boundary strictly inside
it tracks the deepest one: the GDN layer writes the state there into the request's next
ping-pong slot, and the request records the boundary for its commit. Divergences: the
builder is a backend with the shared prepare_metadata entry; the track fields are a slot
and a token count per request (the layer splits its recurrence there) instead of chunk
rows into the kernel's per-chunk states.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ftmetal.attention.base import BaseAttnBackend
from ftmetal.attention.metadata import AttnType, LinearAttnMetadata
from ftmetal.core import Batch
from ftmetal.kvcache.linear_state_pool import SNAPSHOT_CHUNK


class LinearAttnBackend(BaseAttnBackend):
    def prepare_metadata(self, batch: Batch) -> None:
        lens = [req.extend_len for req in batch.reqs]
        slots: list[int] = []
        track_slots: list[int] = []
        track_seqlens: list[int] = []
        for req in batch.reqs:
            assert req.linear_slot_idx is not None, "every request needs a GDN state slot"
            slots.append(req.linear_slot_idx)
            track_slot, track_seqlen = -1, 0
            chunks = (req.extend_len - 1) // SNAPSHOT_CHUNK
            if batch.is_prefill and req.mamba_ping_pong is not None and chunks > 0:
                track_seqlen = req.cached_len + chunks * SNAPSHOT_CHUNK
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
