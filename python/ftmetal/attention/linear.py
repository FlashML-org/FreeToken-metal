"""Linear (GDN) attention: metadata only; the layer computes on the state pool.

Port of FreeToken attention/linear.py (build_fla_metadata). Divergences: the
builder is a backend with the shared prepare_metadata entry; the hybrid-radix track
fields arrive with the radix cache in round 2.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ftmetal.attention.base import BaseAttnBackend
from ftmetal.attention.metadata import AttnType, LinearAttnMetadata
from ftmetal.core import Batch


class LinearAttnBackend(BaseAttnBackend):
    def prepare_metadata(self, batch: Batch) -> None:
        lens = [req.extend_len for req in batch.reqs]
        slots = [req.linear_slot_idx for req in batch.reqs]
        assert all(slot is not None for slot in slots), "every request needs a GDN state slot"
        cu = np.zeros(len(lens) + 1, dtype=np.int32)
        np.cumsum(lens, out=cu[1:])
        batch.attn_metadata[AttnType.LINEAR] = LinearAttnMetadata(
            cu_seqlens=mx.array(cu),
            cache_indices=mx.array(np.asarray(slots, dtype=np.int32)),
            seq_lens_host=lens,
            slots_host=slots,
        )


__all__ = ["LinearAttnBackend"]
