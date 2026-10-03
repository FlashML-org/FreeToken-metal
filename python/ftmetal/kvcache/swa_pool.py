"""Window pool for sliding-window layers: K/V slots reached through a full->swa mapping.

Port of the window half of FreeToken kvcache/hybrid_swa_pool.py (HybridSWAKVCache's swa
group, translate_loc_from_full_to_swa, alloc_swa / free_swa, the slot-0 sentinel and
_naive_swa_num_tokens). Divergences: the full-attention layers live in the FULL type's
own MHAKVCache, so this pool holds only the window layers, token-granular (page size 1);
the mapping and the free list are host numpy, translated when the metadata is built (the
page table is host-side too), so the budget does not price them; the naive window is
sized from the per-forward token budget where FreeToken reads max_seq_len (a whole
262k-token context per request would not fit), which the scheduler's out-of-window frees
during prefill keep sufficient.
"""

from __future__ import annotations

from typing import Self

import mlx.core as mx
import numpy as np

from ftmetal.kvcache.base import PoolConfig, PoolCost
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.models.config import KVCacheGroupSpec

# A full slot mapped to this window slot has no live window K/V; padded metadata reads it.
SENTINEL_SLOT = 0
# FreeToken's alignment of one request's share of the naive window pool.
WINDOW_ALIGN = 32


def naive_swa_num_tokens(sliding_window: int, max_extend_tokens: int, max_running_req: int) -> int:
    """One window plus a forward's tokens per running request, plus one request of headroom
    (FreeToken's _naive_swa_num_tokens with max_forward_len = max_extend_tokens); the
    sentinel is one of these slots."""
    width = -(-(sliding_window + max_extend_tokens + 1) // WINDOW_ALIGN) * WINDOW_ALIGN
    return (max_running_req + 1) * width


class SWAKVCache(MHAKVCache):
    def __init__(
        self, spec: KVCacheGroupSpec, num_swa_tokens: int, num_full_tokens: int, dtype: mx.Dtype
    ) -> None:
        """num_full_tokens: the FULL pool's token slots, dummy page included."""
        assert spec.sliding_window is not None, "the window pool needs a sliding window"
        assert num_swa_tokens > 1, "the window pool needs a slot beyond the sentinel"
        super().__init__(
            num_kv_heads=spec.num_kv_heads,
            num_layers=spec.num_layers,
            head_dim=spec.head_dim,
            num_pages=num_swa_tokens,
            page_size=1,
            dtype=dtype,
            layer_ids=spec.layer_ids,
        )
        self.sliding_window = spec.sliding_window
        self.full_to_swa = np.full(num_full_tokens, SENTINEL_SLOT, dtype=np.int32)
        self.free_swa_slots = np.arange(SENTINEL_SLOT + 1, num_swa_tokens, dtype=np.int32)

    @classmethod
    def kv_cost(cls, spec: KVCacheGroupSpec, config: PoolConfig) -> PoolCost:
        assert spec.sliding_window is not None
        num_swa_tokens = naive_swa_num_tokens(
            spec.sliding_window, config.max_extend_tokens, config.max_running_req
        )
        return PoolCost(0, num_swa_tokens * spec.token_bytes(config.dtype))

    @classmethod
    def from_spec(cls, spec: KVCacheGroupSpec, config: PoolConfig, num_pages: int) -> Self:
        assert spec.sliding_window is not None
        num_swa_tokens = naive_swa_num_tokens(
            spec.sliding_window, config.max_extend_tokens, config.max_running_req
        )
        # + 1: the FULL pool's dummy page, whose slots stay on the sentinel.
        return cls(spec, num_swa_tokens, (num_pages + 1) * config.page_size, config.dtype)

    def translate(self, full_slots: np.ndarray) -> np.ndarray:
        """Full-pool token slots -> window slots (the sentinel where none is live)."""
        return self.full_to_swa[full_slots]

    def alloc_swa(self, full_slots: np.ndarray) -> None:
        """Back each full slot with a window slot of its own."""
        count = len(full_slots)
        if count > len(self.free_swa_slots):
            raise RuntimeError(f"SWA pool exhausted: need {count}, have {len(self.free_swa_slots)}")
        self.full_to_swa[full_slots] = self.free_swa_slots[:count]
        self.free_swa_slots = self.free_swa_slots[count:]

    def free_swa(self, full_slots: np.ndarray) -> None:
        """Return the window slots behind full_slots; slots already on the sentinel are skipped."""
        swa_slots = self.full_to_swa[full_slots]
        self.free_swa_slots = np.concatenate(
            [self.free_swa_slots, swa_slots[swa_slots != SENTINEL_SLOT]]
        )
        self.full_to_swa[full_slots] = SENTINEL_SLOT

    @property
    def swa_available_size(self) -> int:
        return len(self.free_swa_slots)

    @property
    def swa_num_tokens(self) -> int:
        """Window slots, the sentinel included."""
        return self.num_tokens


__all__ = ["SENTINEL_SLOT", "SWAKVCache", "naive_swa_num_tokens"]
