"""Window pool for sliding-window layers: K/V slots reached through a full->swa mapping.

Port of the window half of FreeToken kvcache/hybrid_swa_pool.py (HybridSWAKVCache's swa
group, translate_loc_from_full_to_swa, alloc_swa / free_swa, the slot-0 sentinel,
_naive_swa_num_tokens and, for the SWA radix cache, _swa_paged_num_tokens: a share of the
full pool's tokens, at least what the running requests hold beyond eviction). Divergences:
the full-attention layers live in the FULL type's own MHAKVCache, so this pool holds only
the window layers, token-granular (page size 1); the mapping and the free list are host
numpy, translated when the metadata is built (the page table is host-side too), so the
budget does not price them; the naive window is sized from the per-forward token budget
where FreeToken reads max_seq_len (a whole 262k-token context per request would not fit),
which the scheduler's out-of-window frees during prefill keep sufficient; the radix floor's
decode tail counts the page the next step writes where FreeToken counts its 128-forward
free interval (the frees here run at every drain); no absolute window override.
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
# FreeToken parity: the SWA radix commit keeps this many tokens before a request's last window
# live too, for a next turn whose re-rendered prompt diverges a few tokens before the end.
SWA_RETAIN_GAP = 16


def naive_swa_num_tokens(sliding_window: int, max_extend_tokens: int, max_running_req: int) -> int:
    """One window plus a forward's tokens per running request, plus one request of headroom
    (FreeToken's _naive_swa_num_tokens with max_forward_len = max_extend_tokens); the
    sentinel is one of these slots."""
    width = -(-(sliding_window + max_extend_tokens + 1) // WINDOW_ALIGN) * WINDOW_ALIGN
    return (max_running_req + 1) * width


def swa_radix_floor_tokens(sliding_window: int, page_size: int, max_running_req: int) -> int:
    """Window slots no eviction reaches while max_running_req requests decode (FreeToken's
    _swa_pool_floor): each one's committed window, locked with the retain gap in whole
    pages, and its own tail up to its next free (a window, the page of margin, the page the
    step in flight writes and the one the next step takes)."""
    locked = -(-(sliding_window + SWA_RETAIN_GAP) // page_size) * page_size
    return max_running_req * (locked + sliding_window + 3 * page_size)


def swa_radix_num_tokens(
    sliding_window: int,
    page_size: int,
    max_running_req: int,
    swa_full_tokens_ratio: float,
    num_full_tokens: int,
) -> int:
    """The SWA radix cache's window pool: swa_full_tokens_ratio of the full pool's tokens,
    at least the floor, and the sentinel (FreeToken's _swa_paged_num_tokens)."""
    assert 0.0 < swa_full_tokens_ratio <= 1.0, "swa_full_tokens_ratio must be in (0, 1]"
    floor = swa_radix_floor_tokens(sliding_window, page_size, max_running_req)
    return max(floor, int(swa_full_tokens_ratio * num_full_tokens)) + 1


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
        """Naive: the whole pool, fixed. Radix (the SWA radix cache on a sliding-window
        model): the ratio's share of every page plus the floor, an upper bound on what
        from_spec allocates, as FreeToken prices it."""
        assert spec.sliding_window is not None
        token_bytes = spec.token_bytes(config.dtype)
        if config.cache_type == "naive":
            num_swa_tokens = naive_swa_num_tokens(
                spec.sliding_window, config.max_extend_tokens, config.max_running_req
            )
            return PoolCost(0, num_swa_tokens * token_bytes)
        else:
            floor = swa_radix_floor_tokens(
                spec.sliding_window, config.page_size, config.max_running_req
            )
            page_bytes = int(token_bytes * config.page_size * config.swa_full_tokens_ratio)
            return PoolCost(page_bytes, floor * token_bytes)

    @classmethod
    def from_spec(cls, spec: KVCacheGroupSpec, config: PoolConfig, num_pages: int) -> Self:
        assert spec.sliding_window is not None
        # + 1: the FULL pool's dummy page, whose slots stay on the sentinel.
        num_full_tokens = (num_pages + 1) * config.page_size
        if config.cache_type == "naive":
            num_swa_tokens = naive_swa_num_tokens(
                spec.sliding_window, config.max_extend_tokens, config.max_running_req
            )
        else:
            num_swa_tokens = swa_radix_num_tokens(
                spec.sliding_window,
                config.page_size,
                config.max_running_req,
                config.swa_full_tokens_ratio,
                num_full_tokens,
            )
        return cls(spec, num_swa_tokens, num_full_tokens, config.dtype)

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

    def move_swa(self, target_full_slots: np.ndarray, source_full_slots: np.ndarray) -> None:
        """Hand the window slots behind the source full slots over to the target ones, which
        have none."""
        assert np.all(self.full_to_swa[target_full_slots] == SENTINEL_SLOT)
        self.full_to_swa[target_full_slots] = self.full_to_swa[source_full_slots]
        self.full_to_swa[source_full_slots] = SENTINEL_SLOT

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


__all__ = [
    "SENTINEL_SLOT",
    "SWAKVCache",
    "SWA_RETAIN_GAP",
    "naive_swa_num_tokens",
    "swa_radix_floor_tokens",
    "swa_radix_num_tokens",
]
