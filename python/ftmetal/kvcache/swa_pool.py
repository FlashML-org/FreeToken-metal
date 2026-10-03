"""Window pool for sliding-window layers: K/V slots reached through a full->swa mapping.

Port of the window half of FreeToken kvcache/hybrid_swa_pool.py (HybridSWAKVCache's swa
group, translate_loc_from_full_to_swa, alloc_swa / free_swa, the slot-0 sentinel,
_naive_swa_num_tokens and _swa_pool_floor, what the running requests hold beyond eviction).
Divergences: the full-attention layers live in the FULL type's own MHAKVCache, so this pool
holds only the window layers, token-granular (page size 1); the mapping and the free list
are host numpy, translated when the metadata is built (the page table is host-side too), so
the budget does not price them; the naive window is sized from the per-forward token budget
where FreeToken reads max_seq_len (a whole 262k-token context per request would not fit),
which the scheduler's out-of-window frees during prefill keep sufficient; the SWA radix
cache's pool is a number of whole windows, a fixed cost refused below the floor, where
FreeToken takes a share of the full pool's tokens raised to the floor; the floor's decode
tail counts the page the next step writes where FreeToken counts its 128-forward free
interval (the frees here run at every drain), and it holds two full-size prefill chunks
on top, so a chunk always runs whole.
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
# The SWA radix window pool's default: the fewest windows that hold the floor and these on
# top, for about four finished conversations to stay resumable. 8 is arbitrary.
EXTRA_DEFAULT_WINDOWS = 8


def naive_swa_num_tokens(sliding_window: int, max_extend_tokens: int, max_running_req: int) -> int:
    """One window plus a forward's tokens per running request, plus one request of headroom
    (FreeToken's _naive_swa_num_tokens with max_forward_len = max_extend_tokens); the
    sentinel is one of these slots."""
    width = -(-(sliding_window + max_extend_tokens + 1) // WINDOW_ALIGN) * WINDOW_ALIGN
    return (max_running_req + 1) * width


def swa_radix_floor_tokens(
    sliding_window: int, page_size: int, max_running_req: int, max_extend_tokens: int
) -> int:
    """Window slots no eviction reaches (FreeToken's _swa_pool_floor, and a prefill): each
    running request's committed window, locked with the retain gap in whole pages, and its
    own decode tail up to its next free (a window, the page of margin, the page the step in
    flight writes and the one the next step takes); and the two full-size chunks a request
    in chunked prefill holds, the one draining and the next."""
    locked = -(-(sliding_window + SWA_RETAIN_GAP) // page_size) * page_size
    chunk = -(-max_extend_tokens // page_size) * page_size
    return max_running_req * (locked + sliding_window + 3 * page_size) + 2 * chunk


def swa_radix_num_windows(
    sliding_window: int,
    page_size: int,
    max_running_req: int,
    max_extend_tokens: int,
    swa_windows: int | None,
) -> int:
    """Windows of the SWA radix cache's pool: swa_windows, by default the fewest that hold the
    floor and EXTRA_DEFAULT_WINDOWS; fewer than hold the floor are refused."""
    floor = swa_radix_floor_tokens(sliding_window, page_size, max_running_req, max_extend_tokens)
    fewest = -(-floor // sliding_window)
    if swa_windows is None:
        return fewest + EXTRA_DEFAULT_WINDOWS
    elif swa_windows < fewest:
        raise ValueError(
            f"--swa-windows {swa_windows} holds fewer than the {floor} window slots "
            f"{max_running_req} running requests and {max_extend_tokens}-token prefill chunks "
            f"need: pass --swa-windows {fewest} or more"
        )
    else:
        return swa_windows


def swa_radix_num_tokens(
    sliding_window: int,
    page_size: int,
    max_running_req: int,
    max_extend_tokens: int,
    swa_windows: int | None,
) -> int:
    """The SWA radix cache's window pool slots: whole windows and the sentinel."""
    num_windows = swa_radix_num_windows(
        sliding_window, page_size, max_running_req, max_extend_tokens, swa_windows
    )
    return num_windows * sliding_window + 1


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
    def window_slots(cls, spec: KVCacheGroupSpec, config: PoolConfig) -> int:
        """Naive: a window and a forward per running request. Radix (the SWA radix cache on a
        sliding-window model): whole windows."""
        assert spec.sliding_window is not None
        if config.cache_type == "naive":
            return naive_swa_num_tokens(
                spec.sliding_window, config.max_extend_tokens, config.max_running_req
            )
        else:
            return swa_radix_num_tokens(
                spec.sliding_window,
                config.page_size,
                config.max_running_req,
                config.max_extend_tokens,
                config.swa_windows,
            )

    @classmethod
    def kv_cost(cls, spec: KVCacheGroupSpec, config: PoolConfig) -> PoolCost:
        return PoolCost(0, cls.window_slots(spec, config) * spec.token_bytes(config.dtype))

    @classmethod
    def from_spec(cls, spec: KVCacheGroupSpec, config: PoolConfig, num_pages: int) -> Self:
        # + 1: the FULL pool's dummy page, whose slots stay on the sentinel.
        num_full_tokens = (num_pages + 1) * config.page_size
        return cls(spec, cls.window_slots(spec, config), num_full_tokens, config.dtype)

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
    "EXTRA_DEFAULT_WINDOWS",
    "SENTINEL_SLOT",
    "SWAKVCache",
    "SWA_RETAIN_GAP",
    "naive_swa_num_tokens",
    "swa_radix_floor_tokens",
    "swa_radix_num_tokens",
    "swa_radix_num_windows",
]
