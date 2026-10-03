"""Page allocation over the host page table, and the prefix cache of the configured type.

Port of FreeToken scheduler/cache.py (CacheManager: the radix, hybrid_radix and naive
paths). The radix path commits a request's KV prefix to a RadixPrefixCache. The hybrid
path owns both currencies of the HybridRadixCache, KV pages and GDN state slots: a request
holds a live slot and two ping-pong track slots, a prefill commit donates the frozen track
snapshot to the tree and a finish donates the live slot. On the naive path a
sliding-window model's window slots are the second currency (FreeToken's swa_paged
lifecycle): every token of a new page gets one, the ones behind a request's window are
freed as it slides, the rest with the pages on finish. Divergences: host numpy page table
and free list (page-aligned token slots); naive is refused on a model with linear
attention; radix on a sliding-window model runs naive, as there is no SWA radix cache;
eviction stops at what is evictable and exhaustion raises; no runtime rebuild or
tool-call anchor. lazy_free_region batches the frees of one drained step; the drain
itself is the step fence, so window slots slide out when the step that last extended a
request drains (FreeToken frees them when it schedules the next one, every 128 decode
forwards, relying on stream order).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Literal, NamedTuple

import numpy as np

from ftmetal.core import Req
from ftmetal.engine.config import CacheTypeChoice
from ftmetal.kvcache.hybrid_radix_cache import HybridCacheHandle, HybridRadixCache
from ftmetal.kvcache.linear_state_pool import LinearStatePool
from ftmetal.kvcache.radix_cache import RadixCacheHandle, RadixPrefixCache, align_down
from ftmetal.kvcache.swa_pool import SWAKVCache

CacheType = Literal["naive", "radix", "hybrid_radix"]

logger = logging.getLogger(__name__)


def div_ceil(a: int, b: int) -> int:
    return -(-a // b)


def resolve_cache_type(
    requested: CacheTypeChoice, *, has_linear_attention: bool, has_sliding_window: bool
) -> CacheType:
    """Radix is the radix cache that matches the model: on a model with linear attention the
    hybrid radix, the only tree that can resume a GDN state."""
    if has_linear_attention and requested == "naive":
        raise ValueError("--cache-type naive is not supported on a model with linear attention")
    elif has_linear_attention:
        return "hybrid_radix"
    elif has_sliding_window and requested == "radix":
        # Resolves to swa_radix once the SWA radix cache lands.
        logger.warning(
            "the SWA radix cache is not implemented yet: --cache-type radix runs naive and "
            "prefix reuse is off for this sliding-window model"
        )
        return "naive"
    else:
        return requested


class MatchResult(NamedTuple):
    # The prefix to lock; None on a miss.
    handle: RadixCacheHandle | HybridCacheHandle | None
    # The GDN snapshot slot to restore the live slot from; None on a miss or without one.
    mamba_value: int | None


class CacheManager:
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: np.ndarray,
        cache_type: CacheType,
        linear_state_pool: LinearStatePool | None = None,
        swa_pool: SWAKVCache | None = None,
    ) -> None:
        self.num_pages = num_pages
        self.page_size = page_size
        self.page_table = page_table
        self.cache_type = cache_type
        self.is_hybrid = cache_type == "hybrid_radix"
        assert self.is_hybrid == (linear_state_pool is not None), (
            "the hybrid radix cache, and only it, drives a GDN state pool"
        )
        assert swa_pool is None or cache_type == "naive", (
            f"window slots ride the naive cache only, not {cache_type}"
        )
        self.linear_state_pool = linear_state_pool
        self.swa_pool = swa_pool
        self.prefix_cache: RadixPrefixCache | HybridRadixCache | None = None
        if self.is_hybrid:
            self.prefix_cache = HybridRadixCache(page_size)
        elif cache_type == "radix":
            self.prefix_cache = RadixPrefixCache(page_size)
        self.free_slots = np.arange(num_pages, dtype=np.int32) * page_size
        self.deferred_frees: list[np.ndarray] | None = None

    @property
    def prefill_chunk_align(self) -> int:
        """A snapshot commits only at a page-aligned boundary, so non-final chunks end on one."""
        return self.page_size if self.is_hybrid else 1

    @property
    def evictable_size(self) -> int:
        """Tokens the prefix cache would give back to eviction."""
        if self.is_hybrid:
            return self.hybrid_parts()[0].full_evictable_size
        elif self.cache_type == "radix":
            return self.radix_cache().evictable_size
        else:
            return 0

    @property
    def available_size(self) -> int:
        """Tokens allocatable now: free pages plus the prefix cache's evictable ones."""
        return self.evictable_size + len(self.free_slots) * self.page_size

    @property
    def mamba_available_size(self) -> int:
        """GDN slots allocatable now: free slots plus unlocked tree snapshots."""
        cache, pool = self.hybrid_parts()
        return pool.num_free_slots + cache.mamba_evictable_size

    def page_usage(self) -> tuple[int, int]:
        """(used, total) pages; evictable prefix-cache pages count as free."""
        used = self.num_pages - len(self.free_slots) - self.evictable_size // self.page_size
        return used, self.num_pages

    def hybrid_parts(self) -> tuple[HybridRadixCache, LinearStatePool]:
        cache, pool = self.prefix_cache, self.linear_state_pool
        assert isinstance(cache, HybridRadixCache) and pool is not None, (
            "the prefix cache needs a GDN state pool"
        )
        return cache, pool

    def radix_cache(self) -> RadixPrefixCache:
        cache = self.prefix_cache
        assert isinstance(cache, RadixPrefixCache), f"cache type {self.cache_type} has no radix"
        return cache

    def match_req(self, input_ids: np.ndarray) -> MatchResult:
        """The cached prefix of a prompt, leaving at least its last token to extend."""
        assert len(input_ids) > 0
        prefix = input_ids[: len(input_ids) - 1]
        if self.is_hybrid:
            match = self.hybrid_parts()[0].match_prefix(prefix)
            if match.cached_len == 0:
                return MatchResult(None, None)
            return MatchResult(
                HybridCacheHandle(match.cached_len, match.node, match.kv_indices),
                match.mamba_value,
            )
        elif self.cache_type == "radix":
            match = self.radix_cache().match_prefix(prefix)
            if match.cached_len == 0:
                return MatchResult(None, None)
            return MatchResult(
                RadixCacheHandle(match.cached_len, match.node, match.kv_indices), None
            )
        else:
            return MatchResult(None, None)

    def lock(self, handle: RadixCacheHandle | HybridCacheHandle | None) -> None:
        if handle is None:
            return
        elif self.is_hybrid:
            self.hybrid_parts()[0].inc_lock(handle.node)
        else:
            self.radix_cache().inc_lock(handle.node)

    def unlock(self, handle: RadixCacheHandle | HybridCacheHandle | None) -> None:
        if handle is None:
            return
        elif self.is_hybrid:
            self.hybrid_parts()[0].dec_lock(handle.node)
        else:
            self.radix_cache().dec_lock(handle.node)

    def ensure_mamba_slots(self, num_slots: int) -> None:
        """Evict LRU tree snapshots until num_slots GDN slots are free (or nothing is evictable)."""
        cache, pool = self.hybrid_parts()
        while pool.num_free_slots < num_slots:
            evicted = cache.evict_mamba(num_slots - pool.num_free_slots)
            if not evicted.mamba_slots:
                break
            pool.free(evicted.mamba_slots)
            self.free_pages(evicted.kv_indices)

    def allocate_paged(self, reqs: list[Req]) -> None:
        needed_pages = 0
        allocation: list[tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation.append((req.table_idx, first_page, last_page))
        if needed_pages == 0:
            return
        short_tokens = (needed_pages - len(self.free_slots)) * self.page_size
        if short_tokens > 0 and self.is_hybrid:
            # Evicting KV leaves drops their snapshots too: return both currencies.
            cache, pool = self.hybrid_parts()
            evicted = cache.evict_full(short_tokens)
            self.free_slots = np.concatenate(
                [self.free_slots, evicted.kv_indices[:: self.page_size]]
            )
            pool.free(evicted.mamba_slots)
        elif short_tokens > 0 and self.cache_type == "radix":
            # Clamped to what is evictable: a shortfall left after it raises below.
            cache = self.radix_cache()
            evicted_slots = cache.evict(min(short_tokens, cache.evictable_size))
            self.free_slots = np.concatenate([self.free_slots, evicted_slots[:: self.page_size]])
        if needed_pages > len(self.free_slots):
            raise RuntimeError("KV pages exhausted")
        pages = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        if self.swa_pool is not None:
            # Whole pages, as FreeToken: the finish frees the page's every slot.
            page_offsets = np.arange(self.page_size, dtype=np.int32)
            self.swa_pool.alloc_swa((pages[:, None] + page_offsets).reshape(-1))
        offset = 0
        for table_idx, first_page, last_page in allocation:
            count = last_page - first_page
            page_bases = pages[offset : offset + count]
            slots = (page_bases[:, None] + np.arange(self.page_size, dtype=np.int32)).reshape(-1)
            start, stop = first_page * self.page_size, last_page * self.page_size
            self.page_table[table_idx, start:stop] = slots
            offset += count

    def cache_req(self, req: Req, *, finished: bool) -> None:
        """Commit a request's prefix to the prefix cache and, on finish, free what it owns."""
        if self.is_hybrid:
            self.cache_req_hybrid(req, finished=finished)
        elif self.cache_type == "radix":
            self.cache_req_radix(req, finished=finished)
        elif finished:
            stop = div_ceil(req.cached_len, self.page_size) * self.page_size
            token_slots = self.page_table[req.table_idx, :stop]
            if self.swa_pool is not None:
                self.swa_pool.free_swa(token_slots)
            self.free_pages(token_slots)

    def free_swa_out_of_window(self, req: Req) -> None:
        """Free the window slots of the tokens no later query of req sees, once the step
        that last extended req has drained (FreeToken's maybe_free_swa_out_of_window and
        free_swa_out_of_window_extend, with their window + page margin): the step in flight
        reads no key below cached_len - window."""
        pool = self.swa_pool
        assert pool is not None
        frontier = align_down(req.cached_len - pool.sliding_window - self.page_size, self.page_size)
        if frontier > req.swa_evicted_seqlen:
            pool.free_swa(self.page_table[req.table_idx, req.swa_evicted_seqlen : frontier])
            req.swa_evicted_seqlen = frontier

    def cache_req_radix(self, req: Req, *, finished: bool) -> None:
        """Commit the KV of [0, cached_len) to the radix tree.

        [0, old) is the tree's own, locked since admission; [old, prefix_len) was cached by
        another request meanwhile, so the request's copies are freed; the rest of the aligned
        prefix is donated; the unaligned tail stays with the request until it finishes.
        """
        cache = self.radix_cache()
        old_handle = req.cache_handle
        old_cached_len = 0 if old_handle is None else old_handle.cached_len
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        prefix_len, node = cache.insert(req.input_ids[: req.cached_len], page_indices)
        insert_len = align_down(req.cached_len, self.page_size)
        self.unlock(old_handle)
        self.free_pages(page_indices[old_cached_len:prefix_len])
        if finished:
            self.free_pages(page_indices[insert_len:])
        else:
            kv_indices = cache.collect_kv(node)
            if prefix_len > old_cached_len:
                # The dedup free above returned this request's own pages for the shared span;
                # the row must name the tree's pages the backends read from now on.
                self.page_table[req.table_idx, old_cached_len:prefix_len] = kv_indices[
                    old_cached_len:prefix_len
                ]
            if insert_len > 0:
                req.cache_handle = RadixCacheHandle(insert_len, node, kv_indices)
            else:
                req.cache_handle = None
            self.lock(req.cache_handle)

    def cache_req_hybrid(self, req: Req, *, finished: bool) -> None:
        """Commit KV and the GDN snapshot of a request.

        A prefill commit donates the frozen ping-pong slot (the snapshot its forward wrote at
        mamba_last_track_seqlen) to the tree and replaces it if the tree took it. A finish
        donates the live slot (the state at cached_len, if page-aligned) and frees the rest.
        """
        cache, pool = self.hybrid_parts()
        old_handle = req.cache_handle
        old_cached_len = 0 if old_handle is None else old_handle.cached_len
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        track_seqlen = req.mamba_last_track_seqlen

        if finished:
            # A pending track snapshot (the request finished before its commit) is a shorter
            # prefix than the live donate below: insert it first, so only [old, prefix_len)
            # is this request's duplicate to free. The frozen slot is consumed either way.
            free_upto = old_cached_len
            if (
                track_seqlen is not None
                and 0 < track_seqlen <= req.cached_len
                and align_down(track_seqlen, self.page_size) == track_seqlen
                and req.mamba_ping_pong is not None
            ):
                frozen = req.mamba_ping_pong[1 - req.mamba_next_track_idx]
                prefix_len, mamba_exist = cache.insert(
                    req.input_ids[:track_seqlen], page_indices[:track_seqlen], frozen
                )
                pool.free([slot for slot in req.mamba_ping_pong if mamba_exist or slot != frozen])
                req.mamba_ping_pong = None
                self.free_pages(page_indices[free_upto : max(free_upto, prefix_len)])
                free_upto = max(free_upto, track_seqlen)
            # The live state is at cached_len; attaching it to a shorter (aligned-down) node
            # would restore an over-advanced state, so an unaligned finish donates nothing.
            insert_len = align_down(req.cached_len, self.page_size)
            keep_live = False
            if insert_len == req.cached_len and insert_len > 0:
                assert req.linear_slot_idx is not None
                prefix_len, mamba_exist = cache.insert(
                    req.input_ids[:insert_len], page_indices[:insert_len], req.linear_slot_idx
                )
                self.unlock(old_handle)
                self.free_pages(page_indices[free_upto : max(free_upto, prefix_len)])
                keep_live = not mamba_exist
            else:
                self.unlock(old_handle)
                self.free_pages(page_indices[free_upto:])
            self.free_req_slots(req, keep_live=keep_live)
            return

        if track_seqlen is None:
            return
        if align_down(track_seqlen, self.page_size) != track_seqlen:
            # The insert would align the key down and attach an over-advanced state; the next
            # aligned boundary or the finish commits instead.
            req.mamba_last_track_seqlen = None
            return
        assert req.mamba_ping_pong is not None
        frozen_idx = 1 - req.mamba_next_track_idx
        frozen = req.mamba_ping_pong[frozen_idx]
        prefix_len, mamba_exist = cache.insert(
            req.input_ids[:track_seqlen], page_indices[:track_seqlen], frozen
        )
        self.unlock(old_handle)
        self.free_pages(page_indices[old_cached_len:prefix_len])
        # Lock the committed node before the replacement alloc below, whose eviction could
        # otherwise take this still-unlocked node's pages from under the request.
        match = cache.match_prefix(req.input_ids[:track_seqlen])
        if prefix_len > old_cached_len:
            # The dedup free above returned this request's own pages for the shared span;
            # the row must name the tree's pages the backends read from now on.
            self.page_table[req.table_idx, old_cached_len:prefix_len] = match.kv_indices[
                old_cached_len:prefix_len
            ]
        req.cache_handle = HybridCacheHandle(match.cached_len, match.node, match.kv_indices)
        self.lock(req.cache_handle)
        if not mamba_exist:
            self.ensure_mamba_slots(1)
            ping_pong = list(req.mamba_ping_pong)
            ping_pong[frozen_idx] = pool.alloc(1)[0]
            req.mamba_ping_pong = (ping_pong[0], ping_pong[1])
        req.mamba_last_track_seqlen = None

    def free_req_slots(self, req: Req, *, keep_live: bool = False) -> None:
        """Free a finished request's ping-pong pair and, unless donated, its live slot."""
        _, pool = self.hybrid_parts()
        slots = list(req.mamba_ping_pong) if req.mamba_ping_pong is not None else []
        if not keep_live and req.linear_slot_idx is not None:
            slots.append(req.linear_slot_idx)
        pool.free(slots)
        req.mamba_ping_pong = None
        req.linear_slot_idx = None

    def free_pages(self, token_slots: np.ndarray) -> None:
        if len(token_slots) == 0:
            return
        pages = token_slots[:: self.page_size].copy()
        if self.deferred_frees is not None:
            self.deferred_frees.append(pages)
        else:
            self.free_slots = np.concatenate([self.free_slots, pages])

    @contextmanager
    def lazy_free_region(self) -> Iterator[None]:
        assert self.deferred_frees is None
        self.deferred_frees = []
        try:
            yield
        finally:
            frees = self.deferred_frees
            self.deferred_frees = None
            self.free_slots = np.concatenate([self.free_slots, *frees])

    def check_integrity(self) -> None:
        """Idle only: every page, GDN slot and window slot is free or owned by the prefix
        cache."""
        cache_pages = 0
        if self.is_hybrid:
            cache, pool = self.hybrid_parts()
            cache.check_integrity()
            cache_pages = (cache.full_evictable + cache.full_protected) // self.page_size
            tree_slots = cache.mamba_evictable + cache.mamba_protected
            assert pool.num_free_slots + tree_slots == pool.num_slots - 1, (
                f"GDN slots leaked or double freed: free {pool.num_free_slots} + tree "
                f"{tree_slots} != {pool.num_slots - 1}"
            )
        elif self.cache_type == "radix":
            radix = self.radix_cache()
            radix.check_integrity()
            cache_pages = (radix.evictable_size + radix.protected_size) // self.page_size
        assert len(self.free_slots) + cache_pages == self.num_pages, (
            f"pages leaked or double freed: free {len(self.free_slots)} + cache {cache_pages} "
            f"!= {self.num_pages}"
        )
        assert np.all(self.free_slots % self.page_size == 0)
        if self.swa_pool is not None:
            # Every window slot but the sentinel is free.
            assert self.swa_pool.swa_available_size == self.swa_pool.swa_num_tokens - 1, (
                f"window slots leaked or double freed: free {self.swa_pool.swa_available_size} "
                f"!= {self.swa_pool.swa_num_tokens - 1}"
            )


__all__ = ["CacheManager", "CacheType", "MatchResult", "div_ceil", "resolve_cache_type"]
