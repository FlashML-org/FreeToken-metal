"""Page allocation over the host page table.

Port of FreeToken scheduler/cache.py (CacheManager), round 1: no prefix cache, so a
match always misses, an unfinished request keeps its pages and a finished request
returns every page. The free list holds page-aligned token slots. lazy_free_region
batches the frees of one drained step; the drain itself is the step fence.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np

from ftmetal.core import Req


def div_ceil(a: int, b: int) -> int:
    return -(-a // b)


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: np.ndarray) -> None:
        self.num_pages = num_pages
        self.page_size = page_size
        self.page_table = page_table
        self.free_slots = np.arange(num_pages, dtype=np.int32) * page_size
        self.deferred_frees: list[np.ndarray] | None = None

    @property
    def available_size(self) -> int:
        return len(self.free_slots) * self.page_size

    def match_len(self, input_ids: np.ndarray) -> int:
        """Cached prefix length; always 0 without a prefix cache."""
        return 0

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
        if needed_pages > len(self.free_slots):
            raise RuntimeError("KV pages exhausted")
        pages = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        offset = 0
        for table_idx, first_page, last_page in allocation:
            count = last_page - first_page
            page_bases = pages[offset : offset + count]
            slots = (page_bases[:, None] + np.arange(self.page_size, dtype=np.int32)).reshape(-1)
            start, stop = first_page * self.page_size, last_page * self.page_size
            self.page_table[table_idx, start:stop] = slots
            offset += count

    def cache_req(self, req: Req, *, finished: bool) -> None:
        """Commit a request's pages: without a prefix cache only a finish frees anything."""
        if not finished:
            return
        stop = div_ceil(req.cached_len, self.page_size) * self.page_size
        self.free_pages(self.page_table[req.table_idx, :stop])

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
        assert len(self.free_slots) == self.num_pages, "pages leaked or double freed"
        assert np.all(self.free_slots % self.page_size == 0)


__all__ = ["CacheManager", "div_ceil"]
