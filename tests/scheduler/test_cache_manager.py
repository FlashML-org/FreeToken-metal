"""Page allocation contract: every page returns, rows hold page-aligned slots."""

from __future__ import annotations

import numpy as np
import pytest

from ftmetal.core import Req, SamplingParams
from ftmetal.scheduler.cache import CacheManager


def make_req(table_idx: int, prompt_len: int, output_len: int) -> Req:
    return Req(
        input_ids=np.arange(prompt_len, dtype=np.int32),
        table_idx=table_idx,
        cached_len=0,
        output_len=output_len,
        uid=table_idx,
        sampling_params=SamplingParams(max_tokens=output_len),
    )


def pages_of(row: np.ndarray, page_size: int) -> set[int]:
    return set((row // page_size).tolist())


@pytest.mark.parametrize("page_size", [1, 4, 32])
def test_allocate_then_finish_returns_every_page(page_size: int) -> None:
    num_pages = 64
    page_table = np.zeros((2, 128), dtype=np.int32)
    manager = CacheManager(num_pages, page_size, page_table)
    reqs = [make_req(0, 5, 3), make_req(1, 9, 1)]
    manager.allocate_paged(reqs)
    used_pages = sum(-(-req.device_len // page_size) for req in reqs)
    assert manager.available_size == (num_pages - used_pages) * page_size
    rows = [page_table[req.table_idx, : req.device_len] for req in reqs]
    assert all(len(pages_of(row, page_size)) == -(-len(row) // page_size) for row in rows)
    assert pages_of(rows[0], page_size).isdisjoint(pages_of(rows[1], page_size))
    for _ in range(2):
        for req in reqs:
            req.complete_one()
        manager.allocate_paged(reqs)
    # A request finishes after its last step's complete_one, so the freed span covers
    # every page the last allocate handed out.
    for req in reqs:
        req.complete_one()
    for req in reqs:
        row = page_table[req.table_idx, : req.cached_len]
        assert len(pages_of(row, page_size)) == -(-req.cached_len // page_size)
    with manager.lazy_free_region():
        for req in reqs:
            manager.cache_req(req, finished=True)
        assert manager.available_size < num_pages * page_size
    manager.check_integrity()


def test_pages_exhausted_raises() -> None:
    page_table = np.zeros((1, 64), dtype=np.int32)
    manager = CacheManager(2, 4, page_table)
    with pytest.raises(RuntimeError):
        manager.allocate_paged([make_req(0, 12, 1)])
