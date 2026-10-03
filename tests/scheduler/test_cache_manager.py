"""Page allocation and the prefix caches on CacheManager.

Every page returns and rows hold page-aligned slots. The radix type commits a request's
prefix at its prefill and its finish, frees and re-points duplicates and evicts unlocked
leaves under pressure; with a GDN state pool the hybrid radix commit donates the track
snapshot, a finish donates its live slot, and both currencies evict under pressure. Parts
port FreeToken tests/scheduler/test_hybrid_cache_manager.py and
test_commit_repoints_page_table.py, on a tiny real LinearStatePool.
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.attention.metadata import AttnType
from ftmetal.core import Req, SamplingParams
from ftmetal.kvcache.linear_state_pool import LinearStatePool, linear_pool_num_slots
from ftmetal.models.config import KVCacheGroupSpec
from ftmetal.scheduler.cache import CacheManager, MatchResult, resolve_cache_type
from ftmetal.scheduler.prefill import ChunkedReq, PendingReq, PrefillAdder
from ftmetal.scheduler.table import TableManager


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
    manager = CacheManager(num_pages, page_size, page_table, "naive")
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


@pytest.mark.parametrize("cache_type", ["naive", "radix"])
def test_pages_exhausted_raises(cache_type: Literal["naive", "radix"]) -> None:
    page_table = np.zeros((1, 64), dtype=np.int32)
    manager = CacheManager(2, 4, page_table, cache_type)
    with pytest.raises(RuntimeError):
        manager.allocate_paged([make_req(0, 12, 1)])


def make_pool(num_slots: int = 16) -> LinearStatePool:
    spec = KVCacheGroupSpec(
        name="linear",
        attn_type=AttnType.LINEAR,
        layer_ids=(0,),
        conv_shape=(4, 96),
        state_shape=(1, 32, 32),
    )
    return LinearStatePool(spec, num_slots=num_slots, dtype=mx.float32)


def hybrid_manager(
    num_pages: int = 16, page_size: int = 4, num_slots: int = 16
) -> tuple[CacheManager, LinearStatePool, np.ndarray]:
    pool = make_pool(num_slots)
    page_table = np.zeros((4, 64), dtype=np.int32)
    return CacheManager(num_pages, page_size, page_table, "hybrid_radix", pool), pool, page_table


def prefilled(
    manager: CacheManager,
    pool: LinearStatePool | None,
    table_idx: int,
    prompt: list[int],
    track_seqlen: int | None,
) -> Req:
    """A request admitted by hand and prefilled; with a GDN state pool its snapshot is
    tracked at track_seqlen."""
    match = manager.match_req(np.array(prompt, dtype=np.int32))
    manager.lock(match.handle)
    cached_len = 0 if match.handle is None else match.handle.cached_len
    req = Req(
        input_ids=np.array(prompt, dtype=np.int32),
        table_idx=table_idx,
        cached_len=cached_len,
        output_len=4,
        uid=table_idx,
        sampling_params=SamplingParams(max_tokens=4),
        cache_handle=match.handle,
    )
    if match.handle is not None:
        manager.page_table[table_idx, :cached_len] = match.handle.kv_indices
    if pool is not None:
        req.linear_slot_idx = pool.alloc(1)[0]
        first, second = pool.alloc(2)
        req.mamba_ping_pong = (first, second)
    manager.allocate_paged([req])
    req.complete_one()
    if track_seqlen is not None:  # as the GDN metadata marks a tracked forward
        req.mamba_last_track_seqlen = track_seqlen
        req.mamba_next_track_idx = 1
    return req


def test_prefill_commit_donates_the_track_snapshot_and_a_shared_prefix_hits() -> None:
    manager, pool, page_table = hybrid_manager()
    prompt = list(range(1, 10))
    assert manager.match_req(np.array(prompt, dtype=np.int32)) == MatchResult(None, None)
    req = prefilled(manager, pool, 0, prompt, track_seqlen=8)
    assert req.mamba_ping_pong is not None
    frozen = req.mamba_ping_pong[0]
    free_before = pool.num_free_slots
    with manager.lazy_free_region():
        manager.cache_req(req, finished=False)
    # The tree took the frozen slot and the request got a replacement.
    assert pool.num_free_slots == free_before - 1
    assert req.mamba_ping_pong is not None and frozen not in req.mamba_ping_pong

    match = manager.match_req(np.array(prompt[:8] + [42, 43], dtype=np.int32))
    assert match.handle is not None and match.handle.cached_len == 8
    assert match.mamba_value == frozen
    assert match.handle.kv_indices.tolist() == page_table[0, :8].tolist()


def test_finish_donates_the_live_slot_at_an_aligned_length() -> None:
    manager, pool, _ = hybrid_manager()
    prompt = list(range(1, 10))
    req = prefilled(manager, pool, 0, prompt[:8] + [0], track_seqlen=None)
    for _ in range(3):  # decode to cached_len 12, a page boundary
        req.append_host(7)
        manager.allocate_paged([req])
        req.complete_one()
    live = req.linear_slot_idx
    with manager.lazy_free_region():
        manager.cache_req(req, finished=True)
    assert req.linear_slot_idx is None and req.mamba_ping_pong is None
    match = manager.match_req(np.array([*prompt[:8], 0, 7, 7, 7, 5], dtype=np.int32))
    assert match.handle is not None and match.handle.cached_len == 12
    assert match.mamba_value == live
    manager.check_integrity()


def test_unaligned_finish_keeps_only_the_committed_snapshot() -> None:
    manager, pool, _ = hybrid_manager()
    req = prefilled(manager, pool, 0, list(range(1, 11)), track_seqlen=8)
    with manager.lazy_free_region():
        manager.cache_req(req, finished=False)
        manager.cache_req(req, finished=True)  # cached_len 10: the live state is not donated
    manager.check_integrity()
    assert manager.page_usage() == (0, manager.num_pages)
    assert manager.mamba_available_size == pool.num_slots - 1
    assert pool.num_free_slots == pool.num_slots - 2  # the tree keeps the one snapshot
    match = manager.match_req(np.array(range(1, 11), dtype=np.int32))
    assert match.handle is not None and match.handle.cached_len == 8


def test_duplicate_commit_repoints_the_row_off_its_freed_pages() -> None:
    """Two requests prefilled the same prompt; the later commit dedups, frees its own pages
    and its row must then name the tree's pages, which the backends read from now on."""
    manager, pool, page_table = hybrid_manager()
    prompt = list(range(1, 10))
    reqs = [prefilled(manager, pool, i, prompt, track_seqlen=8) for i in (0, 1)]
    own_pages = set(page_table[1, :8].tolist())
    with manager.lazy_free_region():
        for req in reqs:
            manager.cache_req(req, finished=False)
    free = set((manager.free_slots[:, None] + np.arange(4)).reshape(-1).tolist())
    assert own_pages <= free
    assert page_table[1, :8].tolist() == page_table[0, :8].tolist()
    assert set(page_table[1, :9].tolist()).isdisjoint(free)
    # The duplicate's frozen slot stays with the request; nothing leaks at finish.
    with manager.lazy_free_region():
        for req in reqs:
            manager.cache_req(req, finished=True)
    manager.check_integrity()


def test_page_pressure_evicts_tree_leaves_and_their_snapshots() -> None:
    manager, pool, _ = hybrid_manager(num_pages=6)
    req = prefilled(manager, pool, 0, list(range(1, 11)), track_seqlen=8)
    with manager.lazy_free_region():
        manager.cache_req(req, finished=False)
        manager.cache_req(req, finished=True)
    assert len(manager.free_slots) == 4 and manager.available_size == 6 * 4
    big = Req(
        input_ids=np.arange(100, 121, dtype=np.int32),
        table_idx=2,
        cached_len=0,
        output_len=1,
        uid=2,
        sampling_params=SamplingParams(max_tokens=1),
    )
    manager.allocate_paged([big])  # 6 pages: the tree's two must go
    assert manager.evictable_size == 0
    assert pool.num_free_slots == pool.num_slots - 1
    assert manager.match_req(np.array(range(1, 11), dtype=np.int32)) == MatchResult(None, None)


def test_ensure_mamba_slots_evicts_least_recent_snapshots() -> None:
    manager, pool, _ = hybrid_manager(num_slots=8)
    prompts = [[base + t for t in range(10)] for base in (100, 200)]
    for i, prompt in enumerate(prompts):
        req = prefilled(manager, pool, i, prompt, track_seqlen=8)
        with manager.lazy_free_region():
            manager.cache_req(req, finished=False)
            manager.cache_req(req, finished=True)
    assert pool.num_free_slots == pool.num_slots - 3
    manager.ensure_mamba_slots(pool.num_slots - 2)
    assert pool.num_free_slots == pool.num_slots - 2
    older = manager.match_req(np.array(prompts[0], dtype=np.int32))
    newer = manager.match_req(np.array(prompts[1], dtype=np.int32))
    assert older.handle is None and newer.handle is not None
    manager.check_integrity()


def test_free_req_slots_is_idempotent() -> None:
    manager, pool, _ = hybrid_manager()
    req = make_req(0, 3, 1)
    req.linear_slot_idx = pool.alloc(1)[0]
    first, second = pool.alloc(2)
    req.mamba_ping_pong = (first, second)
    base = pool.num_free_slots
    manager.free_req_slots(req)
    assert pool.num_free_slots == base + 3
    manager.free_req_slots(req)
    assert pool.num_free_slots == base + 3


def test_hybrid_prefill_chunks_end_on_a_page_boundary() -> None:
    manager, _, page_table = hybrid_manager(page_size=32)
    assert manager.prefill_chunk_align == 32
    table = TableManager(max_running_reqs=4, page_table=page_table)
    pending = PendingReq(0, np.arange(300, dtype=np.int32), SamplingParams(max_tokens=1))
    adder = PrefillAdder(
        token_budget=100, reserved_size=0, cache_manager=manager, table_manager=table
    )
    req = adder.try_add_one(pending)
    assert isinstance(req, ChunkedReq) and req.extend_len == 96
    # A budget below one page keeps the unaligned chunk rather than stall the request.
    pending = PendingReq(1, np.arange(300, dtype=np.int32), SamplingParams(max_tokens=1))
    adder = PrefillAdder(
        token_budget=20, reserved_size=0, cache_manager=manager, table_manager=table
    )
    req = adder.try_add_one(pending)
    assert req is not None and req.extend_len == 20


def test_without_a_prefix_cache_chunks_are_not_aligned() -> None:
    page_table = np.zeros((4, 512), dtype=np.int32)
    manager = CacheManager(64, 32, page_table, "naive")
    assert manager.prefill_chunk_align == 1
    table = TableManager(max_running_reqs=4, page_table=page_table)
    pending = PendingReq(0, np.arange(300, dtype=np.int32), SamplingParams(max_tokens=1))
    adder = PrefillAdder(
        token_budget=100, reserved_size=0, cache_manager=manager, table_manager=table
    )
    req = adder.try_add_one(pending)
    assert req is not None and req.extend_len == 100


def test_pool_sizing_covers_the_running_floor() -> None:
    """Four unevictable slots per running request plus the padding slot, even at a tiny ratio."""
    for max_running_req in (1, 8, 64):
        num_slots = linear_pool_num_slots(max_running_req, 0.1)
        assert num_slots >= 4 * max_running_req + 1
    assert linear_pool_num_slots(4, 2.0) == 4 * 4 + 8 + 1


def radix_manager(num_pages: int = 16, page_size: int = 4) -> tuple[CacheManager, np.ndarray]:
    page_table = np.zeros((4, 64), dtype=np.int32)
    return CacheManager(num_pages, page_size, page_table, "radix"), page_table


def test_cache_type_resolves_radix_to_the_hybrid_radix_on_a_gdn_model() -> None:
    assert resolve_cache_type("radix", has_linear_attention=False) == "radix"
    assert resolve_cache_type("naive", has_linear_attention=False) == "naive"
    assert resolve_cache_type("radix", has_linear_attention=True) == "hybrid_radix"
    with pytest.raises(ValueError):
        resolve_cache_type("naive", has_linear_attention=True)


def test_radix_prefill_commit_publishes_the_prompt_and_a_shared_prefix_hits() -> None:
    manager, page_table = radix_manager()
    assert manager.prefill_chunk_align == 1
    prompt = list(range(1, 11))
    assert manager.match_req(np.array(prompt, dtype=np.int32)) == MatchResult(None, None)
    req = prefilled(manager, None, 0, prompt, track_seqlen=None)
    with manager.lazy_free_region():
        manager.cache_req(req, finished=False)
    # The aligned prompt is the tree's now and locked by the request; the tail is its own.
    assert req.cache_handle is not None and req.cache_handle.cached_len == 8
    assert manager.evictable_size == 0

    match = manager.match_req(np.array(prompt[:8] + [42, 43], dtype=np.int32))
    assert match.handle is not None and match.handle.cached_len == 8
    assert match.mamba_value is None
    assert match.handle.kv_indices.tolist() == page_table[0, :8].tolist()


def test_radix_finish_keeps_the_aligned_prefix_and_frees_the_tail() -> None:
    manager, _ = radix_manager()
    req = prefilled(manager, None, 0, list(range(1, 11)), track_seqlen=None)
    with manager.lazy_free_region():
        manager.cache_req(req, finished=False)
    assert req.cache_handle is not None and req.cache_handle.cached_len == 8
    for _ in range(3):  # decode to cached_len 13: the finish commits 12, frees the last page
        req.append_host(7)
        manager.allocate_paged([req])
        req.complete_one()
    with manager.lazy_free_region():
        manager.cache_req(req, finished=True)
    manager.check_integrity()
    assert manager.page_usage() == (0, manager.num_pages)
    assert manager.evictable_size == 12
    assert len(manager.free_slots) == manager.num_pages - 3
    match = manager.match_req(np.array([*range(1, 11), 7, 7, 7, 5], dtype=np.int32))
    assert match.handle is not None and match.handle.cached_len == 12


def test_radix_unfinished_commit_repoints_the_row_off_the_freed_pages() -> None:
    """Two requests prefilled the same prompt; the later commit dedups, frees its own pages
    and its row must then name the tree's pages, which the backends read from now on."""
    manager, page_table = radix_manager()
    prompt = list(range(1, 10))
    reqs = [prefilled(manager, None, i, prompt, track_seqlen=None) for i in (0, 1)]
    own_pages = set(page_table[1, :8].tolist())
    with manager.lazy_free_region():
        for req in reqs:
            manager.cache_req(req, finished=False)
    free = set((manager.free_slots[:, None] + np.arange(4)).reshape(-1).tolist())
    assert own_pages <= free
    assert page_table[1, :8].tolist() == page_table[0, :8].tolist()
    assert set(page_table[1, :9].tolist()).isdisjoint(free)
    handle = reqs[1].cache_handle
    assert handle is not None and handle.kv_indices.tolist() == page_table[1, :8].tolist()
    with manager.lazy_free_region():
        for req in reqs:
            manager.cache_req(req, finished=True)
    manager.check_integrity()


def test_radix_subspan_commit_repoints_only_the_deduped_slice() -> None:
    """The committers admitted on top of a published prefix, so the dedup free and the
    re-point cover only [old cached_len, prefix_len)."""
    manager, page_table = radix_manager(num_pages=32, page_size=1)
    long = list(range(1, 17))
    seed = prefilled(manager, None, 0, long[:8], track_seqlen=None)
    with manager.lazy_free_region():
        manager.cache_req(seed, finished=False)
    second, third = (prefilled(manager, None, i, long, track_seqlen=None) for i in (1, 2))
    assert second.cache_handle is not None and second.cache_handle.cached_len == 8
    own_suffix = set(page_table[2, 8:16].tolist())

    with manager.lazy_free_region():
        manager.cache_req(second, finished=False)  # publishes [8, 16)
        manager.cache_req(third, finished=False)  # dedups against it: frees its own suffix
    free = set(manager.free_slots.tolist())
    assert own_suffix <= free
    assert set(page_table[2, :16].tolist()).isdisjoint(free)
    assert third.cache_handle is not None
    assert page_table[2, :16].tolist() == third.cache_handle.kv_indices.tolist()
    manager.check_integrity()


def test_radix_page_pressure_evicts_only_unlocked_leaves() -> None:
    manager, _ = radix_manager(num_pages=6)
    prompts = [[base + t for t in range(9)] for base in (100, 200)]
    for i, prompt in enumerate(prompts):
        req = prefilled(manager, None, i, prompt, track_seqlen=None)
        with manager.lazy_free_region():
            manager.cache_req(req, finished=False)
            manager.cache_req(req, finished=True)
    assert len(manager.free_slots) == 2 and manager.available_size == 6 * 4
    held = manager.match_req(np.array(prompts[1], dtype=np.int32)).handle
    assert held is not None and held.cached_len == 8
    manager.lock(held)

    big = make_req(2, 16, 1)
    manager.allocate_paged([big])  # 4 pages: the unlocked prompt's two must go
    assert manager.evictable_size == 0
    assert manager.match_req(np.array(prompts[0], dtype=np.int32)) == MatchResult(None, None)
    with pytest.raises(RuntimeError):  # what is left is locked
        manager.allocate_paged([make_req(3, 1, 1)])
    manager.unlock(held)
    assert manager.evictable_size == 8


def test_radix_admission_writes_the_cached_row_and_keeps_unaligned_chunks() -> None:
    page_table = np.zeros((4, 512), dtype=np.int32)
    manager = CacheManager(128, 4, page_table, "radix")
    prompt = list(range(1, 300))
    seed = prefilled(manager, None, 0, prompt[:41], track_seqlen=None)
    with manager.lazy_free_region():
        manager.cache_req(seed, finished=True)

    table = TableManager(max_running_reqs=4, page_table=page_table)
    pending = PendingReq(1, np.array(prompt, dtype=np.int32), SamplingParams(max_tokens=1))
    adder = PrefillAdder(
        token_budget=10, reserved_size=0, cache_manager=manager, table_manager=table
    )
    req = adder.try_add_one(pending)
    assert isinstance(req, ChunkedReq) and (req.cached_len, req.extend_len) == (40, 10)
    assert req.cache_handle is not None
    assert page_table[req.table_idx, :40].tolist() == req.cache_handle.kv_indices.tolist()
