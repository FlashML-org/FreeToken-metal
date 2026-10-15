"""Page allocation, the window slots of a sliding-window model, and the prefix caches on
CacheManager.

Every page returns and rows hold page-aligned slots; a window pool backs every new token
and keeps only a request's window live as it slides. The radix type commits a request's
prefix at its prefill and its finish, frees and re-points duplicates and evicts unlocked
leaves under pressure, and the prefill manager counts a prompt's pending tokens past its
prefix hit; with a GDN state pool the hybrid radix commit donates the track
snapshot, which stays at or below the prompt's reusable prefix end so a next turn resumes
there, a finish donates its live slot, and both currencies evict under pressure; with a
window pool the SWA radix commit keeps the windows a next prompt can resume from, and the
window slots evict on their own; naive with a GDN state pool holds one zeroed live slot per
running request and gives it back on finish and abort. Parts port FreeToken
tests/scheduler/test_hybrid_cache_manager.py, test_commit_repoints_page_table.py,
test_swa_commit_lock_and_floor.py and test_swa_pagesize.py, on tiny real pools.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, NamedTuple

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.attention.linear import LinearAttnBackend
from ftmetal.attention.metadata import AttnType, LinearAttnMetadata
from ftmetal.core import Batch, Req, SamplingParams
from ftmetal.kvcache.linear_state_pool import PADDING_SLOT, LinearStatePool
from ftmetal.kvcache.swa_pool import (
    SENTINEL_SLOT,
    SWA_RETAIN_GAP,
    SWAKVCache,
    swa_radix_floor_tokens,
)
from ftmetal.models.config import KVCacheGroupSpec
from ftmetal.scheduler.cache import CacheManager, MatchResult, resolve_cache_type
from ftmetal.scheduler.decode import DecodeManager
from ftmetal.scheduler.prefill import ChunkedReq, PendingReq, PrefillAdder, PrefillManager
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


def make_window_pool(window: int, num_swa_tokens: int, num_full_tokens: int) -> SWAKVCache:
    spec = KVCacheGroupSpec(
        name="swa",
        attn_type=AttnType.SWA,
        layer_ids=(0,),
        num_kv_heads=1,
        head_dim=8,
        sliding_window=window,
    )
    return SWAKVCache(spec, num_swa_tokens, num_full_tokens, mx.float16)


@pytest.mark.parametrize("page_size", [1, 4])
def test_window_slots_slide_with_the_request_and_return_on_finish(page_size: int) -> None:
    window, num_pages = 8, 32
    pool = make_window_pool(window, 48, (num_pages + 1) * page_size)
    page_table = np.zeros((1, 64), dtype=np.int32)
    manager = CacheManager(num_pages, page_size, page_table, "naive", swa_pool=pool)
    req = make_req(0, 20, 12)
    manager.allocate_paged([req])
    assert pool.swa_available_size == 47 - -(-20 // page_size) * page_size
    for step in range(12):
        req.complete_one()
        # The step that wrote the keys has drained.
        manager.free_swa_out_of_window(req)
        mapped = pool.translate(page_table[0, : req.cached_len])
        # Every key a later query sees keeps its window slot; nothing far behind does.
        assert SENTINEL_SLOT not in mapped[req.cached_len - window :]
        assert np.count_nonzero(mapped) <= window + 2 * page_size
        if step < 11:
            manager.allocate_paged([req])
    with manager.lazy_free_region():
        manager.cache_req(req, finished=True)
    manager.check_integrity()


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


def radix_manager(num_pages: int = 16, page_size: int = 4) -> tuple[CacheManager, np.ndarray]:
    page_table = np.zeros((4, 64), dtype=np.int32)
    return CacheManager(num_pages, page_size, page_table, "radix"), page_table


def test_cache_type_resolves_radix_to_the_hybrid_radix_on_a_gdn_model() -> None:
    resolve = partial(resolve_cache_type, has_sliding_window=False)
    assert resolve("radix", has_linear_attention=False) == "radix"
    assert resolve("naive", has_linear_attention=False) == "naive"
    assert resolve("radix", has_linear_attention=True) == "hybrid_radix"
    assert resolve("naive", has_linear_attention=True) == "naive"


def test_cache_type_resolves_radix_to_the_swa_radix_on_a_sliding_window_model() -> None:
    resolve = partial(resolve_cache_type, has_linear_attention=False, has_sliding_window=True)
    assert resolve("radix") == "swa_radix"
    assert resolve("naive") == "naive"


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


@pytest.mark.parametrize("cache_type", ["radix", "hybrid_radix"])
def test_unfinished_commit_repoints_the_row_off_the_freed_pages(cache_type: str) -> None:
    """Two requests prefilled the same prompt; the later commit dedups, frees its own pages
    and its row must then name the tree's pages, which the backends read from now on. With a
    GDN state pool the duplicate's frozen slot stays with the request; nothing leaks."""
    if cache_type == "radix":
        (manager, page_table), pool = radix_manager(), None
    else:
        manager, pool, page_table = hybrid_manager()
    prompt = list(range(1, 10))
    track_seqlen = None if pool is None else 8
    reqs = [prefilled(manager, pool, i, prompt, track_seqlen) for i in (0, 1)]
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


def test_pending_tokens_count_past_the_cached_prefix() -> None:
    page_table = np.zeros((4, 512), dtype=np.int32)
    manager = CacheManager(128, 4, page_table, "radix")
    prompt = list(range(1, 60))
    seed = prefilled(manager, None, 0, prompt[:41], track_seqlen=None)
    with manager.lazy_free_region():
        manager.cache_req(seed, finished=True)

    table = TableManager(max_running_reqs=4, page_table=page_table)
    prefill = PrefillManager(manager, table, DecodeManager(4))
    params = SamplingParams(max_tokens=1)
    prefill.add_one_req(1, np.array(prompt, dtype=np.int32), params)
    # The cached 40 tokens are not pending.
    assert prefill.has_pending_tokens(19) and not prefill.has_pending_tokens(20)
    prefill.add_one_req(2, np.arange(500, 530, dtype=np.int32), params)
    assert prefill.has_pending_tokens(49) and not prefill.has_pending_tokens(50)
    batch = prefill.schedule_next_batch(10)
    assert batch is not None and [req.extend_len for req in batch.reqs] == [10]
    assert prefill.has_pending_tokens(39) and not prefill.has_pending_tokens(40)


WINDOW = 8
# A token no prompt holds: what the driver samples.
SAMPLED = 999


def swa_manager(
    page_size: int,
    num_pages: int,
    num_swa_tokens: int,
    cache_type: Literal["naive", "swa_radix"],
    max_running_req: int = 2,
) -> tuple[CacheManager, TableManager]:
    pool = make_window_pool(WINDOW, num_swa_tokens, (num_pages + 1) * page_size)
    page_table = np.zeros((max_running_req + 1, 256), dtype=np.int32)
    manager = CacheManager(num_pages, page_size, page_table, cache_type, swa_pool=pool)
    return manager, TableManager(max_running_reqs=max_running_req, page_table=page_table)


def check_window_live(manager: CacheManager, steps: list[tuple[Req, int, int]]) -> None:
    """Every window slot a step reads, [cached_len - window + 1, device_len) at its launch,
    is live."""
    pool = manager.swa_pool
    assert pool is not None
    for req, start, stop in steps:
        window = pool.translate(manager.page_table[req.table_idx, start:stop])
        assert SENTINEL_SLOT not in window, f"{req} reads a freed window slot in [{start}, {stop})"


def check_live_slots(pool: LinearStatePool, batch: Batch) -> None:
    """Every request of a step runs on a live slot of its own, off the free list; a new one
    starts from a zero state."""
    slots = [req.linear_slot_idx for req in batch.reqs]
    assert None not in slots and PADDING_SLOT not in slots, f"{batch.reqs} hold no live slot"
    assert len(set(slots)) == len(slots), f"{batch.reqs} share a live slot"
    assert set(slots).isdisjoint(pool.free_slots), f"{batch.reqs} run on a freed slot"
    for req in batch.reqs:
        if req.cached_len == 0:
            assert req.linear_slot_idx is not None
            state = pool.recurrent_states[0][req.linear_slot_idx]
            assert not mx.any(state).item(), f"{req} starts from a stale state"


class Served(NamedTuple):
    # The prefix each admission found cached.
    cached_lens: list[int]
    # Each prompt's prefill chunk lengths.
    chunk_lens: list[list[int]]
    # The most GDN slots the requests held at once.
    peak_live_slots: int
    # Hybrid radix: the (slot, token count) of each prompt's tracked snapshots, chunk by chunk.
    tracks: list[list[tuple[int, int]]]
    # Hybrid radix: the tree snapshot each admission restores from.
    restore_srcs: list[int | None]


def serve(
    manager: CacheManager,
    table: TableManager,
    prompts: list[list[int]],
    output_len: int,
    max_extend_tokens: int,
    abort_at: int | None,
    arrive_at: list[int] | None = None,
    reuse_lens: list[int | None] | None = None,
) -> Served:
    """Prompts served to completion in the scheduler's order: step N + 1 is allocated before
    step N drains, and a drain slides the window slots of a sliding-window model, commits a
    prompt's last chunk and frees a finished request. With a GDN state pool every step checks
    its live slots and then dirties every slot, as a forward advances the states; under the
    hybrid radix cache the GDN metadata places each prefill chunk's track snapshot. Prompt i
    arrives before iteration arrive_at[i] (all before the first by default) with reuse_lens[i];
    the first prompt is aborted before iteration abort_at."""
    pool = manager.swa_pool
    gdn_pool = manager.linear_state_pool
    if gdn_pool is not None and manager.is_hybrid:
        linear_backend = LinearAttnBackend(gdn_pool, manager.page_size)
    else:
        linear_backend = None
    peak_live_slots = 0
    decode = DecodeManager(manager.page_size)
    prefill = PrefillManager(manager, table, decode)
    params = SamplingParams(max_tokens=output_len, ignore_eos=True)
    arrivals = [1] * len(prompts) if arrive_at is None else arrive_at
    prompt_reuse_lens = [None] * len(prompts) if reuse_lens is None else reuse_lens
    cached_lens: dict[int, int] = {}
    restore_srcs: dict[int, int | None] = {}
    chunk_lens: list[list[int]] = [[] for _ in prompts]
    tracks: list[list[tuple[int, int]]] = [[] for _ in prompts]
    finished: set[Req] = set()
    in_flight: Batch | None = None
    steps: list[tuple[Req, int, int]] = []
    iterations = 0
    last_arrival = max(arrivals)
    while iterations < last_arrival or prefill.runnable or decode.runnable or in_flight is not None:
        iterations += 1
        assert iterations < 1000, "the requests stopped making progress"
        for uid, prompt in enumerate(prompts):
            if arrivals[uid] == iterations:
                prompt_ids = np.array(prompt, dtype=np.int32)
                prefill.add_one_req(uid, prompt_ids, params, prompt_reuse_lens[uid])
        aborted = prefill.abort_req(0) or decode.abort_req(0) if iterations == abort_at else None
        if aborted is not None:
            aborted.aborted = True
            if in_flight is None or aborted not in in_flight.reqs:
                manager.cache_req(aborted, finished=True)
                table.free(aborted.table_idx)
        batch = prefill.schedule_next_batch(max_extend_tokens) or decode.schedule_next_batch()
        steps = []
        if batch is not None:
            manager.allocate_paged(batch.reqs)
            if pool is not None:
                steps = [
                    (req, max(0, req.cached_len - pool.sliding_window + 1), req.device_len)
                    for req in batch.reqs
                ]
                check_window_live(manager, steps)
            if gdn_pool is not None:
                check_live_slots(gdn_pool, batch)
                held_slots = gdn_pool.num_slots - 1 - gdn_pool.num_free_slots
                peak_live_slots = max(peak_live_slots, held_slots)
                gdn_pool.recurrent_states[0] = gdn_pool.recurrent_states[0] + 1
            if linear_backend is not None:
                linear_backend.prepare_metadata(batch)
                metadata = batch.attn_metadata[AttnType.LINEAR]
                assert isinstance(metadata, LinearAttnMetadata)
                for req, slot, seqlen in zip(
                    batch.reqs, metadata.track_slots_host, metadata.track_seqlens_host
                ):
                    if slot >= 0:
                        tracks[req.uid].append((slot, seqlen))
            for req in batch.reqs:
                cached_lens.setdefault(req.uid, req.cached_len)
                restore_srcs.setdefault(req.uid, req.mamba_restore_src)
                if batch.is_prefill:
                    chunk_lens[req.uid].append(req.extend_len)
                req.complete_one()
            decode.filter_reqs(batch.reqs)
        if in_flight is not None:
            new_finished: set[Req] = set()
            with manager.lazy_free_region():
                for req in in_flight.reqs:
                    if req.aborted:
                        manager.cache_req(req, finished=True)
                        table.free(req.table_idx)
                        continue
                    if req in finished:
                        continue
                    if pool is not None and (in_flight.is_decode or isinstance(req, ChunkedReq)):
                        manager.free_swa_out_of_window(req)
                    if isinstance(req, ChunkedReq):
                        continue
                    req.append_host(SAMPLED)
                    if req.generated_len >= req.output_len:
                        decode.remove_req(req)
                        manager.cache_req(req, finished=True)
                        table.free(req.table_idx)
                        new_finished.add(req)
                    elif in_flight.is_prefill:
                        manager.cache_req(req, finished=False)
            finished = new_finished
            if pool is not None:
                # The drain freed nothing the step in flight reads.
                check_window_live(manager, [step for step in steps if step[0] not in finished])
        in_flight = batch
    manager.check_integrity()
    return Served(
        [cached_lens[uid] for uid in range(len(prompts))],
        chunk_lens,
        peak_live_slots,
        tracks,
        [restore_srcs[uid] for uid in range(len(prompts))],
    )


def run(first: int, length: int) -> list[int]:
    return list(range(first, first + length))


class SWACase(NamedTuple):
    label: str
    page_size: int
    num_swa_tokens: int
    max_extend_tokens: int
    output_len: int
    # Prompt rounds: a round's prompts are served together, to completion, before the next.
    rounds: list[list[list[int]]]
    # The prefix each admission found cached, round by round.
    cached_lens: list[list[int]]
    cache_type: Literal["naive", "swa_radix"] = "swa_radix"
    num_pages: int = 128
    # The driver iteration the first round's first prompt is aborted at.
    abort_at: int | None = None


PROMPT = run(1, 40)
SWA_CASES = [
    SWACase(
        "a chunked prompt and its decode run on a pool smaller than their tokens",
        page_size=1,
        num_swa_tokens=48,
        max_extend_tokens=12,
        output_len=20,
        rounds=[[PROMPT]],
        cached_lens=[[0]],
    ),
    SWACase(
        "the same prompt again resumes from the tree",
        page_size=1,
        num_swa_tokens=64,
        max_extend_tokens=12,
        output_len=6,
        rounds=[[PROMPT], [PROMPT]],
        cached_lens=[[0], [39]],
    ),
    SWACase(
        "a finish keeps only the prompt's last window and gap live: a head before them misses",
        page_size=1,
        num_swa_tokens=64,
        max_extend_tokens=64,
        output_len=4,
        rounds=[[run(1, 40)], [run(1, 15) + run(100, 10)], [run(1, 30) + run(200, 10)]],
        cached_lens=[[0], [0], [30]],
    ),
    SWACase(
        "a head inside the slid-out part of a chunked prompt is not resumable",
        page_size=1,
        num_swa_tokens=64,
        max_extend_tokens=12,
        output_len=4,
        rounds=[[PROMPT], [run(1, 15) + run(100, 10)]],
        cached_lens=[[0], [0]],
    ),
    SWACase(
        "the next turn resumes past the decoded tokens",
        page_size=4,
        num_swa_tokens=96,
        max_extend_tokens=12,
        output_len=12,
        rounds=[[PROMPT], [PROMPT + [SAMPLED] * 12 + run(200, 9)]],
        cached_lens=[[0], [48]],
    ),
    SWACase(
        "a second prompt waits for window slots the first one's chunk reserved",
        page_size=4,
        num_swa_tokens=40,
        max_extend_tokens=64,
        output_len=8,
        rounds=[[run(1, 30), run(300, 30)]],
        cached_lens=[[0, 0]],
    ),
    SWACase(
        "two requests decode together on a pool at the floor",
        page_size=4,
        num_swa_tokens=swa_radix_floor_tokens(WINDOW, 4, 2, 16) + 1,
        max_extend_tokens=16,
        output_len=40,
        rounds=[[run(1, 60), run(500, 50)]],
        cached_lens=[[0, 0]],
    ),
    SWACase(
        "two long chunked prompts fit a pool at the floor",
        page_size=1,
        num_swa_tokens=swa_radix_floor_tokens(WINDOW, 1, 2, 64) + 1,
        max_extend_tokens=64,
        output_len=40,
        rounds=[[run(1, 120), run(500, 110)]],
        cached_lens=[[0, 0]],
        num_pages=512,
    ),
    SWACase(
        "a dry window pool takes the oldest windows of the tree",
        page_size=1,
        num_swa_tokens=40,
        max_extend_tokens=16,
        output_len=4,
        rounds=[[run(1, 20)], [run(100, 20)], [run(200, 20)], [run(1, 20), run(200, 20)]],
        cached_lens=[[0], [0], [0], [0, 19]],
    ),
    SWACase(
        "page pressure evicts the tree's KV with its window slots",
        page_size=4,
        num_swa_tokens=96,
        max_extend_tokens=32,
        output_len=4,
        rounds=[[run(1, 40)], [run(100, 40)], [run(1, 40)]],
        cached_lens=[[0], [0], [0]],
        num_pages=16,
    ),
    SWACase(
        "a prompt recomputing a tombstone a running request locks revives it for its window",
        page_size=1,
        num_swa_tokens=200,
        max_extend_tokens=12,
        output_len=30,
        rounds=[[PROMPT], [PROMPT, PROMPT[:30] + run(500, 3)]],
        cached_lens=[[0], [39, 0]],
        num_pages=256,
    ),
    SWACase(
        "an aborted chunk in flight commits only what its window kept",
        page_size=1,
        num_swa_tokens=64,
        max_extend_tokens=12,
        output_len=4,
        rounds=[[PROMPT], [PROMPT]],
        cached_lens=[[0], [24]],
        abort_at=3,
    ),
    SWACase(
        "naive frees every window slot it took",
        page_size=4,
        num_swa_tokens=96,
        max_extend_tokens=12,
        output_len=12,
        rounds=[[PROMPT], [PROMPT]],
        cached_lens=[[0], [0]],
        cache_type="naive",
    ),
]


def test_swa_radix_serves_and_resumes_prompts_and_leaks_nothing() -> None:
    for case in SWA_CASES:
        manager, table = swa_manager(
            case.page_size, case.num_pages, case.num_swa_tokens, case.cache_type
        )
        cached_lens = [
            serve(
                manager,
                table,
                prompts,
                case.output_len,
                case.max_extend_tokens,
                case.abort_at if round_idx == 0 else None,
            ).cached_lens
            for round_idx, prompts in enumerate(case.rounds)
        ]
        assert cached_lens == case.cached_lens, case.label


def test_swa_radix_commit_keeps_only_the_last_window() -> None:
    """A prefill commit tombstones the prompt before its last window and the retain gap and
    gives their window slots back at once; it locks what is left."""
    manager, _ = swa_manager(1, 128, 64, "swa_radix")
    cache, pool = manager.swa_parts()
    req = prefilled(manager, None, 0, run(1, 40), track_seqlen=None)
    with manager.lazy_free_region():
        manager.cache_req(req, finished=False)
    kept = WINDOW + SWA_RETAIN_GAP
    assert (cache.swa_evictable, cache.swa_protected) == (0, kept)
    assert pool.swa_available_size == 63 - kept
    assert manager.match_req(np.array(run(1, 15) + run(100, 10))) == MatchResult(None, None)
    hit = manager.match_req(np.array(run(1, 30) + run(200, 10))).handle
    assert hit is not None and hit.cached_len == 30
    with manager.lazy_free_region():
        manager.cache_req(req, finished=True)
    manager.check_integrity()


def test_four_running_requests_and_a_full_chunk_fit_at_the_floor() -> None:
    """Three requests decode with their tails grown when a long fourth prompt arrives: on a
    pool at exactly the floor, every chunk but its last still runs at full size."""
    chunk = 16
    floor = swa_radix_floor_tokens(WINDOW, 1, 4, chunk)
    manager, table = swa_manager(1, 512, floor + 1, "swa_radix", max_running_req=4)
    prompts = [run(1, 40), run(100, 40), run(200, 40), run(300, 100)]
    served = serve(manager, table, prompts, 60, chunk, None, arrive_at=[1, 1, 1, 30])
    assert served.chunk_lens[3] == [chunk] * 6 + [4]


class NaiveGDNCase(NamedTuple):
    label: str
    max_running_req: int
    max_extend_tokens: int
    output_len: int
    prompts: list[list[int]]
    # Each prompt's prefill chunk lengths.
    chunk_lens: list[list[int]]
    # The most live slots held at once.
    peak_live_slots: int
    arrive_at: list[int] | None = None
    # The driver iteration the first prompt is aborted at.
    abort_at: int | None = None


NAIVE_GDN_CASES = [
    NaiveGDNCase(
        "a chunked prompt prefills in unaligned chunks on its live slot",
        max_running_req=1,
        max_extend_tokens=10,
        output_len=6,
        prompts=[run(1, 25)],
        chunk_lens=[[10, 10, 5]],
        peak_live_slots=1,
    ),
    NaiveGDNCase(
        "max_running_req requests run together on max_running_req + 1 slots",
        max_running_req=3,
        max_extend_tokens=64,
        output_len=8,
        prompts=[run(1, 20), run(100, 20), run(200, 20)],
        chunk_lens=[[20], [20], [20]],
        peak_live_slots=3,
    ),
    NaiveGDNCase(
        "the prompts beyond max_running_req run on the finished ones' slots, zeroed",
        max_running_req=2,
        max_extend_tokens=64,
        output_len=5,
        prompts=[run(1, 15), run(100, 15), run(200, 15), run(300, 15)],
        chunk_lens=[[15], [15], [15], [15]],
        peak_live_slots=2,
    ),
    NaiveGDNCase(
        "an aborted chunk in flight returns its live slot at its drain",
        max_running_req=1,
        max_extend_tokens=12,
        output_len=4,
        prompts=[run(1, 40), run(1, 40)],
        chunk_lens=[[12, 12], [12, 12, 12, 4]],
        peak_live_slots=1,
        abort_at=3,
    ),
    NaiveGDNCase(
        "a request aborted in decode with no step in flight returns its live slot at once",
        max_running_req=2,
        max_extend_tokens=12,
        output_len=20,
        prompts=[run(1, 10), run(100, 30)],
        chunk_lens=[[10], [12, 12, 6]],
        peak_live_slots=2,
        arrive_at=[1, 4],
        abort_at=5,
    ),
]


def test_naive_gdn_holds_one_live_slot_per_running_request() -> None:
    """Naive on a model with linear attention: admission takes a live slot, a finish or an
    abort gives it back, and nothing is reused, on a pool of max_running_req + 1 slots."""
    for case in NAIVE_GDN_CASES:
        page_table = np.zeros((case.max_running_req + 1, 256), dtype=np.int32)
        pool = make_pool(case.max_running_req + 1)
        manager = CacheManager(128, 4, page_table, "naive", linear_state_pool=pool)
        table = TableManager(max_running_reqs=case.max_running_req, page_table=page_table)
        served = serve(
            manager,
            table,
            case.prompts,
            case.output_len,
            case.max_extend_tokens,
            case.abort_at,
            case.arrive_at,
        )
        assert served.cached_lens == [0] * len(case.prompts), case.label
        assert served.chunk_lens == case.chunk_lens, case.label
        assert served.peak_live_slots == case.peak_live_slots, case.label


class TrackCase(NamedTuple):
    label: str
    max_extend_tokens: int
    output_len: int
    # The first turn is run(1, prompt_len); a next turn can share its first reuse_len tokens.
    prompt_len: int
    reuse_len: int | None
    # The next turn shares the first one's first shared_len tokens, then goes its own way.
    shared_len: int
    # The token counts the first turn's prefill chunks track their snapshots at.
    track_seqlens: list[int]
    # The prefix the next turn resumes from.
    cached_len: int


# Page size 4. A chat prompt ends with its generation suffix, five tokens here, of which the
# next turn's re-rendered conversation shares the first three, as Qwen3.6's does.
TRACK_CASES = [
    TrackCase(
        "the generation suffix holds the last stride boundary: the track stays below it",
        max_extend_tokens=512,
        output_len=4,
        prompt_len=257,
        reuse_len=252,
        shared_len=255,
        track_seqlens=[192],
        cached_len=192,
    ),
    TrackCase(
        "the reuse boundary sits in an earlier chunk: the chunks past it track nothing",
        max_extend_tokens=100,
        output_len=4,
        prompt_len=380,
        reuse_len=290,
        shared_len=293,
        track_seqlens=[64, 164, 264],
        cached_len=264,
    ),
    TrackCase(
        "no stride boundary below the reuse end in its chunk: the chunk before commits",
        max_extend_tokens=100,
        output_len=4,
        prompt_len=380,
        reuse_len=260,
        shared_len=263,
        track_seqlens=[64, 164],
        cached_len=164,
    ),
    TrackCase(
        "a prompt finishing at its prefill commits that track before its live state",
        max_extend_tokens=100,
        output_len=1,
        prompt_len=380,
        reuse_len=260,
        shared_len=263,
        track_seqlens=[64, 164],
        cached_len=164,
    ),
    TrackCase(
        "a reuse boundary on a stride boundary is tracked",
        max_extend_tokens=512,
        output_len=4,
        prompt_len=133,
        reuse_len=128,
        shared_len=131,
        track_seqlens=[128],
        cached_len=128,
    ),
    TrackCase(
        "a reusable prefix shorter than a stride tracks nothing",
        max_extend_tokens=512,
        output_len=4,
        prompt_len=70,
        reuse_len=60,
        shared_len=63,
        track_seqlens=[],
        cached_len=0,
    ),
    TrackCase(
        "a raw prompt tracks the deepest stride boundary strictly inside each chunk",
        max_extend_tokens=100,
        output_len=4,
        prompt_len=300,
        reuse_len=None,
        shared_len=300,
        track_seqlens=[64, 164, 264],
        cached_len=264,
    ),
]


def test_track_snapshot_commits_at_or_below_the_reusable_prefix_end() -> None:
    """Each prefill chunk tracks a snapshot at its deepest stride boundary at or below the
    prompt's reuse_len; the last one tracked commits, so the next turn, which re-renders the
    conversation and shares only its reusable prefix, resumes from that very slot."""
    for case in TRACK_CASES:
        pool = make_pool()
        page_table = np.zeros((3, 512), dtype=np.int32)
        manager = CacheManager(256, 4, page_table, "hybrid_radix", pool)
        table = TableManager(max_running_reqs=2, page_table=page_table)
        first = run(1, case.prompt_len)
        served = serve(
            manager,
            table,
            [first],
            case.output_len,
            case.max_extend_tokens,
            None,
            reuse_lens=[case.reuse_len],
        )
        assert [seqlen for _, seqlen in served.tracks[0]] == case.track_seqlens, case.label
        tracked_slots = {seqlen: slot for slot, seqlen in served.tracks[0]}
        next_turn = first[: case.shared_len] + run(1000, 20)
        resumed = serve(manager, table, [next_turn], case.output_len, case.max_extend_tokens, None)
        assert (resumed.cached_lens[0], resumed.restore_srcs[0]) == (
            case.cached_len,
            tracked_slots.get(case.cached_len),
        ), case.label
