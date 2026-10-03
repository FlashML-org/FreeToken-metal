"""SWARadixCache: the window-slot currency on the KV radix tree.

Port of FreeToken tests/kvcache/radix/test_swa_radix.py, without its reference model: each
case drives the cache through hand-picked inserts, matches, locks and evictions, and check()
holds it to its own counters and invariants, to the full-slot ledger (every slot handed to an
insert is in the tree or freed, once) and to the window ledger (exactly the live nodes' slots
keep a window slot). Scenarios are written in pages of PAGE tokens and a WINDOW of two pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from ftmetal.kvcache.radix_cache import RadixTreeNode
from ftmetal.kvcache.swa_radix_cache import SWARadixCache

PAGE = 4
WINDOW = 8


@dataclass
class Ledger:
    """The pools behind the cache: full slots handed out and freed, and the full slots whose
    window slot is live."""

    next_slot: int = 0
    freed: list[int] = field(default_factory=list)
    window_live: set[int] = field(default_factory=set)

    def take(self, count: int, live_from: int = 0) -> np.ndarray:
        """count new full slots; the request freed the window slots of the first live_from."""
        slots = np.arange(self.next_slot, self.next_slot + count, dtype=np.int32)
        self.next_slot += count
        self.window_live.update(slots[live_from:].tolist())
        return slots

    def free(self, full: np.ndarray, window: np.ndarray) -> None:
        self.freed.extend(full.tolist())
        self.window_live.difference_update(window.tolist())


def tokens(num_pages: int, first_page: int = 0, page_size: int = PAGE) -> np.ndarray:
    """Pages of distinct tokens: page i is page_size copies of i + 1."""
    return np.repeat(np.arange(first_page + 1, first_page + num_pages + 1), page_size).astype(
        np.int32
    )


def nodes(cache: SWARadixCache) -> list[RadixTreeNode]:
    out: list[RadixTreeNode] = []
    stack = list(cache.root.children.values())
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(node.children.values())
    return out


def tombstones(cache: SWARadixCache) -> list[int]:
    """The token lengths of the tombstoned nodes, sorted."""
    return sorted(node.length for node in nodes(cache) if node.swa_tombstone)


def node_at(cache: SWARadixCache, ids: np.ndarray, end: int) -> RadixTreeNode:
    """The node whose key ends at token end of ids."""
    node, prefix_len = cache.root, 0
    while prefix_len < end:
        node = node.children[tuple(ids[prefix_len : prefix_len + cache.page_size].tolist())]
        prefix_len += node.length
    assert prefix_len == end, f"no node boundary at {end}"
    return node


def check(cache: SWARadixCache, ledger: Ledger) -> None:
    cache.check_integrity()
    tree = [slot for node in nodes(cache) for slot in node.value.tolist()]
    assert len(set(ledger.freed)) == len(ledger.freed), "a full slot was freed twice"
    assert sorted(tree + ledger.freed) == list(range(ledger.next_slot)), "full slots leaked"
    live = {slot for node in nodes(cache) if not node.swa_tombstone for slot in node.value}
    assert live == ledger.window_live, "the live nodes' slots are not the live window slots"


def insert(
    cache: SWARadixCache,
    ledger: Ledger,
    ids: np.ndarray,
    swa_evicted_seqlen: int = 0,
    reused: np.ndarray | None = None,
) -> np.ndarray:
    """A request's commit: the reused prefix's slots are the tree's, the rest are new and live
    from swa_evicted_seqlen on. Returns the slots the request supplied."""
    own_from = 0 if reused is None else len(reused)
    own = ledger.take(len(ids) - own_from, max(0, swa_evicted_seqlen - own_from))
    slots = own if reused is None else np.concatenate([reused, own])
    _, freed = cache.insert(ids, slots, swa_evicted_seqlen, update_kv_after_len=own_from)
    ledger.free(freed, freed)
    check(cache, ledger)
    return slots


def chain(cache: SWARadixCache, ledger: Ledger, num_pages: int) -> np.ndarray:
    """One live single-page node per page, each commit reusing the one before."""
    ids = tokens(num_pages)
    for end in range(PAGE, len(ids) + 1, PAGE):
        match = cache.match_prefix(ids[:end])
        insert(cache, ledger, ids[:end], reused=match.kv_indices)
    return ids


class MatchCase(NamedTuple):
    label: str
    page_size: int
    window: int
    # One commit of num_tokens whose first swa_evicted_seqlen tokens slid out of its window.
    num_tokens: int
    swa_evicted_seqlen: int
    query_len: int
    cached_len: int


MATCH_CASES = [
    MatchCase("live path shorter than the window", 4, 8, 4, 0, 4, 4),
    MatchCase("live path, query past the tree", 4, 8, 16, 0, 24, 16),
    MatchCase("live run after the tombstone covers the window", 4, 8, 16, 8, 16, 16),
    MatchCase("live run after the tombstone is short", 4, 8, 16, 8, 12, 0),
    MatchCase("query ends in the tombstone", 4, 8, 16, 8, 8, 0),
    MatchCase("token pages, run of exactly the window", 1, 4, 10, 4, 8, 8),
    MatchCase("token pages, run one short of the window", 1, 4, 10, 4, 7, 0),
    MatchCase("unaligned query is matched by whole pages", 4, 8, 16, 0, 15, 12),
]


def test_match_truncates_to_a_windowed_safe_boundary() -> None:
    for case in MATCH_CASES:
        cache, ledger = SWARadixCache(case.page_size, case.window), Ledger()
        ids = tokens(case.num_tokens // case.page_size, page_size=case.page_size)
        slots = insert(cache, ledger, ids, case.swa_evicted_seqlen)
        query = np.concatenate([ids, tokens(4, 50, case.page_size)])[: case.query_len]
        match = cache.match_prefix(query)
        assert match.cached_len == case.cached_len, case.label
        assert match.kv_indices.tolist() == slots[: case.cached_len].tolist(), case.label
        check(cache, ledger)


def test_the_run_before_a_tombstone_commits_at_its_parent() -> None:
    """A: 3 live pages, B: 1 tombstoned page, C: 2 live pages. A covers the window on its own,
    so a match past B keeps A even when C's run is short."""
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = tokens(6)
    insert(cache, ledger, ids)
    cache.match_prefix(ids[: 4 * PAGE])  # node boundaries at 4 pages
    head = cache.match_prefix(ids[: 3 * PAGE])  # and at 3
    swa_uuid = cache.inc_lock(head.node)  # spares A from the trim
    ledger.free(np.array([], dtype=np.int32), cache.trim_head_swa(ids, 4 * PAGE))
    cache.dec_lock(head.node, swa_uuid)
    assert tombstones(cache) == [PAGE]
    check(cache, ledger)

    assert cache.match_prefix(ids).cached_len == 6 * PAGE
    assert cache.match_prefix(ids[: 5 * PAGE]).cached_len == 3 * PAGE
    assert cache.match_prefix(ids[: 4 * PAGE]).cached_len == 3 * PAGE
    check(cache, ledger)


def test_insert_tombstones_the_out_of_window_head() -> None:
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    insert(cache, ledger, tokens(3), swa_evicted_seqlen=PAGE)
    assert tombstones(cache) == [PAGE]
    assert (cache.full_evictable, cache.swa_evictable) == (3 * PAGE, 2 * PAGE)


def test_insert_revives_a_tombstone_whole_or_from_the_frontier() -> None:
    # Whole: the request's window slots are live over the tombstone.
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = chain(cache, ledger, 3)
    cache.match_prefix(ids)  # stamps the root-side page oldest
    ledger.free(*cache.evict_swa(1))
    head = node_at(cache, ids, PAGE)
    stale = head.value.tolist()
    assert head.swa_tombstone
    fresh = insert(cache, ledger, ids, swa_evicted_seqlen=0)
    assert not head.swa_tombstone and head.value.tolist() == fresh[:PAGE].tolist()
    assert set(stale) <= set(ledger.freed)
    assert set(fresh[PAGE:].tolist()) <= set(ledger.freed)

    # From the frontier: the head stays a tombstone, the tail past the frontier is revived.
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = tokens(4)
    first = insert(cache, ledger, ids, swa_evicted_seqlen=3 * PAGE)
    second = insert(cache, ledger, ids, swa_evicted_seqlen=2 * PAGE)
    assert tombstones(cache) == [2 * PAGE]
    tail = node_at(cache, ids, 3 * PAGE)
    assert not tail.swa_tombstone and tail.value.tolist() == second[2 * PAGE : 3 * PAGE].tolist()
    assert set(first[2 * PAGE : 3 * PAGE].tolist()) <= set(ledger.freed)


def test_a_tombstone_below_the_frontier_or_full_locked_is_not_revived() -> None:
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = chain(cache, ledger, 4)
    held = cache.match_prefix(ids)
    swa_uuid = cache.inc_lock(held.node)  # swa-locks the last two pages, full-locks all four
    ledger.free(*cache.evict_swa(10 * PAGE))
    assert tombstones(cache) == [PAGE, PAGE]
    locked = node_at(cache, ids, PAGE).value.tolist()
    fresh = insert(cache, ledger, ids, swa_evicted_seqlen=0)
    assert tombstones(cache) == [PAGE, PAGE]
    assert node_at(cache, ids, PAGE).value.tolist() == locked
    assert set(fresh.tolist()) <= set(ledger.freed)

    cache.dec_lock(held.node, swa_uuid)
    insert(cache, ledger, ids, swa_evicted_seqlen=PAGE)  # page 0 is below this frontier
    assert tombstones(cache) == [PAGE]
    insert(cache, ledger, ids, swa_evicted_seqlen=0)
    assert tombstones(cache) == []


def test_insert_within_the_reused_prefix_frees_nothing() -> None:
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = chain(cache, ledger, 3)
    match = cache.match_prefix(ids)
    swa_uuid = cache.inc_lock(match.node)
    prefix_len, freed = cache.insert(ids, match.kv_indices, 0, update_kv_after_len=len(ids))
    assert (prefix_len, freed.tolist()) == (len(ids), [])
    cache.dec_lock(match.node, swa_uuid)
    check(cache, ledger)


def test_a_split_keeps_the_tombstone_and_moves_the_lock_boundary() -> None:
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = tokens(3)
    insert(cache, ledger, ids, swa_evicted_seqlen=2 * PAGE)
    other = np.concatenate([ids[:PAGE], tokens(1, 9)])
    assert cache.match_prefix(other).cached_len == 0  # splits the tombstone
    assert tombstones(cache) == [PAGE, PAGE]
    check(cache, ledger)

    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = tokens(3)
    insert(cache, ledger, ids)
    held = cache.match_prefix(ids)
    swa_uuid = cache.inc_lock(held.node)
    assert swa_uuid is not None and cache.swa_protected == 3 * PAGE
    cache.match_prefix(other)  # splits the locked node at page 1
    assert len(nodes(cache)) == 2
    cache.dec_lock(held.node, swa_uuid)
    assert (cache.full_protected, cache.swa_protected) == (0, 0)
    check(cache, ledger)


def test_a_window_lock_pins_one_window_and_releases_only_its_own() -> None:
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = chain(cache, ledger, 4)
    short = cache.match_prefix(ids[:PAGE])
    assert cache.inc_lock(short.node) is None  # the live path is shorter than the window
    assert cache.swa_protected == PAGE
    cache.dec_lock(short.node, None)

    deep = cache.match_prefix(ids)
    shallow = cache.match_prefix(ids[: 2 * PAGE])
    deep_uuid, shallow_uuid = cache.inc_lock(deep.node), cache.inc_lock(shallow.node)
    assert None not in (deep_uuid, shallow_uuid) and deep_uuid != shallow_uuid
    assert (cache.full_protected, cache.swa_protected) == (4 * PAGE, 4 * PAGE)
    cache.dec_lock(deep.node, deep_uuid)
    assert (cache.full_protected, cache.swa_protected) == (2 * PAGE, 2 * PAGE)
    cache.dec_lock(shallow.node, shallow_uuid)
    assert (cache.full_protected, cache.swa_protected) == (0, 0)
    check(cache, ledger)


def test_evict_full_takes_unlocked_leaves_and_cascades_through_tombstones() -> None:
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = chain(cache, ledger, 3)
    held = cache.match_prefix(ids)
    swa_uuid = cache.inc_lock(held.node)
    ledger.free(*cache.evict_full(100))
    assert ledger.freed == []  # the only leaf is locked
    cache.dec_lock(held.node, swa_uuid)

    cache.match_prefix(ids[:PAGE])  # the root-side node is the newest, still the leaf goes
    ledger.free(*cache.evict_full(PAGE))
    assert cache.full_evictable == 2 * PAGE
    check(cache, ledger)

    ids = chain(cache, ledger, 3)
    cache.match_prefix(ids)
    ledger.free(*cache.evict_swa(2 * PAGE))  # tombstones the two internal nodes
    assert tombstones(cache) == [PAGE, PAGE]
    ledger.free(*cache.evict_full(PAGE))  # the leaf, then the tombstones it exposes
    assert nodes(cache) == []
    check(cache, ledger)


def test_evict_swa_tombstones_internal_nodes_in_place_and_frees_leaves() -> None:
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = chain(cache, ledger, 3)
    cache.match_prefix(ids)
    ledger.free(*cache.evict_swa(1))
    assert tombstones(cache) == [PAGE] and ledger.freed == []
    match = cache.match_prefix(ids)  # the live tail covers the window: the KV is still served
    assert match.cached_len == 3 * PAGE
    check(cache, ledger)

    # Match recency, not insert order: the stale node goes, not the freshly matched head.
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = chain(cache, ledger, 4)
    cache.match_prefix(ids[: 2 * PAGE])
    ledger.free(*cache.evict_swa(1))
    assert node_at(cache, ids, 3 * PAGE).swa_tombstone
    check(cache, ledger)

    # A free leaf goes with its full KV, and the tombstone it exposes after it.
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = chain(cache, ledger, 2)
    cache.match_prefix(ids)
    ledger.free(*cache.evict_swa(2 * PAGE))
    assert nodes(cache) == []
    check(cache, ledger)


def test_trim_head_swa_keeps_the_window_and_skips_what_it_must() -> None:
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = chain(cache, ledger, 4)
    held = cache.match_prefix(ids)
    swa_uuid = cache.inc_lock(held.node)
    assert cache.trim_head_swa(ids, 0).tolist() == []
    # Asked for the whole path: the locked window and the leaf stay live.
    ledger.free(np.array([], dtype=np.int32), cache.trim_head_swa(ids, 4 * PAGE))
    assert tombstones(cache) == [PAGE, PAGE]
    assert cache.trim_head_swa(ids, 4 * PAGE).tolist() == []  # tombstones are skipped
    cache.dec_lock(held.node, swa_uuid)
    check(cache, ledger)
    match = cache.match_prefix(ids)
    assert match.cached_len == 4 * PAGE and cache.full_evictable == 4 * PAGE


def test_pressure_returns_every_slot_exactly_once() -> None:
    cache, ledger = SWARadixCache(PAGE, WINDOW), Ledger()
    ids = chain(cache, ledger, 6)
    insert(cache, ledger, np.concatenate([ids[: 2 * PAGE], tokens(3, 20)]), PAGE)
    cache.match_prefix(ids)
    for _ in range(40):
        if cache.swa_evictable == 0:
            break
        ledger.free(*cache.evict_swa(PAGE))
        check(cache, ledger)
    for _ in range(40):
        if cache.full_evictable == 0:
            break
        ledger.free(*cache.evict_full(PAGE))
        check(cache, ledger)
    assert nodes(cache) == [] and ledger.window_live == set()
