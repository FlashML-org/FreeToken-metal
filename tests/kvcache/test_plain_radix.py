"""RadixPrefixCache: the single-currency KV prefix cache.

Port of FreeToken tests/kvcache/radix/test_plain_radix.py. Expectations come from the
page-keyed model in radix_harness; every Session op compares against it and Session.check()
runs the invariant battery. The assertions here add the class's own contracts: which slots
come back, the running sizes, what split_at does to node identity and how the class refuses
an illegal request. Not ported: the cold-handle get_matched_indices wart (a match carries
its KV slots, empty on a miss) and reset (the class has none).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from ftmetal.kvcache.radix_cache import RadixPrefixCache
from radix_harness import Session, node_end_path, node_path_slots, page_ids

EMPTY = {"full_evictable": 0, "full_protected": 0}


def plain_session(page_size: int) -> Session:
    return Session(page_size, RadixPrefixCache(page_size))


@pytest.fixture(params=[1, 4], ids=["p1", "p4"])
def plain(request: pytest.FixtureRequest) -> Session:
    """A plain radix cache, its reference model and slot ledgers, at page size 1 and 4."""
    page_size: int = request.param
    return plain_session(page_size)


@pytest.fixture
def plain4() -> Session:
    """One session at page size 4, for behaviours that do not depend on the page size."""
    return plain_session(4)


def cache_of(session: Session) -> RadixPrefixCache:
    cache = session.adapter.cache
    assert isinstance(cache, RadixPrefixCache)
    return cache


def tree_slots(session: Session) -> list[int]:
    return [slot for _, record in session.adapter.records() for slot in record.slots]


def array(values: Sequence[int]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int32)


def test_cold_match_is_empty_and_a_commit_round_trips(plain: Session) -> None:
    page_size = plain.page_size
    ids = page_ids(page_size, 1, 2, 3)

    cold, expected = plain.do_match(ids)
    assert (cold.cached_len, cold.indices) == (0, [])
    assert expected.group is None
    assert cold.node.is_root()  # the lock target of a cold match is the root
    assert plain.adapter.counters() == EMPTY
    assert plain.adapter.nodes() == []

    slots = plain.kv.take(len(ids))
    got, inserted = plain.do_insert(ids, slots=slots)
    assert got.matched_len == 0
    assert inserted.adopted == slots

    warm, _ = plain.do_match(ids)
    assert (warm.cached_len, warm.indices) == (len(ids), slots)
    assert plain.adapter.counters() == {"full_evictable": len(ids), "full_protected": 0}
    assert len(plain.adapter.nodes()) == 1  # three pages, one radix node
    assert plain.model.events["node.add"] == 1 and plain.model.events["node.split"] == 0
    plain.check()


def test_insert_drops_a_trailing_partial_page(plain: Session) -> None:
    page_size = plain.page_size
    ids = page_ids(page_size, 1, 2) + (9,) * (page_size - 1)  # a ragged tail unless page 1
    kept = 2 * page_size
    slots = plain.kv.take(len(ids))

    _, inserted = plain.do_insert(ids, slots=slots)
    assert inserted.adopted == slots[:kept]

    match, _ = plain.do_match(ids)
    assert (match.cached_len, match.indices) == (kept, slots[:kept])
    # The partial page's slots stay the caller's: insert truncates before it stores.
    assert set(slots[kept:]) <= plain.kv.free
    assert not set(slots[kept:]) & set(tree_slots(plain))
    plain.check()


def test_mid_node_divergence_rounds_down_to_the_page_boundary(plain: Session) -> None:
    page_size = plain.page_size
    ids = page_ids(page_size, 1, 2)
    slots = plain.kv.take(len(ids))
    plain.do_insert(ids, slots=slots)

    divergent = ids[:-1] + (999,)  # equal for 2P - 1 tokens, differs in the last
    match, _ = plain.do_match(divergent)
    assert (match.cached_len, match.indices) == (page_size, slots[:page_size])
    if page_size > 1:  # 2P - 1 tokens compared equal, only P are reusable
        assert match.cached_len < len(ids) - 1
    assert plain.model.events["node.split"] == 1
    plain.check()


def test_pages_sharing_leading_tokens_do_not_share_a_tree_edge(plain4: Session) -> None:
    """The reuse unit is a whole page: two pages with a common token prefix get their own
    edges, or a match would hand back the other page's slots."""
    one, two = (1, 7, 7, 1), (1, 7, 7, 2)  # equal for 3 of 4 tokens, different pages
    slots_one, slots_two = plain4.kv.take(4), plain4.kv.take(4)
    plain4.do_insert(one, slots=slots_one)
    plain4.do_insert(two, slots=slots_two)

    assert len(cache_of(plain4).root.children) == 2
    assert plain4.model.events["node.split"] == 0
    match_one, _ = plain4.do_match(one)
    match_two, _ = plain4.do_match(two)
    assert (match_one.cached_len, match_one.indices) == (4, slots_one)
    assert (match_two.cached_len, match_two.indices) == (4, slots_two)
    plain4.check()


def test_reinserting_a_cached_prefix_adopts_nothing(plain4: Session) -> None:
    ids = page_ids(4, 1, 2)
    first = plain4.kv.take(len(ids))
    plain4.do_insert(ids, slots=first)

    second = plain4.kv.take(len(ids))
    got, inserted = plain4.do_insert(ids, slots=second)
    assert got.matched_len == len(ids)
    assert inserted.adopted == []
    # Nothing is stored for a cached prefix and nothing handed back: the duplicates stay the
    # caller's to free (CacheManager frees page_indices[old_cached_len:prefix_len]).
    assert inserted.dups == second
    assert set(second) <= plain4.kv.free

    match, _ = plain4.do_match(ids)
    assert match.indices == first  # the tree kept its own slots
    assert len(plain4.adapter.nodes()) == 1

    # The end node insert returns is the commit's lock target: it spans the whole inserted
    # prefix and its path names the tree's slots, never the supplied ones.
    fake = range(7_000_000, 7_000_000 + len(ids))  # never registered with the ledger
    prefix_len, node = cache_of(plain4).insert(array(ids), array(fake))
    assert prefix_len == len(ids)
    assert node_end_path(node) == ids
    assert node_path_slots(node) == first
    plain4.check()


def test_request_lifecycle_reuses_the_cached_prefix(plain: Session) -> None:
    page_size = plain.page_size
    full = page_ids(page_size, 1, 2, 3)
    head_slots = plain.kv.take(page_size)
    plain.do_insert(full[:page_size], slots=head_slots)

    plain.do_request(full, prompt_pages=1)  # match, lock, extend and commit, unlock
    assert plain.kv.free == set()  # a reused prefix duplicates nothing and leaks nothing

    match, _ = plain.do_match(full)
    assert match.cached_len == 3 * page_size
    assert match.indices[:page_size] == head_slots  # the reused page keeps its slots
    assert plain.adapter.counters() == {"full_evictable": 3 * page_size, "full_protected": 0}
    assert plain.model.events["node.add"] == 2 and plain.model.events["node.split"] == 0
    plain.check()


def test_split_turns_the_original_node_into_the_suffix(plain: Session) -> None:
    page_size = plain.page_size
    ids = page_ids(page_size, 1, 2)
    slots = plain.kv.take(len(ids))
    plain.do_insert(ids, slots=slots)
    whole, _ = plain.do_match(ids)
    node = whole.node
    assert node.length == 2 * page_size

    prefix, _ = plain.do_match(ids[:page_size])  # forces split_at(page_size)
    assert plain.model.events["node.split"] == 1
    assert len(plain.adapter.nodes()) == 2
    assert prefix.node is not node
    # A new prefix node, the original object kept as the suffix: a handle taken before the
    # split still names the same end boundary, which lock and unlock symmetry rides on.
    assert node.length == page_size
    assert node.parent is prefix.node
    assert node_end_path(prefix.node) == ids[:page_size]
    assert node_end_path(node) == ids
    assert prefix.node.value.tolist() == slots[:page_size]
    assert node.value.tolist() == slots[page_size:]
    assert node_path_slots(node) == slots

    again, _ = plain.do_match(ids)  # the split is transparent to a full match
    assert (again.cached_len, again.indices) == (2 * page_size, slots)
    assert again.node is node
    plain.check()


def test_lock_survives_a_split(plain: Session) -> None:
    page_size = plain.page_size
    ids = page_ids(page_size, 1, 2)
    plain.do_insert(ids)

    held = plain.do_lock(ids)
    assert plain.adapter.counters() == {"full_evictable": 0, "full_protected": 2 * page_size}

    plain.do_match(ids[:page_size])  # splits the locked node in two
    assert plain.model.events["node.split"] == 1
    # split_at copies ref_count into the new prefix half: the protected total is unchanged.
    assert plain.adapter.counters() == {"full_evictable": 0, "full_protected": 2 * page_size}
    plain.check()

    plain.do_unlock(held)  # the handle names the suffix; the release walks up through both
    assert plain.adapter.counters() == {"full_evictable": 2 * page_size, "full_protected": 0}
    plain.check()


def test_lock_accounting_walks_to_the_root(plain: Session) -> None:
    page_size = plain.page_size
    head, tail = page_ids(page_size, 1), page_ids(page_size, 1, 2)
    plain.do_insert(head)
    plain.do_insert(tail)
    assert len(plain.adapter.nodes()) == 2
    assert plain.adapter.counters() == {"full_evictable": 2 * page_size, "full_protected": 0}

    deep = plain.do_lock(tail)  # protects the leaf and everything up to the root
    assert plain.adapter.counters() == {"full_evictable": 0, "full_protected": 2 * page_size}
    again = plain.do_lock(tail)  # ref 2: still exactly one protected region
    assert plain.adapter.counters() == {"full_evictable": 0, "full_protected": 2 * page_size}
    plain.do_unlock(again)
    assert plain.adapter.counters() == {"full_evictable": 0, "full_protected": 2 * page_size}
    plain.do_unlock(deep)
    assert plain.adapter.counters() == {"full_evictable": 2 * page_size, "full_protected": 0}

    shallow = plain.do_lock(head)  # a lock protects the root path, not the descendants
    assert plain.adapter.counters() == {"full_evictable": page_size, "full_protected": page_size}
    plain.do_unlock(shallow)
    assert plain.adapter.counters() == {"full_evictable": 2 * page_size, "full_protected": 0}
    plain.check()


def test_eviction_is_lru_ordered_over_leaves(plain: Session) -> None:
    page_size = plain.page_size
    a, b, c = page_ids(page_size, 1), page_ids(page_size, 2), page_ids(page_size, 3)
    slots_a, slots_b, slots_c = (plain.kv.take(page_size) for _ in range(3))
    for ids, slots in ((a, slots_a), (b, slots_b), (c, slots_c)):
        plain.do_insert(ids, slots=slots)
    plain.do_match(a)  # a becomes the most recently used

    plain.do_evict_full(page_size)
    assert set(slots_b) <= plain.kv.free  # b was the oldest
    assert not (set(slots_a) | set(slots_c)) & plain.kv.free

    plain.do_evict_full(page_size)
    assert set(slots_c) <= plain.kv.free
    still, _ = plain.do_match(a)
    assert (still.cached_len, still.indices) == (page_size, slots_a)
    gone, _ = plain.do_match(b)
    assert gone.cached_len == 0

    plain.do_evict_full(page_size)
    assert set(slots_a) <= plain.kv.free
    assert plain.adapter.nodes() == [] and plain.adapter.counters() == EMPTY
    plain.check()


def two_level_tree(session: Session) -> tuple[tuple[int, ...], list[int], list[int]]:
    """A one-page head node and its one-page leaf; returns the head ids, the head's slots and
    the leaf's (the tail insert's slots for the head page were duplicates)."""
    page_size = session.page_size
    head, tail = page_ids(page_size, 1), page_ids(page_size, 1, 2)
    head_slots = session.kv.take(page_size)
    session.do_insert(head, slots=head_slots)
    tail_slots = session.kv.take(2 * page_size)
    session.do_insert(tail, slots=tail_slots)
    return head, head_slots, tail_slots[page_size:]


def test_eviction_takes_the_leaf_before_the_parent(plain: Session) -> None:
    page_size = plain.page_size
    head, head_slots, leaf_slots = two_level_tree(plain)

    plain.do_evict_full(page_size)  # only leaves are collectible
    assert set(leaf_slots) <= plain.kv.free
    assert not set(head_slots) & plain.kv.free
    assert len(plain.adapter.nodes()) == 1
    assert plain.adapter.counters() == {"full_evictable": page_size, "full_protected": 0}
    match, _ = plain.do_match(head)
    assert (match.cached_len, match.indices) == (page_size, head_slots)

    plain.do_evict_full(page_size)  # now the parent is a leaf itself
    assert set(head_slots) <= plain.kv.free
    assert plain.adapter.nodes() == []
    plain.check()


def test_eviction_cascades_to_a_newly_childless_parent(plain: Session) -> None:
    page_size = plain.page_size
    _, head_slots, leaf_slots = two_level_tree(plain)

    # One call: the parent is no candidate when the walk starts; it is pushed onto the heap
    # only once its last child is unlinked.
    plain.do_evict_full(2 * page_size)
    assert plain.model.events["evict.parent_exposed"] == 1
    assert set(leaf_slots) <= plain.kv.free and set(head_slots) <= plain.kv.free
    assert plain.adapter.nodes() == [] and plain.adapter.counters() == EMPTY
    plain.check()


def test_locked_nodes_are_never_evicted(plain: Session) -> None:
    page_size = plain.page_size
    a, b = page_ids(page_size, 1), page_ids(page_size, 2)
    slots_a, slots_b = plain.kv.take(page_size), plain.kv.take(page_size)
    plain.do_insert(a, slots=slots_a)
    plain.do_insert(b, slots=slots_b)

    held = plain.do_lock(a)
    assert plain.adapter.counters() == {"full_evictable": page_size, "full_protected": page_size}

    plain.do_evict_full(10**6)  # clamped to the evictable size
    assert set(slots_b) <= plain.kv.free
    assert not set(slots_a) & plain.kv.free
    match, _ = plain.do_match(a)
    assert (match.cached_len, match.indices) == (page_size, slots_a)
    with pytest.raises(AssertionError):  # a locked node is not evictable capacity
        cache_of(plain).evict(1)

    plain.do_unlock(held)
    assert plain.adapter.counters() == {"full_evictable": page_size, "full_protected": 0}
    plain.do_evict_full(page_size)
    assert set(slots_a) <= plain.kv.free
    plain.check()


def test_evicting_more_than_evictable_raises_assertion_error(plain4: Session) -> None:
    ids = page_ids(4, 1)
    slots = plain4.kv.take(4)
    plain4.do_insert(ids, slots=slots)

    with pytest.raises(AssertionError, match="Cannot evict"):
        cache_of(plain4).evict(4 + 1)

    plain4.check()  # the guard fires before anything is unlinked
    match, _ = plain4.do_match(ids)
    assert (match.cached_len, match.indices) == (4, slots)


def test_evict_zero_is_a_no_op_and_evict_all_keeps_the_root(plain4: Session) -> None:
    cache = cache_of(plain4)
    assert len(cache.evict(0)) == 0  # safe on an empty tree

    ids = page_ids(4, 1, 2)
    slots = plain4.kv.take(len(ids))
    plain4.do_insert(ids, slots=slots)
    assert len(cache.evict(0)) == 0
    assert tree_slots(plain4) == slots

    plain4.do_evict_full(len(ids))
    assert set(slots) <= plain4.kv.free
    assert cache.root.is_root() and cache.root.children == {}
    assert cache.root.ref_count == 1  # the root is protected, never a victim
    assert plain4.adapter.counters() == EMPTY
    cold, _ = plain4.do_match(ids)
    assert cold.cached_len == 0
    plain4.check()


def test_insert_owns_its_indices_and_survives_caller_mutation(plain4: Session) -> None:
    """The tree must copy the slots it stores: a slice is a view, and the caller passes its
    page-table row, which it rewrites for the next request."""
    ids = list(range(1, 9))
    slots = plain4.kv.take(len(ids))
    caller_buffer = array(slots)

    cache_of(plain4).insert(array(ids), caller_buffer)
    caller_buffer.fill(-1)

    assert plain4.adapter.match(ids).indices == slots
    assert -1 not in tree_slots(plain4)
