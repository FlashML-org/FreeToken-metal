"""HybridRadixCache: the GDN snapshot currency on the KV radix tree.

Port of FreeToken tests/kvcache/radix/test_hybrid_radix.py. A snapshot is one slot id at a
node's end boundary, so a prefix of a snapshot-bearing node is not reusable; a match is
truncated to the deepest live snapshot; split_at leaves the snapshot on the suffix; insert
dedups an existing snapshot and refills a tombstone; evict_mamba counts snapshots,
tombstones internal nodes in place and cascades through the tombstone leaves it exposes.
Expectations come from the page-keyed model in radix_harness; every Session op compares
against it and Session.check() runs the invariant battery.
"""

from __future__ import annotations

import pytest

from radix_harness import Session, page_ids

PAGE = 4


def ids(*labels: int) -> tuple[int, ...]:
    return page_ids(PAGE, *labels)


def donated(session: Session) -> int:
    """The GDN slot the harness donated on the most recent insert."""
    return session.mamba.handed[-1]


def full_size(session: Session) -> int:
    return session.model.counters()["full_evictable"]


def live_snapshots(session: Session) -> int:
    counters = session.model.counters()
    return counters["mamba_evictable"] + counters["mamba_protected"]


@pytest.fixture
def hybrid() -> Session:
    return Session(PAGE)


def two_node_tree(session: Session) -> tuple[tuple[int, ...], int, int]:
    """X = [page1, page2] with snapshot mx, its child Y = [page3, page4] with snapshot my.

    Built through the request lifecycle, so Y's insert reuses X's own slots and X is older.
    """
    session.do_insert(ids(1, 2))
    mx = donated(session)
    session.do_request(ids(1, 2, 3, 4), prompt_pages=2)
    my = donated(session)
    session.check()
    return ids(1, 2, 3, 4), mx, my


def test_insert_attaches_snapshot_and_match_restores_it(hybrid: Session) -> None:
    slots = hybrid.kv.take(2 * PAGE)
    got, expected = hybrid.do_insert(ids(1, 2), slots=slots)
    assert got == (0, False)
    assert expected.adopted == slots and expected.dups == []
    hybrid.check()

    match, _ = hybrid.do_match(ids(1, 2))
    assert match.cached_len == 2 * PAGE
    assert match.indices == slots
    assert match.second == donated(hybrid)
    assert hybrid.model.events["insert.snapshot_attach"] == 1
    assert live_snapshots(hybrid) == 1


def test_deepest_snapshot_wins_and_unmatched_suffix_is_dropped(hybrid: Session) -> None:
    chain, mx, my = two_node_tree(hybrid)
    assert mx != my

    match, _ = hybrid.do_match(chain)
    assert (match.cached_len, match.second) == (4 * PAGE, my)
    # A query running past the end of the tree resumes from the deepest snapshot.
    match, _ = hybrid.do_match(chain + ids(5))
    assert (match.cached_len, match.second) == (4 * PAGE, my)
    assert hybrid.model.events["match.snapshot_truncation"] == 0
    hybrid.check()


def test_split_leaves_the_snapshot_on_the_suffix_half(hybrid: Session) -> None:
    chain, mx, my = two_node_tree(hybrid)

    match, _ = hybrid.do_match(ids(1, 2, 3, 5))  # diverges inside Y: Y splits into [3] + [4]
    assert hybrid.model.events["node.split"] == 1
    assert hybrid.model.events["match.snapshot_truncation"] == 1
    assert (match.cached_len, match.second) == (2 * PAGE, mx)
    hybrid.check()

    match, _ = hybrid.do_match(chain)  # the suffix half kept my
    assert (match.cached_len, match.second) == (4 * PAGE, my)
    assert live_snapshots(hybrid) == 2
    assert full_size(hybrid) == 4 * PAGE


def test_prefix_of_a_snapshot_node_is_not_reusable(hybrid: Session) -> None:
    slots = hybrid.kv.take(4 * PAGE)
    hybrid.do_insert(ids(1, 2, 3, 4), slots=slots)
    snapshot = donated(hybrid)
    hybrid.check()

    match, _ = hybrid.do_match(ids(1, 2))  # splits the node; the [1, 2] half has no snapshot
    assert (match.cached_len, match.indices, match.second) == (0, [], None)
    assert hybrid.model.events["match.no_snapshot"] == 1

    match, _ = hybrid.do_match(ids(1, 2, 3, 4))
    assert (match.cached_len, match.indices, match.second) == (4 * PAGE, slots, snapshot)
    assert full_size(hybrid) == 4 * PAGE
    hybrid.check()


def test_insert_dedups_and_the_caller_frees_the_donated_slot(hybrid: Session) -> None:
    hybrid.do_insert(ids(1, 2))
    mx = donated(hybrid)

    slots = hybrid.kv.take(2 * PAGE)
    got, expected = hybrid.do_insert(ids(1, 2), slots=slots)
    assert got == (2 * PAGE, True)
    assert expected.dups == slots and expected.adopted == []
    assert hybrid.model.events["insert.snapshot_dedup"] == 1
    assert live_snapshots(hybrid) == 1
    hybrid.check()

    match, _ = hybrid.do_match(ids(1, 2))
    assert match.second == mx
    assert donated(hybrid) in hybrid.mamba.free


@pytest.mark.parametrize("page_size", [1, 4, 64])
def test_insert_stores_whole_pages_only(page_size: int) -> None:
    """Fewer tokens than a page reach only the root (no snapshot, exists=True); a ragged tail
    stays with the caller."""
    session = Session(page_size)
    stub = page_ids(page_size, 1)[: page_size - 1]
    got, _ = session.do_insert(stub)
    assert got == (0, True)
    assert full_size(session) == 0 and live_snapshots(session) == 0
    session.check()

    slots = session.kv.take(2 * page_size + page_size - 1)
    tokens = page_ids(page_size, 1, 2) + page_ids(page_size, 3)[: page_size - 1]
    got, expected = session.do_insert(tokens, slots=slots)
    assert got == (0, False)
    assert expected.adopted == slots[: 2 * page_size]
    session.check()

    match, _ = session.do_match(page_ids(page_size, 1, 2))
    assert (match.cached_len, match.indices) == (2 * page_size, slots[: 2 * page_size])
    assert full_size(session) == 2 * page_size


def test_evict_mamba_tombstones_an_internal_node_and_keeps_its_kv(hybrid: Session) -> None:
    chain, mx, my = two_node_tree(hybrid)

    hybrid.do_evict_mamba(1)  # X is the LRU snapshot and internal
    assert hybrid.model.events["evict_mamba.tombstone_in_place"] == 1
    assert mx in hybrid.mamba.free and hybrid.kv.free == set()
    assert full_size(hybrid) == 4 * PAGE and live_snapshots(hybrid) == 1
    hybrid.check()

    match, _ = hybrid.do_match(ids(1, 2))  # a tombstoned boundary is no longer resumable
    assert (match.cached_len, match.second) == (0, None)
    match, _ = hybrid.do_match(chain)
    assert (match.cached_len, match.second) == (4 * PAGE, my)


def test_insert_refills_a_tombstoned_node(hybrid: Session) -> None:
    two_node_tree(hybrid)
    hybrid.do_evict_mamba(1)
    hybrid.check()

    slots = hybrid.kv.take(2 * PAGE)
    got, expected = hybrid.do_insert(ids(1, 2), slots=slots)
    assert got == (2 * PAGE, False)  # attaches, does not dedup
    assert expected.dups == slots  # the KV stays canonical
    assert hybrid.model.events["insert.snapshot_attach"] == 3
    hybrid.check()

    match, _ = hybrid.do_match(ids(1, 2))
    assert (match.cached_len, match.second) == (2 * PAGE, donated(hybrid))
    assert live_snapshots(hybrid) == 2


def test_evict_mamba_on_a_leaf_frees_kv_and_cascades_through_tombstones(hybrid: Session) -> None:
    chain, _, _ = two_node_tree(hybrid)
    hybrid.do_evict_mamba(1)  # X becomes a KV-only tombstone
    hybrid.do_evict_mamba(1)  # Y is the only snapshot node left, and a leaf
    assert hybrid.model.events["evict_mamba.leaf_free"] == 1
    assert hybrid.model.events["evict_mamba.cascade"] == 1
    hybrid.check()

    assert full_size(hybrid) == 0 and live_snapshots(hybrid) == 0
    assert hybrid.kv.in_use() == set() and hybrid.mamba.in_use() == set()
    match, _ = hybrid.do_match(chain)
    assert (match.cached_len, match.second) == (0, None)


def test_evict_mamba_counts_snapshots_not_tokens(hybrid: Session) -> None:
    hybrid.do_insert(ids(1))
    hybrid.do_request(ids(1, 2), prompt_pages=1)
    hybrid.do_request(ids(1, 2, 3), prompt_pages=2)
    hybrid.check()
    assert live_snapshots(hybrid) == 3 and full_size(hybrid) == 3 * PAGE

    hybrid.do_evict_mamba(2)
    assert len(hybrid.mamba.free) == 2
    assert hybrid.model.events["evict_mamba.tombstone_in_place"] == 2
    assert hybrid.kv.free == set() and full_size(hybrid) == 3 * PAGE
    assert live_snapshots(hybrid) == 1
    hybrid.check()


def test_snapshot_slots_are_conserved_across_eviction_waves(hybrid: Session) -> None:
    count = 8
    hybrid.do_insert(ids(1))
    for k in range(2, count + 1):
        hybrid.do_request(ids(*range(1, k + 1)), prompt_pages=k - 1)
    hybrid.check()
    assert live_snapshots(hybrid) == count and full_size(hybrid) == count * PAGE
    handed = list(hybrid.mamba.handed)

    for _ in range(10):
        if live_snapshots(hybrid) == 0:
            break
        hybrid.do_evict_mamba(3)
        hybrid.check()
    assert live_snapshots(hybrid) == 0, "evict_mamba did not converge"

    assert sorted(hybrid.mamba.free) == sorted(handed)
    assert hybrid.kv.in_use() == set() and full_size(hybrid) == 0
    assert hybrid.do_match(ids(*range(1, count + 1)))[0].cached_len == 0


def test_evict_full_takes_the_leaf_snapshot_and_leaves_the_ancestor_usable(
    hybrid: Session,
) -> None:
    chain, mx, my = two_node_tree(hybrid)

    hybrid.do_evict_full(2 * PAGE)  # the only unlocked leaf is Y
    assert my in hybrid.mamba.free and mx not in hybrid.mamba.free
    assert hybrid.model.events["evict_full.cascade"] == 0
    assert full_size(hybrid) == 2 * PAGE and live_snapshots(hybrid) == 1
    hybrid.check()

    match, _ = hybrid.do_match(chain)
    assert (match.cached_len, match.second) == (2 * PAGE, mx)


def test_lock_pins_the_snapshot_and_the_whole_kv_path(hybrid: Session) -> None:
    """A lock pins the matched node's snapshot and the KV up to the root, but not an
    ancestor's snapshot (full_ref >= mamba_ref, not the other way round)."""
    chain, mx, _ = two_node_tree(hybrid)
    held = hybrid.do_lock(chain)
    assert held is not None
    hybrid.check()

    hybrid.do_evict_full(4 * PAGE)  # Y is locked and X internal: nothing is evictable
    assert hybrid.kv.free == set() and full_size(hybrid) == 0

    hybrid.do_evict_mamba(1)  # X's snapshot is unprotected: tombstoned in place
    assert hybrid.mamba.free == {mx}
    hybrid.do_evict_mamba(1)  # Y's snapshot is pinned
    assert hybrid.mamba.free == {mx}
    hybrid.check()

    hybrid.do_unlock(held)
    assert full_size(hybrid) == 4 * PAGE
    hybrid.do_evict_full(2 * PAGE)  # Y is evictable again; X cascades behind it
    assert hybrid.model.events["evict_full.cascade"] == 1
    assert hybrid.kv.in_use() == set() and hybrid.mamba.in_use() == set()
    hybrid.check()
