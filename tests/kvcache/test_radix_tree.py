"""The shared tree machinery, and a self-test of the radix harness.

Port of FreeToken tests/kvcache/radix/test_tree_and_harness.py (hybrid cache only):
split_at identity semantics, the page key, get_match_len and the page-aligned walk; then a
cache broken on purpose must be caught by the check whose name says so.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from ftmetal.kvcache.hybrid_radix_cache import EvictResult, HybridMatch, HybridRadixCache
from ftmetal.kvcache.radix_cache import RadixTreeNode, align_down, page_key, tree_walk
from radix_harness import HarnessFailure, Session, node_path_slots, page_ids


def array(values: Sequence[int]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int32)


def bare_node(
    page_size: int, key: Sequence[int], value: Sequence[int], parent: RadixTreeNode | None = None
) -> RadixTreeNode:
    node = RadixTreeNode(page_size)
    node.set_key_value(array(key), array(value))
    if parent is not None:
        node.set_parent(parent)
    return node


def test_split_at_keeps_the_original_object_as_the_suffix() -> None:
    """Handles issued before a split (lock targets, deeper nodes) must still resolve to the
    same slots, so the matched node object becomes the suffix and a new node the prefix."""
    root = RadixTreeNode(2)
    node = bare_node(2, [1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16], root)
    deep = bare_node(2, [7, 8], [17, 18], node)
    node.ref_count = 2
    node.mamba_value = 99
    node.mamba_ref_count = 1
    stamp = node.timestamp
    path_before = node_path_slots(deep)

    prefix = node.split_at(4)

    assert prefix is not node
    assert node.key.tolist() == [5, 6] and node.value.tolist() == [15, 16]
    assert prefix.key.tolist() == [1, 2, 3, 4] and prefix.value.tolist() == [11, 12, 13, 14]
    assert deep.parent is node
    assert node_path_slots(deep) == path_before == [11, 12, 13, 14, 15, 16, 17, 18]
    assert prefix.parent is root and node.parent is prefix
    assert root.children[page_key(prefix.key, 2)] is prefix
    assert prefix.children[page_key(node.key, 2)] is node
    assert len(root.children) == 1 and len(prefix.children) == 1
    # A full lock covers both halves; a snapshot cannot be split; LRU age is inherited.
    assert prefix.ref_count == 2 and node.ref_count == 2
    assert prefix.mamba_value is None and node.mamba_value == 99
    assert prefix.mamba_ref_count == 0 and node.mamba_ref_count == 1
    assert prefix.timestamp == stamp

    with pytest.raises(AssertionError):
        node.split_at(0)
    with pytest.raises(AssertionError):
        node.split_at(node.length)


def test_page_key_buckets_a_whole_page_not_a_token() -> None:
    """A child is registered under its first page: a longer query finds the same bucket and
    two pages sharing leading tokens are distinct buckets."""
    assert page_key(array([5, 9, 9]), 1) == (5,)
    assert page_key(array([1, 7]), 1) == page_key(array([1, 9]), 1)
    assert page_key(array([1, 7, 7, 2, 3, 3]), 4) == page_key(array([1, 7, 7, 2]), 4)
    assert page_key(array([1, 7, 7, 2]), 4) != page_key(array([1, 7, 7, 3]), 4)

    left = (1, 3, 1, 5) + (3, 2, 2, 9)
    right = (1, 3, 1, 6) + (0, 0, 0, 0)
    session = Session(4)
    session.do_insert(left)
    got, _ = session.do_insert(right)
    session.check()
    assert got[0] == 0, "two distinct pages shared a prefix"
    assert len(session.adapter.nodes()) == 2


@pytest.mark.parametrize("page_size", [1, 4, 32])
def test_match_len_and_walk_reuse_whole_pages_only(page_size: int) -> None:
    node = bare_node(page_size, list(range(100, 108)), list(range(200, 208)))
    assert node.get_match_len(array(range(100, 108))) == 8
    assert node.get_match_len(array([*range(100, 106), 0, 0])) == 6
    assert node.get_match_len(array([100, 101, 102])) == 3
    assert node.get_match_len(array([0, 101, 102])) == 0
    assert align_down(6, page_size) == 6 // page_size * page_size

    # The walk truncates a divergence inside a page to the page below it.
    blocks = page_ids(page_size, 1, 2)
    root = RadixTreeNode(page_size)
    bare_node(page_size, blocks, range(len(blocks)), root)
    ragged = array(blocks[: 2 * page_size - 1])
    walked, prefix_len = tree_walk(root, ragged)
    assert prefix_len == align_down(len(ragged), page_size) == page_size
    assert walked.length == page_size and walked.value.tolist() == list(range(page_size))


class BreakableCache(HybridRadixCache):
    """Behaves until broken is set; subclasses then break exactly one thing."""

    broken = False


class ForgetfulInsert(BreakableCache):
    """insert links the new node but forgets to grow full_evictable."""

    def insert(
        self, input_ids: np.ndarray, kv_indices: np.ndarray, mamba_value: int
    ) -> tuple[int, bool]:
        before = self.full_evictable
        result = super().insert(input_ids, kv_indices, mamba_value)
        if self.broken:
            self.full_evictable = before
        return result


class LeakyEvict(BreakableCache):
    """evict_full also unlinks one more leaf without returning its slots."""

    def evict_full(self, num_tokens: int) -> EvictResult:
        result = super().evict_full(num_tokens)
        if self.broken:
            leaf = next(leaf for leaf in self.leaves() if leaf.ref_count == 0)
            self.unlink(leaf)
            self.full_evictable -= leaf.length
            self.free_node_mamba(leaf, [])
        return result


class NewestFirstEvict(BreakableCache):
    """evict_full takes the most recently used leaf: every counter and slot stays conserved."""

    def evict_full(self, num_tokens: int) -> EvictResult:
        if not self.broken:
            return super().evict_full(num_tokens)
        victim = max(self.leaves(), key=lambda leaf: leaf.timestamp)
        self.unlink(victim)
        self.full_evictable -= victim.length
        mamba: list[int] = []
        self.free_node_mamba(victim, mamba)
        return EvictResult(victim.value, mamba)


class ShortMatch(BreakableCache):
    """match_prefix stops one node early: a silent loss of prefix reuse."""

    def match_prefix(self, input_ids: np.ndarray) -> HybridMatch:
        match = super().match_prefix(input_ids)
        if not self.broken or match.node.is_root():
            return match
        parent = match.node.parent
        return HybridMatch(
            self.collect_kv(parent),
            match.cached_len - match.node.length,
            parent.mamba_value,
            parent,
        )


BROKEN: list[tuple[type[BreakableCache], str, str]] = [
    (ForgetfulInsert, "insert", "counters"),
    (LeakyEvict, "evict", "conservation.kv"),
    (NewestFirstEvict, "evict", "model.evict"),
    (ShortMatch, "match", "match.cached_len"),
]


@pytest.mark.parametrize(
    "cache_class, trigger, tag", BROKEN, ids=[entry[0].__name__ for entry in BROKEN]
)
def test_harness_catches_a_broken_implementation(
    cache_class: type[BreakableCache], trigger: str, tag: str
) -> None:
    """Each break must be caught by the check named for it, not by a downstream avalanche."""
    page_size = 4
    cache = cache_class(page_size)
    session = Session(page_size, cache)
    keys = [page_ids(page_size, label) for label in (1, 2, 3)]
    for key in keys:  # three one-page leaves with distinct LRU stamps
        session.do_insert(key)
        session.check()
    assert len(session.adapter.nodes()) == 3

    cache.broken = True
    with pytest.raises(HarnessFailure) as failure:
        if trigger == "insert":
            session.do_insert(page_ids(page_size, 4))
        elif trigger == "evict":
            session.do_evict_full(page_size)
        else:
            session.do_match(keys[0])
        session.check()
    assert failure.value.tag == tag, failure.value.detail
