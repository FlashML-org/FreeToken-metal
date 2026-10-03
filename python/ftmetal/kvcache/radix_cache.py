"""Radix tree nodes over whole pages of token ids, the page-aligned walk, the plain prefix cache.

Port of FreeToken kvcache/radix_cache.py (RadixTreeNode, the tree walk, the key per page
size, RadixPrefixCache). Divergences: keys and values are host numpy int32 arrays; a child
is keyed by the tuple of its first page's tokens at every page size; RadixPrefixCache takes
HybridRadixCache's shapes (a match carries its KV slots, inc_lock/dec_lock take the node,
insert returns the end node), has no reset and check_integrity recomputes its size counters.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

# The tokens of a child's first page: the key it is registered under in its parent.
PageKey = tuple[int, ...]

EMPTY_IDS = np.empty(0, dtype=np.int32)


def align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def page_key(ids: np.ndarray, page_size: int) -> PageKey:
    return tuple(ids[:page_size].tolist())


class RadixTreeNode:
    def __init__(self, page_size: int, timestamp: int | None = None) -> None:
        self.page_size = page_size
        self.children: dict[PageKey, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count = 0
        self.timestamp = time.monotonic_ns() if timestamp is None else timestamp
        # The hybrid cache's second currency: a GDN state snapshot slot for the state at this
        # node's end boundary, with its own lock count. split_at leaves it on the suffix.
        self.mamba_value: int | None = None
        self.mamba_ref_count = 0
        # The SWA cache's second currency is the window slots behind value: a tombstone's were
        # freed (its full KV stays), swa_ref_count locks the trailing window and swa_uuid marks
        # the node a window lock stops at.
        self.swa_tombstone = False
        self.swa_ref_count = 0
        self.swa_uuid: int | None = None
        self.key = EMPTY_IDS
        self.value = EMPTY_IDS

    def set_key_value(self, key: np.ndarray, value: np.ndarray) -> None:
        assert len(key) == len(value)
        self.key = key
        self.value = value

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[page_key(self.key, self.page_size)] = self

    @property
    def length(self) -> int:
        return len(self.key)

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: np.ndarray) -> int:
        """Leading tokens of input_ids equal to this node's key."""
        common = min(self.length, len(input_ids))
        mismatches = np.flatnonzero(self.key[:common] != input_ids[:common])
        return int(mismatches[0]) if len(mismatches) > 0 else common

    def split_at(self, pos: int) -> RadixTreeNode:
        """A new node for key[:pos] between this node and its parent; this node keeps key[pos:]."""
        assert 0 < pos < self.length
        parent = self.parent
        new_node = RadixTreeNode(self.page_size, self.timestamp)
        new_node.set_key_value(self.key[:pos], self.value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count
        # Both halves keep the tombstone and the window lock; the lock boundary moves to the
        # root-side half, which an unlock walking up reaches last.
        new_node.swa_tombstone = self.swa_tombstone
        new_node.swa_ref_count = self.swa_ref_count
        new_node.swa_uuid = self.swa_uuid
        self.swa_uuid = None
        self.set_key_value(self.key[pos:], self.value[pos:])
        self.set_parent(new_node)
        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


def tree_walk(root: RadixTreeNode, input_ids: np.ndarray) -> tuple[RadixTreeNode, int]:
    """The deepest node matching a page-aligned prefix of input_ids and that prefix's length.

    A partially matched node is split so the returned node ends exactly at the prefix; every
    node on the path is stamped with one clock read.
    """
    page_size = root.page_size
    prefix_len = 0
    node = root
    tic = time.monotonic_ns()
    while prefix_len < len(input_ids):
        child = node.children.get(page_key(input_ids[prefix_len:], page_size))
        if child is None:
            return node, prefix_len
        node = child
        # The key lookup matched a whole page, so match_len >= page_size.
        match_len = align_down(node.get_match_len(input_ids[prefix_len:]), page_size)
        prefix_len += match_len
        if match_len != node.length:
            node = node.split_at(match_len)
            node.timestamp = tic
            return node, prefix_len
        node.timestamp = tic
    return node, prefix_len


@dataclass(frozen=True)
class RadixCacheHandle:
    """A locked matched prefix: the lock target node and the KV slots of [0, cached_len)."""

    cached_len: int
    node: RadixTreeNode
    kv_indices: np.ndarray


class RadixMatch(NamedTuple):
    # KV slots of [0, cached_len).
    kv_indices: np.ndarray
    cached_len: int
    # The matched node (lock target); the root on a miss.
    node: RadixTreeNode


class RadixPrefixCache:
    def __init__(self, page_size: int) -> None:
        self.page_size = page_size
        self.root = RadixTreeNode(page_size)
        self.root.ref_count = 1  # the root is always protected
        self.evictable_size = 0
        self.protected_size = 0

    def match_prefix(self, input_ids: np.ndarray) -> RadixMatch:
        """The longest page-aligned cached prefix of input_ids."""
        node, prefix_len = tree_walk(self.root, input_ids)
        return RadixMatch(self.collect_kv(node), prefix_len, node)

    def insert(self, input_ids: np.ndarray, kv_indices: np.ndarray) -> tuple[int, RadixTreeNode]:
        """Insert the page-aligned prefix, donating the KV slots of its uncached part.

        Returns (length already cached before the insert, the node ending the inserted
        prefix). The caller keeps, and frees, its own slots for the already cached part.
        """
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        node, prefix_len = tree_walk(self.root, input_ids)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.page_size)
            new_node.set_key_value(input_ids[prefix_len:].copy(), kv_indices[prefix_len:].copy())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return prefix_len, node

    def inc_lock(self, node: RadixTreeNode) -> None:
        """Pin the KV of root..node."""
        while not node.is_root():
            if node.ref_count == 0:
                self.evictable_size -= node.length
                self.protected_size += node.length
            node.ref_count += 1
            node = node.parent

    def dec_lock(self, node: RadixTreeNode) -> None:
        while not node.is_root():
            node.ref_count -= 1
            assert node.ref_count >= 0
            if node.ref_count == 0:
                self.evictable_size += node.length
                self.protected_size -= node.length
            node = node.parent

    def evict(self, num_tokens: int) -> np.ndarray:
        """Evict at least num_tokens of KV by LRU over unlocked leaves; returns their slots."""
        if num_tokens == 0:
            return EMPTY_IDS
        assert num_tokens <= self.evictable_size, (
            f"Cannot evict {num_tokens}, only {self.evictable_size} is evictable"
        )
        leaves = [node for node in self.leaves() if node.ref_count == 0]
        heapq.heapify(leaves)
        evicted: list[np.ndarray] = []
        evicted_size = 0
        while evicted_size < num_tokens:
            assert leaves, f"Cannot evict enough cache, need {num_tokens}, evicted {evicted_size}"
            node = heapq.heappop(leaves)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            parent.children.pop(page_key(node.key, self.page_size))
            # The root is always protected, so it is never pushed.
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leaves, parent)
        return np.concatenate(evicted)

    def check_integrity(self) -> None:
        evictable = protected = 0
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            assert node.ref_count >= 0
            if node.ref_count == 0:
                evictable += node.length
            else:
                protected += node.length
            stack.extend(node.children.values())
        assert (evictable, protected) == (self.evictable_size, self.protected_size), (
            f"size counters ({self.evictable_size}, {self.protected_size}) drifted from the "
            f"nodes ({evictable}, {protected})"
        )

    def collect_kv(self, node: RadixTreeNode) -> np.ndarray:
        values: list[np.ndarray] = []
        while not node.is_root():
            values.append(node.value)
            node = node.parent
        values.reverse()
        return np.concatenate(values) if values else EMPTY_IDS

    def leaves(self) -> list[RadixTreeNode]:
        out: list[RadixTreeNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf():
                if not node.is_root():
                    out.append(node)
            else:
                stack.extend(node.children.values())
        return out


__all__ = [
    "EMPTY_IDS",
    "PageKey",
    "RadixCacheHandle",
    "RadixMatch",
    "RadixPrefixCache",
    "RadixTreeNode",
    "align_down",
    "page_key",
    "tree_walk",
]
