"""Radix tree nodes over whole pages of token ids, and the page-aligned walk.

Port of FreeToken kvcache/radix_cache.py (RadixTreeNode, the tree walk, the key per page
size). Divergences: keys and values are host numpy int32 arrays; a child is keyed by the
tuple of its first page's tokens at every page size; the node carries no SWA fields and
there is no plain RadixPrefixCache (the hybrid cache is the only tree).
"""

from __future__ import annotations

import time

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


__all__ = ["EMPTY_IDS", "PageKey", "RadixTreeNode", "align_down", "page_key", "tree_walk"]
