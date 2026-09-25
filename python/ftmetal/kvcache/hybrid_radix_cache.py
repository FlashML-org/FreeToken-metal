"""Hybrid radix prefix cache: KV pages plus GDN state snapshots at node boundaries.

Port of FreeToken kvcache/hybrid_radix_cache.py (HybridRadixCache). A node's snapshot is a
LinearStatePool slot holding the recurrent + conv state after the node's last token; a match
reuses the prefix only up to the deepest node owning a live snapshot. Donate-not-copy
inserts with dedup, dual LRU eviction (KV leaves, snapshots on any node), internal nodes
tombstoned in place and leaves always carrying a live snapshot. The cache stores slot ids
and page indices only; the caller frees them in the pools. Divergences: host numpy arrays;
the snapshot-chunk vs page-size check lives with the caller that places the snapshots.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ftmetal.kvcache.radix_cache import (
    EMPTY_IDS,
    RadixTreeNode,
    align_down,
    page_key,
    tree_walk,
)


@dataclass(frozen=True)
class HybridCacheHandle:
    """A locked matched prefix: the lock target node and the KV slots of [0, cached_len)."""

    cached_len: int
    node: RadixTreeNode
    kv_indices: np.ndarray


class HybridMatch(NamedTuple):
    # KV slots of [0, cached_len).
    kv_indices: np.ndarray
    # Truncated to the deepest live-snapshot boundary on the matched path.
    cached_len: int
    # The snapshot slot to restore from; None on a miss.
    mamba_value: int | None
    # The matched node (lock target); the root on a miss.
    node: RadixTreeNode


class EvictResult(NamedTuple):
    kv_indices: np.ndarray
    mamba_slots: list[int]


class HybridRadixCache:
    def __init__(self, page_size: int) -> None:
        self.page_size = page_size
        self.root = RadixTreeNode(page_size)
        self.root.ref_count = 1  # the root is always protected
        self.full_evictable = 0
        self.full_protected = 0
        # Live snapshots, unlocked and locked.
        self.mamba_evictable = 0
        self.mamba_protected = 0

    def match_prefix(self, input_ids: np.ndarray) -> HybridMatch:
        """The matched prefix, truncated to the deepest node on the path with a live snapshot."""
        node, _ = tree_walk(self.root, input_ids)
        end_len = self.path_len(node)
        while not node.is_root():
            if node.mamba_value is not None:
                return HybridMatch(self.collect_kv(node), end_len, node.mamba_value, node)
            end_len -= node.length
            node = node.parent
        return HybridMatch(EMPTY_IDS, 0, None, self.root)

    def insert(
        self, input_ids: np.ndarray, kv_indices: np.ndarray, mamba_value: int
    ) -> tuple[int, bool]:
        """Insert the page-aligned KV prefix and donate mamba_value at its end node.

        Returns (length already cached before the insert, mamba_exist). mamba_exist means the
        end node already owns a snapshot (or is the root) and the caller keeps its slot.
        """
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        node, prefix_len = tree_walk(self.root, input_ids)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.page_size)
            new_node.set_key_value(input_ids[prefix_len:].copy(), kv_indices[prefix_len:].copy())
            new_node.set_parent(node)
            self.full_evictable += new_node.length
            node = new_node
        if node.is_root() or node.mamba_value is not None:
            return prefix_len, True
        node.mamba_value = mamba_value
        if node.mamba_ref_count == 0:
            self.mamba_evictable += 1
        return prefix_len, False

    def inc_lock(self, node: RadixTreeNode) -> None:
        """Pin the node's snapshot and the KV of root..node (full_ref >= mamba_ref)."""
        if node.mamba_value is not None:
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1
                self.mamba_protected += 1
            node.mamba_ref_count += 1
        while not node.is_root():
            if node.ref_count == 0:
                self.full_evictable -= node.length
                self.full_protected += node.length
            node.ref_count += 1
            node = node.parent

    def dec_lock(self, node: RadixTreeNode) -> None:
        if node.mamba_value is not None and node.mamba_ref_count > 0:
            node.mamba_ref_count -= 1
            if node.mamba_ref_count == 0:
                self.mamba_evictable += 1
                self.mamba_protected -= 1
        while not node.is_root():
            node.ref_count -= 1
            assert node.ref_count >= 0
            if node.ref_count == 0:
                self.full_evictable += node.length
                self.full_protected -= node.length
            node = node.parent

    def evict_full(self, num_tokens: int) -> EvictResult:
        """Evict KV by LRU over unlocked leaves, with their snapshots and exposed tombstones."""
        leaves = [node for node in self.leaves() if node.ref_count == 0]
        heapq.heapify(leaves)
        kv: list[np.ndarray] = []
        mamba: list[int] = []
        freed = 0
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue
            freed += node.length
            kv.append(node.value)
            self.full_evictable -= node.length
            self.free_node_mamba(node, mamba)
            parent, cascaded = self.cascade_tombstone_leaves(self.unlink(node), kv)
            freed += cascaded
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaves, parent)
        return EvictResult(np.concatenate(kv) if kv else EMPTY_IDS, mamba)

    def evict_mamba(self, num_snapshots: int) -> EvictResult:
        """Evict snapshots by LRU over unlocked snapshot nodes, internal ones included.

        An internal (or KV-locked) node becomes a tombstone: its slot is freed, its KV and
        children stay. An unlocked leaf frees its KV too, then the tombstone leaves it exposes
        upward, so a leaf always carries a live snapshot.
        """
        candidates = [node for node in self.snapshot_nodes() if node.mamba_ref_count == 0]
        heapq.heapify(candidates)
        kv: list[np.ndarray] = []
        mamba: list[int] = []
        freed = 0
        while freed < num_snapshots and candidates:
            node = heapq.heappop(candidates)
            if node.mamba_value is None or node.mamba_ref_count != 0 or node.is_root():
                continue
            freed += 1
            if node.is_leaf() and node.ref_count == 0:
                kv.append(node.value)
                self.full_evictable -= node.length
                self.free_node_mamba(node, mamba)
                self.cascade_tombstone_leaves(self.unlink(node), kv)
            else:
                self.free_node_mamba(node, mamba)
        return EvictResult(np.concatenate(kv) if kv else EMPTY_IDS, mamba)

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable

    def check_integrity(self) -> None:
        for node in self.snapshot_nodes():
            assert node.mamba_value is not None
            assert node.mamba_ref_count >= 0 and node.ref_count >= 0

    def free_node_mamba(self, node: RadixTreeNode, freed_slots: list[int]) -> None:
        if node.mamba_value is not None:
            freed_slots.append(node.mamba_value)
            node.mamba_value = None
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1

    def unlink(self, node: RadixTreeNode) -> RadixTreeNode:
        parent = node.parent
        parent.children.pop(page_key(node.key, self.page_size))
        return parent

    def cascade_tombstone_leaves(
        self, parent: RadixTreeNode, kv_out: list[np.ndarray]
    ) -> tuple[RadixTreeNode, int]:
        """Free the unlocked snapshot-less leaves exposed upward from parent.

        Returns the highest surviving ancestor and the tokens freed.
        """
        freed = 0
        while (
            parent.mamba_value is None
            and parent.is_leaf()
            and parent.ref_count == 0
            and not parent.is_root()
        ):
            kv_out.append(parent.value)
            self.full_evictable -= parent.length
            freed += parent.length
            parent = self.unlink(parent)
        return parent, freed

    def path_len(self, node: RadixTreeNode) -> int:
        total = 0
        while not node.is_root():
            total += node.length
            node = node.parent
        return total

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

    def snapshot_nodes(self) -> list[RadixTreeNode]:
        out: list[RadixTreeNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.mamba_value is not None and not node.is_root():
                out.append(node)
            stack.extend(node.children.values())
        return out


__all__ = ["EvictResult", "HybridCacheHandle", "HybridMatch", "HybridRadixCache"]
