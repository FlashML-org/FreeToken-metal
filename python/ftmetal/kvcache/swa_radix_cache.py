"""SWA radix prefix cache: KV pages plus the window slots behind them.

Port of FreeToken kvcache/swa_radix_cache.py (SWARadixCache, SWACacheHandle). A node's value
(full-pool slots) is canonical; its window K/V sits behind the window pool's full->swa
mapping and is freed on its own as tokens slide out of the window, which leaves the node a
tombstone that keeps its full KV. A match is reusable for the window layers only up to the
deepest node whose live run back to the last tombstone covers the window, or whose path to
the root has no tombstone. Two-tier lock (full ref >= swa ref; the swa lock ends at the
swa_uuid node) and two-tier eviction (full: unlocked leaves; swa: unlocked live nodes,
internal ones tombstoned in place, a leaf always live). A match stamps its path with
timestamps decreasing toward the root, so the swa pass takes root-side nodes first. The
cache stores slot ids only; the caller frees them in the pools. Divergences: host numpy
arrays; dec_lock has no skip_swa and there is no revive counter (neither has a caller);
check_integrity recomputes the size counters; a tombstone whose full KV a running request
locks is revived too, keeping its full slots and taking over the incoming request's window
slots (SGLang's _recover_tombstone_keeping_locked_full), where FreeToken leaves it a
tombstone and a commit's windowed match can then stop short of the request's own prefix.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ftmetal.kvcache.radix_cache import EMPTY_IDS, RadixTreeNode, align_down, page_key

# A match or insert event's clock reading is scaled by this, leaving room below it for the
# per-depth offsets of one path stamp.
EVENT_STRIDE = 1 << 24


@dataclass
class SWACacheHandle:
    """A locked matched prefix: the lock target node, the KV slots of [0, cached_len) and the
    window boundary CacheManager.lock records for the unlock."""

    cached_len: int
    node: RadixTreeNode
    kv_indices: np.ndarray
    swa_uuid: int | None = None


class SWAMatch(NamedTuple):
    # KV slots of [0, cached_len).
    kv_indices: np.ndarray
    # Truncated to the deepest windowed-safe boundary on the matched path.
    cached_len: int
    # The matched node (lock target); the root on a miss.
    node: RadixTreeNode


class SWAInsertResult(NamedTuple):
    # Length already cached before the insert.
    prefix_len: int
    # Full slots the caller frees in both pools: the tree's displaced slots and the request's
    # duplicates.
    freed: np.ndarray
    # The caller first moves the window slots behind swa_sources (request slots, among freed)
    # to swa_targets (the tree's full slots of a locked tombstone the insert revived).
    swa_targets: np.ndarray
    swa_sources: np.ndarray


class SWAEvictResult(NamedTuple):
    # Full slots that left the tree; their pages return to the free list.
    kv_indices: np.ndarray
    # Full slots whose window slots return to the window pool.
    swa_indices: np.ndarray


class SWARadixCache:
    def __init__(self, page_size: int, sliding_window: int) -> None:
        assert sliding_window > 0, "the SWA radix cache needs a sliding window"
        self.page_size = page_size
        self.sliding_window = sliding_window
        self.root = RadixTreeNode(page_size)
        self.root.ref_count = 1  # the root is always protected
        self.full_evictable = 0
        self.full_protected = 0
        # Tokens of live (non-tombstone) nodes, unlocked and window-locked.
        self.swa_evictable = 0
        self.swa_protected = 0
        # A logical clock, so stamps never tie.
        self.clock = 0
        self.last_swa_uuid = 0

    def match_prefix(self, input_ids: np.ndarray) -> SWAMatch:
        """The matched prefix, truncated to the deepest node whose live run back to the last
        tombstone is at least the window (any depth while the path has no tombstone)."""
        node = self.root
        values: list[np.ndarray] = []
        live_run = math.inf
        best_count, best_node = 0, self.root
        prefix_len = 0
        while prefix_len < len(input_ids):
            child = node.children.get(page_key(input_ids[prefix_len:], self.page_size))
            if child is None:
                break
            if child.swa_tombstone:
                # The run up to the tombstone may already cover the window: commit its parent.
                if live_run >= self.sliding_window:
                    best_count, best_node = len(values), node
                live_run = 0
            # The key lookup matched a whole page, so match_len >= page_size.
            match_len = align_down(child.get_match_len(input_ids[prefix_len:]), self.page_size)
            partial = match_len < child.length
            if partial:
                child = child.split_at(match_len)
            values.append(child.value)
            if not child.swa_tombstone:
                live_run += child.length
            node = child
            prefix_len += match_len
            if partial:
                break
        if live_run >= self.sliding_window:
            best_count, best_node = len(values), node
        self.stamp_path(best_node)
        kv_indices = np.concatenate(values[:best_count]) if best_count else EMPTY_IDS
        return SWAMatch(kv_indices, len(kv_indices), best_node)

    def insert(
        self,
        input_ids: np.ndarray,
        kv_indices: np.ndarray,
        swa_evicted_seqlen: int = 0,
        update_kv_after_len: int = 0,
    ) -> SWAInsertResult:
        """Insert the page-aligned prefix with the request's full slots kv_indices.

        Nodes past update_kv_after_len (the request's reused prefix) meet the request's own
        slots, whose window slots are live from swa_evicted_seqlen on (the request freed the
        ones below). A tombstone there is revived from that frontier on (split at it) with
        the request's live window: by adopting the request's slots, or, while a running
        request reads the node's full slots, by moving the request's window slots behind
        them. The new suffix is a tombstone below swa_evicted_seqlen and a live leaf after it.
        """
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        freed: list[np.ndarray] = []
        swa_targets: list[np.ndarray] = []
        swa_sources: list[np.ndarray] = []
        node = self.root
        total = 0
        while total < insert_len:
            child = node.children.get(page_key(input_ids[total:], self.page_size))
            if child is None:
                break
            match_len = align_down(child.get_match_len(input_ids[total:]), self.page_size)
            partial = match_len < child.length
            if partial:
                child = child.split_at(match_len)
            segment = kv_indices[total : total + match_len]
            if update_kv_after_len < total + match_len:
                if not child.swa_tombstone or swa_evicted_seqlen >= total + match_len:
                    # The tree's slots are canonical, or the node is wholly behind the
                    # request's window: drop the request's copy.
                    freed.append(segment.copy())
                else:
                    start = max(swa_evicted_seqlen - total, 0)
                    if start > 0:
                        # The head behind the request's frontier stays a tombstone.
                        child.split_at(start)
                        freed.append(segment[:start].copy())
                    incoming = segment[start:].copy()
                    if child.ref_count > 0:
                        swa_targets.append(child.value)
                        swa_sources.append(incoming)
                        freed.append(incoming)
                    else:
                        freed.append(child.value)
                        child.set_key_value(child.key, incoming)
                    self.revive(child)
            total += match_len
            node = child
            if partial:
                break
        if total < insert_len:
            suffix_ids, suffix_kv = input_ids[total:], kv_indices[total:].copy()
            # A leaf is never a tombstone: the frees' page margin leaves a live last page.
            boundary = max(0, min(swa_evicted_seqlen, insert_len) - total)
            boundary = min(boundary, max(0, len(suffix_ids) - self.page_size))
            if boundary > 0:
                node = self.add_child(
                    node, suffix_ids[:boundary].copy(), suffix_kv[:boundary], tombstone=True
                )
                suffix_ids, suffix_kv = suffix_ids[boundary:], suffix_kv[boundary:]
            self.add_child(node, suffix_ids.copy(), suffix_kv, tombstone=False)
        return SWAInsertResult(
            total,
            np.concatenate(freed) if freed else EMPTY_IDS,
            np.concatenate(swa_targets) if swa_targets else EMPTY_IDS,
            np.concatenate(swa_sources) if swa_sources else EMPTY_IDS,
        )

    def revive(self, node: RadixTreeNode) -> None:
        assert node.swa_ref_count == 0, "a tombstone holds no window lock"
        node.swa_tombstone = False
        node.timestamp = self.tick()
        self.swa_evictable += node.length

    def add_child(
        self, parent: RadixTreeNode, ids: np.ndarray, kv_indices: np.ndarray, *, tombstone: bool
    ) -> RadixTreeNode:
        child = RadixTreeNode(self.page_size, self.tick())
        child.set_key_value(ids, kv_indices)
        child.set_parent(parent)
        child.swa_tombstone = tombstone
        self.full_evictable += child.length
        if not tombstone:
            self.swa_evictable += child.length
        return child

    def inc_lock(self, node: RadixTreeNode) -> int | None:
        """Pin the KV of root..node and the live window behind node.

        Returns the uuid stamped on the node where the window lock is complete, None when the
        live path is shorter than the window.
        """
        swa_uuid: int | None = None
        swa_locked = 0
        while not node.is_root():
            if node.ref_count == 0:
                self.full_evictable -= node.length
                self.full_protected += node.length
            node.ref_count += 1
            if swa_locked < self.sliding_window and not node.swa_tombstone:
                if node.swa_ref_count == 0:
                    self.swa_evictable -= node.length
                    self.swa_protected += node.length
                node.swa_ref_count += 1
                swa_locked += node.length
                if swa_locked >= self.sliding_window:
                    if node.swa_uuid is None:
                        self.last_swa_uuid += 1
                        node.swa_uuid = self.last_swa_uuid
                    swa_uuid = node.swa_uuid
            node = node.parent
        return swa_uuid

    def dec_lock(self, node: RadixTreeNode, swa_uuid: int | None) -> None:
        """Undo inc_lock; the window unlock stops at the node stamped swa_uuid."""
        dec_swa = True
        while not node.is_root():
            node.ref_count -= 1
            assert node.ref_count >= 0
            if node.ref_count == 0:
                self.full_evictable += node.length
                self.full_protected -= node.length
            if dec_swa and not node.swa_tombstone and node.swa_ref_count > 0:
                node.swa_ref_count -= 1
                if node.swa_ref_count == 0:
                    self.swa_evictable += node.length
                    self.swa_protected -= node.length
                if swa_uuid is not None and node.swa_uuid == swa_uuid:
                    dec_swa = False
            node = node.parent

    def evict_full(self, num_tokens: int) -> SWAEvictResult:
        """Evict KV by LRU over unlocked leaves, with their window slots and the tombstone
        leaves they expose."""
        leaves = [node for node in self.leaves() if node.ref_count == 0]
        heapq.heapify(leaves)
        kv: list[np.ndarray] = []
        swa: list[np.ndarray] = []
        freed = 0
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue
            freed += node.length
            kv.append(node.value)
            self.full_evictable -= node.length
            if not node.swa_tombstone:
                swa.append(node.value)
                self.swa_evictable -= node.length
            parent, cascaded = self.cascade_tombstone_leaves(self.unlink(node), kv)
            freed += cascaded
            # The cascade may have unlinked node's parent: push the survivor it returned.
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaves, parent)
        return SWAEvictResult(
            np.concatenate(kv) if kv else EMPTY_IDS, np.concatenate(swa) if swa else EMPTY_IDS
        )

    def evict_swa(self, num_tokens: int) -> SWAEvictResult:
        """Evict window slots by LRU over unlocked live nodes, internal ones included.

        An internal (or full-locked) node becomes a tombstone in place: its window slots are
        freed, its KV and children stay. An unlocked leaf frees its KV too, then the tombstone
        leaves it exposes upward.
        """
        candidates = [node for node in self.live_nodes() if node.swa_ref_count == 0]
        heapq.heapify(candidates)
        kv: list[np.ndarray] = []
        swa: list[np.ndarray] = []
        freed = 0
        while freed < num_tokens and candidates:
            node = heapq.heappop(candidates)
            if node.swa_tombstone or node.swa_ref_count != 0 or node.is_root():
                continue
            swa.append(node.value)
            self.swa_evictable -= node.length
            freed += node.length
            node.swa_tombstone = True
            if node.is_leaf() and node.ref_count == 0:
                kv.append(node.value)
                self.full_evictable -= node.length
                self.cascade_tombstone_leaves(self.unlink(node), kv)
        return SWAEvictResult(
            np.concatenate(kv) if kv else EMPTY_IDS, np.concatenate(swa) if swa else EMPTY_IDS
        )

    def trim_head_swa(self, input_ids: np.ndarray, keep_from: int) -> np.ndarray:
        """Tombstone the path below the page-aligned keep_from, keeping its full KV; returns
        the full slots whose window slots the caller frees. Locked, tombstoned and leaf nodes
        are left alone."""
        if keep_from <= 0:
            return EMPTY_IDS
        self.match_prefix(input_ids[:keep_from])  # splits a node boundary at keep_from
        freed: list[np.ndarray] = []
        node, prefix_len = self.root, 0
        while prefix_len < keep_from:
            child = node.children.get(page_key(input_ids[prefix_len:], self.page_size))
            if child is None or prefix_len + child.length > keep_from:
                break
            if not child.swa_tombstone and child.swa_ref_count == 0 and not child.is_leaf():
                freed.append(child.value)
                self.swa_evictable -= child.length
                child.swa_tombstone = True
            node, prefix_len = child, prefix_len + child.length
        return np.concatenate(freed) if freed else EMPTY_IDS

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def swa_evictable_size(self) -> int:
        return self.swa_evictable

    def check_integrity(self) -> None:
        sizes = {"full_evictable": 0, "full_protected": 0, "swa_evictable": 0, "swa_protected": 0}
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            assert node.ref_count >= node.swa_ref_count >= 0, "full ref must be >= swa ref"
            if node.swa_tombstone:
                assert node.swa_ref_count == 0, "a tombstone holds no window lock"
                assert not node.is_leaf(), "a leaf is never a tombstone"
            full = "full_evictable" if node.ref_count == 0 else "full_protected"
            sizes[full] += node.length
            if not node.swa_tombstone:
                swa = "swa_evictable" if node.swa_ref_count == 0 else "swa_protected"
                sizes[swa] += node.length
            stack.extend(node.children.values())
        counters = {
            "full_evictable": self.full_evictable,
            "full_protected": self.full_protected,
            "swa_evictable": self.swa_evictable,
            "swa_protected": self.swa_protected,
        }
        assert counters == sizes, f"size counters {counters} drifted from the nodes {sizes}"

    def unlink(self, node: RadixTreeNode) -> RadixTreeNode:
        parent = node.parent
        parent.children.pop(page_key(node.key, self.page_size))
        return parent

    def cascade_tombstone_leaves(
        self, parent: RadixTreeNode, kv_out: list[np.ndarray]
    ) -> tuple[RadixTreeNode, int]:
        """Free the unlocked tombstone leaves exposed upward from parent, so a leaf stays live.

        Returns the highest surviving ancestor and the tokens freed.
        """
        freed = 0
        while (
            parent.swa_tombstone
            and parent.is_leaf()
            and parent.ref_count == 0
            and not parent.is_root()
        ):
            kv_out.append(parent.value)
            self.full_evictable -= parent.length
            freed += parent.length
            parent = self.unlink(parent)
        return parent, freed

    def stamp_path(self, node: RadixTreeNode) -> None:
        """Stamp root..node newest at node, older toward the root."""
        self.clock += 1
        base = self.clock * EVENT_STRIDE
        depth = 0
        while not node.is_root():
            node.timestamp = base - depth
            depth += 1
            node = node.parent

    def tick(self) -> int:
        self.clock += 1
        return self.clock * EVENT_STRIDE

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

    def live_nodes(self) -> list[RadixTreeNode]:
        out: list[RadixTreeNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not node.is_root() and not node.swa_tombstone:
                out.append(node)
            stack.extend(node.children.values())
        return out


__all__ = [
    "EVENT_STRIDE",
    "SWACacheHandle",
    "SWAEvictResult",
    "SWAInsertResult",
    "SWAMatch",
    "SWARadixCache",
]
