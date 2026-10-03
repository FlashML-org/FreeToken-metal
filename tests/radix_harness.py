"""Radix cache harness: a page-keyed reference model, an invariant battery, a Session.

Port of FreeToken tests/kvcache/radix/{model,adapters,driver}.py, reduced to the plain and
the hybrid cache. Every op is asked of the cache and of the reference model and the answers
are compared field by field; Session.check() then runs the battery: counters recomputed
from raw node fields, slot conservation per currency, prefix closure, the cache's own
check_integrity and node-for-node agreement with the model.

The model is keyed by page, not by token: the tree's reuse unit is a whole page and two
different pages may share leading tokens. A page is named by its page-aligned prefix
tuple(ids[:(p + 1) * page_size]). The model never reads the cache's bookkeeping; the only
thing it learns from the implementation is which of several equally old LRU candidates an
eviction picked. Slot ids are globally unique and never reused, so every returned slot
maps back to one node.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ftmetal.kvcache.hybrid_radix_cache import HybridRadixCache
from ftmetal.kvcache.radix_cache import RadixPrefixCache, RadixTreeNode, page_key

Path = tuple[int, ...]


class HarnessFailure(AssertionError):
    """A harness-detected failure with a stable tag naming the check that fired."""

    def __init__(self, tag: str, message: str) -> None:
        super().__init__(f"[{tag}] {message}")
        self.tag = tag
        self.detail = message


class ModelMismatch(HarnessFailure):
    """The implementation disagreed with the page-keyed model."""


class InvariantViolation(HarnessFailure):
    """An accounting, conservation or structure invariant broke."""


@dataclass(frozen=True)
class NodeRecord:
    length: int
    slots: tuple[int, ...]
    ref: int
    mamba: int | None
    mamba_ref: int


def sizes(records: Sequence[NodeRecord], has_snapshots: bool) -> dict[str, int]:
    """The cache's running sizes, recomputed from per-node records."""
    out = {
        "full_evictable": sum(record.length for record in records if record.ref == 0),
        "full_protected": sum(record.length for record in records if record.ref > 0),
    }
    if has_snapshots:
        live = [record for record in records if record.mamba is not None]
        out["mamba_evictable"] = sum(1 for record in live if record.mamba_ref == 0)
        out["mamba_protected"] = sum(1 for record in live if record.mamba_ref > 0)
    return out


# ------------------------------------------------------------------------ reference model
class MGroup:
    """One node of the model: a run of pages sharing a tree edge.

    paths[i] is the page-aligned prefix ending at the i-th page, so paths[-1] names the node's
    end boundary; slots holds page_size slot ids per page.
    """

    def __init__(self, paths: list[Path], slots: list[int], stamp: int) -> None:
        self.paths = paths
        self.slots = slots
        self.stamp = stamp
        self.ref = 0
        self.mamba: int | None = None
        self.mamba_ref = 0

    @property
    def length(self) -> int:
        return len(self.slots)

    @property
    def n_pages(self) -> int:
        return len(self.paths)

    @property
    def end(self) -> Path:
        return self.paths[-1]

    @property
    def start(self) -> Path:
        return self.paths[0]

    def record(self) -> NodeRecord:
        return NodeRecord(self.length, tuple(self.slots), self.ref, self.mamba, self.mamba_ref)


class PageTrie:
    """Page-keyed trie: kids[path][page] = child path; the root's path is ()."""

    def __init__(self, page_size: int) -> None:
        self.page_size = page_size
        self.page_group: dict[Path, MGroup] = {}
        self.kids: dict[Path, dict[Path, Path]] = {(): {}}
        self.group_list: list[MGroup] = []
        # Branch counters, so a scenario can assert it reached the branch it claims to test.
        self.events: Counter[str] = Counter()

    def parent_group(self, group: MGroup) -> MGroup | None:
        parent_path = group.start[: -self.page_size]
        return None if not parent_path else self.page_group[parent_path]

    def is_leaf(self, group: MGroup) -> bool:
        return not self.kids.get(group.end)

    def walk(self, ids: Sequence[int]) -> list[Path]:
        """Page paths of the longest page-aligned prefix of ids in the trie."""
        out: list[Path] = []
        current: Path = ()
        for i in range(len(ids) // self.page_size):
            page = tuple(ids[i * self.page_size : (i + 1) * self.page_size])
            child = self.kids.get(current, {}).get(page)
            if child is None:
                break
            out.append(child)
            current = child
        return out

    def path_slots(self, group: MGroup | None) -> list[int]:
        chain: list[MGroup] = []
        while group is not None:
            chain.append(group)
            group = self.parent_group(group)
        return [slot for node in reversed(chain) for slot in node.slots]

    def add_group(
        self, parent: MGroup | None, pages: list[Path], slots: list[int], stamp: int
    ) -> MGroup:
        base: Path = () if parent is None else parent.end
        paths: list[Path] = []
        current = base
        for page in pages:
            current = current + page
            paths.append(current)
        group = MGroup(paths, slots, stamp)
        for path in paths:
            self.page_group[path] = group
        self.kids.setdefault(base, {})[pages[0]] = paths[0]
        for i in range(len(paths) - 1):
            self.kids.setdefault(paths[i], {})[pages[i + 1]] = paths[i + 1]
        self.group_list.append(group)
        self.events["node.add"] += 1
        return group

    def split(self, group: MGroup, n_pages: int) -> MGroup:
        """Mirror split_at: a new prefix node; group becomes the suffix and keeps the snapshot."""
        assert 0 < n_pages < group.n_pages
        prefix = MGroup(group.paths[:n_pages], group.slots[: n_pages * self.page_size], group.stamp)
        prefix.ref = group.ref
        group.paths = group.paths[n_pages:]
        group.slots = group.slots[n_pages * self.page_size :]
        for path in prefix.paths:
            self.page_group[path] = prefix
        self.group_list.insert(self.group_list.index(group), prefix)
        self.events["node.split"] += 1
        return prefix

    def remove(self, group: MGroup) -> MGroup | None:
        """Unlink a leaf; returns its parent (None for the root)."""
        assert self.is_leaf(group)
        parent = self.parent_group(group)
        base: Path = () if parent is None else parent.end
        self.kids[base].pop(group.start[len(base) :])
        for path in group.paths:
            self.page_group.pop(path)
            self.kids.pop(path, None)
        self.group_list.remove(group)
        self.events["node.remove"] += 1
        return parent


@dataclass
class ExpMatch:
    cached_len: int
    indices: list[int]
    group: MGroup | None
    second: int | None


@dataclass
class ExpInsert:
    matched_len: int
    # What the caller must free by convention: its supplied slots for [reused_len, matched).
    dups: list[int]
    # Supplied slots the tree took ownership of.
    adopted: list[int]
    # The hybrid cache's dedup flag; None for the plain cache.
    second_exists: bool | None


class RadixModel:
    """RadixPrefixCache's semantics at page granularity: the trie, the commit and the leaf
    eviction HybridModel extends with the snapshot currency."""

    has_snapshots = False

    def __init__(self, page_size: int) -> None:
        self.page_size = page_size
        self.trie = PageTrie(page_size)
        self.clock = 0

    @property
    def events(self) -> Counter[str]:
        return self.trie.events

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def cover(self, ids: Sequence[int]) -> list[MGroup]:
        """Nodes covering the longest page-aligned prefix, splitting the last one if needed;
        stamped with one clock read like tree_walk."""
        chain = self.trie.walk(ids)
        out: list[MGroup] = []
        i = 0
        while i < len(chain):
            group = self.trie.page_group[chain[i]]
            covered = min(group.n_pages, len(chain) - i)
            if covered < group.n_pages:
                group = self.trie.split(group, covered)
            out.append(group)
            i += covered
        stamp = self.tick()
        for group in out:
            group.stamp = stamp
        return out

    def pages_of(self, ids: Sequence[int], start: int, stop: int) -> list[Path]:
        size = self.page_size
        return [tuple(ids[t : t + size]) for t in range(start, stop, size)]

    def match(self, ids: Sequence[int]) -> ExpMatch:
        groups = self.cover(ids)
        if not groups:
            return ExpMatch(0, [], None, None)
        last = groups[-1]
        return ExpMatch(len(last.end), self.trie.path_slots(last), last, None)

    def commit(
        self, ids: Sequence[int], slots: Sequence[int], reused_len: int
    ) -> tuple[MGroup | None, ExpInsert]:
        """Walk, then append the missing page-aligned suffix; returns the end node too.

        The tree stores nothing for the already cached prefix: the caller owns, and frees,
        its supplied slots for [reused_len, matched).
        """
        insert_len = len(ids) // self.page_size * self.page_size
        ids, slots = list(ids[:insert_len]), list(slots[:insert_len])
        groups = self.cover(ids)
        node = groups[-1] if groups else None
        matched = 0 if node is None else len(node.end)
        if matched != insert_len:
            node = self.trie.add_group(
                node, self.pages_of(ids, matched, insert_len), slots[matched:], self.tick()
            )
        return node, ExpInsert(matched, slots[reused_len:matched], slots[matched:], None)

    def insert(
        self, ids: Sequence[int], slots: Sequence[int], reused_len: int, mamba: int | None
    ) -> ExpInsert:
        assert mamba is None, "the plain cache takes no snapshot"
        return self.commit(ids, slots, reused_len)[1]

    def inc_lock(self, group: MGroup | None) -> None:
        while group is not None:
            group.ref += 1
            group = self.trie.parent_group(group)

    def dec_lock(self, group: MGroup | None) -> None:
        while group is not None:
            if group.ref <= 0:
                raise ModelMismatch("model.dec_lock", f"full ref underflow at node end={group.end}")
            group.ref -= 1
            group = self.trie.parent_group(group)

    def records(self) -> list[tuple[Path, NodeRecord]]:
        return [(group.end, group.record()) for group in self.trie.group_list]

    def counters(self) -> dict[str, int]:
        return sizes([record for _, record in self.records()], self.has_snapshots)

    # -- eviction: the model follows the cache's reported victims and checks each is legal
    def owner(self, slot: int, tag: str) -> MGroup:
        for group in self.trie.group_list:
            if slot in group.slots:
                return group
        raise ModelMismatch(tag, f"cache freed slot {slot} which the model does not own")

    def pick_victim(self, candidates: list[MGroup], victim: MGroup, tag: str) -> None:
        """The observed victim must be eligible and minimal in stamp (ties are legal)."""
        if victim not in candidates:
            raise ModelMismatch(tag, f"evicted node end={victim.end} was not eligible")
        if victim.stamp != min(group.stamp for group in candidates):
            older = [(group.end, group.stamp) for group in candidates if group.stamp < victim.stamp]
            raise ModelMismatch(tag, f"non-LRU victim end={victim.end}; older: {older}")

    def take(self, observed: Sequence[int], pos: int, group: MGroup, tag: str) -> int:
        chunk = list(observed[pos : pos + group.length])
        if chunk != group.slots:
            raise ModelMismatch(tag, f"freed {chunk} for node end={group.end}, owns {group.slots}")
        return pos + group.length

    def take_mamba(self, observed: Sequence[int], pos: int, group: MGroup, tag: str) -> int:
        if group.mamba is None:
            return pos
        if pos >= len(observed) or observed[pos] != group.mamba:
            raise ModelMismatch(
                tag, f"expected snapshot {group.mamba} of node end={group.end} to be freed"
            )
        group.mamba = None
        group.mamba_ref = 0
        return pos + 1

    @staticmethod
    def drained(observed: Sequence[int], pos: int, tag: str) -> None:
        if pos != len(observed):
            raise ModelMismatch(tag, f"returned unpredicted slots {list(observed[pos:])}")

    def cascadable(self, group: MGroup) -> bool:
        """Whether an exposed unlocked leaf is reclaimed at once instead of queued as a
        candidate: never in the plain cache."""
        return False

    def cascade(
        self,
        parent: MGroup | None,
        observed_kv: Sequence[int],
        pos: int,
        candidates: list[MGroup],
        tag: str,
        label: str,
    ) -> tuple[MGroup | None, int, int]:
        """Reclaim exposed unlocked cascadable leaves upward."""
        freed = 0
        while (
            parent is not None
            and parent.ref == 0
            and self.trie.is_leaf(parent)
            and self.cascadable(parent)
        ):
            self.events[f"{label}.cascade"] += 1
            pos = self.take(observed_kv, pos, parent, tag)
            freed += parent.length
            if parent in candidates:
                candidates.remove(parent)
            parent = self.trie.remove(parent)
        return parent, pos, freed

    def evict_full(
        self, num_tokens: int, observed_kv: Sequence[int], observed_mamba: Sequence[int]
    ) -> None:
        tag = "model.evict"
        candidates = [
            group for group in self.trie.group_list if group.ref == 0 and self.trie.is_leaf(group)
        ]
        freed, kv_pos, mamba_pos = 0, 0, 0
        while freed < num_tokens and candidates:
            if kv_pos >= len(observed_kv):
                raise ModelMismatch(tag, f"evict_full stopped at {freed} with leaves left")
            victim = self.owner(observed_kv[kv_pos], tag)
            self.pick_victim(candidates, victim, tag)
            kv_pos = self.take(observed_kv, kv_pos, victim, tag)
            freed += victim.length
            mamba_pos = self.take_mamba(observed_mamba, mamba_pos, victim, tag)
            candidates.remove(victim)
            parent, kv_pos, extra = self.cascade(
                self.trie.remove(victim), observed_kv, kv_pos, candidates, tag, "evict_full"
            )
            freed += extra
            if (
                parent is not None
                and parent.ref == 0
                and self.trie.is_leaf(parent)
                and parent not in candidates
            ):
                self.events["evict.parent_exposed"] += 1
                candidates.append(parent)
        self.drained(observed_kv, kv_pos, tag)
        self.drained(observed_mamba, mamba_pos, tag)


class HybridModel(RadixModel):
    """HybridRadixCache's semantics at page granularity: an optional snapshot slot at a node's
    end boundary, which a match truncates to and an insert attaches or dedups."""

    has_snapshots = True

    def match(self, ids: Sequence[int]) -> ExpMatch:
        groups = self.cover(ids)
        for depth, group in enumerate(reversed(groups)):
            if group.mamba is not None:
                if depth:
                    self.events["match.snapshot_truncation"] += 1
                return ExpMatch(len(group.end), self.trie.path_slots(group), group, group.mamba)
        self.events["match.no_snapshot"] += 1
        return ExpMatch(0, [], None, None)

    def insert(
        self, ids: Sequence[int], slots: Sequence[int], reused_len: int, mamba: int | None
    ) -> ExpInsert:
        assert mamba is not None, "the hybrid cache takes a snapshot with every insert"
        node, expected = self.commit(ids, slots, reused_len)
        expected.second_exists = node is None or node.mamba is not None
        if expected.second_exists:
            self.events["insert.snapshot_dedup"] += 1
        else:
            assert node is not None
            self.events["insert.snapshot_attach"] += 1
            node.mamba = mamba
        return expected

    def inc_lock(self, group: MGroup | None) -> None:
        if group is not None and group.mamba is not None:
            group.mamba_ref += 1
        super().inc_lock(group)

    def dec_lock(self, group: MGroup | None) -> None:
        if group is not None and group.mamba is not None and group.mamba_ref > 0:
            group.mamba_ref -= 1
        super().dec_lock(group)

    def cascadable(self, group: MGroup) -> bool:
        return group.mamba is None

    def evict_mamba(
        self, num_snapshots: int, observed_kv: Sequence[int], observed_mamba: Sequence[int]
    ) -> None:
        tag = "model.evict_mamba"
        candidates = [
            group
            for group in self.trie.group_list
            if group.mamba is not None and group.mamba_ref == 0
        ]
        freed, kv_pos, mamba_pos = 0, 0, 0
        while freed < num_snapshots and candidates:
            if mamba_pos >= len(observed_mamba):
                raise ModelMismatch(tag, f"evict_mamba stopped at {freed} with snapshots left")
            owners = [group for group in candidates if group.mamba == observed_mamba[mamba_pos]]
            if not owners:
                raise ModelMismatch(tag, f"freed snapshot {observed_mamba[mamba_pos]} not eligible")
            victim = owners[0]
            self.pick_victim(candidates, victim, tag)
            candidates.remove(victim)
            freed += 1
            if self.trie.is_leaf(victim) and victim.ref == 0:
                self.events["evict_mamba.leaf_free"] += 1
                kv_pos = self.take(observed_kv, kv_pos, victim, tag)
                mamba_pos = self.take_mamba(observed_mamba, mamba_pos, victim, tag)
                _, kv_pos, _ = self.cascade(
                    self.trie.remove(victim), observed_kv, kv_pos, candidates, tag, "evict_mamba"
                )
            else:
                self.events["evict_mamba.tombstone_in_place"] += 1
                mamba_pos = self.take_mamba(observed_mamba, mamba_pos, victim, tag)
        self.drained(observed_kv, kv_pos, tag)
        self.drained(observed_mamba, mamba_pos, tag)


# ------------------------------------------------------------------------ implementation side
@dataclass
class MatchOut:
    cached_len: int
    indices: list[int]
    node: RadixTreeNode
    second: int | None


class InsertOut(NamedTuple):
    matched_len: int
    # The hybrid cache's mamba_exist; None for the plain cache.
    second_exists: bool | None


@dataclass
class EvictOut:
    kv: list[int]
    mamba: list[int]


def iter_nodes(root: RadixTreeNode) -> list[tuple[RadixTreeNode, RadixTreeNode]]:
    """Every non-root node with its parent, depth first."""
    out: list[tuple[RadixTreeNode, RadixTreeNode]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children.values():
            out.append((child, node))
            stack.append(child)
    return out


def node_end_path(node: RadixTreeNode) -> Path:
    keys: list[list[int]] = []
    while not node.is_root():
        keys.append(node.key.tolist())
        node = node.parent
    return tuple(token for key in reversed(keys) for token in key)


def node_path_slots(node: RadixTreeNode) -> list[int]:
    values: list[list[int]] = []
    while not node.is_root():
        values.append(node.value.tolist())
        node = node.parent
    return [slot for value in reversed(values) for slot in value]


def ids_array(ids: Sequence[int]) -> np.ndarray:
    return np.asarray(list(ids), dtype=np.int32)


def tree_records(root: RadixTreeNode) -> list[tuple[Path, NodeRecord]]:
    """One record per node, keyed by its end boundary: the shape the model also emits."""
    return [
        (
            node_end_path(node),
            NodeRecord(
                node.length,
                tuple(node.value.tolist()),
                node.ref_count,
                node.mamba_value,
                node.mamba_ref_count,
            ),
        )
        for node, _ in iter_nodes(root)
    ]


class PlainAdapter:
    """The plain cache under test, asked in plain lists."""

    has_snapshots = False

    def __init__(self, cache: RadixPrefixCache) -> None:
        self.cache = cache
        self.page_size = cache.page_size

    def match(self, ids: Sequence[int]) -> MatchOut:
        match = self.cache.match_prefix(ids_array(ids))
        return MatchOut(match.cached_len, match.kv_indices.tolist(), match.node, None)

    def insert(self, ids: Sequence[int], slots: Sequence[int], mamba: int | None) -> InsertOut:
        assert mamba is None, "the plain cache takes no snapshot"
        prefix_len, _ = self.cache.insert(ids_array(ids), ids_array(slots))
        return InsertOut(prefix_len, None)

    def evict_full(self, num_tokens: int) -> EvictOut:
        return EvictOut(self.cache.evict(num_tokens).tolist(), [])

    def counters(self) -> dict[str, int]:
        return {
            "full_evictable": self.cache.evictable_size,
            "full_protected": self.cache.protected_size,
        }

    def nodes(self) -> list[RadixTreeNode]:
        return [node for node, _ in iter_nodes(self.cache.root)]

    def records(self) -> list[tuple[Path, NodeRecord]]:
        return tree_records(self.cache.root)


class HybridAdapter:
    """The hybrid cache under test, asked in plain lists."""

    has_snapshots = True

    def __init__(self, cache: HybridRadixCache) -> None:
        self.cache = cache
        self.page_size = cache.page_size

    def match(self, ids: Sequence[int]) -> MatchOut:
        match = self.cache.match_prefix(ids_array(ids))
        return MatchOut(match.cached_len, match.kv_indices.tolist(), match.node, match.mamba_value)

    def insert(self, ids: Sequence[int], slots: Sequence[int], mamba: int | None) -> InsertOut:
        assert mamba is not None, "the hybrid cache takes a snapshot with every insert"
        return InsertOut(*self.cache.insert(ids_array(ids), ids_array(slots), mamba))

    def evict_full(self, num_tokens: int) -> EvictOut:
        evicted = self.cache.evict_full(num_tokens)
        return EvictOut(evicted.kv_indices.tolist(), list(evicted.mamba_slots))

    def evict_mamba(self, num_snapshots: int) -> EvictOut:
        evicted = self.cache.evict_mamba(num_snapshots)
        return EvictOut(evicted.kv_indices.tolist(), list(evicted.mamba_slots))

    def counters(self) -> dict[str, int]:
        cache = self.cache
        return {
            "full_evictable": cache.full_evictable,
            "full_protected": cache.full_protected,
            "mamba_evictable": cache.mamba_evictable,
            "mamba_protected": cache.mamba_protected,
        }

    def nodes(self) -> list[RadixTreeNode]:
        return [node for node, _ in iter_nodes(self.cache.root)]

    def records(self) -> list[tuple[Path, NodeRecord]]:
        return tree_records(self.cache.root)


class SlotLedger:
    """Globally unique, never reused slot ids, and the book of what came back."""

    def __init__(self, base: int, name: str) -> None:
        self.name = name
        self.next_slot = base
        self.handed: list[int] = []
        self.handed_set: set[int] = set()
        self.free: set[int] = set()

    def take(self, count: int) -> list[int]:
        out = list(range(self.next_slot, self.next_slot + count))
        self.next_slot += count
        self.handed.extend(out)
        self.handed_set.update(out)
        return out

    def release(self, slots: Sequence[int]) -> None:
        for slot in slots:
            if slot not in self.handed_set:
                raise InvariantViolation(f"conservation.{self.name}", f"{slot} never handed out")
            if slot in self.free:
                raise InvariantViolation(f"conservation.{self.name}", f"double free of {slot}")
            self.free.add(slot)

    def in_use(self) -> set[int]:
        return self.handed_set - self.free


def check_conservation(name: str, live: Sequence[int], ledger: SlotLedger) -> None:
    """{live in the tree} + {handed back} == {handed out}; no slot in two places."""
    live_set = set(live)
    problems = (
        (".aliasing", [slot for slot, n in Counter(live).items() if n > 1], "owned twice"),
        ("", sorted(live_set & ledger.free), "live in the tree and handed back"),
        ("", sorted(ledger.handed_set - live_set - ledger.free), "leaked"),
        ("", sorted(live_set - ledger.handed_set), "live but never handed out"),
    )
    for suffix, bad, why in problems:
        if bad:
            raise InvariantViolation(f"conservation.{name}{suffix}", f"slots {bad[:8]} are {why}")


def check_prefix_closure(adapter: PlainAdapter | HybridAdapter) -> None:
    seen: set[int] = set()
    for node, parent in iter_nodes(adapter.cache.root):
        checks = (
            ("cycle", id(node) not in seen, "is reachable twice"),
            ("parent", node.parent is parent, "has a parent other than its container"),
            (
                "key",
                parent.children.get(page_key(node.key, adapter.page_size)) is node,
                "is not registered under its first page",
            ),
            ("page_align", node.length % adapter.page_size == 0, "is not whole pages"),
            ("key_value", len(node.key) == len(node.value), "has key and value lengths apart"),
        )
        for tag, ok, why in checks:
            if not ok:
                raise InvariantViolation(f"structure.{tag}", f"node {node_end_path(node)} {why}")
        seen.add(id(node))


def check_all(
    adapter: PlainAdapter | HybridAdapter, model: RadixModel, kv: SlotLedger, mamba: SlotLedger
) -> None:
    records = adapter.records()
    check_prefix_closure(adapter)
    recomputed = sizes([record for _, record in records], adapter.has_snapshots)
    if adapter.counters() != recomputed:
        raise InvariantViolation(
            "counters", f"maintained {adapter.counters()} vs raw nodes {recomputed}"
        )
    try:
        adapter.cache.check_integrity()
    except AssertionError as exc:
        raise InvariantViolation("check_integrity", str(exc)) from exc
    check_conservation("kv", [slot for _, record in records for slot in record.slots], kv)
    check_conservation(
        "mamba", [record.mamba for _, record in records if record.mamba is not None], mamba
    )
    implementation, reference = dict(records), dict(model.records())
    if set(implementation) != set(reference):
        raise InvariantViolation(
            "model.structure",
            f"only in cache: {sorted(set(implementation) - set(reference))[:4]}; "
            f"only in model: {sorted(set(reference) - set(implementation))[:4]}",
        )
    for end, record in implementation.items():
        if record != reference[end]:
            raise InvariantViolation("model.node", f"node {end}: {record} vs {reference[end]}")
    if adapter.counters() != model.counters():
        raise InvariantViolation("model.counters", f"{adapter.counters()} vs {model.counters()}")


# ------------------------------------------------------------------------ session
def page_ids(page_size: int, *labels: int) -> tuple[int, ...]:
    """Whole pages that share leading tokens (the lead repeats every three labels): the shape a
    token-keyed expectation gets wrong once page_size > 1."""
    out: list[int] = []
    for label in labels:
        if page_size == 1:
            out.append(label)
        else:
            out.extend([(label % 3) + 1] + [7] * (page_size - 2) + [label])
    return tuple(out)


class Session:
    """One cache and its reference model driven op by op; check() runs the battery.

    The cache is a hybrid one unless a RadixPrefixCache is passed in.
    """

    def __init__(
        self, page_size: int, cache: RadixPrefixCache | HybridRadixCache | None = None
    ) -> None:
        self.page_size = page_size
        self.adapter: PlainAdapter | HybridAdapter
        self.model: RadixModel
        if isinstance(cache, RadixPrefixCache):
            self.adapter = PlainAdapter(cache)
            self.model = RadixModel(page_size)
        else:
            self.adapter = HybridAdapter(HybridRadixCache(page_size) if cache is None else cache)
            self.model = HybridModel(page_size)
        self.kv = SlotLedger(1_000_000, "kv")
        self.mamba = SlotLedger(9_000_000, "mamba")

    def check(self) -> None:
        check_all(self.adapter, self.model, self.kv, self.mamba)

    @staticmethod
    def same(tag: str, what: str, got: object, want: object) -> None:
        if got != want:
            raise ModelMismatch(tag, f"{what}: cache says {got!r}, model says {want!r}")

    def do_match(self, ids: Sequence[int]) -> tuple[MatchOut, ExpMatch]:
        expected = self.model.match(ids)
        got = self.adapter.match(ids)
        query = f"match_prefix({tuple(ids)})"
        self.same("match.cached_len", query, got.cached_len, expected.cached_len)
        self.same("match.indices", query, got.indices, expected.indices)
        self.same("match.node", query, node_path_slots(got.node), expected.indices)
        self.same("match.second", query, got.second, expected.second)
        return got, expected

    def do_insert(
        self, ids: Sequence[int], *, slots: Sequence[int] | None = None, reused_len: int = 0
    ) -> tuple[InsertOut, ExpInsert]:
        if slots is None:
            slots = self.kv.take(len(ids))
        mamba = self.mamba.take(1)[0] if self.model.has_snapshots else None
        expected = self.model.insert(ids, slots, reused_len, mamba)
        got = self.adapter.insert(ids, slots, mamba)
        query = f"insert({tuple(ids)})"
        self.same("insert.matched_len", query, got.matched_len, expected.matched_len)
        self.same("insert.second_exists", query, got.second_exists, expected.second_exists)
        # Everything the caller keeps goes back to the ledgers.
        aligned = len(ids) // self.page_size * self.page_size
        self.kv.release(expected.dups + list(slots[aligned:]))
        if mamba is not None and expected.second_exists:
            self.mamba.release([mamba])
        return got, expected

    def do_lock(self, ids: Sequence[int]) -> tuple[RadixTreeNode, MGroup] | None:
        got, expected = self.do_match(ids)
        if expected.group is None:
            return None
        self.adapter.cache.inc_lock(got.node)
        self.model.inc_lock(expected.group)
        return got.node, expected.group

    def do_unlock(self, held: tuple[RadixTreeNode, MGroup] | None) -> None:
        if held is not None:
            self.adapter.cache.dec_lock(held[0])
            self.model.dec_lock(held[1])

    def do_evict_full(self, num_tokens: int) -> None:
        if isinstance(self.adapter, PlainAdapter):
            # RadixPrefixCache.evict asserts it is asked for no more than is evictable.
            num_tokens = min(num_tokens, self.adapter.cache.evictable_size)
        got = self.adapter.evict_full(num_tokens)
        self.model.evict_full(num_tokens, got.kv, got.mamba)
        self.kv.release(got.kv)
        self.mamba.release(got.mamba)

    def do_evict_mamba(self, num_snapshots: int) -> None:
        assert isinstance(self.adapter, HybridAdapter) and isinstance(self.model, HybridModel)
        got = self.adapter.evict_mamba(num_snapshots)
        self.model.evict_mamba(num_snapshots, got.kv, got.mamba)
        self.kv.release(got.kv)
        self.mamba.release(got.mamba)

    def do_request(self, ids: Sequence[int], prompt_pages: int) -> None:
        """Match a prompt, lock it, extend and commit, unlock: the supplied slots for the
        reused prefix are the tree's own, so the cache must neither free nor re-own them."""
        got, expected = self.do_match(tuple(ids[: prompt_pages * self.page_size]))
        held = None
        if expected.group is not None:
            self.adapter.cache.inc_lock(got.node)
            self.model.inc_lock(expected.group)
            held = (got.node, expected.group)
        slots = list(got.indices) + self.kv.take(len(ids) - got.cached_len)
        self.do_insert(ids, slots=slots, reused_len=got.cached_len)
        self.do_unlock(held)


__all__ = [
    "HarnessFailure",
    "HybridAdapter",
    "HybridModel",
    "InsertOut",
    "InvariantViolation",
    "ModelMismatch",
    "NodeRecord",
    "PlainAdapter",
    "RadixModel",
    "Session",
    "SlotLedger",
    "node_end_path",
    "node_path_slots",
    "page_ids",
]
