"""MoeLookup and its mailbox handshake with the coordinator, on offload_harness's identifying
checkpoints: every packed weight word of an expert's projection names (layer, projection,
expert).

A chain of layers, each a lookup of the step's routed ids followed by gathers of the gate_up and
down bank rows at its slots and at its slot map, runs one step per eval; each layer's ids depend
on the previous layer's gathers, as a router's do. The cache holds a fraction of the experts, so
most layers miss and the coordinator reads victims between the layers of one command buffer.
Every real pair's and used expert's rows must name it, the slots and the cache's slot_of and
last_use must equal the reference LRU's for the same calls (the GPU's hit marks reach the victim
choice, and no slot the request uses is a victim), padding rows and unused experts must read a
slot the layer uses, and the counts must add up; with one pool for every layer, the layers of a
step share its slots. The chain runs eagerly, and captured once as a command graph (banks and
slot_of external, mailbox, last_use and hit_stats pools, moe_step and the ids step inputs) and
replayed. Then the handshake's edges: no real rows post nothing, a probe round trip, a hit in
the least recently used slot keeps it, a failed read and a layer routing more experts than its
pool holds fail their requests, a corrupted sequence is a protocol error, a shutdown releases a
waiting lookup, also one whose post overwrote the shutdown's refusal, the watchdog fails a
request nobody answers, and with no CPU to answer at all the lookup gives up past the mailbox's
spin limit, after which nothing posts. Last moe_lookup_issue, step by step on a filled cache:
the resident pairs get their slots, an absent expert's pairs -1 - expert, the padding rows the
first pair's slot or, while it is absent, PENDING_FALLBACK_SLOT; issued names the request, which
the coordinator answers with the misses' slots in slot_of; a request without misses or real rows
posts nothing, nor one after the shutdown word.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import numpy.typing as npt
import pytest

from ftmetal.command_graph import capture, replay
from ftmetal.kernel._ext import _C
from ftmetal.kernel.moe.moe_lookup import moe_lookup_issue
from ftmetal.moe.offload_cache import STALL_TIMEOUT_S, OffloadMoeCache
from ftmetal.moe.slot_pools import MoePoolGrouping
from offload_harness import CheckpointSpec, Naming, ReferenceLru, build_cache, expert_code

Word = _C.MailboxWord
State = _C.MailboxState
Kind = _C.MoeLookupKind
# The smallest affine 4-bit group-64 expert.
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 64
# A summary row: the slot, then the min and max word of the gate, up and down parts.
SUMMARY_COLUMNS = 7
# A spin limit the tests' unanswered waits reach in milliseconds.
TEST_SPIN_LIMIT = 100_000


@dataclass(frozen=True)
class Case:
    num_layers: int
    num_experts: int
    top_k: int
    num_rows: int
    cache_size: int
    num_steps: int
    kind: _C.MoeLookupKind = Kind.DECODE
    naming: Naming = "qwen"
    is_filled: bool = True
    is_replayed: bool = False
    grouping: MoePoolGrouping = "per-layer"

    def __str__(self) -> str:
        return (
            f"{self.naming}-L{self.num_layers}-E{self.num_experts}-k{self.top_k}"
            f"-T{self.num_rows}-S{self.cache_size}-{self.grouping}-{self.kind.name.lower()}"
            f"{'' if self.is_filled else '-empty'}{'-replay' if self.is_replayed else '-eager'}"
        )


CASES = [
    # Most layers miss every step; the real rows run from none to all.
    Case(4, 32, 4, 3, 56, 1000),
    Case(4, 32, 4, 3, 56, 1000, is_replayed=True),
    # Qwen3.6's 256 experts, top-8.
    Case(3, 256, 8, 4, 288, 200),
    # Gemma4's 128 experts, top-8, in its weight names.
    Case(3, 128, 8, 8, 200, 200, naming="gemma4"),
    # A copy-miss prefill of 128 rows on an empty cache: requests name a whole layer.
    Case(2, 256, 8, 128, 520, 8, kind=Kind.PREFILL_COPY_MISS, is_filled=False),
    # One pool: a step's 4 x 12 experts fit its 56 slots, which every layer's misses share.
    Case(4, 32, 4, 3, 56, 1000, grouping="unified"),
    Case(4, 32, 4, 3, 56, 1000, is_replayed=True, grouping="unified"),
    Case(3, 256, 8, 4, 288, 200, grouping="unified"),
]


def mailbox_words(cache: OffloadMoeCache) -> npt.NDArray[np.uint32]:
    """A view of the mailbox the GPU and the coordinator share."""
    return np.asarray(memoryview(cache.mailbox)).view(np.uint32)


def chain(cache: OffloadMoeCache, case: Case, topk_ids: mx.array, moe_step: mx.array) -> mx.array:
    """uint32 [layers, T * top_k + E, SUMMARY_COLUMNS]: per layer, a summary row of each pair's
    slot in its pool, then of each expert's slot map entry."""
    after_previous = mx.zeros((1, 1), dtype=mx.int32)
    summaries = []
    for layer_id in range(case.num_layers):
        gate_up = cache.layer_banks(layer_id)["gate_up_proj.weight"]
        down = cache.layer_banks(layer_id)["down_proj.weight"]
        half = gate_up.shape[1] // 2
        slots, slot_map = cache.lookup(
            layer_id, topk_ids[layer_id] + after_previous, moe_step, case.kind
        )
        rows = mx.concatenate([slots.reshape(-1), slot_map])
        gate_up_rows = mx.take(gate_up, rows, axis=0)
        columns = [rows.astype(mx.uint32)]
        for part in (gate_up_rows[:, :half], gate_up_rows[:, half:], mx.take(down, rows, axis=0)):
            flat = part.reshape(rows.shape[0], -1)
            columns += [flat.min(axis=1), flat.max(axis=1)]
        summary = mx.stack(columns, axis=1)
        summaries.append(summary)
        after_previous = (summary[:1, :1] * 0).astype(mx.int32)
    return mx.stack(summaries)


def expected_row(layer_id: int, expert: int, slot: int) -> list[int]:
    codes = [expert_code(layer_id, projection, expert) for projection in range(3)]
    return [slot, *(code for code in codes for _ in range(2))]


def check_step(
    case: Case,
    summary: npt.NDArray[np.uint32],
    topk_ids: npt.NDArray[np.int32],
    num_real_rows: int,
    step: int,
    reference: ReferenceLru,
) -> tuple[list[int], list[int]]:
    """Check one step's summary against the reference LRU's calls; returns each layer's
    distinct experts and loads."""
    num_pairs = case.num_rows * case.top_k
    distinct_counts: list[int] = []
    load_counts: list[int] = []
    for layer_id in range(case.num_layers):
        real = topk_ids[layer_id, :num_real_rows]
        distinct = sorted({int(expert) for expert in real.reshape(-1)})
        slot_of: dict[int, int] = {}
        loads: list[tuple[int, int]] = []
        if distinct:
            slots, loads = reference.resolve(layer_id, distinct, step)
            # The lookups return slots of the layer's pool.
            first_slot = reference.pool_slots(layer_id).start
            slot_of = {expert: slot - first_slot for expert, slot in zip(distinct, slots)}
        distinct_counts.append(len(distinct))
        load_counts.append(len(loads))
        pair_rows = summary[layer_id, :num_pairs]
        expert_rows = summary[layer_id, num_pairs:]
        pair_experts = topk_ids[layer_id].reshape(-1).tolist()
        used_slots = set(slot_of.values())
        for i in range(num_pairs):
            expert = pair_experts[i]
            if i < num_real_rows * case.top_k:
                expected = expected_row(layer_id, expert, slot_of[expert])
                assert pair_rows[i].tolist() == expected, (step, layer_id, i)
            elif used_slots:
                assert int(pair_rows[i, 0]) in used_slots, (step, layer_id, i)
            else:
                assert 0 <= int(pair_rows[i, 0]) < len(reference.pool_slots(layer_id))
        for expert in range(case.num_experts):
            if expert in slot_of:
                expected = expected_row(layer_id, expert, slot_of[expert])
                assert expert_rows[expert].tolist() == expected, (step, layer_id, expert)
            elif used_slots:
                assert int(expert_rows[expert, 0]) in used_slots, (step, layer_id, expert)
    return distinct_counts, load_counts


@pytest.mark.parametrize("case", CASES, ids=str)
def test_chain_matches_reference(tmp_path: Path, case: Case) -> None:
    spec = CheckpointSpec(
        num_layers=case.num_layers,
        num_experts=case.num_experts,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        naming=case.naming,
        is_identifying=True,
    )
    cache, _ = build_cache(
        tmp_path / "checkpoint",
        spec,
        cache_size=case.cache_size,
        prefill_overlap=False,
        grouping=case.grouping,
    )
    fill: tuple[int, ...] = ()
    if case.is_filled:
        cache.fill_initial()
        slot_of = np.array(cache.slot_of)
        fill = tuple(int(np.flatnonzero(slot_of == slot)[0]) for slot in range(cache.cache_size))
    layout = cache.slot_layout
    reference = ReferenceLru(case.num_experts, layout.layer_pools, layout.pool_bounds, fill)
    shape = (case.num_layers, case.num_rows, case.top_k)

    graph = None
    if case.is_replayed:
        moe_step = mx.zeros((2,), dtype=mx.int32)
        topk_ids = mx.zeros(shape, dtype=mx.int32)
        mx.eval(moe_step, topk_ids)
        pools = [cache.mailbox, cache.last_use, cache.hit_stats]
        summary = chain(cache, case, topk_ids, moe_step)
        graph = capture(
            [summary, cache.mailbox, cache.last_use, cache.hit_stats],
            [moe_step, topk_ids],
            pools,
            [*cache.banks.values(), cache.slot_of],
        )

    rng = np.random.default_rng(case.num_experts * 7 + case.num_rows)
    distinct_totals = np.zeros(case.num_layers, dtype=np.int64)
    load_totals = np.zeros(case.num_layers, dtype=np.int64)
    num_requests = 0
    for step in range(1, case.num_steps + 1):
        num_real_rows = int(rng.integers(0, case.num_rows + 1))
        ids = np.argsort(rng.random((*shape[:2], case.num_experts)), axis=2)[..., : case.top_k]
        ids = ids.astype(np.int32)
        moe_step = mx.array([num_real_rows, step], dtype=mx.int32)
        topk_ids = mx.array(ids)
        cache.arm()
        if graph is None:
            summary = chain(cache, case, topk_ids, moe_step)
        else:
            mx.eval(moe_step, topk_ids)
            summary, pools = replay(
                graph, [moe_step, topk_ids], [cache.mailbox, cache.last_use, cache.hit_stats]
            )
            cache.mailbox, cache.last_use, cache.hit_stats = pools
        mx.eval(summary, cache.mailbox, cache.last_use, cache.hit_stats)
        cache.check_healthy()
        distinct, loads = check_step(case, np.array(summary), ids, num_real_rows, step, reference)
        distinct_totals += distinct
        load_totals += loads
        num_requests += sum(1 for count in loads if count > 0)

    expected_slot_of = [-1] * (case.num_layers * case.num_experts)
    for slot, flat_id in enumerate(reference.id_of_slot):
        if flat_id >= 0:
            expected_slot_of[flat_id] = slot
    assert np.array(cache.slot_of).tolist() == expected_slot_of
    assert np.array(cache.last_use).tolist() == reference.last_use
    stats = cache.stats()
    num_lookups = case.num_steps + (1 if case.is_replayed else 0)
    assert stats.lookups == [num_lookups] * case.num_layers
    assert stats.misses == load_totals.tolist()
    assert all(
        (served > 0) == (requests > 0) for served, requests in zip(stats.serve_ns, stats.requests)
    )
    assert (np.array(stats.lookup_hits) + load_totals).tolist() == distinct_totals.tolist()
    words = mailbox_words(cache)
    assert words[Word.SEQUENCE] == words[Word.ACKNOWLEDGED] == num_requests
    assert words[Word.STATE] == State.IDLE
    assert words[Word.ERROR] == _C.MailboxError.NONE


def small_cache(
    tmp_path: Path, *, run_coordinator: bool = True, stall_timeout_s: float = STALL_TIMEOUT_S
) -> OffloadMoeCache:
    """An empty cache of 8 slots over 2 layers of 8 experts."""
    spec = CheckpointSpec(
        num_layers=2,
        num_experts=8,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        is_identifying=True,
    )
    cache, _ = build_cache(
        tmp_path / "checkpoint",
        spec,
        cache_size=8,
        prefill_overlap=False,
        stall_timeout_s=stall_timeout_s,
        run_coordinator=run_coordinator,
    )
    return cache


def lookup(
    cache: OffloadMoeCache, topk_ids: list[list[int]], num_real_rows: int, step: int
) -> mx.array:
    """Layer 0's slots, unevaluated."""
    slots, _ = cache.lookup(
        0,
        mx.array(topk_ids, dtype=mx.int32),
        mx.array([num_real_rows, step], dtype=mx.int32),
        Kind.DECODE,
    )
    return slots


def test_no_real_rows_post_nothing(tmp_path: Path) -> None:
    cache = small_cache(tmp_path)
    slots = lookup(cache, [[0, 1], [2, 3]], 0, 1)
    mx.eval(slots, cache.mailbox, cache.last_use, cache.hit_stats)
    words = mailbox_words(cache)
    assert words[Word.SEQUENCE] == 0 and words[Word.STATE] == State.IDLE
    assert (np.array(cache.slot_of) == -1).all() and (np.array(cache.last_use) == 0).all()
    assert ((np.array(slots) >= 0) & (np.array(slots) < cache.cache_size)).all()
    stats = cache.stats()
    assert (stats.lookups, stats.lookup_hits, stats.misses) == ([1, 0], [0, 0], [0, 0])


def test_probe_round_trip(tmp_path: Path) -> None:
    cache = small_cache(tmp_path)
    cache.probe()
    words = mailbox_words(cache)
    assert words[Word.SEQUENCE] == words[Word.ACKNOWLEDGED] == 1
    assert words[Word.STATE] == State.IDLE and words[Word.ERROR] == _C.MailboxError.NONE
    assert words[Word.KIND] == Kind.PROBE and words[Word.NUM_IDS] == 0
    stats = cache.stats()
    assert (stats.lookups, stats.misses) == ([0, 0], [0, 0])


def test_hit_in_the_least_recently_used_slot_keeps_it(tmp_path: Path) -> None:
    cache = small_cache(tmp_path)
    cache.fill_initial()
    slot_of = np.array(cache.slot_of)
    assert slot_of[:4].tolist() == [3, 2, 1, 0], "layer 0's experts 0-3 fill its pool"
    # Step 1 uses every resident expert but layer 0's expert 0, which becomes the least
    # recently used; step 2's request hits it and misses expert 4.
    mx.eval(lookup(cache, [[1, 2], [3, 1]], 2, 1), cache.mailbox)
    layer_1_slots, _ = cache.lookup(
        1, mx.array([[0, 1], [2, 3]], dtype=mx.int32), mx.array([2, 1], dtype=mx.int32), Kind.DECODE
    )
    mx.eval(layer_1_slots, cache.mailbox)
    slots = lookup(cache, [[0, 4]], 1, 2)
    gate_up = mx.take(cache.layer_banks(0)["gate_up_proj.weight"], slots.reshape(-1), axis=0)
    mx.eval(gate_up, cache.mailbox)
    slot_0, slot_4 = np.array(slots).reshape(-1).tolist()
    assert slot_0 == 3 and slot_4 != 3
    codes = np.array(gate_up)[:, 0, 0].tolist()
    assert codes == [expert_code(0, 0, 0), expert_code(0, 0, 4)]
    assert mailbox_words(cache)[Word.NUM_LOADED] == 1
    cache.check_healthy()


def test_failed_read_fails_the_request(tmp_path: Path) -> None:
    cache = small_cache(tmp_path)
    last_file = tmp_path / "checkpoint" / "model-00001.safetensors"
    os.truncate(last_file, last_file.stat().st_size // 2)
    # Layer 1's experts lie in the truncated file.
    slots, _ = cache.lookup(
        1, mx.array([[6, 7]], dtype=mx.int32), mx.array([1, 1], dtype=mx.int32), Kind.DECODE
    )
    mx.eval(slots, cache.mailbox)
    words = mailbox_words(cache)
    assert words[Word.STATE] == State.IDLE and words[Word.ACKNOWLEDGED] == 1
    assert words[Word.ERROR] == _C.MailboxError.IO
    with pytest.raises(RuntimeError, match="model-00001.safetensors at .*: unexpected end of file"):
        cache.check_healthy()


def test_layer_beyond_its_pool_fails_the_request(tmp_path: Path) -> None:
    cache = small_cache(tmp_path)
    # Layer 0's pool is 4 of the 8 slots: a step routing 5 of its experts cannot fit.
    mx.eval(lookup(cache, [[0, 1, 2, 3], [4, 0, 1, 2]], 2, 1), cache.mailbox)
    words = mailbox_words(cache)
    assert words[Word.SEQUENCE] == words[Word.ACKNOWLEDGED] == 1
    assert words[Word.ERROR] == _C.MailboxError.CAPACITY
    with pytest.raises(RuntimeError, match="pool 0's 4 slots cannot hold layer 0's 5 experts"):
        cache.check_healthy()


def test_corrupted_sequence_is_a_protocol_error(tmp_path: Path) -> None:
    cache = small_cache(tmp_path)
    words = mailbox_words(cache)
    words[Word.SEQUENCE] = 41
    mx.eval(lookup(cache, [[0, 1]], 1, 1), cache.mailbox)
    assert words[Word.STATE] == State.IDLE and words[Word.ACKNOWLEDGED] == 42
    assert words[Word.ERROR] == _C.MailboxError.PROTOCOL
    with pytest.raises(
        RuntimeError, match="protocol error: lookup request 42 .* follows request 0"
    ):
        cache.check_healthy()
    # Every later request fails at once with the first failure's error.
    mx.eval(lookup(cache, [[2, 3]], 1, 2), cache.mailbox)
    assert words[Word.ACKNOWLEDGED] == 43 and words[Word.ERROR] == _C.MailboxError.PROTOCOL


def test_shutdown_releases_a_waiting_lookup(tmp_path: Path) -> None:
    cache = small_cache(tmp_path, run_coordinator=False, stall_timeout_s=600.0)
    words = mailbox_words(cache)
    slots = lookup(cache, [[0, 1]], 1, 1)
    mx.async_eval(slots, cache.mailbox)
    deadline = time.monotonic() + 10.0
    while words[Word.STATE] != State.POSTED:
        assert time.monotonic() < deadline, "the lookup never posted"
        time.sleep(0.001)
    cache.shutdown()
    mx.eval(slots, cache.mailbox)
    assert words[Word.STATE] == State.SHUTDOWN
    assert words[Word.ERROR] == _C.MailboxError.SHUTDOWN
    with pytest.raises(RuntimeError, match="shut down"):
        cache.check_healthy()
    # Nobody answers after a shutdown: a later lookup posts nothing.
    mx.eval(lookup(cache, [[2, 3]], 1, 2), cache.mailbox)
    assert words[Word.SEQUENCE] == 1 and words[Word.STATE] == State.SHUTDOWN


def test_shutdown_word_releases_a_post_that_overwrote_the_refusal(tmp_path: Path) -> None:
    """A lookup that read the mailbox just before a shutdown posts anyway, overwriting the
    SHUTDOWN state: the shutdown word, which only the CPU writes, still releases it."""
    cache = small_cache(tmp_path, run_coordinator=False, stall_timeout_s=600.0)
    words = mailbox_words(cache)
    slots = lookup(cache, [[0, 1]], 1, 1)
    mx.async_eval(slots, cache.mailbox)
    deadline = time.monotonic() + 10.0
    while words[Word.STATE] != State.POSTED:
        assert time.monotonic() < deadline, "the lookup never posted"
        time.sleep(0.001)
    # The shutdown as the waiting lookup sees it once its post has overwritten the state.
    words[Word.SHUTDOWN] = 1
    mx.eval(slots, cache.mailbox)
    assert words[Word.STATE] == State.SHUTDOWN and words[Word.SEQUENCE] == 1
    # Nothing posts once the word is set, whatever the state says.
    words[Word.STATE] = State.IDLE
    mx.eval(lookup(cache, [[2, 3]], 1, 2), cache.mailbox)
    assert words[Word.SEQUENCE] == 1


def test_watchdog_fails_an_unanswered_request(tmp_path: Path) -> None:
    cache = small_cache(tmp_path, run_coordinator=False, stall_timeout_s=0.5)
    mx.eval(lookup(cache, [[0, 1]], 1, 1), cache.mailbox)
    words = mailbox_words(cache)
    assert words[Word.STATE] == State.IDLE and words[Word.ACKNOWLEDGED] == 1
    assert words[Word.ERROR] == _C.MailboxError.TIMEOUT
    with pytest.raises(RuntimeError, match="unanswered"):
        cache.check_healthy()


def test_lookup_gives_up_past_its_spin_limit(tmp_path: Path) -> None:
    """Neither coordinator nor watchdog answers, as when the CPU is gone: the lookup gives up after
    the mailbox's spin limit, its misses fall back, nothing later posts and check_healthy
    raises."""
    cache = small_cache(tmp_path, run_coordinator=False, stall_timeout_s=600.0)
    words = mailbox_words(cache)
    assert words[Word.SPIN_LIMIT] == _C.MAILBOX_SPIN_LIMIT
    words[Word.SPIN_LIMIT] = TEST_SPIN_LIMIT
    slots = lookup(cache, [[0, 1]], 1, 1)
    mx.eval(slots, cache.mailbox)
    assert np.array(slots).tolist() == [[0, 0]]
    assert words[Word.ABANDONED] == 1
    assert words[Word.SEQUENCE] == 1 and words[Word.STATE] == State.POSTED
    with pytest.raises(RuntimeError, match="gave up waiting for the CPU"):
        cache.check_healthy()
    mx.eval(lookup(cache, [[2, 3]], 1, 2), cache.mailbox)
    assert words[Word.SEQUENCE] == 1


@dataclass(frozen=True)
class IssueStep:
    """One issued lookup of layer 0 and what it must return: each pair's resident expert, -1 -
    its absent expert, FIRST for the first pair's slot or PENDING for PENDING_FALLBACK_SLOT; the
    misses posted (0: none)."""

    topk_ids: list[list[int]]
    num_real_rows: int
    expected: list[list[int | str]]
    num_misses: int


FIRST = "first"
PENDING = "pending"
# After fill_initial layer 0's pool holds its experts 0-3, expert e in slot 3 - e.
ISSUE_STEPS = [
    # Experts 4 and 5 absent; padding rows take the first pair's slot, expert 1's.
    IssueStep([[1, 4], [2, 5], [6, 7]], 2, [[1, -5], [2, -6], [FIRST, FIRST]], 2),
    # The first pair's expert absent: the padding row's slot is left to the waiters.
    IssueStep([[6, 1], [0, 2]], 1, [[-7, 1], [PENDING, PENDING]], 1),
    # Every expert resident, then no real row: nothing posted.
    IssueStep([[1, 6]], 1, [[1, 6]], 0),
    IssueStep([[3, 7], [0, 1]], 0, [[FIRST, FIRST], [FIRST, FIRST]], 0),
]


def issue(cache: OffloadMoeCache, step: IssueStep, step_number: int) -> tuple[mx.array, mx.array]:
    """Layer 0's issued slots and issued words, evaluated."""
    outputs = moe_lookup_issue(
        mx.array(step.topk_ids, dtype=mx.int32),
        mx.array([step.num_real_rows, step_number], dtype=mx.int32),
        cache.slot_of,
        cache.mailbox,
        cache.last_use,
        cache.hit_stats,
        layer_id=0,
        slot_base=0,
    )
    cache.set_table_arrays([outputs.mailbox, outputs.last_use, outputs.hit_stats])
    mx.eval(outputs.slots, outputs.issued, cache.table_arrays())
    return outputs.slots, outputs.issued


def test_issue_leaves_absent_pairs_to_the_waiters(tmp_path: Path) -> None:
    cache = small_cache(tmp_path)
    cache.fill_initial()
    words = mailbox_words(cache)
    num_posted = 0
    for step_number, step in enumerate(ISSUE_STEPS, start=1):
        slot_of = np.array(cache.slot_of)
        resident = {expert: int(slot_of[expert]) for expert in range(8) if slot_of[expert] >= 0}
        cache.arm()
        slots, issued = issue(cache, step, step_number)
        first = resident.get(step.topk_ids[0][0], 0) if step.num_real_rows > 0 else 0

        def expected_slot(entry: int | str) -> int:
            if entry == FIRST:
                return first
            elif entry == PENDING:
                return _C.PENDING_FALLBACK_SLOT
            else:
                assert isinstance(entry, int)
                return resident[entry] if entry >= 0 else entry

        expected = [[expected_slot(entry) for entry in row] for row in step.expected]
        assert np.array(slots).tolist() == expected, step_number
        num_posted += step.num_misses > 0
        sequence = num_posted if step.num_misses > 0 else 0
        assert np.array(issued).tolist() == [sequence, step.num_misses], step_number
        assert words[Word.SEQUENCE] == num_posted, step_number
        hits = {e for row in step.topk_ids[: step.num_real_rows] for e in row} & resident.keys()
        last_use = np.array(cache.last_use)
        assert all(last_use[resident[expert]] == step_number for expert in hits), step_number
        deadline = time.monotonic() + 10.0
        while words[Word.ACKNOWLEDGED] != num_posted:
            assert time.monotonic() < deadline, f"request {num_posted} was never answered"
            time.sleep(0.001)
        cache.check_healthy()
        # The answer reads the absent experts into slots.
        absent = [-1 - e for row in step.expected for e in row if isinstance(e, int) and e < 0]
        assert (np.array(cache.slot_of)[absent] >= 0).all(), step_number
    # Nobody returns an issued request's answer to IDLE.
    assert words[Word.STATE] == State.DONE
    stats = cache.stats()
    assert stats.lookups[0] == len(ISSUE_STEPS) and stats.misses[0] == 3


def test_issue_posts_nothing_after_the_shutdown_word(tmp_path: Path) -> None:
    cache = small_cache(tmp_path, run_coordinator=False, stall_timeout_s=600.0)
    words = mailbox_words(cache)
    words[Word.SHUTDOWN] = 1
    step = IssueStep([[0, 1]], 1, [[-1, -2]], 0)
    slots, issued = issue(cache, step, 1)
    assert np.array(slots).tolist() == [[-1, -2]]
    assert np.array(issued).tolist() == [0, 0]
    assert words[Word.SEQUENCE] == 0 and words[Word.STATE] == State.IDLE
