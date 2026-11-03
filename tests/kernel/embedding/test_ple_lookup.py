"""PleRowStore, PleService's n-gram hash and the ple_post / ple_wait handshake, on small
synthetic PLE tables.

A table is shards of three parts (packed weights uint32 [rows, 20], scales and biases bf16
[rows, 5]: one 160-wide 4-bit group-32 row) in safetensors files, each part a tensor of its
own, the parts of one shard spread over the files behind odd-sized filler tensors; its rows
are random bytes. The store's gathers must equal those bytes, cold and cached, with repeated
ids, at shard boundaries and on many threads. The service's hash must equal a numpy int64
reference. A chain of decode steps, each a post, a stand-in for the layers before the PLE
layer and a wait, must return each row's hashed table rows and advance its slot's context,
eagerly and replayed as a command graph. Then the failure paths: a shutdown releases a
waiting ple_wait and refuses later posts, the watchdog fails a post nobody answers, a
failed read and a corrupted sequence fail theirs, each with zero rows and check_healthy
raising.
"""

from __future__ import annotations

import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import numpy.typing as npt
import pytest

from ftmetal.command_graph import capture, replay
from ftmetal.kernel._ext import _C
from ftmetal.kernel.embedding.ple_lookup import PleRowParts, ple_post, ple_wait

Word = _C.PleMailboxWord
State = _C.MailboxState
Error = _C.MailboxError
ROW_BYTES = (80, 10, 10)
PART_NAMES = ("weight", "scales", "biases")
EOS = 248044
# The checkpoint's layer_multipliers: their products wrap in int64.
MULTIPLIERS = (23703573157769, 20109073645365, 8052911324071)
# 8 bigram and 8 trigram heads, laid end to end as the checkpoint's are, over 816 rows.
HEAD_SIZES = (19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83)
HEAD_OFFSETS = tuple(int(offset) for offset in np.cumsum((0, *HEAD_SIZES[:-1])))
# 855 rows; the heads use the first 816.
SHARD_ROWS = (131, 7, 263, 5, 449)
STALL_TIMEOUT_S = 10.0
# The largest token ids the vocabulary holds.
VOCAB_SIZE = 248320


@dataclass(frozen=True)
class Table:
    files: list[str]
    # int64 [shards, 3, 3]: each part's file index, offset of row 0 and row stride.
    parts: npt.NDArray[np.int64]
    shard_starts: list[int]
    # Each part's bytes of every row, uint8 [rows, row bytes].
    rows: tuple[npt.NDArray[np.uint8], ...]


def write_table(directory: Path, shard_rows: tuple[int, ...], num_files: int) -> Table:
    """Random rows in safetensors files; part p of shard s in file (3 * s + p) % num_files."""
    directory.mkdir()
    rng = np.random.default_rng(sum(shard_rows) * 31 + num_files)
    rows = tuple(
        rng.integers(0, 256, (sum(shard_rows), width), dtype=np.uint8) for width in ROW_BYTES
    )
    starts = [0, *np.cumsum(shard_rows).tolist()]
    contents: list[dict[str, mx.array]] = [
        {"filler": mx.zeros((1000 + 37 * i,), dtype=mx.uint8)} for i in range(num_files)
    ]
    for shard in range(len(shard_rows)):
        for part, name in enumerate(PART_NAMES):
            data = rows[part][starts[shard] : starts[shard + 1]]
            array = (
                mx.array(data.view(np.uint32))
                if part == 0
                else mx.array(data.view(np.uint16)).view(mx.bfloat16)
            )
            contents[(3 * shard + part) % num_files][f"shards.{shard}.{name}"] = array
    files: list[str] = []
    parts = np.zeros((len(shard_rows), 3, 3), dtype=np.int64)
    for index, content in enumerate(contents):
        path = directory / f"model-{index:05d}.safetensors"
        mx.save_safetensors(str(path), content)
        with open(path, "rb") as file:
            header_bytes = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(header_bytes))
        for name, entry in header.items():
            if name.startswith("shards."):
                _, shard, part_name = name.split(".")
                part = PART_NAMES.index(part_name)
                offset = 8 + header_bytes + entry["data_offsets"][0]
                parts[int(shard), part] = (index, offset, ROW_BYTES[part])
        files.append(str(path))
    return Table(files, parts, starts, rows)


def open_store(table: Table, *, cache_rows: int = 1024, io_threads: int = 4) -> _C.PleRowStore:
    return _C.PleRowStore(
        table.files, table.parts, table.shard_starts, list(ROW_BYTES), cache_rows, io_threads
    )


def gather(store: _C.PleRowStore, ids: list[int]) -> list[npt.NDArray[np.uint8]]:
    """The store's rows ids, each part uint8 [len(ids), row bytes]."""
    parts = [np.full((len(ids), width), 0xA5, dtype=np.uint8) for width in ROW_BYTES]
    store.gather(np.array(ids, dtype=np.int64), *parts)
    return parts


@dataclass(frozen=True)
class GatherCase:
    name: str
    ids: tuple[int, ...]
    cache_rows: int = 1024
    io_threads: int = 4
    passes: int = 2


SHARD_BOUNDARY_IDS = (0, 130, 131, 137, 138, 400, 401, 405, 406, 854)
RANDOM_IDS = tuple(np.random.default_rng(7).integers(0, sum(SHARD_ROWS), 3000).tolist())
GATHER_CASES = [
    # Every row once on the caller alone, then from the cache.
    GatherCase("every-row-caller-only", tuple(range(sum(SHARD_ROWS))), io_threads=0),
    GatherCase("shard-boundaries", SHARD_BOUNDARY_IDS),
    GatherCase("repeated-ids", (5,) * 40 + (854,) * 3 + (5,), io_threads=8),
    # Many threads, the cache far smaller than the rows: slots evicted mid-gather.
    GatherCase("random-many-threads-tiny-cache", RANDOM_IDS, cache_rows=4, io_threads=32),
    GatherCase("random-many-threads", RANDOM_IDS, cache_rows=256, io_threads=16),
    GatherCase("no-ids", (), passes=1),
]


@pytest.mark.parametrize("case", GATHER_CASES, ids=lambda case: case.name)
def test_gather_matches_the_table(tmp_path: Path, case: GatherCase) -> None:
    table = write_table(tmp_path / "table", SHARD_ROWS, num_files=3)
    store = open_store(table, cache_rows=case.cache_rows, io_threads=case.io_threads)
    assert store.num_rows == sum(SHARD_ROWS) and store.row_bytes == list(ROW_BYTES)
    ids = np.array(case.ids, dtype=np.int64)
    for _ in range(case.passes):
        counts = store.hits + store.misses
        parts = gather(store, list(case.ids))
        for part, rows in zip(parts, table.rows, strict=True):
            np.testing.assert_array_equal(part, rows[ids])
        assert store.hits + store.misses == counts + len(case.ids)


def test_concurrent_gathers_match_the_table(tmp_path: Path) -> None:
    """A prefill's gather and the coordinator's share one store and its IO threads."""
    table = write_table(tmp_path / "table", SHARD_ROWS, num_files=3)
    store = open_store(table, cache_rows=16, io_threads=8)
    rng = np.random.default_rng(3)
    id_lists = [rng.integers(0, sum(SHARD_ROWS), 500).tolist() for _ in range(6)]
    with ThreadPoolExecutor(len(id_lists)) as executor:
        results = list(executor.map(lambda ids: gather(store, ids), id_lists))
    for ids, parts in zip(id_lists, results, strict=True):
        for part, rows in zip(parts, table.rows, strict=True):
            np.testing.assert_array_equal(part, rows[ids])


def test_gathered_row_is_cached(tmp_path: Path) -> None:
    table = write_table(tmp_path / "table", SHARD_ROWS, num_files=2)
    store = open_store(table)
    gather(store, [401])
    assert (store.hits, store.misses) == (0, 1) and store.bytes_read >= sum(ROW_BYTES)
    bytes_read = store.bytes_read
    parts = gather(store, [401])
    assert (store.hits, store.misses, store.bytes_read) == (1, 1, bytes_read)
    for part, rows in zip(parts, table.rows, strict=True):
        np.testing.assert_array_equal(part[0], rows[401])


def test_gather_refuses_rows_outside_the_table(tmp_path: Path) -> None:
    store = open_store(write_table(tmp_path / "table", SHARD_ROWS, num_files=1))
    with pytest.raises(IndexError, match="row 855 is not in"):
        gather(store, [3, 855])
    with pytest.raises(IndexError, match="row -1 is not in"):
        gather(store, [-1])


def test_store_refuses_a_file_shorter_than_its_rows(tmp_path: Path) -> None:
    table = write_table(tmp_path / "table", SHARD_ROWS, num_files=1)
    os.truncate(table.files[0], os.path.getsize(table.files[0]) - 1)
    with pytest.raises(RuntimeError, match="ends before shard"):
        open_store(table)


def test_gather_from_a_truncated_file_raises(tmp_path: Path) -> None:
    table = write_table(tmp_path / "table", SHARD_ROWS, num_files=1)
    store = open_store(table)
    os.truncate(table.files[0], int(table.parts[:, :, 1].min()))
    with pytest.raises(
        RuntimeError, match=r"model-00000\.safetensors at .*: unexpected end of file"
    ):
        gather(store, [0, 500, 854])


def reference_ngram_ids(triples: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """int64 [n, heads] of (token, previous, before previous) triples [n, 3]."""
    token, previous, before_previous = triples.T
    before_previous = np.where(previous == EOS, EOS, before_previous)
    bigram = (token * MULTIPLIERS[0]) ^ (previous * MULTIPLIERS[1])
    trigram = bigram ^ (before_previous * MULTIPLIERS[2])
    half = len(HEAD_SIZES) // 2
    mixed = np.concatenate(
        [np.repeat(bigram[:, None], half, axis=1), np.repeat(trigram[:, None], half, axis=1)],
        axis=1,
    )
    return mixed % np.array(HEAD_SIZES) + np.array(HEAD_OFFSETS)


@dataclass(frozen=True)
class Lookup:
    """A service over a store, with its mailbox and staging buffers."""

    service: _C.PleService
    mailbox: mx.array
    staging: PleRowParts

    def words(self) -> npt.NDArray[np.uint32]:
        """A view of the mailbox the GPU and the coordinator share."""
        return np.frombuffer(memoryview(self.mailbox), dtype=np.uint32)


def start_lookup(
    store: _C.PleRowStore,
    *,
    run_coordinator: bool = True,
    stall_timeout_s: float = STALL_TIMEOUT_S,
) -> Lookup:
    heads = len(HEAD_SIZES)
    staging = PleRowParts(
        *(mx.zeros((_C.MAX_PLE_ROWS, heads, width), dtype=mx.uint8) for width in ROW_BYTES)
    )
    mailbox = mx.zeros((_C.PLE_MAILBOX_WORDS,), dtype=mx.uint32)
    mx.eval(mailbox, *staging)
    service = _C.PleService(
        mailbox,
        list(staging),
        store,
        list(MULTIPLIERS),
        list(HEAD_SIZES),
        list(HEAD_OFFSETS),
        EOS,
        stall_timeout_s,
        run_coordinator,
    )
    return Lookup(service, mailbox, staging)


HASH_TRIPLES = [
    (0, 0, 0),
    (1, 2, 3),
    # A fresh request's first token, and an eos after it.
    (17, EOS, EOS),
    (EOS, EOS, EOS),
    # A token after an eos: the token before the eos is not part of the trigram.
    (5, EOS, 17),
    (5, EOS, 248000),
    (5, 17, EOS),
    (VOCAB_SIZE - 1, VOCAB_SIZE - 2, VOCAB_SIZE - 3),
    # int32's largest: every product wraps.
    (2**31 - 1, 2**31 - 2, 2**31 - 3),
    (151643, 872, 12),
]


def test_ngram_ids_match_reference(tmp_path: Path) -> None:
    store = open_store(write_table(tmp_path / "table", SHARD_ROWS, num_files=1))
    lookup = start_lookup(store)
    assert lookup.service.num_heads == len(HEAD_SIZES)
    triples = np.array(HASH_TRIPLES, dtype=np.int64)
    ids = lookup.service.ngram_ids(triples)
    np.testing.assert_array_equal(ids, reference_ngram_ids(triples))
    assert ids[4].tolist() == ids[5].tolist(), "the token before an eos is not hashed"
    assert ((ids >= 0) & (ids < sum(HEAD_SIZES))).all()


def lookup_step(
    lookup: Lookup, input_ids: mx.array, slots: mx.array, context: mx.array
) -> tuple[mx.array, mx.array]:
    """(the step's table rows uint8 [num_rows, heads, row bytes of every part], context)."""
    posted = ple_post(input_ids, slots, context, lookup.mailbox)
    # The layers before the PLE layer, on the step's tokens.
    hidden = mx.cumsum(posted.input_ids.astype(mx.float32) * 0.5)
    parts = ple_wait(
        posted.ticket, lookup.mailbox, lookup.staging, input_ids.shape[0], after=[hidden]
    )
    return mx.concatenate(list(parts), axis=-1), posted.context


@dataclass(frozen=True)
class StepsCase:
    name: str
    slots: tuple[int, ...]
    num_slots: int
    num_steps: int
    is_replayed: bool = False


STEPS_CASES = [
    StepsCase("one-row", (0,), 1, 8),
    StepsCase("four-rows", (5, 2, 0, 3), 6, 6),
    # Padding rows share the sink slot 0 with nothing else.
    StepsCase("padded", (1, 2, 3, 0, 0), 4, 6),
    StepsCase("max-rows", tuple(range(_C.MAX_PLE_ROWS - 1, -1, -1)), _C.MAX_PLE_ROWS, 3),
    StepsCase("four-rows-replayed", (5, 2, 0, 3), 6, 6, is_replayed=True),
]


@pytest.mark.parametrize("case", STEPS_CASES, ids=lambda case: case.name)
def test_steps_return_hashed_rows_and_advance_context(tmp_path: Path, case: StepsCase) -> None:
    table = write_table(tmp_path / "table", SHARD_ROWS, num_files=3)
    lookup = start_lookup(open_store(table, cache_rows=64, io_threads=8))
    rng = np.random.default_rng(len(case.slots) * 101 + case.num_steps)
    expected_context = rng.integers(0, VOCAB_SIZE, (case.num_slots, 2))
    expected_context[rng.random(expected_context.shape) < 0.3] = EOS
    context = mx.array(expected_context.astype(np.int32))
    slots = mx.array(np.array(case.slots, dtype=np.int32))
    mx.eval(context, slots)
    graph = None
    for step in range(1, case.num_steps + 1):
        tokens = rng.integers(0, VOCAB_SIZE, len(case.slots))
        tokens[rng.random(len(case.slots)) < 0.25] = EOS
        input_ids = mx.array(tokens.astype(np.int32))
        mx.eval(input_ids)
        lookup.service.arm()
        if graph is None:
            rows, advanced = lookup_step(lookup, input_ids, slots, context)
            if case.is_replayed:
                graph = capture(
                    [rows, advanced],
                    [input_ids, slots],
                    [context],
                    [lookup.mailbox, *lookup.staging],
                )
            context = advanced
        else:
            rows, (context,) = replay(graph, [input_ids, slots], [context])
        mx.eval(rows, context)
        lookup.service.check_healthy()

        triples = np.stack(
            [tokens, expected_context[list(case.slots), 1], expected_context[list(case.slots), 0]],
            axis=1,
        ).astype(np.int64)
        for row, slot in enumerate(case.slots):
            expected_context[slot] = (triples[row, 1], tokens[row])
        ids = reference_ngram_ids(triples)
        expected_rows = np.concatenate([rows_of_part[ids] for rows_of_part in table.rows], axis=-1)
        np.testing.assert_array_equal(np.array(rows), expected_rows, err_msg=f"step {step}")
        np.testing.assert_array_equal(np.array(context), expected_context, err_msg=f"step {step}")
    words = lookup.words()
    assert words[Word.SEQUENCE] == words[Word.ACKNOWLEDGED] == case.num_steps
    assert words[Word.STATE] == State.IDLE and words[Word.ERROR] == Error.NONE
    assert words[Word.NUM_ROWS] == len(case.slots)


def one_row_step(lookup: Lookup, token: int) -> tuple[mx.array, mx.array]:
    """A one-row step on a fresh context: (rows, ticket), unevaluated."""
    context = mx.full((1, 2), EOS, dtype=mx.int32)
    posted = ple_post(
        mx.array([token], dtype=mx.int32), mx.array([0], dtype=mx.int32), context, lookup.mailbox
    )
    parts = ple_wait(posted.ticket, lookup.mailbox, lookup.staging, 1)
    return mx.concatenate(list(parts), axis=-1), posted.ticket


def wait_for_post(lookup: Lookup) -> None:
    deadline = time.monotonic() + 10.0
    while lookup.words()[Word.STATE] != State.POSTED:
        assert time.monotonic() < deadline, "the step never posted"
        time.sleep(0.001)


def test_shutdown_releases_a_waiting_step(tmp_path: Path) -> None:
    store = open_store(write_table(tmp_path / "table", SHARD_ROWS, num_files=1))
    lookup = start_lookup(store, run_coordinator=False, stall_timeout_s=600.0)
    rows, ticket = one_row_step(lookup, 11)
    mx.async_eval(rows, ticket)
    wait_for_post(lookup)
    lookup.service.shutdown()
    mx.eval(rows, ticket)
    assert ticket.item() == 1 and not np.array(rows).any()
    words = lookup.words()
    assert words[Word.STATE] == State.SHUTDOWN and words[Word.ERROR] == Error.SHUTDOWN
    with pytest.raises(RuntimeError, match="shut down"):
        lookup.service.check_healthy()
    # Nobody answers after a shutdown: a later step posts nothing and gets zero rows.
    rows, ticket = one_row_step(lookup, 12)
    mx.eval(rows, ticket)
    assert ticket.item() == 0 and not np.array(rows).any()
    assert words[Word.SEQUENCE] == 1 and words[Word.STATE] == State.SHUTDOWN


def test_watchdog_fails_an_unanswered_post(tmp_path: Path) -> None:
    store = open_store(write_table(tmp_path / "table", SHARD_ROWS, num_files=1))
    lookup = start_lookup(store, run_coordinator=False, stall_timeout_s=0.5)
    rows, ticket = one_row_step(lookup, 11)
    mx.eval(rows, ticket)
    assert ticket.item() == 1 and not np.array(rows).any()
    words = lookup.words()
    assert words[Word.STATE] == State.IDLE and words[Word.ACKNOWLEDGED] == 1
    assert words[Word.ERROR] == Error.TIMEOUT
    with pytest.raises(RuntimeError, match="PLE post 1 was left unanswered"):
        lookup.service.check_healthy()


def test_failed_read_fails_the_post(tmp_path: Path) -> None:
    table = write_table(tmp_path / "table", SHARD_ROWS, num_files=1)
    lookup = start_lookup(open_store(table))
    os.truncate(table.files[0], int(table.parts[:, :, 1].min()))
    rows, _ = one_row_step(lookup, 11)
    mx.eval(rows)
    assert not np.array(rows).any()
    words = lookup.words()
    assert words[Word.STATE] == State.IDLE and words[Word.ACKNOWLEDGED] == 1
    assert words[Word.ERROR] == Error.IO
    with pytest.raises(RuntimeError, match="PLE post 1: .* unexpected end of file"):
        lookup.service.check_healthy()
    # Every later post fails at once with the first failure's error.
    rows, _ = one_row_step(lookup, 12)
    mx.eval(rows)
    assert not np.array(rows).any()
    assert words[Word.ACKNOWLEDGED] == 2 and words[Word.ERROR] == Error.IO


def test_corrupted_sequence_is_a_protocol_error(tmp_path: Path) -> None:
    lookup = start_lookup(open_store(write_table(tmp_path / "table", SHARD_ROWS, num_files=1)))
    words = lookup.words()
    words[Word.SEQUENCE] = 41
    rows, _ = one_row_step(lookup, 11)
    mx.eval(rows)
    assert not np.array(rows).any()
    assert words[Word.ACKNOWLEDGED] == 42 and words[Word.ERROR] == Error.PROTOCOL
    with pytest.raises(RuntimeError, match="protocol error: PLE post 42 follows post 0"):
        lookup.service.check_healthy()


def test_sequence_wraps_past_the_refused_ticket(tmp_path: Path) -> None:
    lookup = start_lookup(open_store(write_table(tmp_path / "table", SHARD_ROWS, num_files=1)))
    expected, _ = one_row_step(lookup, 11)
    mx.eval(expected)
    words = lookup.words()
    words[Word.SEQUENCE] = words[Word.ACKNOWLEDGED] = 2**32 - 1
    rows, ticket = one_row_step(lookup, 11)
    mx.eval(rows, ticket)
    assert ticket.item() == 1 and np.array_equal(np.array(rows), np.array(expected))
    assert words[Word.ACKNOWLEDGED] == 1
    lookup.service.check_healthy()


def test_service_refuses_heads_outside_the_table(tmp_path: Path) -> None:
    store = open_store(write_table(tmp_path / "table", (400,), num_files=1))
    with pytest.raises(ValueError, match="head 10's rows fall outside the table's 400"):
        start_lookup(store)
