"""Chunk-window planning over image spans (rows, jobs, cache hits, multi-span items, a
repeated image across a cut) and the mrope positions of a batch."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.core import Req, SamplingParams
from ftmetal.mm.encoder_cache import EncoderCache
from ftmetal.mm.item import MMGatherRow, MMItem
from ftmetal.scheduler.mm import make_mrope_positions, mm_rows_after, plan_mm_batch, plan_mm_chunk


def image(item_hash: int, offsets: list[tuple[int, int]]) -> MMItem:
    feature = np.zeros((1, 1), dtype=np.float32)
    return MMItem(
        modality="image", hash=item_hash, pad_value=item_hash, offsets=offsets, feature=feature
    )


def encoded_cache(hashes: list[int]) -> EncoderCache:
    """A cache holding an encoding of each of hashes."""
    cache = EncoderCache()
    for item_hash in hashes:
        cache.register(item_hash, 0, 1)
        cache.put(item_hash, mx.zeros((1, 1)))
    return cache


def request(
    uid: int, cached_len: int, device_len: int, items: list[MMItem] | None, delta: int = 0
) -> Req:
    return Req(
        input_ids=np.zeros(device_len, dtype=np.int32),
        table_idx=0,
        cached_len=cached_len,
        output_len=4,
        uid=uid,
        sampling_params=SamplingParams(),
        mm_items=items,
        mrope_delta=delta,
    )


# (item offsets, chunk window, encoded hashes, whether the item is a job, whether it keeps its
# pixels, the gather plan as (row_lo, row_hi, position) rows)
CHUNK_CASES: list[
    tuple[list[tuple[int, int]], tuple[int, int], list[int], bool, bool, list[tuple[int, int, int]]]
] = [
    # One span cut by the chunk end, then its last rows.
    ([(10, 20)], (0, 16), [], True, True, [(0, 6, 10)]),
    ([(10, 20)], (16, 32), [], True, True, [(6, 10, 0)]),
    # The span is past the window: encoded by a later chunk.
    ([(10, 20)], (0, 8), [], False, True, []),
    # The span is entirely in the cached prefix.
    ([(10, 20)], (20, 40), [], False, False, []),
    # Encoded already: gather only.
    ([(10, 20)], (0, 40), [7], False, False, [(0, 10, 10)]),
    # Two spans in one window are one job; their rows follow each other.
    ([(2, 5), (8, 12)], (0, 10), [], True, True, [(0, 3, 2), (3, 5, 8)]),
    ([(2, 5), (8, 12)], (10, 20), [], True, True, [(5, 7, 0)]),
]


@pytest.mark.parametrize(
    ("offsets", "window", "encoded", "is_job", "keeps_pixels", "rows"), CHUNK_CASES
)
def test_chunk_plan_gathers_the_rows_inside_the_window(
    offsets: list[tuple[int, int]],
    window: tuple[int, int],
    encoded: list[int],
    is_job: bool,
    keeps_pixels: bool,
    rows: list[tuple[int, int, int]],
) -> None:
    item = image(7, offsets)
    jobs, plan = plan_mm_chunk(1, [item], window[0], window[1], encoded_cache(encoded))
    assert jobs == ([item] if is_job else [])
    assert (item.feature is not None) is keeps_pixels
    num_tokens = sum(end - start for start, end in offsets)
    assert plan == [MMGatherRow(1, 7, lo, hi, num_tokens, position) for lo, hi, position in rows]


# (item offsets, cached prefix, the rows prefill still gathers)
ROWS_AFTER_CASES: list[tuple[list[tuple[int, int]], int, int]] = [
    ([(2, 5), (8, 12)], 0, 7),
    # The hit ends inside the first span.
    ([(2, 5), (8, 12)], 4, 5),
    ([(2, 5), (8, 12)], 12, 0),
]


@pytest.mark.parametrize(("offsets", "cached_len", "rows"), ROWS_AFTER_CASES)
def test_rows_after_the_cached_prefix(
    offsets: list[tuple[int, int]], cached_len: int, rows: int
) -> None:
    assert mm_rows_after(image(1, offsets), cached_len) == rows


def test_repeated_image_survives_a_chunk_cut_between_its_occurrences() -> None:
    cache = EncoderCache()
    items = [image(7, [(0, 4)]), image(7, [(5, 9)])]
    for item in items:
        cache.register(item.hash, 1, mm_rows_after(item, 0))
    encoded: list[int] = []

    def run_chunk(lo: int, hi: int) -> list[MMItem]:
        jobs, plan = plan_mm_chunk(1, items, lo, hi, cache)
        for item in jobs:
            if not cache.has(item.hash):
                assert item.feature is not None
                encoded.append(item.hash)
                cache.put(item.hash, mx.ones((4, 8)))
            item.feature = None
        for row in plan:
            cache.get_slice(row.item_hash, row.row_lo, row.row_hi)
            cache.consume(row.item_hash, row.uid, row.row_hi - row.row_lo)
        return jobs

    # Both occurrences overlap the first window: one job, and the second drops its pixels.
    assert len(run_chunk(0, 6)) == 1
    assert items[1].feature is None
    # The second occurrence still has rows to gather, from the cache.
    assert cache.has(7)
    assert run_chunk(6, 10) == []
    assert encoded == [7] and not cache.has(7)


def test_batch_rows_follow_the_requests_in_batch_order() -> None:
    # Request 1: 6 tokens, an image on [2, 5); a text request; request 3: the chunk [4, 10) of
    # a prompt whose image spans [3, 8).
    reqs = [
        request(1, 0, 6, [image(7, [(2, 5)])]),
        request(2, 0, 3, None),
        request(3, 4, 10, [image(8, [(3, 8)])]),
    ]
    jobs, plan, rows = plan_mm_batch(reqs, EncoderCache())
    assert [job.hash for job in jobs] == [7, 8]
    assert plan == [MMGatherRow(1, 7, 0, 3, 3, 2), MMGatherRow(3, 8, 1, 5, 5, 0)]
    # Request 3 starts at batch row 9: its image rows 1..4 land on its first four tokens.
    assert rows == [2, 3, 4, 9, 10, 11, 12]


def test_mrope_positions_take_the_prompt_columns_then_index_plus_delta() -> None:
    prompt = request(1, 2, 5, [image(7, [(1, 3)])], delta=-3)
    prompt.mrope_positions_full = np.array(
        [[0, 1, 1, 4, 5, 6], [0, 1, 2, 4, 5, 6], [0, 1, 3, 4, 5, 6]], dtype=np.int32
    )
    decoding = request(2, 6, 7, [image(7, [(1, 3)])], delta=-3)
    decoding.mrope_positions_full = prompt.mrope_positions_full
    text = request(3, 0, 2, None)
    positions = make_mrope_positions([prompt, decoding, text])
    assert positions.dtype == np.int32 and positions.flags.c_contiguous
    assert positions.tolist() == [[1, 4, 5, 3, 0, 1], [2, 4, 5, 3, 0, 1], [3, 4, 5, 3, 0, 1]]
