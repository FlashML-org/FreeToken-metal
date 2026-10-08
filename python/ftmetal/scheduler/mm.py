"""Scheduler-side multimodal planning: the embedding rows a chunk gathers, the items it
encodes, and the mrope positions of a batch.

Port of FreeToken scheduler/mm.py, with _make_mrope_positions of scheduler/scheduler.py.
Divergences: no mm_chunk_end or cut_image_spans and no block ends (bidirectional image
blocks are Gemma4's, which serves no images); gather rows are MMGatherRows; the positions
are a host numpy array; an item this request will not encode drops its pixels at planning
(FreeToken frees them only after an encode, which a cache or prefix hit never runs).
"""

from __future__ import annotations

import numpy as np

from ftmetal.core import Req
from ftmetal.mm.encoder_cache import EncoderCache
from ftmetal.mm.item import MMGatherRow, MMItem


def mm_rows_after(item: MMItem, cached_len: int) -> int:
    """Embedding rows of the item that prefill gathers once the first cached_len tokens are
    skipped."""
    return sum(max(0, span_hi - max(span_lo, cached_len)) for span_lo, span_hi in item.offsets)


def plan_mm_chunk(
    uid: int,
    items: list[MMItem],
    window_lo: int,
    window_hi: int,
    encoder_cache: EncoderCache,
) -> tuple[list[MMItem], list[MMGatherRow]]:
    """Encoder jobs and the gather plan for the items overlapping the chunk window
    [window_lo, window_hi); a repeated image is one job. Drops the pixels of every item this
    request will not encode."""
    jobs: list[MMItem] = []
    queued: set[int] = set()
    plan: list[MMGatherRow] = []
    for item in items:
        needs_encode = item.hash not in queued and not encoder_cache.has(item.hash)
        # Its rows are cached (held for this request until it gathers them), queued by an
        # earlier occurrence, or all behind the window; the pixels share unified memory.
        if not needs_encode or mm_rows_after(item, window_lo) == 0:
            item.feature = None
        row_base = 0
        for span_lo, span_hi in item.offsets:
            lo, hi = max(span_lo, window_lo), min(span_hi, window_hi)
            if lo < hi:
                if needs_encode:
                    jobs.append(item)
                    queued.add(item.hash)
                    needs_encode = False
                plan.append(
                    MMGatherRow(
                        uid=uid,
                        item_hash=item.hash,
                        row_lo=row_base + lo - span_lo,
                        row_hi=row_base + hi - span_lo,
                        num_tokens=item.num_tokens,
                        position=lo - window_lo,
                    )
                )
            row_base += span_hi - span_lo
    return jobs, plan


def plan_mm_batch(
    reqs: list[Req], encoder_cache: EncoderCache
) -> tuple[list[MMItem], list[MMGatherRow], list[int]]:
    """Jobs, plan and the batch row of every gathered embedding row, over the reqs in batch
    order (each spans [cached_len, device_len))."""
    jobs: list[MMItem] = []
    plan: list[MMGatherRow] = []
    rows: list[int] = []
    offset = 0
    for req in reqs:
        if req.mm_items:
            req_jobs, req_plan = plan_mm_chunk(
                req.uid, req.mm_items, req.cached_len, req.device_len, encoder_cache
            )
            jobs.extend(req_jobs)
            plan.extend(req_plan)
            for gather in req_plan:
                start = offset + gather.position
                rows.extend(range(start, start + gather.row_hi - gather.row_lo))
        offset += req.extend_len
    return jobs, plan, rows


def make_mrope_positions(reqs: list[Req]) -> np.ndarray:
    """int32 [3, N] rope rows: an image request's prompt tokens use their precomputed
    columns, everything else is the sequence index plus the request's delta."""
    columns: list[np.ndarray] = []
    for req in reqs:
        full = req.mrope_positions_full
        if full is not None and req.device_len <= full.shape[1]:
            columns.append(full[:, req.cached_len : req.device_len])
        else:
            row = np.arange(req.cached_len, req.device_len, dtype=np.int32) + req.mrope_delta
            columns.append(np.broadcast_to(row, (3, len(row))))
    return np.ascontiguousarray(np.concatenate(columns, axis=1), dtype=np.int32)


__all__ = ["make_mrope_positions", "mm_rows_after", "plan_mm_batch", "plan_mm_chunk"]
