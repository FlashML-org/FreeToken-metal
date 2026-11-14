"""Scheduler behaviour under many requests, aborts and page pressure on the tiny model.

Every completed request must still decode greedily to what mlx-lm produces alone,
and every resource (pages, table rows, GDN slots) must be back in its free list or
held evictable by the prefix cache.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.message.backend import (
    BaseBackendMsg,
    BatchBackendMsg,
    ExitMsg,
    FlushCacheBackendMsg,
    UserMsg,
)
from ftmetal.message.tokenizer import (
    BaseTokenizerMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    FlushCacheResultMsg,
    FlushStatus,
)
from ftmetal.scheduler.scheduler import Scheduler, TokenEvent
from tiny_model import build_reference, reference_greedy, save_checkpoint


def make_scheduler(path: Path, max_running_req: int, num_pages: int) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max_running_req,
        page_size=4,
        num_page_override=num_pages,
        max_extend_tokens_override=32,
    )
    return Scheduler(config, eos_token_ids={127})


def assert_all_freed(scheduler: Scheduler) -> None:
    scheduler.cache_manager.check_integrity()
    assert scheduler.table_manager.available_size == scheduler.config.max_running_req
    pool = scheduler.engine.ctx.linear_state_pool
    assert pool is not None
    assert scheduler.cache_manager.mamba_available_size == pool.num_slots - 1


def tokens_by_uid(events: list[TokenEvent], uids: list[int]) -> list[list[int]]:
    return [[event.token for event in events if event.uid == uid] for uid in uids]


def test_requests_beyond_max_running_queue_and_match(tmp_path: Path) -> None:
    model = build_reference(seed=10)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(10)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in (9, 41, 17, 33, 5)]
    steps = [6, 11, 3, 8, 14]
    expected = [reference_greedy(model, p, s)[1] for p, s in zip(prompts, steps)]

    scheduler = make_scheduler(tmp_path, max_running_req=2, num_pages=64)
    for uid, (prompt, n) in enumerate(zip(prompts, steps)):
        params = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events = scheduler.run_until_idle()
    assert tokens_by_uid(events, list(range(len(prompts)))) == expected
    finishes = [event.finish_reason for event in events if event.finished]
    assert finishes == ["length"] * len(prompts)
    assert_all_freed(scheduler)


def test_abort_pending_request_never_runs(tmp_path: Path) -> None:
    model = build_reference(seed=12)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(12)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in (15, 7)]
    steps = 5
    expected = reference_greedy(model, prompts[0], steps)[1]

    scheduler = make_scheduler(tmp_path, max_running_req=1, num_pages=64)
    for uid, prompt in enumerate(prompts):
        params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    data = scheduler.overlap_loop(None)
    scheduler.abort_request(1)
    events = scheduler.take_results()
    while scheduler.runnable or data is not None:
        data = scheduler.overlap_loop(data)
        events += scheduler.take_results()
    assert {event.uid for event in events} == {0}
    assert tokens_by_uid(events, [0]) == [expected]
    assert_all_freed(scheduler)


def test_page_pressure_admits_requests_one_after_another(tmp_path: Path) -> None:
    model = build_reference(seed=13)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(13)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in (22, 18, 26)]
    steps = 6
    expected = [reference_greedy(model, p, steps)[1] for p in prompts]

    # 9 pages of 4 tokens: room for one request (prompt + output) at a time.
    scheduler = make_scheduler(tmp_path, max_running_req=3, num_pages=9)
    for uid, prompt in enumerate(prompts):
        params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events = scheduler.run_until_idle()
    assert tokens_by_uid(events, [0, 1, 2]) == expected
    assert_all_freed(scheduler)


def test_abort_in_flight_frees_at_the_drain_and_the_row_is_reusable(tmp_path: Path) -> None:
    """An abort of a request whose step is in flight marks it; the drain frees it. The new
    request admitted in between must not see its pages, row or GDN slots recycled early."""
    model = build_reference(seed=14)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(14)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in (14, 23, 19)]
    steps = 8
    expected = [reference_greedy(model, p, steps)[1] for p in prompts]

    scheduler = make_scheduler(tmp_path, max_running_req=2, num_pages=64)
    for uid in (0, 1):
        params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompts[uid], dtype=np.int32), params)
    data = None
    events: list[TokenEvent] = []
    for _ in range(3):
        data = scheduler.overlap_loop(data)
        events += scheduler.take_results()
    assert data is not None and any(req.uid == 0 for req in data.forward_input.batch.reqs)
    rows_before = scheduler.table_manager.available_size
    pages_before = scheduler.cache_manager.available_size
    scheduler.abort_request(0)
    # In flight: only marked, nothing freed yet.
    assert scheduler.table_manager.available_size == rows_before
    assert scheduler.cache_manager.available_size == pages_before
    # The freed row is what uid 2 needs (max_running_req=2, uid 1 holds the other one).
    scheduler.add_request(
        2,
        np.array(prompts[2], dtype=np.int32),
        SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True),
    )
    data = scheduler.overlap_loop(data)
    events += scheduler.take_results()
    # This iteration scheduled before it drained: uid 2 could not take the row yet.
    assert scheduler.table_manager.available_size == rows_before + 1
    data = scheduler.overlap_loop(data)
    events += scheduler.take_results()
    assert scheduler.table_manager.available_size == rows_before  # uid 2 admitted on it
    aborted_events = len([event for event in events if event.uid == 0])
    while scheduler.runnable or data is not None:
        data = scheduler.overlap_loop(data)
        events += scheduler.take_results()
    assert len([event for event in events if event.uid == 0]) == aborted_events
    assert tokens_by_uid(events, [1, 2]) == expected[1:]
    assert_all_freed(scheduler)


def test_abort_mid_chunked_prefill_frees_once_and_the_prompt_still_runs(tmp_path: Path) -> None:
    model = build_reference(seed=15)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(15)
    prompt = [int(t) for t in rng.integers(1, 120, size=70)]  # > max_extend_tokens: chunked
    steps = 5
    expected = reference_greedy(model, prompt, steps)[1]

    scheduler = make_scheduler(tmp_path, max_running_req=1, num_pages=64)
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    scheduler.add_request(0, np.array(prompt, dtype=np.int32), params)
    data = scheduler.overlap_loop(None)
    assert data is not None and data.forward_input.batch.is_prefill
    assert data.forward_input.batch.reqs[0].extend_len < len(prompt)
    scheduler.abort_request(0)
    assert scheduler.table_manager.available_size == 0  # in flight: deferred
    data = scheduler.overlap_loop(data)
    assert scheduler.table_manager.available_size == 1  # drained: freed exactly once
    assert data is None and not scheduler.runnable
    assert scheduler.take_results() == []
    scheduler.cache_manager.check_integrity()

    # The same prompt again may hit the aborted chunk's prefix and must still be exact.
    scheduler.add_request(1, np.array(prompt, dtype=np.int32), params)
    events = scheduler.run_until_idle()
    assert tokens_by_uid(events, [1]) == [expected]
    assert_all_freed(scheduler)


# (timings_per_token, stop strings, whether every event carries the timings or only the last)
TIMINGS_CASES: list[tuple[bool, list[str], bool]] = [
    (False, [], False),
    (True, [], True),
    # The detokenizer may end it on any token.
    (False, ["never"], True),
]


@pytest.mark.parametrize(("timings_per_token", "stop_strs", "every_event"), TIMINGS_CASES)
def test_timings_count_the_prompt_and_every_committed_token(
    tmp_path: Path, timings_per_token: bool, stop_strs: list[str], every_event: bool
) -> None:
    save_checkpoint(build_reference(seed=10), tmp_path, quantization=None)
    scheduler = make_scheduler(tmp_path, max_running_req=1, num_pages=64)
    prompt = np.random.default_rng(10).integers(1, 120, size=130).astype(np.int32)
    cached_lens = [0, 132]
    for uid, cached_len in enumerate(cached_lens):
        # The first turn finishes at 132 cached tokens, a page boundary, so the tree keeps them
        # for the next turn, which goes on past them.
        params = SamplingParams(
            max_tokens=3,
            ignore_eos=True,
            stop_strs=stop_strs,
            timings_per_token=timings_per_token,
        )
        scheduler.add_request(uid, prompt, params)
        events = scheduler.run_until_idle()
        assert [event.timings is not None for event in events] == [every_event] * 2 + [True]
        timings = [event.timings for event in events if event.timings is not None]
        assert [t["predicted_n"] for t in timings] == ([1, 2, 3] if every_event else [3])
        last = timings[-1]
        assert (last["cache_n"], last["prompt_n"]) == (cached_len, len(prompt) - cached_len)
        assert min(last["queue_ms"], last["prompt_ms"], last["predicted_ms"]) >= 0
        generated = np.array([event.token for event in events], dtype=np.int32)
        prompt = np.concatenate([prompt, generated, prompt[:5]])


# (a request queued with the flush, the flush's timeout, its status)
FLUSH_CASES: list[tuple[bool, float, FlushStatus]] = [
    (False, 0.0, "ok"),
    (True, 0.0, "busy"),
    # Waits for the request to finish.
    (True, 60.0, "ok"),
]


@pytest.mark.parametrize(("with_request", "timeout_s", "status"), FLUSH_CASES)
def test_flush_empties_the_prefix_cache_once_idle(
    tmp_path: Path, with_request: bool, timeout_s: float, status: FlushStatus
) -> None:
    save_checkpoint(build_reference(seed=10), tmp_path, quantization=None)
    scheduler = make_scheduler(tmp_path, max_running_req=1, num_pages=64)
    # Past a 64-token boundary: the hybrid tree keeps a snapshot there.
    prompt = np.random.default_rng(10).integers(1, 120, size=130).astype(np.int32)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    scheduler.add_request(0, prompt, params)
    scheduler.run_until_idle()
    assert scheduler.cache_manager.evictable_size > 0
    recv_backend: queue.Queue[BaseBackendMsg] = queue.Queue()
    send_tokenizer: queue.Queue[BaseTokenizerMsg | None] = queue.Queue()
    thread = threading.Thread(
        target=scheduler.run_forever, args=(recv_backend, send_tokenizer), daemon=True
    )
    thread.start()
    request = UserMsg(
        uid=1, input_ids=prompt[::-1].copy(), sampling_params=params, reuse_len=len(prompt)
    )
    flush = FlushCacheBackendMsg(request_id="flush", timeout_s=timeout_s)
    recv_backend.put(BatchBackendMsg(data=[request, flush] if with_request else [flush]))
    results: list[FlushCacheResultMsg] = []
    is_request_done = not with_request
    try:
        while not (results and is_request_done):
            batch = send_tokenizer.get(timeout=60)
            assert isinstance(batch, BatchTokenizerMsg)
            results += [msg for msg in batch.data if isinstance(msg, FlushCacheResultMsg)]
            is_request_done |= any(
                isinstance(msg, DetokenizeMsg) and msg.finished for msg in batch.data
            )
    finally:
        recv_backend.put(ExitMsg())
        thread.join(timeout=60)
    assert [(result.request_id, result.status) for result in results] == [("flush", status)]
    if status == "ok":
        assert scheduler.cache_manager.evictable_size == 0
    assert_all_freed(scheduler)
