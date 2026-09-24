"""Scheduler behaviour under many requests, aborts and page pressure on the tiny model.

Every completed request must still decode greedily to what mlx-lm produces alone,
and every resource (pages, table rows, GDN slots) must be back in its free list.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.scheduler.scheduler import Scheduler, TokenEvent
from tiny_model import build_reference, reference_greedy, save_checkpoint


def make_scheduler(path: Path, max_running_req: int, num_pages: int) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max_running_req,
        page_size=4,
        num_page_override=num_pages,
        max_extend_tokens=32,
    )
    return Scheduler(config, eos_token_ids={127})


def assert_all_freed(scheduler: Scheduler) -> None:
    scheduler.cache_manager.check_integrity()
    assert scheduler.table_manager.available_size == scheduler.config.max_running_req
    pool = scheduler.engine.linear_state_pool
    assert pool is not None and pool.num_free_slots == pool.num_slots - 1


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


def test_abort_running_request_frees_it_and_leaves_the_other_intact(tmp_path: Path) -> None:
    model = build_reference(seed=11)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(11)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in (12, 20)]
    steps = 10
    expected = reference_greedy(model, prompts[1], steps)[1]

    scheduler = make_scheduler(tmp_path, max_running_req=2, num_pages=64)
    for uid, prompt in enumerate(prompts):
        params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    data = None
    events: list[TokenEvent] = []
    for _ in range(4):
        data = scheduler.overlap_loop(data)
        events += scheduler.take_results()
    aborted_before = len([event for event in events if event.uid == 0])
    scheduler.abort_request(0)
    while scheduler.runnable or data is not None:
        data = scheduler.overlap_loop(data)
        events += scheduler.take_results()
    assert len([event for event in events if event.uid == 0]) == aborted_before
    assert tokens_by_uid(events, [1]) == [expected]
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
