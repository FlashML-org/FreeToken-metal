"""The hybrid radix prefix cache end to end on the tiny model.

A prompt sharing a snapshot boundary with an earlier one is admitted with that prefix
cached (KV pages reused, GDN state restored from the snapshot) and must still decode
greedily to what mlx-lm produces alone; so must every request while both currencies are
evicted under a small page and slot budget. The tiny model's greedy tokens hardly depend
on the GDN state, so a hit's prefill logits are also held to mlx-lm's at 1e-4.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import qwen3_5 as reference

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.scheduler.scheduler import Scheduler
from tiny_model import build_reference, reference_greedy, save_checkpoint

STEPS = 8


def make_scheduler(
    path: Path,
    *,
    page_size: int = 32,
    num_pages: int = 64,
    max_running_req: int = 2,
    max_extend_tokens: int = 512,
    linear_state_cache_ratio: float = 2.0,
) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max_running_req,
        page_size=page_size,
        num_page_override=num_pages,
        max_extend_tokens=max_extend_tokens,
        linear_state_cache_ratio=linear_state_cache_ratio,
    )
    return Scheduler(config, eos_token_ids={127})


def run_alone(scheduler: Scheduler, uid: int, prompt: list[int]) -> tuple[int, list[int]]:
    """One prompt to completion; returns the prefix its admission found cached and its tokens."""
    params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
    scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    cached_len = forward_input.batch.reqs[0].cached_len
    data = scheduler.forward(forward_input)
    while scheduler.runnable or data is not None:
        data = scheduler.overlap_loop(data)
    return cached_len, [event.token for event in scheduler.take_results()]


def assert_hit_logits_match(
    scheduler: Scheduler, model: reference.Model, prompt: list[int], cached_len: int
) -> None:
    """Admit prompt, restore its snapshot and run its prefill forward alone against mlx-lm.

    Leaves the scheduler mid-request: call it last.
    """
    params = SamplingParams(temperature=0.0, max_tokens=1)
    scheduler.add_request(99, np.array(prompt, dtype=np.int32), params)
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    batch = forward_input.batch
    assert batch.reqs[0].cached_len == cached_len
    scheduler.restore_linear_states(batch)
    logits = scheduler.engine.model.forward(batch)[0]
    expected = reference_greedy(model, prompt, 1)[0][-1]
    assert mx.allclose(logits, expected, atol=1e-4, rtol=1e-4)


def random_tokens(seed: int, count: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(1, 120, size=count)]


@pytest.mark.parametrize("page_size", [4, 32])
def test_same_prompt_twice_hits_and_decodes_the_same(tmp_path: Path, page_size: int) -> None:
    model = build_reference(seed=20)
    save_checkpoint(model, tmp_path, quantization=None)
    prompt = random_tokens(20, 150)
    _, expected = reference_greedy(model, prompt, STEPS)

    scheduler = make_scheduler(tmp_path, page_size=page_size, num_pages=2048 // page_size)
    assert run_alone(scheduler, 0, prompt) == (0, expected)
    # The prefill tracked the deepest 64-token boundary strictly inside the prompt.
    assert run_alone(scheduler, 1, prompt) == (128, expected)
    scheduler.cache_manager.check_integrity()
    assert_hit_logits_match(scheduler, model, prompt, 128)


def test_next_turn_resumes_from_the_finished_live_state(tmp_path: Path) -> None:
    """At page size 1 a finish donates its live state at any length: the next turn resumes
    there (157, off the conv ring's period) and tracks 64 tokens further from it."""
    model = build_reference(seed=28)
    save_checkpoint(model, tmp_path, quantization=None)
    first = random_tokens(28, 150)
    scheduler = make_scheduler(tmp_path, page_size=1, num_pages=2048)
    _, tokens = run_alone(scheduler, 0, first)
    assert tokens == reference_greedy(model, first, STEPS)[1]
    second = first + tokens[: STEPS - 1] + random_tokens(29, 100)
    assert run_alone(scheduler, 1, second) == (157, reference_greedy(model, second, STEPS)[1])
    third = second[:221] + random_tokens(30, 20)
    assert_hit_logits_match(scheduler, model, third, 221)


def test_prompts_sharing_a_snapshot_prefix_reuse_it(tmp_path: Path) -> None:
    model = build_reference(seed=21)
    save_checkpoint(model, tmp_path, quantization=None)
    shared = random_tokens(21, 128)
    first = shared + random_tokens(22, 20)  # its snapshot lands at 128
    second = shared + random_tokens(23, 100)  # hits 128, then tracks 192
    third = second[:192] + random_tokens(24, 30)  # hits the snapshot the hit committed
    scheduler = make_scheduler(tmp_path)
    for uid, (prompt, cached_len) in enumerate([(first, 0), (second, 128), (third, 192)]):
        _, expected = reference_greedy(model, prompt, STEPS)
        assert run_alone(scheduler, uid, prompt) == (cached_len, expected)
    scheduler.cache_manager.check_integrity()
    assert_hit_logits_match(scheduler, model, third, 192)


def test_identical_prompts_in_one_batch_dedup_their_commits(tmp_path: Path) -> None:
    model = build_reference(seed=25)
    save_checkpoint(model, tmp_path, quantization=None)
    prompt = random_tokens(25, 100)
    _, expected = reference_greedy(model, prompt, STEPS)
    scheduler = make_scheduler(tmp_path)
    params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
    for uid in (0, 1):
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events = scheduler.run_until_idle()
    for uid in (0, 1):
        assert [event.token for event in events if event.uid == uid] == expected
    scheduler.cache_manager.check_integrity()
    assert run_alone(scheduler, 2, prompt) == (64, expected)
    assert_hit_logits_match(scheduler, model, prompt, 64)


def test_chunked_prompt_commits_its_last_tracked_snapshot(tmp_path: Path) -> None:
    """Chunks of 128: [0, 128) tracks 64, [128, 256) tracks 192, and the last chunk of 44
    tokens tracks nothing, so the commit takes 192 from the chunk before it."""
    model = build_reference(seed=26)
    save_checkpoint(model, tmp_path, quantization=None)
    prompt = random_tokens(26, 300)
    _, expected = reference_greedy(model, prompt, STEPS)
    scheduler = make_scheduler(tmp_path, max_extend_tokens=128)
    assert run_alone(scheduler, 0, prompt) == (0, expected)
    assert run_alone(scheduler, 1, prompt) == (192, expected)
    # The hit's own prefill tracked 256 from its restored state.
    assert_hit_logits_match(scheduler, model, prompt, 256)


def test_eviction_under_small_budgets_keeps_outputs(tmp_path: Path) -> None:
    """16 pages and 9 GDN slots: each finished prompt leaves 4 pages and a snapshot behind,
    so admissions evict both currencies as the prompts keep coming."""
    model = build_reference(seed=27)
    save_checkpoint(model, tmp_path, quantization=None)
    prompts = [random_tokens(270 + i, 150) for i in range(6)]
    expected = [reference_greedy(model, prompt, STEPS)[1] for prompt in prompts]
    scheduler = make_scheduler(
        tmp_path, num_pages=16, max_running_req=1, linear_state_cache_ratio=0.0
    )
    pool = scheduler.engine.linear_state_pool
    assert pool is not None and pool.num_slots == 9
    for uid, prompt in enumerate(prompts):
        assert run_alone(scheduler, uid, prompt) == (0, expected[uid])
        scheduler.cache_manager.check_integrity()
    # The oldest prompt's snapshot was evicted; the newest one still hits.
    assert run_alone(scheduler, 6, prompts[0]) == (0, expected[0])
    assert run_alone(scheduler, 7, prompts[5]) == (128, expected[5])
    scheduler.cache_manager.check_integrity()
    assert_hit_logits_match(scheduler, model, prompts[5], 128)
