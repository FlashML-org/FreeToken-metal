"""The radix prefix caches end to end on the tiny model.

A prompt sharing a snapshot boundary with an earlier one is admitted with that prefix
cached (KV pages reused, GDN state restored from the snapshot) and must still decode
greedily to what mlx-lm produces alone; so must every request while both currencies are
evicted under a small page and slot budget. The tiny model's greedy tokens hardly depend
on the GDN state, so a hit's prefill logits are also held to mlx-lm's at 1e-4. Without
linear layers the engine takes the plain radix cache, held to the same contract. A GDN
snapshot cache smaller than one conversation's snapshots fails at startup. A prompt is
reported admitted once, with the prefix its first chunk found cached.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import qwen3_5 as reference

from ftmetal.attention.linear import SNAPSHOT_STRIDE
from ftmetal.core import SamplingParams
from ftmetal.engine.config import CacheTypeChoice, EngineConfig
from ftmetal.message.tokenizer import PromptAdmittedMsg
from ftmetal.scheduler.scheduler import Scheduler
from tiny_model import (
    TINY_FULL_ATTENTION_CONFIG,
    build_reference,
    reference_greedy,
    save_checkpoint,
    uncached_greedy,
)

STEPS = 8


def make_scheduler(
    path: Path,
    *,
    page_size: int = 32,
    num_pages: int = 64,
    max_running_req: int = 2,
    max_extend_tokens: int = 512,
    linear_state_checkpoints: int | None = None,
    cache_type: CacheTypeChoice = "radix",
) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max_running_req,
        page_size=page_size,
        num_page_override=num_pages,
        max_extend_tokens_override=max_extend_tokens,
        linear_state_checkpoints=linear_state_checkpoints,
        cache_type=cache_type,
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


def prefill_logits(scheduler: Scheduler, prompt: list[int]) -> tuple[int, mx.array]:
    """Admit prompt, restore its snapshot and run its prefill forward alone; returns the
    prefix its admission found cached and the last position's logits.

    Leaves the scheduler mid-request: call it last.
    """
    params = SamplingParams(temperature=0.0, max_tokens=1)
    scheduler.add_request(99, np.array(prompt, dtype=np.int32), params)
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    batch = forward_input.batch
    scheduler.restore_linear_states(batch)
    return batch.reqs[0].cached_len, scheduler.engine.model.forward(batch)[0]


def assert_hit_logits_match(
    scheduler: Scheduler, model: reference.Model, prompt: list[int], cached_len: int
) -> None:
    """The prompt's prefill hits cached_len and its logits match mlx-lm's."""
    hit_len, logits = prefill_logits(scheduler, prompt)
    assert hit_len == cached_len
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
    # The prefill tracked the deepest stride boundary strictly inside the prompt.
    snapshot = SNAPSHOT_STRIDE * ((len(prompt) - 1) // SNAPSHOT_STRIDE)
    assert run_alone(scheduler, 1, prompt) == (snapshot, expected)
    scheduler.cache_manager.check_integrity()
    assert_hit_logits_match(scheduler, model, prompt, snapshot)


def test_next_turn_resumes_from_the_finished_live_state(tmp_path: Path) -> None:
    """At page size 1 a finish donates its live state at any length: the next turn resumes
    there (157, off the conv ring's period) and tracks one stride past it (221)."""
    model = build_reference(seed=28)
    save_checkpoint(model, tmp_path, quantization=None)
    first = random_tokens(28, 150)
    scheduler = make_scheduler(tmp_path, page_size=1, num_pages=2048)
    _, tokens = run_alone(scheduler, 0, first)
    assert tokens == reference_greedy(model, first, STEPS)[1]
    second = first + tokens[: STEPS - 1] + random_tokens(29, 100)
    assert run_alone(scheduler, 1, second) == (157, reference_greedy(model, second, STEPS)[1])
    third = second[:256] + random_tokens(30, 20)
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


def test_next_turn_survives_a_divergence_just_before_the_prompt_end(tmp_path: Path) -> None:
    """Qwen's template re-renders the previous turn without its "<think>\n", so the next
    turn diverges two tokens before the previous prompt's end. The snapshot sits at the
    deepest 64-token stride boundary (128 for a 150-token prompt), before the divergence,
    so the turn resumes there instead of from the previous prefill chunk."""
    model = build_reference(seed=31)
    save_checkpoint(model, tmp_path, quantization=None)
    first = random_tokens(31, 150)
    scheduler = make_scheduler(tmp_path, page_size=1, num_pages=2048)
    _, tokens = run_alone(scheduler, 0, first)
    assert tokens == reference_greedy(model, first, STEPS)[1]
    second = first[:-2] + random_tokens(32, 2) + tokens + random_tokens(33, 40)
    assert run_alone(scheduler, 1, second) == (128, reference_greedy(model, second, STEPS)[1])
    scheduler.cache_manager.check_integrity()


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
    """Chunks of 128 at page 32: [0, 128) tracks 64, [128, 256) tracks 192, and the last
    chunk of 44 tokens holds no stride boundary, so the carried 192 commits; the finish at
    308 is not page-aligned and donates nothing. The hit's own 108-token prefill tracks
    one more stride at 256."""
    model = build_reference(seed=26)
    save_checkpoint(model, tmp_path, quantization=None)
    prompt = random_tokens(26, 300)
    _, expected = reference_greedy(model, prompt, STEPS)
    scheduler = make_scheduler(tmp_path, max_extend_tokens=128)
    assert run_alone(scheduler, 0, prompt) == (0, expected)
    assert run_alone(scheduler, 1, prompt) == (192, expected)
    # Each prompt is admitted once, at its first chunk, with the hit that chunk found.
    assert scheduler.prompt_admissions == [
        PromptAdmittedMsg(uid=0, prompt_tokens=300, cached_tokens=0),
        PromptAdmittedMsg(uid=1, prompt_tokens=300, cached_tokens=192),
    ]
    assert_hit_logits_match(scheduler, model, prompt, 256)


def test_eviction_under_small_budgets_keeps_outputs(tmp_path: Path) -> None:
    """16 pages and 9 GDN slots: each finished prompt leaves 4 pages and a snapshot behind,
    so admissions evict both currencies as the prompts keep coming."""
    model = build_reference(seed=27)
    save_checkpoint(model, tmp_path, quantization=None)
    prompts = [random_tokens(270 + i, 150) for i in range(6)]
    expected = [reference_greedy(model, prompt, STEPS)[1] for prompt in prompts]
    scheduler = make_scheduler(
        tmp_path, num_pages=16, max_running_req=1, linear_state_checkpoints=4
    )
    pool = scheduler.engine.ctx.linear_state_pool
    assert pool is not None and pool.num_slots == 9
    for uid, prompt in enumerate(prompts):
        assert run_alone(scheduler, uid, prompt) == (0, expected[uid])
        scheduler.cache_manager.check_integrity()
    # The oldest prompt's snapshot was evicted; the newest one still hits.
    assert run_alone(scheduler, 6, prompts[0]) == (0, expected[0])
    assert run_alone(scheduler, 7, prompts[5]) == (128, expected[5])
    scheduler.cache_manager.check_integrity()
    assert_hit_logits_match(scheduler, model, prompts[5], 128)


def test_plain_radix_reuses_a_shared_prefix_without_linear_layers(tmp_path: Path) -> None:
    """No linear layers, no GDN state pool: the engine takes the plain radix cache. A prompt
    sharing a prefix with an earlier one reuses its pages up to the page boundary below the
    divergence, decodes to mlx-lm's uncached tokens, and prefills to the logits of a naive
    engine that reuses nothing."""
    model = build_reference(seed=40, text_config=TINY_FULL_ATTENTION_CONFIG)
    save_checkpoint(model, tmp_path, quantization=None)
    shared = random_tokens(40, 101)
    first = shared + random_tokens(41, 20)
    second = shared + random_tokens(42, 30)
    third = shared + random_tokens(43, 10)
    scheduler = make_scheduler(tmp_path, page_size=4)
    assert scheduler.engine.ctx.linear_state_pool is None
    assert scheduler.cache_manager.cache_type == "radix"
    for uid, (prompt, cached_len) in enumerate([(first, 0), (second, 100)]):
        assert run_alone(scheduler, uid, prompt) == (
            cached_len,
            uncached_greedy(model, prompt, STEPS),
        )
        scheduler.cache_manager.check_integrity()
    hit_len, hit_logits = prefill_logits(scheduler, third)
    assert hit_len == 100
    mx.eval(hit_logits)  # before the next engine replaces the global context

    naive = make_scheduler(tmp_path, page_size=4, cache_type="naive")
    assert naive.cache_manager.prefix_cache is None
    miss_len, miss_logits = prefill_logits(naive, third)
    assert miss_len == 0
    assert mx.allclose(hit_logits, miss_logits, atol=1e-4, rtol=1e-4)
    expected = model(mx.array(third)[None])[0, -1]
    assert mx.allclose(hit_logits, expected, atol=1e-4, rtol=1e-4)


def test_too_few_snapshots_fail_at_startup_and_naive_ignores_the_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    save_checkpoint(build_reference(seed=29), tmp_path, quantization=None)
    with pytest.raises(ValueError, match="--linear-state-checkpoints 2 or more"):
        make_scheduler(tmp_path, linear_state_checkpoints=1)
    with caplog.at_level(logging.WARNING, logger="ftmetal.scheduler.scheduler"):
        scheduler = make_scheduler(tmp_path, linear_state_checkpoints=1, cache_type="naive")
    assert "--linear-state-checkpoints" in caplog.text
    pool = scheduler.engine.ctx.linear_state_pool
    assert pool is not None and pool.num_slots == 2 + 1  # a live slot per running request
