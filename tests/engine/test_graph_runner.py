"""GraphRunner on the tiny model: replayed decode against eager decode, token for token.

Replay needs the Metal kernels, so these run on the GPU like the rest of the suite.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.attention.linear import LinearAttnBackend
from ftmetal.core import Batch, SamplingParams
from ftmetal.engine import engine, graph
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.engine import Engine
from ftmetal.engine.graph import GraphRunner, determine_graph_bs
from ftmetal.kernel._ext import _C
from ftmetal.kernel.command_graph import check_same_graph
from ftmetal.scheduler.scheduler import Scheduler, TokenEvent
from tiny_model import build_reference, reference_greedy, save_checkpoint

# Tokens of KV cache every test engine gets, whatever its page size.
NUM_TOKENS = 512


class ReplayLog:
    """Wraps a runner's replay to record (batch size, padded size) of every replay."""

    def __init__(self, runner: GraphRunner) -> None:
        self.runner_replay = runner.replay
        self.sizes: list[tuple[int, int]] = []

    def __call__(self, batch: Batch) -> mx.array:
        self.sizes.append((batch.size, batch.padded_size))
        return self.runner_replay(batch)


@pytest.mark.parametrize(
    ("max_running_req", "ladder"),
    [
        (1, [1]),
        (3, [1, 2, 3]),
        (4, [1, 2, 4]),
        (6, [1, 2, 4, 6]),
        (8, [1, 2, 4, 8]),
        (12, [1, 2, 4, 8, 12]),
        (24, [1, 2, 4, 8, 16, 24]),
    ],
)
def test_graph_ladder_ends_at_max_running_req(max_running_req: int, ladder: list[int]) -> None:
    assert determine_graph_bs(max_running_req) == ladder


def engine_config(
    path: Path,
    max_running_req: int,
    command_graph: bool,
    page_size: int = 4,
    attention_backend: str = "metal",
) -> EngineConfig:
    return EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max_running_req,
        page_size=page_size,
        attention_backend=attention_backend,
        num_page_override=NUM_TOKENS // page_size,
        max_extend_tokens=128,
        command_graph=command_graph,
    )


def make_scheduler(
    path: Path, max_running_req: int, command_graph: bool, page_size: int = 4
) -> Scheduler:
    config = engine_config(path, max_running_req, command_graph, page_size)
    return Scheduler(config, eos_token_ids={127})


def generate(scheduler: Scheduler, prompts: list[list[int]], steps: list[int]) -> list[list[int]]:
    for uid, (prompt, max_tokens) in enumerate(zip(prompts, steps)):
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events: list[TokenEvent] = scheduler.run_until_idle()
    scheduler.cache_manager.check_integrity()
    return [[event.token for event in events if event.uid == uid] for uid in range(len(prompts))]


@pytest.mark.parametrize(
    ("max_running_req", "page_size", "lengths", "steps", "replayed_sizes"),
    [
        (1, 4, [23], [12], {(1, 1)}),
        (4, 4, [9, 41, 17, 33], [6, 11, 3, 14], {(4, 4), (3, 4), (2, 2), (1, 1)}),
        (4, 1, [9, 41, 17, 33], [6, 11, 3, 14], {(4, 4), (3, 4), (2, 2), (1, 1)}),
        # Off the ladder: the top bucket is max_running_req itself, and five rows pad to it.
        (
            6,
            4,
            [9, 41, 17, 33, 25, 12],
            [6, 11, 3, 14, 8, 5],
            {(6, 6), (5, 6), (4, 4), (3, 4), (2, 2), (1, 1)},
        ),
    ],
)
def test_replay_matches_eager_across_bucket_switches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_running_req: int,
    page_size: int,
    lengths: list[int],
    steps: list[int],
    replayed_sizes: set[tuple[int, int]],
) -> None:
    model = build_reference(seed=20)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(20)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in lengths]
    expected = [reference_greedy(model, p, n)[1] for p, n in zip(prompts, steps)]

    eager = make_scheduler(tmp_path, max_running_req, command_graph=False, page_size=page_size)
    assert eager.engine.graph_runner.graphs == {}
    assert generate(eager, prompts, steps) == expected

    replayed = make_scheduler(tmp_path, max_running_req, command_graph=True, page_size=page_size)
    runner = replayed.engine.graph_runner
    assert sorted(runner.graphs) == determine_graph_bs(max_running_req)
    log = ReplayLog(runner)
    monkeypatch.setattr(runner, "replay", log)
    assert generate(replayed, prompts, steps) == expected
    # Every decode step replays; as requests finish the batch shrinks through the buckets.
    assert set(log.sizes) >= replayed_sizes


def test_buckets_share_one_tight_arena(tmp_path: Path) -> None:
    model = build_reference(seed=23)
    save_checkpoint(model, tmp_path, quantization=None)
    runner = make_scheduler(tmp_path, max_running_req=4, command_graph=True).engine.graph_runner
    assert sorted(runner.graphs) == [1, 2, 4] and runner.arena is not None
    assert runner.arena.nbytes == max(plan.arena_bytes for plan in runner.plans.values())
    for bs, captured in runner.graphs.items():
        plan, stats = runner.plans[bs], captured.stats
        assert plan.arena_bytes <= 1.1 * plan.lower_bound
        assert stats.num_pinned_temporaries == 0
        assert stats.pinned_bytes < stats.capture_pinned_bytes


def test_graph_reserve_exceeded_fails_the_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build_reference(seed=24)
    save_checkpoint(model, tmp_path, quantization=None)
    # Room for the arena of batch size 2, not for that of 4.
    monkeypatch.setattr(engine, "graph_reserve_bytes", lambda max_graph_bs: 20_000)
    with pytest.raises(RuntimeError, match="graph reserve"):
        Engine(engine_config(tmp_path, max_running_req=4, command_graph=True))


@pytest.mark.parametrize("failing_part", ["backend", "recorder"])
def test_capture_failure_fails_the_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_part: str
) -> None:
    model = build_reference(seed=21)
    save_checkpoint(model, tmp_path, quantization=None)
    real_capture = graph.capture

    def prepare_failing(self: LinearAttnBackend, batch: Batch) -> None:
        raise RuntimeError("capture failed: injected")

    def capture_failing_at_two(
        outputs: list[mx.array],
        step_inputs: list[mx.array],
        pools: list[mx.array],
        externals: list[mx.array],
    ) -> _C.Graph:
        captured = real_capture(outputs, step_inputs, pools, externals)
        if outputs[0].shape[0] == 2:
            raise RuntimeError("capture failed: injected")
        return captured

    if failing_part == "backend":
        monkeypatch.setattr(LinearAttnBackend, "prepare_for_capture", prepare_failing)
    else:
        monkeypatch.setattr(graph, "capture", capture_failing_at_two)
    with pytest.raises(RuntimeError, match="capture failed: injected"):
        Engine(engine_config(tmp_path, max_running_req=4, command_graph=True))


def test_uncapturable_backend_decodes_eagerly_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = build_reference(seed=26)
    save_checkpoint(model, tmp_path, quantization=None)
    config = engine_config(tmp_path, 2, command_graph=True, attention_backend="mlx")
    with caplog.at_level(logging.WARNING, logger=engine.__name__):
        runner = Engine(config).graph_runner
    assert runner.graphs == {}
    assert "CommandGraph needs a capturable attention backend" in caplog.text


def prefilled_scheduler(path: Path, prompts: list[list[int]], command_graph: bool) -> Scheduler:
    """A scheduler whose requests have all prefilled, at their first decode step."""
    scheduler = make_scheduler(path, max_running_req=4, command_graph=command_graph)
    for uid, prompt in enumerate(prompts):
        params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    while scheduler.prefill_manager.runnable:
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        scheduler.process_last_data(scheduler.forward(forward_input))
    return scheduler


def pad_decode_batches(
    monkeypatch: pytest.MonkeyPatch, scheduler: Scheduler, padded_size: int
) -> None:
    """Pad every decode batch to padded_size with dummy requests, graphs or not."""
    dummy_req = scheduler.engine.dummy_req

    def pad_batch(batch: Batch) -> None:
        extra = padded_size - batch.size if batch.is_decode else 0
        batch.padded_reqs = batch.reqs + [dummy_req] * extra

    monkeypatch.setattr(scheduler.engine.graph_runner, "pad_batch", pad_batch)


def test_padded_eager_decode_matches_unpadded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eager decode of a batch padded with dummy rows: the metadata covers the padded rows
    like the step inputs, and the real rows get the unpadded batch's logits and tokens."""
    model = build_reference(seed=25)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(25)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in (14, 37, 22)]
    steps = [9, 5, 12]
    expected = [reference_greedy(model, p, n)[1] for p, n in zip(prompts, steps)]

    logits: dict[int, mx.array] = {}
    for padded_size in (3, 4):
        scheduler = prefilled_scheduler(tmp_path, prompts, command_graph=False)
        pad_decode_batches(monkeypatch, scheduler, padded_size)
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        batch = forward_input.batch
        assert batch.is_decode and (batch.size, batch.padded_size) == (3, padded_size)
        logits[padded_size] = scheduler.engine.model.forward(batch)[: batch.size]
    assert mx.array_equal(logits[4], logits[3]).item()

    scheduler = make_scheduler(tmp_path, max_running_req=4, command_graph=False)
    pad_decode_batches(monkeypatch, scheduler, 4)
    assert generate(scheduler, prompts, steps) == expected


def test_padded_decode_step_captures_the_replayed_graph(tmp_path: Path) -> None:
    """The double-capture check of a real padded decode step (three requests in bucket 4)
    against the graph the engine replays, which was captured on dummy rows."""
    model = build_reference(seed=22)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(22)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in (10, 27, 18)]
    scheduler = prefilled_scheduler(tmp_path, prompts, command_graph=True)
    runner = scheduler.engine.graph_runner
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    batch = forward_input.batch
    assert batch.is_decode and (batch.size, batch.padded_size) == (3, 4)
    for backend in runner.attn_backends.values():
        backend.prepare_for_replay(batch)
    check_same_graph(runner.graphs[batch.padded_size], runner.capture(batch))
