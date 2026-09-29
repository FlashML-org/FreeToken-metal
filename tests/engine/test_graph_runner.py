"""GraphRunner on the tiny model: replayed decode against eager decode, token for token.

Replay needs the Metal kernels, so these run on the GPU like the rest of the suite.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.core import Batch, SamplingParams
from ftmetal.engine import graph
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.graph import GraphRunner
from ftmetal.kernel._ext import _C
from ftmetal.kernel.command_graph import check_same_graph
from ftmetal.scheduler.scheduler import Scheduler, TokenEvent
from tiny_model import build_reference, reference_greedy, save_checkpoint


class ReplayLog:
    """Wraps a runner's replay to record (batch size, padded size) of every replay."""

    def __init__(self, runner: GraphRunner) -> None:
        self.runner_replay = runner.replay
        self.sizes: list[tuple[int, int]] = []

    def __call__(self, batch: Batch) -> mx.array:
        self.sizes.append((batch.size, batch.padded_size))
        return self.runner_replay(batch)


def make_scheduler(path: Path, max_running_req: int, command_graph: bool) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max_running_req,
        page_size=4,
        num_page_override=128,
        max_extend_tokens=128,
        command_graph=command_graph,
    )
    return Scheduler(config, eos_token_ids={127})


def generate(scheduler: Scheduler, prompts: list[list[int]], steps: list[int]) -> list[list[int]]:
    for uid, (prompt, max_tokens) in enumerate(zip(prompts, steps)):
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events: list[TokenEvent] = scheduler.run_until_idle()
    scheduler.cache_manager.check_integrity()
    return [[event.token for event in events if event.uid == uid] for uid in range(len(prompts))]


@pytest.mark.parametrize(
    ("max_running_req", "lengths", "steps"),
    [(1, [23], [12]), (4, [9, 41, 17, 33], [6, 11, 3, 14])],
)
def test_replay_matches_eager_across_bucket_switches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_running_req: int,
    lengths: list[int],
    steps: list[int],
) -> None:
    model = build_reference(seed=20)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(20)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in lengths]
    expected = [reference_greedy(model, p, n)[1] for p, n in zip(prompts, steps)]

    eager = make_scheduler(tmp_path, max_running_req, command_graph=False)
    assert eager.engine.graph_runner.graphs == {}
    assert generate(eager, prompts, steps) == expected

    replayed = make_scheduler(tmp_path, max_running_req, command_graph=True)
    runner = replayed.engine.graph_runner
    assert sorted(runner.graphs) == [bs for bs in (1, 2, 4) if bs <= max_running_req]
    log = ReplayLog(runner)
    monkeypatch.setattr(runner, "replay", log)
    assert generate(replayed, prompts, steps) == expected
    # Every decode step replays; as requests finish the batch shrinks through the buckets,
    # three live requests padded to four.
    sizes = set(log.sizes)
    if max_running_req == 1:
        assert sizes == {(1, 1)}
    else:
        assert {(4, 4), (3, 4), (2, 2), (1, 1)} <= sizes


def test_failed_capture_leaves_its_bucket_eager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build_reference(seed=21)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(21)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in (12, 30)]
    steps = [4, 9]
    expected = [reference_greedy(model, p, n)[1] for p, n in zip(prompts, steps)]

    real_capture = graph.capture

    def capture_failing_at_two(
        outputs: list[mx.array],
        step_inputs: list[mx.array],
        pools: list[mx.array],
        externals: list[mx.array],
    ) -> _C.Graph:
        captured = real_capture(outputs, step_inputs, pools, externals)
        if outputs[0].shape[0] == 2:
            raise RuntimeError("CommandGraph capture failed: injected")
        return captured

    monkeypatch.setattr(graph, "capture", capture_failing_at_two)
    scheduler = make_scheduler(tmp_path, max_running_req=4, command_graph=True)
    runner = scheduler.engine.graph_runner
    assert sorted(runner.graphs) == [1, 4]
    log = ReplayLog(runner)
    monkeypatch.setattr(runner, "replay", log)
    assert generate(scheduler, prompts, steps) == expected
    # Two live requests run eagerly (their bucket has no graph and they are not padded);
    # one alone replays.
    assert set(log.sizes) == {(1, 1)}


def test_padded_decode_steps_capture_the_same_graph(tmp_path: Path) -> None:
    """The double-capture check on two real padded decode steps (three requests in bucket 4)."""
    model = build_reference(seed=22)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(22)
    scheduler = make_scheduler(tmp_path, max_running_req=4, command_graph=True)
    for uid, length in enumerate((10, 27, 18)):
        params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
        prompt = rng.integers(1, 120, size=length).astype(np.int32)
        scheduler.add_request(uid, prompt, params)
    while scheduler.prefill_manager.runnable:
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        scheduler.process_last_data(scheduler.forward(forward_input))

    runner = scheduler.engine.graph_runner
    captures: list[_C.Graph] = []
    for _ in range(2):
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        batch = forward_input.batch
        assert batch.is_decode and (batch.size, batch.padded_size) == (3, 4)
        for backend in runner.attn_backends.values():
            backend.prepare_for_replay(batch)
        captures.append(runner.capture(batch))
        for req in batch.reqs:
            req.complete_one()
    check_same_graph(captures[0], captures[1])
