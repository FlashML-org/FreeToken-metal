"""CommandGraph check on the real model: bs=1 decode, eager against the engine's replay.

    uv run python tools/command_graph_m1.py /path/to/Qwen3.6-35B-A3B-MLX \\
        [--context 2048] [--steps 18] [--timing-steps 64]

The engine starts with CommandGraph on at max_running_req 1, so its graph runner
captures batch size 1 on a dummy row and replays it from the shared arena. One request
prefills a real prompt (the engine's own source, tokenized) through the scheduler. From
its first decode position the script runs the decode steps eagerly through
model.forward, captures that first step once more for the double-capture check against
the runner's graph, then runs the same steps as replays of the runner's graph, each run
from the same state (the GDN slot is saved in a spare slot and restored; KV slots are
rewritten before they are read). Every step feeds its greedy token back, and both runs
build their metadata with the backends' prepare_for_replay, so they run the same
kernels. The script reports logits and token agreement, the graph's statistics and
arena plan, and per step the dispatches and command buffers (counted by the recorder's
hooks) and the host time: launch wall and thread CPU time up to the async_eval return,
and the whole step with a busy-wait for the token.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

import ftmetal
from ftmetal.core import Batch, Req, SamplingParams, host_ints
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.graph import GraphRunner
from ftmetal.kernel._ext import _C
from ftmetal.kernel.command_graph import (
    ArenaPlan,
    begin_count,
    check_same_graph,
    end_count,
    wait_until_available,
)
from ftmetal.scheduler.scheduler import Scheduler
from ftmetal.tokenizer import TokenizeManager

# Timed steps skipped at the start of each timing run.
WARMUP_STEPS = 8
# Decode positions reserved past the prompt beyond the steps the script runs.
SEQ_LEN_SLACK = 16


@dataclass
class StepTiming:
    launch_wall: float
    launch_cpu: float
    step_wall: float
    dispatches: int
    commits: int
    encoders: int


def prompt_tokens(model_path: str, length: int) -> np.ndarray:
    """The first length tokens of the engine's own Python source."""
    package = Path(ftmetal.__file__).resolve().parent
    text = "\n".join(path.read_text() for path in sorted(package.rglob("*.py")))
    ids = TokenizeManager(model_path).tokenizer.encode(text, add_special_tokens=False).ids
    assert len(ids) >= length, f"the source holds only {len(ids)} tokens"
    return np.asarray(ids[:length], dtype=np.int32)


def prefill(scheduler: Scheduler, prompt: np.ndarray, max_tokens: int) -> Req:
    """Prefill one request through the scheduler; returns it at its first decode position."""
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    scheduler.add_request(0, prompt, params)
    while scheduler.prefill_manager.runnable:
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        scheduler.process_last_data(scheduler.forward(forward_input))
    (req,) = scheduler.decode_manager.running_reqs
    return req


def decode_batch(scheduler: Scheduler, req: Req, token: mx.array, allocate: bool) -> Batch:
    """The decode batch of req's next token, padded to its bucket, without metadata."""
    if allocate:
        scheduler.cache_manager.allocate_paged([req])
    batch = Batch(reqs=[req], phase="decode")
    scheduler.engine.graph_runner.pad_batch(batch)
    batch.input_ids = token
    batch.positions = mx.array([req.cached_len], dtype=mx.int32)
    return batch


def prepare_for_replay(runner: GraphRunner, batch: Batch) -> None:
    for backend in runner.attn_backends.values():
        backend.prepare_for_replay(batch)


def greedy(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1).astype(mx.int32)


def host_logits(logits: mx.array) -> np.ndarray:
    return np.array(logits.astype(mx.float32))


def eager_step(scheduler: Scheduler, batch: Batch) -> mx.array:
    """model.forward on the metadata a replay of batch's bucket would get."""
    prepare_for_replay(scheduler.engine.graph_runner, batch)
    logits = scheduler.engine.model.forward(batch)
    batch.reqs[0].complete_one()
    return logits


def replay_step(scheduler: Scheduler, batch: Batch) -> mx.array:
    logits = scheduler.engine.graph_runner.replay(batch)
    batch.reqs[0].complete_one()
    return logits


def timed_steps(
    scheduler: Scheduler,
    req: Req,
    token: mx.array,
    num_steps: int,
    replayed: bool,
) -> tuple[list[StepTiming], list[int]]:
    """num_steps pipelined-free steps, each waited for by busy-waiting on its token."""
    engine = scheduler.engine
    timings: list[StepTiming] = []
    tokens: list[int] = []
    for _ in range(num_steps):
        begin_count()
        started_wall, started_cpu = time.perf_counter(), time.thread_time()
        batch = decode_batch(scheduler, req, token, allocate=not replayed)
        if replayed:
            logits = replay_step(scheduler, batch)
        else:
            logits = eager_step(scheduler, batch)
        token = greedy(logits)
        mx.async_eval(token, *engine.pool_arrays())
        launched_wall, launched_cpu = time.perf_counter(), time.thread_time()
        wait_until_available([token])
        finished_wall = time.perf_counter()
        counts = end_count()
        timings.append(
            StepTiming(
                launch_wall=launched_wall - started_wall,
                launch_cpu=launched_cpu - started_cpu,
                step_wall=finished_wall - started_wall,
                dispatches=counts.dispatches,
                commits=counts.commits,
                encoders=counts.encoders,
            )
        )
        tokens.extend(host_ints(token))
    return timings[WARMUP_STEPS:], tokens


def summarize(name: str, timings: list[StepTiming]) -> None:
    def median(values: list[float]) -> float:
        return 1000.0 * statistics.median(values)

    print(
        f"{name}: launch wall {median([t.launch_wall for t in timings]):.3f} ms, "
        f"launch cpu {median([t.launch_cpu for t in timings]):.3f} ms, "
        f"step wall {median([t.step_wall for t in timings]):.3f} ms, "
        f"dispatches {statistics.median([t.dispatches for t in timings]):.0f}, "
        f"command buffers {statistics.median([t.commits for t in timings]):.0f} "
        f"(min {min(t.commits for t in timings)}, max {max(t.commits for t in timings)}), "
        f"encoders {statistics.median([t.encoders for t in timings]):.0f} "
        f"over {len(timings)} steps",
        flush=True,
    )


def print_graph(graph: _C.Graph, plan: ArenaPlan, hidden_size: int) -> None:
    stats = graph.stats
    temporaries = stats.temporaries
    calls = stats.calls
    print(
        f"graph: {stats.num_dispatches} dispatches in {stats.num_encoders} recorded encoders, "
        f"{stats.num_barriers} barriers, {stats.num_bindings} bindings, "
        f"{stats.num_bytes_arguments} setBytes",
        flush=True,
    )
    print(
        f"graph buffers: {stats.num_step_input_buffers} step inputs, "
        f"{stats.num_pool_buffers} pools, {stats.num_external_buffers} externals, "
        f"{stats.num_constant_buffers} constants, {stats.num_intermediate_buffers} intermediates, "
        f"{stats.num_temporary_buffers} temporaries; {stats.num_held_arrays} held arrays, "
        f"pinned {stats.capture_pinned_bytes / 2**20:.2f} MiB at capture, "
        f"{stats.pinned_bytes / 2**20:.3f} MiB on the arena "
        f"({stats.num_pinned_temporaries} temporaries fail the poison test)",
        flush=True,
    )
    residual = [
        plan.placements[block].offset
        for block, label in enumerate(graph.block_labels)
        if label.startswith("Add ") and label.endswith(f"[1, {hidden_size}]")
    ]
    print(
        f"arena: {plan.arena_bytes / 2**20:.3f} MiB for {len(graph.block_sizes)} blocks of "
        f"{sum(graph.block_sizes) / 2**20:.2f} MiB, lower bound {plan.lower_bound / 2**20:.3f} "
        f"MiB (x{plan.arena_bytes / plan.lower_bound:.4f}); residual stream: {len(residual)} "
        f"adds at {len(set(residual))} offsets, first {residual[:8]}",
        flush=True,
    )
    nbytes = sum(nbytes for nbytes, _ in temporaries)
    lengths = sum(length for _, length in temporaries)
    print(
        f"temporaries: {len(temporaries)}, nbytes {nbytes / 2**20:.2f} MiB, "
        f"buffer length {lengths / 2**20:.2f} MiB",
        flush=True,
    )
    print(
        f"capture calls: {calls.dispatches} dispatches, {calls.commits} commits, "
        f"{calls.encoders} encoders, {calls.barriers} barriers, "
        f"{calls.fence_updates} fence updates, {calls.fence_waits} fence waits, "
        f"{calls.signal_events} signal events, {calls.completion_handlers} completion handlers",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--timing-steps", type=int, default=64)
    parser.add_argument("--attention-backend", default="metal")
    parser.add_argument("--page-size", type=int, default=1)
    args = parser.parse_args()
    assert args.steps >= 1, "at least one replay"
    decode_steps = args.steps + args.timing_steps

    config = EngineConfig(
        model_path=args.model_path,
        max_running_req=1,
        attention_backend=args.attention_backend,
        max_seq_len_override=args.context + decode_steps + SEQ_LEN_SLACK,
        page_size=args.page_size,
        command_graph=True,
    )
    scheduler = Scheduler(config, eos_token_ids=set())
    engine = scheduler.engine
    runner = engine.graph_runner
    assert 1 in runner.graphs, "the engine did not capture batch size 1"
    req = prefill(scheduler, prompt_tokens(args.model_path, args.context), decode_steps + 1)
    first_token = mx.array([int(req.input_ids[-1])], dtype=mx.int32)
    saved_lens = (req.cached_len, req.device_len)
    state_pool = engine.ctx.linear_state_pool
    live_slot = req.linear_slot_idx
    assert state_pool is not None and live_slot is not None
    (spare_slot,) = state_pool.alloc(1)
    state_pool.copy_from(live_slot, spare_slot)
    mx.eval(*engine.pool_arrays())
    print(
        f"prefilled {args.context} tokens; decoding from position {req.cached_len}, "
        f"page table width {engine.aligned_max_seq_len // args.page_size}",
        flush=True,
    )

    def restore() -> None:
        state_pool.copy_from(spare_slot, live_slot)
        mx.eval(*engine.pool_arrays())
        req.cached_len, req.device_len = saved_lens

    eager_logits: list[np.ndarray] = []
    token = first_token
    for _ in range(args.steps):
        logits = eager_step(scheduler, decode_batch(scheduler, req, token, allocate=True))
        token = greedy(logits)
        eager_logits.append(host_logits(logits))
    eager_timings, eager_tokens = timed_steps(
        scheduler, req, token, args.timing_steps, replayed=False
    )

    restore()
    batch = decode_batch(scheduler, req, first_token, allocate=False)
    prepare_for_replay(runner, batch)
    captured = runner.capture(batch)
    restore()
    replay_logits: list[np.ndarray] = []
    token = first_token
    for _ in range(args.steps):
        logits = replay_step(scheduler, decode_batch(scheduler, req, token, allocate=False))
        token = greedy(logits)
        replay_logits.append(host_logits(logits))
    replay_timings, replay_tokens = timed_steps(
        scheduler, req, token, args.timing_steps, replayed=True
    )

    print_graph(runner.graphs[1], runner.plans[1], engine.model_config.hidden_size)
    try:
        check_same_graph(runner.graphs[1], captured)
    except RuntimeError as error:
        same_graph = False
        print(f"double-capture check FAILED: {error}", flush=True)
    else:
        same_graph = True
        print("double-capture check passed: the real step records the runner's graph", flush=True)
    exact = [np.array_equal(e, r) for e, r in zip(eager_logits, replay_logits)]
    max_diff = max(float(np.abs(e - r).max()) for e, r in zip(eager_logits, replay_logits))
    eager_top = [int(e.argmax()) for e in eager_logits]
    replay_top = [int(r.argmax()) for r in replay_logits]
    print(
        f"logits: {sum(exact)} of {len(exact)} replayed steps bit-exact, "
        f"max |diff| {max_diff:.3g}; greedy tokens "
        f"{'identical' if eager_top == replay_top else 'DIFFER'}: {replay_top}",
        flush=True,
    )
    print(
        f"timing-run tokens {'identical' if eager_tokens == replay_tokens else 'DIFFER'} "
        f"over {len(replay_tokens)} further steps",
        flush=True,
    )
    summarize("eager", eager_timings)
    summarize("replay", replay_timings)
    passed = same_graph and all(exact) and eager_tokens == replay_tokens
    return 0 if passed else 1


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    # A model-holding process must not linger in MLX teardown.
    os._exit(status)
