"""CommandGraph M1 check on the real model: bs=1 decode, eager against capture and replay.

    uv run python tools/command_graph_m1.py /path/to/Qwen3.6-35B-A3B-MLX \\
        [--context 2048] [--steps 18] [--timing-steps 64]

One request prefills a real prompt (the engine's own source, tokenized) through the
scheduler. From its first decode position the script then runs the decode steps twice
from the same state (the GDN slot is saved in a spare slot and restored; KV slots are
rewritten before they are read): eagerly through model.forward, then as two captures
(the double-capture check compares them) followed by replays of the first graph. Every
step feeds its greedy token back. Each decode batch has the fixed shapes a graph needs:
the page table is aligned_max_seq_len / page_size columns wide and flash-decode's
partition count follows max_seqlen_k = aligned_max_seq_len, so both runs use the same
kernels. The script reports logits and token agreement, the graph's statistics, and
per step the dispatches and command buffers (counted by the recorder's hooks) and the
host time: launch wall and thread CPU time up to the async_eval return, and the whole
step with a busy-wait for the token.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

import ftmetal
from ftmetal.attention.full.common import FullAttnBackend, FullAttnMetadata
from ftmetal.attention.metadata import AttnType, LinearAttnMetadata
from ftmetal.core import Batch, Req, SamplingParams, host_ints
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.engine import Engine
from ftmetal.kernel._ext import _C
from ftmetal.kernel.command_graph import (
    begin_capture,
    begin_count,
    check_same_graph,
    end_capture,
    end_count,
    replay,
    wait_until_available,
)
from ftmetal.kvcache.linear_state_pool import LinearStatePool
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.scheduler.scheduler import Scheduler
from ftmetal.tokenizer import TokenizeManager

# Timed steps skipped at the start of each timing run.
WARMUP_STEPS = 8
# Decode positions reserved past the prompt beyond the steps the script runs.
SEQ_LEN_SLACK = 16


@dataclass
class DecodeStep:
    batch: Batch
    # The graph's step inputs, in capture order: input_ids, positions, out_loc,
    # cu_seqlens_q, seq_lens, page_table, cu_seqlens, cache_indices, track_slots.
    step_inputs: list[mx.array]


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


def decode_step(scheduler: Scheduler, req: Req, token: mx.array, allocate: bool) -> DecodeStep:
    """The decode batch of req's next token with bucket-fixed metadata shapes."""
    engine = scheduler.engine
    if allocate:
        scheduler.cache_manager.allocate_paged([req])
    page_size = engine.config.page_size
    full_backend = engine.attn_backends[AttnType.FULL]
    assert isinstance(full_backend, FullAttnBackend)
    row = engine.page_table[req.table_idx]
    pages = -(-req.device_len // page_size)
    page_table = np.full(
        (1, engine.aligned_max_seq_len // page_size), full_backend.dummy_page, dtype=np.int32
    )
    page_table[0, :pages] = row[: pages * page_size : page_size] // page_size
    assert req.linear_slot_idx is not None
    batch = Batch(reqs=[req], phase="decode")
    batch.input_ids = token
    batch.positions = mx.array([req.cached_len], dtype=mx.int32)
    batch.out_loc = mx.array(row[req.cached_len : req.device_len])
    full = FullAttnMetadata(
        cu_seqlens_q=mx.array([0, 1], dtype=mx.int32),
        seq_lens=mx.array([req.device_len], dtype=mx.int32),
        page_table=mx.array(page_table),
        max_seqlen_q=1,
        max_seqlen_k=engine.aligned_max_seq_len,
        seq_lens_host=[req.device_len],
        extend_lens_host=[1],
        cached_lens_host=[req.cached_len],
    )
    linear = LinearAttnMetadata(
        cu_seqlens=mx.array([0, 1], dtype=mx.int32),
        cache_indices=mx.array([req.linear_slot_idx], dtype=mx.int32),
        track_slots=mx.array([-1], dtype=mx.int32),
        seq_lens_host=[1],
        slots_host=[req.linear_slot_idx],
        track_slots_host=[-1],
        track_seqlens_host=[0],
    )
    batch.attn_metadata[AttnType.FULL] = full
    batch.attn_metadata[AttnType.LINEAR] = linear
    step_inputs = [
        batch.input_ids,
        batch.positions,
        batch.out_loc,
        full.cu_seqlens_q,
        full.seq_lens,
        full.page_table,
        linear.cu_seqlens,
        linear.cache_indices,
        linear.track_slots,
    ]
    return DecodeStep(batch, step_inputs)


def set_pool_arrays(engine: Engine, arrays: list[mx.array]) -> None:
    """Rebind the pools to arrays, in engine.pool_arrays() order."""
    offset = 0
    for pool in engine.pools.values():
        if isinstance(pool, MHAKVCache):
            count = len(pool.k)
            pool.k = arrays[offset : offset + count]
            pool.v = arrays[offset + count : offset + 2 * count]
            offset += 2 * count
        elif isinstance(pool, LinearStatePool):
            count = len(pool.conv_states)
            pool.conv_states = arrays[offset : offset + count]
            pool.recurrent_states = arrays[offset + count : offset + 2 * count]
            offset += 2 * count
        else:
            raise TypeError(f"unknown pool {type(pool).__name__}")
    assert offset == len(arrays)


def greedy(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1).astype(mx.int32)


def host_logits(logits: mx.array) -> np.ndarray:
    return np.array(logits.astype(mx.float32))


def eager_step(engine: Engine, step: DecodeStep) -> mx.array:
    logits = engine.model.forward(step.batch)
    step.batch.reqs[0].complete_one()
    return logits


def capture_step(
    engine: Engine, step: DecodeStep, externals: list[mx.array]
) -> tuple[_C.Graph, mx.array]:
    """One real decode step under the recorder; returns its graph and logits."""
    pools = engine.pool_arrays()
    logits = engine.model.forward(step.batch)
    updated = engine.pool_arrays()
    begin_capture([logits, *updated], step.step_inputs, pools, externals)
    mx.eval(logits, *updated)
    graph = end_capture()
    step.batch.reqs[0].complete_one()
    return graph, logits


def replay_step(engine: Engine, step: DecodeStep, graph: _C.Graph) -> mx.array:
    logits, pools = replay(graph, step.step_inputs, engine.pool_arrays())
    set_pool_arrays(engine, pools)
    step.batch.reqs[0].complete_one()
    return logits


def timed_steps(
    scheduler: Scheduler,
    req: Req,
    token: mx.array,
    num_steps: int,
    allocate: bool,
    graph: _C.Graph | None,
) -> tuple[list[StepTiming], list[int]]:
    """num_steps pipelined-free steps, each waited for by busy-waiting on its token."""
    engine = scheduler.engine
    timings: list[StepTiming] = []
    tokens: list[int] = []
    for _ in range(num_steps):
        begin_count()
        started_wall, started_cpu = time.perf_counter(), time.thread_time()
        step = decode_step(scheduler, req, token, allocate)
        if graph is None:
            logits = eager_step(engine, step)
        else:
            logits = replay_step(engine, step, graph)
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


def print_graph(graph: _C.Graph) -> None:
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
        f"pinned {stats.pinned_bytes / 2**20:.2f} MiB",
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
    assert args.steps >= 3, "two captures and at least one replay"
    decode_steps = args.steps + args.timing_steps

    config = EngineConfig(
        model_path=args.model_path,
        max_running_req=1,
        attention_backend=args.attention_backend,
        max_seq_len_override=args.context + decode_steps + SEQ_LEN_SLACK,
        page_size=args.page_size,
    )
    scheduler = Scheduler(config, eos_token_ids=set())
    engine = scheduler.engine
    req = prefill(scheduler, prompt_tokens(args.model_path, args.context), decode_steps + 1)
    first_token = mx.array([int(req.input_ids[-1])], dtype=mx.int32)
    saved_lens = (req.cached_len, req.device_len)
    state_pool = engine.ctx.linear_state_pool
    assert state_pool is not None and req.linear_slot_idx is not None
    (spare_slot,) = state_pool.alloc(1)
    state_pool.copy_from(req.linear_slot_idx, spare_slot)
    mx.eval(*engine.pool_arrays())
    print(
        f"prefilled {args.context} tokens; decoding from position {req.cached_len}, "
        f"page table width {engine.aligned_max_seq_len // args.page_size}",
        flush=True,
    )

    eager_logits: list[np.ndarray] = []
    token = first_token
    for _ in range(args.steps):
        logits = eager_step(engine, decode_step(scheduler, req, token, allocate=True))
        token = greedy(logits)
        eager_logits.append(host_logits(logits))
    eager_timings, eager_tokens = timed_steps(
        scheduler, req, token, args.timing_steps, allocate=True, graph=None
    )

    state_pool.copy_from(spare_slot, req.linear_slot_idx)
    mx.eval(*engine.pool_arrays())
    req.cached_len, req.device_len = saved_lens
    externals = [
        value for _, value in tree_flatten(engine.model.parameters()) if isinstance(value, mx.array)
    ]
    graphs: list[_C.Graph] = []
    replay_logits: list[np.ndarray] = []
    token = first_token
    for _ in range(2):
        step = decode_step(scheduler, req, token, allocate=False)
        mx.eval(*step.step_inputs)
        graph, logits = capture_step(engine, step, externals)
        graphs.append(graph)
        token = greedy(logits)
        replay_logits.append(host_logits(logits))
    for _ in range(args.steps - 2):
        logits = replay_step(engine, decode_step(scheduler, req, token, allocate=False), graphs[0])
        token = greedy(logits)
        replay_logits.append(host_logits(logits))
    replay_timings, replay_tokens = timed_steps(
        scheduler, req, token, args.timing_steps, allocate=False, graph=graphs[0]
    )

    print_graph(graphs[0])
    try:
        check_same_graph(graphs[0], graphs[1])
    except RuntimeError as error:
        print(f"double-capture check FAILED: {error}", flush=True)
    else:
        print("double-capture check passed", flush=True)
    exact = [np.array_equal(e, r) for e, r in zip(eager_logits, replay_logits)]
    max_diff = max(float(np.abs(e - r).max()) for e, r in zip(eager_logits, replay_logits))
    eager_top = [int(e.argmax()) for e in eager_logits]
    replay_top = [int(r.argmax()) for r in replay_logits]
    print(
        f"logits: {sum(exact)} of {len(exact)} steps bit-exact (2 captures, "
        f"{len(exact) - 2} replays), max |diff| {max_diff:.3g}; greedy tokens "
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
    passed = all(exact) and eager_tokens == replay_tokens
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
