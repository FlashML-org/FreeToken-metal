"""The overlap loop: schedule step N+1, launch it, then drain step N.

Port of FreeToken scheduler/scheduler.py. Divergences: one process and one MLX
stream, so the launch is an async_eval of the sampled tokens plus every pool array
and the drain is reading those tokens; requests arrive by add_request and results
leave through send_result as TokenEvents (the message transport and run_forever
land in round 1's server phase).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import mlx.core as mx
import numpy as np

from ftmetal.core import Batch, Req, SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.engine import Engine, ForwardOutput
from ftmetal.engine.sample import BatchSamplingArgs
from ftmetal.scheduler.cache import CacheManager
from ftmetal.scheduler.decode import DecodeManager
from ftmetal.scheduler.prefill import ChunkedReq, PrefillManager
from ftmetal.scheduler.table import TableManager

logger = logging.getLogger(__name__)


class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    write_ids: mx.array
    num_tokens: int


class ForwardData(NamedTuple):
    forward_input: ForwardInput
    forward_output: ForwardOutput


@dataclass
class TokenEvent:
    uid: int
    token: int
    finished: bool
    finish_reason: Literal["stop", "length"] | None = None


@dataclass
class StepMetrics:
    """One overlap iteration: the batch it launched, and where its wall time went."""

    phase: str
    size: int
    num_tokens: int
    # Scheduling, graph build and dispatch of the launched batch.
    host_ms: float
    # Blocked on the previous batch's tokens: GPU time not hidden behind the host.
    wait_ms: float
    # Whole iteration; throughput is size / iteration_ms.
    iteration_ms: float


@dataclass
class SchedulerMetrics:
    steps: list[StepMetrics] = field(default_factory=list)

    def decode_tokens_per_second(self, last: int = 50) -> float:
        recent = [s for s in self.steps if s.phase == "decode"][-last:]
        if not recent:
            return 0.0
        return sum(s.size for s in recent) / (sum(s.iteration_ms for s in recent) / 1000.0)


class Scheduler:
    def __init__(self, config: EngineConfig, eos_token_ids: set[int]) -> None:
        self.engine = Engine(config)
        self.config = config
        self.eos_token_ids = eos_token_ids
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages,
            config.page_size,
            self.engine.page_table,
            self.engine.linear_state_pool,
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        self.finished_reqs: set[Req] = set()
        self.metrics = SchedulerMetrics()
        # Results not yet taken by the caller; the server phase replaces this with a queue.
        self.results: list[TokenEvent] = []

    def add_request(self, uid: int, input_ids: np.ndarray, sampling_params: SamplingParams) -> None:
        max_output_len = self.engine.max_seq_len - len(input_ids)
        if max_output_len <= 0:
            raise ValueError(
                f"prompt is too long: {len(input_ids)} tokens > {self.engine.max_seq_len}"
            )
        if sampling_params.max_tokens > max_output_len:
            sampling_params.max_tokens = max_output_len
        self.prefill_manager.add_one_req(uid, input_ids, sampling_params)

    def abort_request(self, uid: int) -> None:
        """Drop a pending or running request; a step already launched with it still runs."""
        req = self.prefill_manager.abort_req(uid) or self.decode_manager.abort_req(uid)
        if req is None:
            return
        req.aborted = True
        self.free_req_resources(req)

    @property
    def runnable(self) -> bool:
        return self.prefill_manager.runnable or self.decode_manager.runnable

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """One iteration: launch the next batch, then drain the previous one."""
        started = time.perf_counter()
        forward_input = self.schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = self.forward(forward_input)
        launched = time.perf_counter()
        self.process_last_data(last_data)
        drained = time.perf_counter()
        if forward_input is not None:
            self.metrics.steps.append(
                StepMetrics(
                    phase=forward_input.batch.phase,
                    size=forward_input.batch.size,
                    num_tokens=forward_input.num_tokens,
                    host_ms=(launched - started) * 1000.0,
                    wait_ms=(drained - launched) * 1000.0,
                    iteration_ms=(drained - started) * 1000.0,
                )
            )
        return ongoing_data

    def send_result(self, events: list[TokenEvent]) -> None:
        self.results.extend(events)

    def take_results(self) -> list[TokenEvent]:
        events, self.results = self.results, []
        return events

    def run_until_idle(self) -> list[TokenEvent]:
        data = None
        while self.runnable or data is not None:
            data = self.overlap_loop(data)
        return self.take_results()

    def generate(self, input_ids: np.ndarray, sampling_params: SamplingParams) -> list[int]:
        """Offline convenience: one prompt to completion, returns the sampled tokens."""
        self.add_request(0, input_ids, sampling_params)
        return [event.token for event in self.run_until_idle()]

    def schedule_next_batch(self) -> ForwardInput | None:
        batch = (
            self.prefill_manager.schedule_next_batch(self.config.max_extend_tokens)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is None:
            return None
        return self.prepare_batch(batch)

    def prepare_batch(self, batch: Batch) -> ForwardInput:
        self.cache_manager.allocate_paged(batch.reqs)
        table = self.table_manager
        positions = np.concatenate(
            [np.arange(req.cached_len, req.device_len, dtype=np.int32) for req in batch.reqs]
        )
        input_flat = np.concatenate(
            [table.flat_ids(req.table_idx, req.cached_len, req.device_len) for req in batch.reqs]
        )
        write_flat = np.array(
            [
                table.flat_ids(req.table_idx, req.device_len, req.device_len + 1)[0]
                if req.can_decode
                else table.flat_ids(table.scratch_row, 0, 1)[0]
                for req in batch.reqs
            ],
            dtype=np.int32,
        )
        batch.positions = mx.array(positions)
        batch.input_ids = table.gather_tokens(mx.array(input_flat))
        self.engine.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            write_ids=mx.array(write_flat),
            num_tokens=batch.num_tokens,
        )

    def forward(self, forward_input: ForwardInput) -> ForwardData:
        batch, sample_args, write_ids, _ = forward_input
        output = self.engine.forward_batch(batch, sample_args)
        self.table_manager.write_sampled(output.next_tokens, write_ids)
        mx.async_eval(output.next_tokens, self.table_manager.token_pool, *self.engine.pool_arrays())
        self.decode_manager.filter_reqs(batch.reqs)
        return ForwardData(forward_input, output)

    def process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        batch = last_data.forward_input.batch
        next_tokens = last_data.forward_output.next_tokens.tolist()
        events: list[TokenEvent] = []
        new_finished: set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for req, next_token in zip(batch.reqs, next_tokens):
                if isinstance(req, ChunkedReq) or req.aborted:
                    continue
                if req in self.finished_reqs:
                    continue
                req.append_host(next_token)
                hit_length = req.generated_len >= req.output_len
                hit_eos = not req.sampling_params.ignore_eos and next_token in self.eos_token_ids
                finished = hit_length or hit_eos
                reason = None
                if finished:
                    reason = "stop" if hit_eos else "length"
                events.append(TokenEvent(req.uid, next_token, finished, reason))
                if finished:
                    self.decode_manager.remove_req(req)
                    self.free_req_resources(req)
                    new_finished.add(req)
                elif batch.is_prefill:
                    self.cache_manager.cache_req(req, finished=False)
        self.finished_reqs = new_finished
        self.send_result(events)

    def free_req_resources(self, req: Req) -> None:
        if req.table_idx == -1:
            return
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1


__all__ = [
    "ForwardData",
    "ForwardInput",
    "Scheduler",
    "SchedulerMetrics",
    "StepMetrics",
    "TokenEvent",
]
