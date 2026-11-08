"""The overlap loop: schedule step N+1, launch it, then drain step N.

Port of FreeToken scheduler/scheduler.py. Divergences: one process and one MLX
stream, so the launch is an async_eval of the sampled tokens plus every pool array
and the drain is reading those tokens (a prefix hit's snapshot restore is an in-place
copy in the launched graph, ordered by the pool arrays it rebinds); requests arrive by
add_request and results leave through send_result as TokenEvents; run_forever is the
server's scheduler thread body over the message queues. A padded decode batch's dummy
rows read the token pool's scratch row; only real requests allocate pages and write
sampled tokens. The multimodal gather plan, the image block ends and the mrope positions
are built in numpy by scheduler/mm.py. A drained step's tokens are used only once the
engine's offload cache passes its health check (a failed expert read leaves stand-in
experts behind).
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import Literal, NamedTuple

import mlx.core as mx
import numpy as np

from ftmetal.core import Batch, Req, SamplingParams, host_ints
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.engine import Engine, ForwardOutput
from ftmetal.engine.sample import BatchSamplingArgs
from ftmetal.kvcache.linear_state_pool import gdn_snapshot_slots
from ftmetal.kvcache.swa_pool import swa_radix_num_windows
from ftmetal.message.backend import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    ExitMsg,
    UserMsg,
)
from ftmetal.message.tokenizer import (
    BaseTokenizerMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    PromptAdmittedMsg,
)
from ftmetal.mm.item import MMItem
from ftmetal.scheduler.cache import CacheManager, resolve_cache_type
from ftmetal.scheduler.decode import DecodeManager
from ftmetal.scheduler.mm import make_mrope_positions, mm_block_ends, plan_mm_batch
from ftmetal.scheduler.prefill import ChunkedReq, PrefillManager
from ftmetal.scheduler.status import SchedulerStatusReporter
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


class Scheduler:
    def __init__(self, config: EngineConfig, eos_token_ids: set[int]) -> None:
        # Before the engine loads anything: too few --swa-windows or --gdn-snapshots fails fast.
        swa_group = config.model_config.swa_attention_group()
        cache_type = resolve_cache_type(
            config.cache_type,
            has_linear_attention=config.model_config.has_linear_attention,
            has_sliding_window=swa_group is not None,
        )
        if cache_type == "swa_radix" and swa_group is not None:
            num_windows = swa_radix_num_windows(
                swa_group.sliding_window,
                config.page_size,
                config.max_running_req,
                config.max_extend_tokens,
                config.swa_windows,
            )
            logger.info(f"window pool: {num_windows} windows of {swa_group.sliding_window}")
        elif config.swa_windows is not None:
            logger.warning(
                f"--swa-windows sizes the SWA radix cache's window pool: ignored with the "
                f"{cache_type} cache"
            )
        if cache_type == "hybrid_radix":
            num_snapshots = gdn_snapshot_slots(config.gdn_snapshots)
            logger.info(f"GDN snapshot cache: {num_snapshots} slots")
        elif config.gdn_snapshots is not None:
            logger.warning(
                f"--gdn-snapshots sizes the hybrid radix cache's GDN snapshot cache: ignored "
                f"with the {cache_type} cache"
            )
        self.engine = Engine(config)
        self.config = config
        # The engine's MoE offload cache, for the idle line's hit rate; None while resident.
        self.moe_offload_cache = self.engine.moe_offload_cache
        self.eos_token_ids = eos_token_ids
        # An offload cache holds a decode's experts in one step only up to max_decode_rows
        # rows, capped through the requests admitted; a looked-up prefill's tokens are capped
        # through its budget (prefill_budget).
        max_admitted = config.max_running_req
        cache = self.moe_offload_cache
        max_rows = None if cache is None else cache.slot_layout.max_decode_rows
        if max_rows is not None and max_rows < max_admitted:
            logger.warning(
                f"MoE offload: {max_rows} requests run at once, not {config.max_running_req}: a "
                f"decode of more rows could route more experts than a pool of the cache holds"
            )
            max_admitted = max_rows
        self.table_manager = TableManager(
            config.max_running_req, self.engine.page_table, max_admitted=max_admitted
        )
        assert self.engine.dummy_req.table_idx == self.table_manager.scratch_row
        self.cache_manager = CacheManager(
            self.engine.num_pages,
            config.page_size,
            self.engine.page_table,
            cache_type,
            self.engine.ctx.linear_state_pool,
            self.engine.ctx.swa_pool,
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            encoder_cache=self.engine.encoder_cache,
            keep_images_whole=self.engine.model_config.has_bidirectional_mm_blocks,
        )
        self.finished_reqs: set[Req] = set()
        self.status_reporter = SchedulerStatusReporter(
            logger.info, decode_log_interval=config.decode_log_interval
        )
        # Results not yet taken by the caller (run_forever forwards them as messages).
        self.results: list[TokenEvent] = []
        # Prompts whose first prefill batch is prepared, not yet reported by run_forever.
        self.prompt_admissions: list[PromptAdmittedMsg] = []
        # Requests whose tokens may still come; reported as failed if run_forever dies.
        self.live_uids: set[int] = set()
        # The launched-but-not-drained batch (overlap). An abort of one of its requests only
        # marks the request; process_last_data frees it once the forward has landed.
        self.inflight_data: ForwardData | None = None

    @property
    def max_seq_len(self) -> int:
        """The longest prompt plus output a request may hold."""
        return self.engine.max_seq_len

    def add_request(
        self,
        uid: int,
        input_ids: np.ndarray,
        sampling_params: SamplingParams,
        reuse_len: int | None = None,
        mm_items: list[MMItem] | None = None,
        mrope_positions: np.ndarray | None = None,
        mrope_delta: int = 0,
    ) -> None:
        """Queue a prompt; reuse_len is the tokens of it a next turn can share (None: all);
        mm_items, mrope_positions and mrope_delta are its images' tokenizer outputs."""
        assert reuse_len is None or 0 <= reuse_len <= len(input_ids)
        assert not mm_items or self.engine.encoder_cache is not None, "no vision tower"
        max_output_len = self.engine.max_seq_len - len(input_ids)
        if max_output_len <= 0:
            raise ValueError(
                f"prompt is too long: {len(input_ids)} tokens > {self.engine.max_seq_len}"
            )
        if sampling_params.max_tokens > max_output_len:
            sampling_params.max_tokens = max_output_len
        self.prefill_manager.add_one_req(
            uid, input_ids, sampling_params, reuse_len, mm_items, mrope_positions, mrope_delta
        )

    def abort_request(self, uid: int) -> None:
        """Drop a pending or running request; a step already launched with it still runs.

        FreeToken's fence: a request in the in-flight batch is only marked, and
        process_last_data frees it after the forward has landed, so its pages, table row
        and GDN slots are never handed to a new request while the forward still writes
        them. A request with no forward in flight is freed here; deferring it would leak
        until its next batch, which strict prefill priority puts arbitrarily far away.
        """
        req = self.prefill_manager.abort_req(uid) or self.decode_manager.abort_req(uid)
        if req is None:
            return
        if req.mm_items and self.engine.encoder_cache is not None:
            # Drop the request's claims; entries it held alone die here.
            self.engine.encoder_cache.release(uid, [item.hash for item in req.mm_items])
        req.aborted = True
        inflight = (
            self.inflight_data is not None and req in self.inflight_data.forward_input.batch.reqs
        )
        if not inflight:
            self.free_req_resources(req)

    @property
    def runnable(self) -> bool:
        return self.prefill_manager.runnable or self.decode_manager.runnable

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """One iteration: launch the next batch, then drain the previous one."""
        forward_input = self.schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            self.report_status(forward_input)
            ongoing_data = self.forward(forward_input)
        self.process_last_data(last_data)
        self.inflight_data = ongoing_data
        return ongoing_data

    def report_status(self, forward_input: ForwardInput) -> None:
        kv_used_pages, kv_total_pages = self.kv_page_usage()
        self.status_reporter.report_batch(
            forward_input.batch,
            new_tokens=forward_input.num_tokens,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=kv_used_pages,
            kv_total_pages=kv_total_pages,
            page_size=self.config.page_size,
            linear_slots=self.mamba_slot_usage(),
            swa_tokens=self.swa_token_usage(),
        )

    def kv_page_usage(self) -> tuple[int, int]:
        """(used, total) KV pages; evictable prefix-cache pages are not used."""
        return self.cache_manager.page_usage()

    def mamba_slot_usage(self) -> tuple[int, int] | None:
        """(used, total) GDN slots; free slots and evictable snapshots are not used."""
        pool = self.engine.ctx.linear_state_pool
        if pool is None:
            return None
        total = pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def swa_token_usage(self) -> tuple[int, int] | None:
        """(used, total) window slots; free slots and the tree's unlocked live ones are not
        used."""
        pool = self.engine.ctx.swa_pool
        if pool is None:
            return None
        total = pool.swa_num_tokens - 1
        return total - self.cache_manager.swa_available_size, total

    def run_forever(
        self,
        recv_backend: queue.Queue[BaseBackendMsg],
        send_tokenizer: queue.Queue[BaseTokenizerMsg | None],
    ) -> None:
        """Serve backend messages until ExitMsg: admit, abort, step, and emit detokenize messages.

        Blocks on the queue when idle. live_uids are the requests whose tokens may still
        come; the caller reports them if this loop dies. Every abort is acknowledged with an
        ErrorReplyMsg after the step's tokens, the terminal reply the frontend's accounting
        waits for.
        """
        self.live_uids.clear()
        data: ForwardData | None = None
        while True:
            is_idle = data is None and not self.runnable
            pending = [recv_backend.get()] if is_idle else []
            while not recv_backend.empty():
                pending.append(recv_backend.get_nowait())
            msgs: list[BaseBackendMsg] = []
            for msg in pending:
                msgs.extend(msg.data if isinstance(msg, BatchBackendMsg) else [msg])

            replies: list[BaseTokenizerMsg] = []
            abort_acks: list[BaseTokenizerMsg] = []
            for msg in msgs:
                if isinstance(msg, ExitMsg):
                    return
                elif isinstance(msg, UserMsg):
                    try:
                        self.add_request(
                            msg.uid,
                            msg.input_ids,
                            msg.sampling_params,
                            msg.reuse_len,
                            msg.mm_items,
                            msg.mrope_positions,
                            msg.mrope_delta,
                        )
                    except ValueError as exc:
                        replies.append(
                            ErrorReplyMsg(
                                uid=msg.uid, error=str(exc), code="context_length_exceeded"
                            )
                        )
                    else:
                        self.live_uids.add(msg.uid)
                elif isinstance(msg, AbortBackendMsg):
                    self.abort_request(msg.uid)
                    self.live_uids.discard(msg.uid)
                    abort_acks.append(ErrorReplyMsg(uid=msg.uid, error="request aborted"))
                else:
                    raise NotImplementedError(f"unexpected backend message {msg!r}")

            data = self.overlap_loop(data)
            replies.extend(self.prompt_admissions)
            self.prompt_admissions = []
            kv_used_pages, kv_total_pages = self.kv_page_usage()
            mamba_used_slots, mamba_total_slots = self.mamba_slot_usage() or (0, 0)
            swa_used_tokens, swa_total_tokens = self.swa_token_usage() or (0, 0)
            for event in self.take_results():
                replies.append(
                    DetokenizeMsg(
                        uid=event.uid,
                        next_token=event.token,
                        finished=event.finished,
                        finish_reason=event.finish_reason,
                        kv_used_pages=kv_used_pages,
                        kv_total_pages=kv_total_pages,
                        mamba_used_slots=mamba_used_slots,
                        mamba_total_slots=mamba_total_slots,
                        swa_used_tokens=swa_used_tokens,
                        swa_total_tokens=swa_total_tokens,
                    )
                )
                if event.finished:
                    self.live_uids.discard(event.uid)
            replies.extend(abort_acks)
            if replies:
                send_tokenizer.put(BatchTokenizerMsg(data=replies))

            if data is None and not self.runnable:
                cache = self.moe_offload_cache
                self.status_reporter.report_idle(
                    moe_experts=None if cache is None else cache.hit_counts()
                )

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
            self.prefill_manager.schedule_next_batch(self.prefill_budget())
            or self.decode_manager.schedule_next_batch()
        )
        if batch is None:
            return None
        forward_input = self.prepare_batch(batch)
        # Reported only once the batch is prepared, so a prompt is admitted exactly once.
        self.prompt_admissions.extend(
            PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
            for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
        )
        return forward_input

    def prefill_budget(self) -> int:
        """max_extend_tokens, or with an offload cache, for a prefill that will look its experts
        up (no prefill overlap, or fewer pending tokens than its threshold), the cache's
        max_lookup_tokens; the cache reads a longer one's layers ahead whatever its length."""
        budget = self.config.max_extend_tokens
        cache = self.moe_offload_cache
        max_tokens = None if cache is None else cache.slot_layout.max_lookup_tokens
        if cache is None or max_tokens is None:
            return budget
        elif cache.prefill_overlap and self.prefill_manager.has_pending_tokens(
            cache.prefill_overlap_min_tokens
        ):
            return budget
        else:
            return min(budget, max_tokens)

    def prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        table = self.table_manager
        positions = np.concatenate(
            [np.arange(req.cached_len, req.device_len, dtype=np.int32) for req in batch.padded_reqs]
        )
        input_flat = np.concatenate(
            [
                table.flat_ids(req.table_idx, req.cached_len, req.device_len)
                for req in batch.padded_reqs
            ]
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
        encoder_cache = self.engine.encoder_cache
        if batch.is_prefill and encoder_cache is not None:
            jobs, plan, rows = plan_mm_batch(batch.padded_reqs, encoder_cache)
            if plan:
                batch.mm_encoder_jobs = jobs
                batch.mm_gather_plan = plan
                batch.mm_rows = mx.array(np.array(rows, dtype=np.int32))
                if self.engine.model_config.has_bidirectional_mm_blocks:
                    batch.mm_block_ends = mm_block_ends(batch.padded_reqs)
        if self.engine.model_config.is_mrope:
            batch.mrope_positions = mx.array(make_mrope_positions(batch.padded_reqs))
        self.engine.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            write_ids=mx.array(write_flat),
            num_tokens=batch.num_tokens,
        )

    def forward(self, forward_input: ForwardInput) -> ForwardData:
        batch, sample_args, write_ids, _ = forward_input
        self.restore_linear_states(batch)
        output = self.engine.forward_batch(batch, sample_args)
        self.table_manager.write_sampled(output.next_tokens, write_ids)
        mx.async_eval(output.next_tokens, self.table_manager.token_pool, *self.engine.pool_arrays())
        self.decode_manager.filter_reqs(batch.reqs)
        return ForwardData(forward_input, output)

    def restore_linear_states(self, batch: Batch) -> None:
        """Copy each prefix hit's tree snapshot into its live slot before the forward reads it."""
        pool = self.engine.ctx.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                assert req.linear_slot_idx is not None
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None

    def process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        batch = last_data.forward_input.batch
        next_tokens = host_ints(last_data.forward_output.next_tokens)
        self.engine.check_healthy()
        events: list[TokenEvent] = []
        new_finished: set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for req, next_token in zip(batch.reqs, next_tokens):
                if req.aborted:
                    # Aborted while this step was in flight: the abort popped it from its
                    # manager (no further step launches) and this drain frees it exactly
                    # once, now that the forward has landed. No event: the abort is the
                    # request's terminal reply.
                    self.free_req_resources(req)
                    continue
                if req in self.finished_reqs:
                    continue
                if self.cache_manager.swa_pool is not None and (
                    batch.is_decode or isinstance(req, ChunkedReq)
                ):
                    # This step has drained: its keys behind the window are read no more. A
                    # prompt's last chunk commits with the frontier of the chunks before it,
                    # as FreeToken's does; its own slots slide out from the next drain.
                    self.cache_manager.free_swa_out_of_window(req)
                if isinstance(req, ChunkedReq):
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
    "TokenEvent",
]
