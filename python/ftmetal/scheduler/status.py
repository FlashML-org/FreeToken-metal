"""Periodic scheduler status lines from accumulators.

Port of FreeToken scheduler/status.py (SchedulerStatusReporter). Divergences: reported
at schedule time, before the launch, so the batch's cached/new token counts are read
straight from the reqs; and an idle line summarising the burst since the last idle, with
the MoE offload cache's hit rate over it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ftmetal.core import Batch


@dataclass
class SchedulerStatusReporter:
    log: Callable[[str], None]
    clock: Callable[[], float] = time.perf_counter
    decode_log_interval: int = 40
    _last_prefill_time: float = field(init=False)
    _last_decode_time: float = field(init=False)
    _decode_forward_count: int = field(default=0, init=False)
    _decode_generated_tokens: int = field(default=0, init=False)
    # Since the last idle line.
    _burst_start: float | None = field(default=None, init=False)
    _burst_steps: int = field(default=0, init=False)
    _burst_decode_tokens: int = field(default=0, init=False)
    _burst_decode_time: float = field(default=0.0, init=False)
    _last_step_time: float | None = field(default=None, init=False)
    # The offload cache's (hit, missed) expert counts at the last idle line.
    _last_moe_experts: tuple[int, int] = field(default=(0, 0), init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self._last_prefill_time = now
        self._last_decode_time = now
        self.decode_log_interval = max(1, self.decode_log_interval)

    def report_batch(
        self,
        batch: Batch,
        *,
        new_tokens: int,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        linear_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
    ) -> None:
        now = self.clock()
        if self._burst_start is None:
            self._burst_start = now
        self._burst_steps += 1
        if batch.is_prefill:
            self._last_step_time = None
            self._report_prefill(
                batch,
                now,
                new_tokens=new_tokens,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                linear_slots=linear_slots,
                swa_tokens=swa_tokens,
            )
        else:
            # Decode step time is the gap between consecutive decode schedules.
            if self._last_step_time is not None:
                self._burst_decode_tokens += batch.size
                self._burst_decode_time += now - self._last_step_time
            self._last_step_time = now
            self._report_decode(
                batch,
                now,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                page_size=page_size,
                linear_slots=linear_slots,
                swa_tokens=swa_tokens,
            )

    def report_idle(self, moe_experts: tuple[int, int] | None = None) -> None:
        """One line for the burst since the last idle; no-op if nothing ran. moe_experts: the
        offload cache's (hit, missed) expert counts so far."""
        if self._burst_start is None:
            return
        elapsed = self.clock() - self._burst_start
        decode_tok_s = (
            self._burst_decode_tokens / self._burst_decode_time
            if self._burst_decode_time > 0
            else 0.0
        )
        moe_msg = ""
        if moe_experts is not None:
            hits = moe_experts[0] - self._last_moe_experts[0]
            misses = moe_experts[1] - self._last_moe_experts[1]
            self._last_moe_experts = moe_experts
            moe_msg = f", moe hit rate: {_usage_ratio(hits, hits + misses):.3f}"
        self.log(
            f"Idle after {self._burst_steps} steps in {elapsed:.1f} s, "
            f"decode throughput (token/s): {decode_tok_s:.1f}{moe_msg}"
        )
        self._burst_start = None
        self._burst_steps = 0
        self._burst_decode_tokens = 0
        self._burst_decode_time = 0.0
        self._last_step_time = None

    def _report_prefill(
        self,
        batch: Batch,
        now: float,
        *,
        new_tokens: int,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        linear_slots: tuple[int, int] | None,
        swa_tokens: tuple[int, int] | None,
    ) -> None:
        gap = now - self._last_prefill_time
        self._last_prefill_time = now
        cached_tokens = sum(req.cached_len for req in batch.reqs)
        input_throughput = new_tokens / gap if gap > 0 else 0.0
        self.log(
            f"Prefill batch, "
            f"#new-seq: {batch.size}, "
            f"#new-token: {new_tokens}, "
            f"#cached-token: {cached_tokens}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_linear_msg(linear_slots)}"
            f"#running-req: {running_reqs}, "
            f"#queue-req: {queue_reqs}, "
            f"input throughput (token/s): {input_throughput:.2f}"
        )

    def _report_decode(
        self,
        batch: Batch,
        now: float,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        linear_slots: tuple[int, int] | None,
        swa_tokens: tuple[int, int] | None,
    ) -> None:
        self._decode_forward_count += 1
        self._decode_generated_tokens += batch.size
        if self._decode_forward_count % self.decode_log_interval != 0:
            return
        gap = now - self._last_decode_time
        self._last_decode_time = now
        gen_throughput = self._decode_generated_tokens / gap if gap > 0 else 0.0
        self._decode_generated_tokens = 0
        self.log(
            f"Decode batch, "
            f"#running-req: {running_reqs}, "
            f"#token: {kv_used_pages * page_size}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_linear_msg(linear_slots)}"
            f"gen throughput (token/s): {gen_throughput:.2f}, "
            f"#queue-req: {queue_reqs}"
        )


def _usage_ratio(used: int, total: int) -> float:
    return used / total if total > 0 else 0.0


def _linear_msg(linear_slots: tuple[int, int] | None) -> str:
    """GDN state slot occupancy for hybrid models; empty for the rest."""
    if linear_slots is None:
        return ""
    used, total = linear_slots
    return f"gdn slots: {used}/{total}, "


def _swa_msg(swa_tokens: tuple[int, int] | None) -> str:
    """Window slot occupancy for sliding-window models; empty for the rest."""
    if swa_tokens is None:
        return ""
    used, total = swa_tokens
    return f"swa tokens: {used}/{total}, "


__all__ = ["SchedulerStatusReporter"]
