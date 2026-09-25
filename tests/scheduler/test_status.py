"""SchedulerStatusReporter: prefill lines every batch, decode lines per interval, idle summary."""

from __future__ import annotations

import numpy as np

from ftmetal.core import Batch, Req, SamplingParams
from ftmetal.scheduler.status import SchedulerStatusReporter


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def make_req(uid: int, cached_len: int, device_len: int) -> Req:
    return Req(
        input_ids=np.zeros(device_len, dtype=np.int32),
        table_idx=uid,
        cached_len=cached_len,
        output_len=8,
        uid=uid,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
    )


def report(reporter: SchedulerStatusReporter, batch: Batch, new_tokens: int) -> None:
    reporter.report_batch(
        batch,
        new_tokens=new_tokens,
        running_reqs=2,
        queue_reqs=1,
        kv_used_pages=5,
        kv_total_pages=20,
        page_size=32,
        linear_slots=(3, 12),
    )


def test_prefill_decode_and_idle_lines() -> None:
    lines: list[str] = []
    clock = FakeClock()
    reporter = SchedulerStatusReporter(lines.append, clock=clock, decode_log_interval=2)

    clock.now += 0.5
    prefill = Batch(reqs=[make_req(0, 64, 128), make_req(1, 0, 32)], phase="prefill")
    report(reporter, prefill, new_tokens=96)
    assert len(lines) == 1
    assert "#new-seq: 2, #new-token: 96, #cached-token: 64" in lines[0]
    assert "token usage: 0.25" in lines[0] and "gdn slots: 3/12" in lines[0]
    assert "input throughput (token/s): 192.00" in lines[0]

    decode = Batch(reqs=[make_req(0, 128, 129), make_req(1, 32, 33)], phase="decode")
    for _ in range(4):
        clock.now += 0.01
        report(reporter, decode, new_tokens=2)
    # Two decode lines: one per two decode batches, with tokens since the previous line.
    assert len(lines) == 3
    assert "#running-req: 2, #token: 160" in lines[1]
    assert "gen throughput (token/s): 200.00" in lines[2]
    assert "#queue-req: 1" in lines[2]

    clock.now += 1.0
    reporter.report_idle()
    assert lines[-1] == "Idle after 5 steps in 1.0 s, decode throughput (token/s): 200.0"
    # Idle with nothing since is silent.
    reporter.report_idle()
    assert len(lines) == 4
