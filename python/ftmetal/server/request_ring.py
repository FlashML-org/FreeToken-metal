"""In-memory ring of recent API requests for /v1/requests and the /v1/stats latency figures.

A bounded deque + a monotonic all-time cursor: clients pull incrementally with
?since=<next_cursor>. Records are appended by the generation layer and an HTTP middleware.
Purely in-process; request_logger.py owns the on-disk JSONL.

Port of FreeToken server/request_ring.py. Divergence: the FrontendManager owns its ring, so
there is no module singleton (nor its test-only reset).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

from pydantic import JsonValue


@dataclass
class RequestRecord:
    ts: str
    method: str
    path: str
    status: int
    model: str | None
    duration_ms: int
    ttft_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    stream: bool | None
    error: str | None


class RequestRing:
    def __init__(self, capacity: int = 512) -> None:
        self.buffer: deque[tuple[int, RequestRecord]] = deque(maxlen=capacity)
        # All-time monotonic cursor: survives eviction.
        self.next_cursor = 0

    def add(self, record: RequestRecord) -> None:
        self.buffer.append((self.next_cursor, record))
        self.next_cursor += 1

    def since(self, cursor: int, limit: int) -> tuple[list[dict[str, JsonValue]], int]:
        """Records with id >= cursor (up to limit) and the cursor to poll with next: past the
        last returned record when limit truncated the result, else the all-time count."""
        matched = [(index, record) for index, record in self.buffer if index >= cursor]
        out = [asdict(record) for _, record in matched[:limit]]
        if len(out) < len(matched):
            next_cursor = matched[len(out) - 1][0] + 1
        else:
            next_cursor = self.next_cursor
        return out, next_cursor

    def p95_ms(self) -> int:
        durations = sorted(record.duration_ms for _, record in self.buffer)
        if not durations:
            return 0
        k = max(0, math.ceil(0.95 * len(durations)) - 1)
        return int(durations[k])

    def ttft_mean_ms(self) -> int:
        """Mean TTFT over the records that have one."""
        values = [record.ttft_ms for _, record in self.buffer if record.ttft_ms is not None]
        if not values:
            return 0
        return int(round(sum(values) / len(values)))


__all__ = ["RequestRecord", "RequestRing"]
