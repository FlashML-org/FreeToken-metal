"""Optional inbound-request logging.

With FTMETAL_API_LOG_DIR set, every request
dispatched to an API handler (/v1/chat/completions, /v1/completions, /v1/messages,
/v1/messages/count_tokens, /v1/responses, /generate) is appended as one JSON line to a
per-process requests-<start>-<pid>.jsonl under that directory. Requests that fail body
validation never reach a handler, so they are not recorded. A writer thread drains a bounded
queue, so a slow log disk never blocks the event loop; when the queue is full, records drop.
Unset, log_request is a no-op.

Port of FreeToken server/request_logger.py. Divergences: the directory is read when the app
starts, not at import, and the log is stopped (its queue written out) when the app stops, so
one process can run several apps; the writer is a RequestLog object rather than module
globals; the enqueue path guards only against a full queue (every caller hands it a request
model, which always serializes).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time

from fastapi import Request
from pydantic import BaseModel

from ftmetal.env import read_env

logger = logging.getLogger(__name__)

# Bounded so a stalled log disk drops records instead of growing memory without limit (and
# never blocks the event loop on a full queue).
MAX_QUEUE = 10_000
# How long stopping the log waits for the writer to drain the queue.
CLOSE_TIMEOUT_S = 2.0


class RequestLog:
    """One app's request log: an owner-only JSONL file and the thread that writes it."""

    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        # One file per process: the PID keeps two servers sharing a directory (or a restart
        # within the same second) from colliding on one path.
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self.path = os.path.join(log_dir, f"requests-{stamp}-{os.getpid()}.jsonl")
        # 0o600: request bodies can carry prompts and personal data.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self.file = os.fdopen(fd, "a", encoding="utf-8")
        self.queue: queue.Queue[str | None] = queue.Queue(maxsize=MAX_QUEUE)
        self.has_warned = False
        self.writer = threading.Thread(target=self.write_loop, name="reqlog-writer", daemon=True)
        self.writer.start()

    def warn_once(self, message: str) -> None:
        if not self.has_warned:
            logger.warning(message)
            self.has_warned = True

    def write_loop(self) -> None:
        while (line := self.queue.get()) is not None:
            try:
                self.file.write(line)
                self.file.flush()
            except OSError as exc:  # a full or vanished log disk must not kill the writer
                self.warn_once(f"failed to write request log: {exc}")
        self.file.close()

    def enqueue(self, line: str) -> None:
        try:
            self.queue.put_nowait(line)
        except queue.Full:
            self.warn_once("request log queue full; dropping records (slow log disk?)")

    def close(self) -> None:
        """Write out what is queued and stop the writer, giving up on a stalled disk."""
        try:
            self.queue.put(None, timeout=CLOSE_TIMEOUT_S)
        except queue.Full:
            return
        self.writer.join(timeout=CLOSE_TIMEOUT_S)


# The running app's log; None when FTMETAL_API_LOG_DIR is unset.
active_log: RequestLog | None = None


def init() -> None:
    """Open the log FTMETAL_API_LOG_DIR names, at app start, so a bad path is reported at boot
    instead of on the first request; a path that cannot be opened disables logging."""
    global active_log
    raw_dir = read_env("API_LOG_DIR")
    if raw_dir is None:
        active_log = None
    else:
        log_dir = os.path.expanduser(os.path.expandvars(raw_dir))
        try:
            active_log = RequestLog(log_dir)
        except OSError as exc:
            logger.warning(f"cannot open request log under {log_dir!r}: {exc}")
            active_log = None
        else:
            logger.info(f"logging API requests to {active_log.path}")


def shutdown() -> None:
    """Write out the running app's log and stop it, at app stop."""
    global active_log
    request_log, active_log = active_log, None
    if request_log is not None:
        request_log.close()


def log_request(endpoint: str, req: BaseModel, request: Request | None = None) -> None:
    """Enqueue one record for an inbound API request: the fields the client set. A no-op
    unless FTMETAL_API_LOG_DIR is set; never blocks."""
    request_log = active_log
    if request_log is None:
        return
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "endpoint": endpoint,
        "client": request.client.host if request and request.client else None,
        "request": req.model_dump(mode="json", exclude_unset=True),
    }
    request_log.enqueue(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = ["RequestLog", "init", "log_request", "shutdown"]
