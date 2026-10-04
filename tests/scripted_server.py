"""The real API server over a scheduler that samples a fixed script for every request.

ScriptedScheduler replaces the scheduler thread's Scheduler, so the routes, the message
flow, the detokenizer and the parsers run as served, without a model. The tiny checkpoint
encodes one token per byte (plus <think>, </think> and the ChatML specials), so a scripted
completion streams character by character and every marker arrives split across chunks.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import JsonValue
from tokenizers import Tokenizer

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.message.tokenizer import PromptAdmittedMsg
from ftmetal.scheduler.scheduler import ForwardData, Scheduler, TokenEvent
from ftmetal.scheduler.status import SchedulerStatusReporter
from ftmetal.server.args import ServerArgs

SCRIPT = "Let me think.</think>\n\nThe answer is 42."
# One token per byte except </think>, plus EOS.
SCRIPT_TOKENS = len(SCRIPT.encode()) - len("</think>") + 2
WAIT_SECONDS = 5.0


class ScriptedScheduler(Scheduler):
    """Samples script's tokens then EOS for every request, one token per step."""

    script: ClassVar[str] = SCRIPT
    step_seconds: ClassVar[float] = 0.0
    max_prompt_tokens: ClassVar[int] = 4096
    # The prefix-cache hit every admitted prompt reports.
    cached_tokens: ClassVar[int] = 0
    # Whether a step raises, which kills the scheduler thread.
    fails: ClassVar[bool] = False
    # The pools each step reports: used KV pages are the requests admitted so far, used GDN
    # slots the running ones; no GDN pool when its total is 0.
    kv_total_pages: ClassVar[int] = 64
    mamba_total_slots: ClassVar[int] = 0
    instances: ClassVar[list[ScriptedScheduler]] = []

    def __init__(self, config: EngineConfig, eos_token_ids: set[int]) -> None:
        self.tokenizer = Tokenizer.from_file(str(Path(config.model_path) / "tokenizer.json"))
        self.eos_token_id = min(eos_token_ids)
        self.results: list[TokenEvent] = []
        self.prompt_admissions: list[PromptAdmittedMsg] = []
        self.live_uids: set[int] = set()
        self.status_reporter = SchedulerStatusReporter(lambda line: None)
        # uid -> (tokens left, whether the last one is EOS)
        self.running: dict[int, tuple[list[int], bool]] = {}
        self.admitted: dict[int, tuple[list[int], SamplingParams]] = {}
        self.reuse_lens: dict[int, int | None] = {}
        self.aborted: list[int] = []
        ScriptedScheduler.instances.append(self)

    def add_request(
        self,
        uid: int,
        input_ids: np.ndarray,
        sampling_params: SamplingParams,
        reuse_len: int | None = None,
    ) -> None:
        if len(input_ids) > self.max_prompt_tokens:
            raise ValueError(f"prompt is too long: {len(input_ids)} tokens")
        self.admitted[uid] = (input_ids.tolist(), sampling_params)
        self.reuse_lens[uid] = reuse_len
        self.prompt_admissions.append(
            PromptAdmittedMsg(
                uid=uid, prompt_tokens=len(input_ids), cached_tokens=self.cached_tokens
            )
        )
        script_tokens = [*self.tokenizer.encode(self.script).ids, self.eos_token_id]
        # Sampling past EOS repeats the script until max_tokens.
        repeats = sampling_params.max_tokens // len(script_tokens) + 1
        tokens = script_tokens * repeats if sampling_params.ignore_eos else script_tokens
        self.running[uid] = (
            tokens[: sampling_params.max_tokens],
            len(tokens) <= sampling_params.max_tokens,
        )

    def abort_request(self, uid: int) -> None:
        self.running.pop(uid, None)
        self.aborted.append(uid)

    @property
    def runnable(self) -> bool:
        return len(self.running) > 0

    def kv_page_usage(self) -> tuple[int, int]:
        return len(self.admitted), self.kv_total_pages

    def mamba_slot_usage(self) -> tuple[int, int] | None:
        return (len(self.running), self.mamba_total_slots) if self.mamba_total_slots else None

    def swa_token_usage(self) -> tuple[int, int] | None:
        return None

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        time.sleep(self.step_seconds)
        if self.fails and self.running:
            raise RuntimeError("scripted step failure")
        for uid, (tokens, ends_with_eos) in list(self.running.items()):
            token = tokens.pop(0)
            finish_reason: Literal["stop", "length"] | None = None
            if not tokens:
                finish_reason = "stop" if ends_with_eos else "length"
                self.running.pop(uid)
            self.results.append(TokenEvent(uid, token, not tokens, finish_reason))
        return None

    def prompt_text(self, uid: int) -> str:
        """The prompt uid was admitted with, decoded."""
        return self.tokenizer.decode(self.admitted[uid][0], skip_special_tokens=False)


def wait_for_abort(scheduler: ScriptedScheduler, uid: int) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    while uid not in scheduler.aborted:
        assert time.monotonic() < deadline, f"request {uid} was never aborted"
        time.sleep(0.01)


@contextmanager
def running_server(app: FastAPI) -> Iterator[int]:
    """app served over HTTP on a free loopback port, for clients that need a real socket;
    yields the port."""
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + WAIT_SECONDS
    while not server.started:
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.01)
    try:
        yield server.servers[0].sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        thread.join()


def tiny_server_args(
    tiny_model_dir: Path,
    tool_call_parser: str = "qwen3_coder",
    reasoning_parser: str | None = "qwen3",
    enable_cache_report: bool = False,
) -> ServerArgs:
    """The tiny checkpoint mimics Qwen3.6's template, so it gets Qwen3.6's parsers by default;
    the model id is the directory name, as parse_args sets it."""
    return ServerArgs(
        model_path=str(tiny_model_dir),
        served_model_name=tiny_model_dir.name,
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        enable_cache_report=enable_cache_report,
    )


def sse_events(body: str) -> list[str]:
    """The data payloads of a server-sent event stream."""
    return [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]


def named_sse_events(body: str) -> list[tuple[str, dict[str, JsonValue]]]:
    """(event name, payload) of a stream whose frames are named, as Anthropic and Responses
    send them."""
    events: list[tuple[str, dict[str, JsonValue]]] = []
    for frame in body.split("\n\n"):
        lines = frame.splitlines()
        if not lines:
            continue
        assert lines[0].startswith("event: ") and lines[1].startswith("data: "), frame
        payload = json.loads(lines[1].removeprefix("data: "))
        assert isinstance(payload, dict)
        events.append((lines[0].removeprefix("event: "), payload))
    return events
