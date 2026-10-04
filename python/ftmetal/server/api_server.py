"""The FastAPI app, the frontend state its routes share, and the two threads behind it: the
tokenizer worker and the scheduler.

Port of FreeToken server/api_server.py (FrontendManager, app, lifespan, run_api_server,
/generate), with the message loops of tokenizer/server.py (tokenize_worker) and of
scheduler/scheduler.py (run_forever and its message handling). Divergences: one process;
the tokenizer worker and the scheduler are threads fed by queue.Queue instead of processes
over ZMQ, and the scheduler thread is the only MLX caller; replies reach the event loop
through call_soon_threadsafe and wait per request on an asyncio.Queue; the scheduler is
built before the server accepts requests, so maintenance_state is "serving" while both
threads run and "failed" once one has died (no loading gate, no cache rebuild); both
TokenizeManagers probe their chat template's thinking and effort controls before serving
(FreeToken probes on first use, and warms the frontend's on the first cache poll); new_user
refuses with EngineUnavailableError, answered 503 as FreeToken answers a closed admission;
closing a request's reply stream before its terminal reply also aborts it, and an abort is
sent once; /generate reports a failed request as an {"error": ...} event before [DONE]; a
stop string matched by the detokenizer aborts the request; the decode state of an aborted
request is dropped when the tokenizer worker relays the abort, so tokens that race it are
dropped too; no supervisor, stats, CORS, control, accounting or cache routes and no request
log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from jinja2 import TemplateError
from pydantic import BaseModel

from ftmetal import __version__
from ftmetal.core import SamplingParams
from ftmetal.effort import ThinkingProfile
from ftmetal.message.backend import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    ExitMsg,
)
from ftmetal.message.frontend import BaseFrontendMsg, BatchFrontendMsg, UserReply
from ftmetal.message.tokenizer import (
    AbortMsg,
    BaseTokenizerMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    PromptAdmittedMsg,
    TokenizeMsg,
)
from ftmetal.scheduler.scheduler import Scheduler
from ftmetal.server.anthropic_api import register_anthropic_routes
from ftmetal.server.args import ServerArgs
from ftmetal.server.openai_api import register_openai_routes
from ftmetal.server.responses_api import register_responses_routes
from ftmetal.tokenizer import (
    DetokenizeManager,
    TokenizeManager,
    load_eos_token_ids,
    load_sampling_defaults,
)

logger = logging.getLogger(__name__)


class EngineUnavailableError(RuntimeError):
    """The scheduler or tokenizer thread is not running, so no request can be admitted."""


@dataclass(kw_only=True)
class FrontendManager:
    config: ServerArgs
    # The tokenizer worker's inbox; None stops it.
    send_tokenizer: queue.Queue[BaseTokenizerMsg | None]
    # Set by the lifespan once the scheduler thread is up.
    loop: asyncio.AbstractEventLoop | None = None
    backend_threads: list[threading.Thread] = field(default_factory=list)
    uid_counter: int = 0
    ack_map: dict[int, asyncio.Queue[UserReply]] = field(default_factory=dict)
    # The frontend's own TokenizeManager (count_tokens, the render check before a stream),
    # built on first use; the worker's stays on its thread.
    frontend_tokenize_manager: TokenizeManager | None = None
    frontend_tokenizer_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def maintenance_state(self) -> Literal["serving", "failed"]:
        alive = bool(self.backend_threads) and all(
            thread.is_alive() for thread in self.backend_threads
        )
        return "serving" if alive else "failed"

    def frontend_tokenizer(self) -> TokenizeManager:
        """Built and cached on first use; called from a worker thread, never the event loop."""
        with self.frontend_tokenizer_lock:
            if self.frontend_tokenize_manager is None:
                self.frontend_tokenize_manager = TokenizeManager(self.config.model_path)
            return self.frontend_tokenize_manager

    def warm_frontend_tokenizer(self) -> ThinkingProfile:
        """Build the frontend tokenizer and probe its chat template's thinking and effort
        controls; the lifespan runs it before serving."""
        return self.frontend_tokenizer().thinking_profile()

    def new_user(self) -> int:
        if self.maintenance_state != "serving":
            raise EngineUnavailableError(f"server unavailable: engine is {self.maintenance_state}")
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = asyncio.Queue()
        return uid

    async def send_one(self, msg: BaseTokenizerMsg) -> None:
        self.send_tokenizer.put(msg)

    def post(self, msg: BaseFrontendMsg) -> None:
        """Hand replies to the event loop; called from the tokenizer thread."""
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.deliver, msg)

    def deliver(self, msg: BaseFrontendMsg) -> None:
        replies = msg.data if isinstance(msg, BatchFrontendMsg) else [msg]
        for reply in replies:
            assert isinstance(reply, UserReply)
            # A request whose stream closed has left the map; its late replies drop.
            pending = self.ack_map.get(reply.uid)
            if pending is not None:
                pending.put_nowait(reply)

    async def wait_for_ack(self, uid: int) -> AsyncGenerator[UserReply]:
        """uid's replies up to the terminal one; closing the stream earlier aborts uid."""
        pending = self.ack_map[uid]
        finished = False
        try:
            while not finished:
                reply = await pending.get()
                finished = reply.finished
                yield reply
        finally:
            if self.ack_map.pop(uid, None) is not None and not finished:
                self.send_tokenizer.put(AbortMsg(uid=uid))

    async def stream_generate(self, uid: int) -> AsyncGenerator[bytes]:
        """uid's raw text as {"text": ...} server-sent events, then [DONE]."""
        async for ack in self.wait_for_ack(uid):
            if ack.error is not None:
                event = json.dumps({"error": ack.error}, ensure_ascii=False)
                yield f"data: {event}\n\n".encode()
            elif ack.incremental_output:
                # SSE: JSON-encode (so a newline inside a token can't break the frame)
                # and terminate every event with a blank line.
                event = json.dumps({"text": ack.incremental_output}, ensure_ascii=False)
                yield f"data: {event}\n\n".encode()
            if ack.finished:
                break
        yield b"data: [DONE]\n\n"

    async def stream_with_cancellation(
        self, generator: AsyncIterator[bytes | str], request: Request, uid: int
    ) -> AsyncGenerator[bytes | str]:
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info(f"client disconnected for request {uid}")
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int) -> None:
        await asyncio.sleep(0.1)
        if self.ack_map.pop(uid, None) is not None:
            logger.warning(f"aborting request {uid}")
            await self.send_one(AbortMsg(uid=uid))


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


def run_tokenizer(
    recv_tokenizer: queue.Queue[BaseTokenizerMsg | None],
    send_backend: queue.Queue[BaseBackendMsg],
    frontend: FrontendManager,
    tokenize_manager: TokenizeManager,
    detokenize_manager: DetokenizeManager,
) -> None:
    """The tokenizer worker: encodes prompts, decodes sampled tokens, relays aborts."""
    while True:
        pending = [recv_tokenizer.get()]
        while not recv_tokenizer.empty():
            pending.append(recv_tokenizer.get_nowait())
        msgs: list[BaseTokenizerMsg] = []
        for msg in pending:
            if msg is None:
                return
            msgs.extend(msg.data if isinstance(msg, BatchTokenizerMsg) else [msg])

        backend_msgs: list[BaseBackendMsg] = []
        replies: list[BaseFrontendMsg] = []
        for msg in msgs:
            if isinstance(msg, DetokenizeMsg):
                reply = detokenize_manager.detokenize(msg)
                if reply is not None:
                    replies.append(reply)
                    if reply.matched_stop is not None and not msg.finished:
                        backend_msgs.append(AbortBackendMsg(uid=msg.uid))
            elif isinstance(msg, PromptAdmittedMsg):
                replies.append(
                    UserReply(
                        uid=msg.uid,
                        incremental_output="",
                        finished=False,
                        prompt_tokens_delta=msg.prompt_tokens,
                        cached_tokens=msg.cached_tokens,
                    )
                )
            elif isinstance(msg, ErrorReplyMsg):
                detokenize_manager.discard(msg.uid)
                replies.append(
                    UserReply(
                        uid=msg.uid,
                        incremental_output="",
                        finished=True,
                        error=msg.error,
                        error_code=msg.code,
                    )
                )
            elif isinstance(msg, TokenizeMsg):
                # A conversation the template rejects fails its own request, not the worker.
                try:
                    user_msg = tokenize_manager.tokenize(msg)
                except (TemplateError, TypeError, ValueError) as exc:
                    logger.warning(f"could not encode request {msg.uid}: {exc!r}")
                    replies.append(
                        UserReply(
                            uid=msg.uid,
                            incremental_output="",
                            finished=True,
                            error=f"could not encode request: {exc}",
                        )
                    )
                else:
                    detokenize_manager.add_request(msg.uid, msg.sampling_params.stop_strs)
                    backend_msgs.append(user_msg)
            elif isinstance(msg, AbortMsg):
                detokenize_manager.discard(msg.uid)
                backend_msgs.append(AbortBackendMsg(uid=msg.uid))
            else:
                raise NotImplementedError(f"unexpected tokenizer message {msg!r}")
        if backend_msgs:
            send_backend.put(BatchBackendMsg(data=backend_msgs))
        if replies:
            frontend.post(BatchFrontendMsg(data=replies))


def run_scheduler(
    config: ServerArgs,
    eos_token_ids: set[int],
    recv_backend: queue.Queue[BaseBackendMsg],
    send_tokenizer: queue.Queue[BaseTokenizerMsg | None],
    started: Future[None],
) -> None:
    """The scheduler thread, the only MLX caller: admits and aborts, runs the overlap loop."""
    try:
        scheduler = Scheduler(config, eos_token_ids)
    except Exception as exc:
        # The lifespan re-raises it and the server refuses to start.
        started.set_exception(exc)
        return
    started.set_result(None)

    try:
        scheduler.run_forever(recv_backend, send_tokenizer)
    except Exception:
        # The thread ends here: running requests fail and new ones get a 503.
        logger.exception("the scheduler thread failed")
        failed: list[BaseTokenizerMsg] = [
            ErrorReplyMsg(uid=uid, error="the engine failed") for uid in sorted(scheduler.live_uids)
        ]
        send_tokenizer.put(BatchTokenizerMsg(data=failed))


def create_app(config: ServerArgs) -> FastAPI:
    tokenize_manager = TokenizeManager(config.model_path)
    eos_token_ids = load_eos_token_ids(config.model_path, tokenize_manager.tokenizer)
    detokenize_manager = DetokenizeManager(tokenize_manager.tokenizer, eos_token_ids)
    recv_tokenizer: queue.Queue[BaseTokenizerMsg | None] = queue.Queue()
    recv_backend: queue.Queue[BaseBackendMsg] = queue.Queue()
    frontend = FrontendManager(config=config, send_tokenizer=recv_tokenizer)
    engine_started: Future[None] = Future()
    engine_thread = threading.Thread(
        target=run_scheduler,
        args=(config, eos_token_ids, recv_backend, recv_tokenizer, engine_started),
        name="ftmetal-scheduler",
        daemon=True,
    )
    tokenizer_thread = threading.Thread(
        target=run_tokenizer,
        args=(recv_tokenizer, recv_backend, frontend, tokenize_manager, detokenize_manager),
        name="ftmetal-tokenizer",
        daemon=True,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine_thread.start()
        # Both TokenizeManagers probe their template once, before the first request.
        await asyncio.to_thread(tokenize_manager.thinking_profile)
        await asyncio.to_thread(frontend.warm_frontend_tokenizer)
        await asyncio.wrap_future(engine_started)
        frontend.loop = asyncio.get_running_loop()
        tokenizer_thread.start()
        frontend.backend_threads = [engine_thread, tokenizer_thread]
        logger.info(f"serving {config.model_path}")
        yield
        recv_backend.put(ExitMsg())
        await asyncio.to_thread(engine_thread.join)
        recv_tokenizer.put(None)
        await asyncio.to_thread(tokenizer_thread.join)

    defaults = load_sampling_defaults(config.model_path)
    model_sampling: dict[str, float | int] = {
        "temperature": defaults.temperature,
        "top_k": defaults.top_k,
        "top_p": defaults.top_p,
    }
    app = FastAPI(title="ftmetal", version=__version__, lifespan=lifespan)
    register_openai_routes(app, lambda: frontend, lambda: model_sampling)
    register_anthropic_routes(app, lambda: frontend, lambda: model_sampling)
    register_responses_routes(app, lambda: frontend, lambda: model_sampling)

    @app.exception_handler(EngineUnavailableError)
    async def engine_unavailable(request: Request, exc: EngineUnavailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    @app.post("/generate", response_model=None)
    async def generate(req: GenerateRequest, request: Request) -> StreamingResponse | JSONResponse:
        """A raw prompt sampled with the framework defaults, streamed as {"text": ...} events."""
        if req.max_tokens < 1:
            return JSONResponse(
                {"error": f"max_tokens must be at least 1, got {req.max_tokens}"}, status_code=400
            )
        uid = frontend.new_user()
        sampling_params = SamplingParams(ignore_eos=req.ignore_eos, max_tokens=req.max_tokens)
        await frontend.send_one(
            TokenizeMsg(uid=uid, text=req.prompt, sampling_params=sampling_params)
        )
        return StreamingResponse(
            frontend.stream_with_cancellation(frontend.stream_generate(uid), request, uid),
            media_type="text/event-stream",
        )

    return app


def run_api_server(config: ServerArgs) -> None:
    uvicorn.run(create_app(config), host=config.server_host, port=config.server_port)


__all__ = [
    "EngineUnavailableError",
    "FrontendManager",
    "GenerateRequest",
    "create_app",
    "run_api_server",
    "run_scheduler",
    "run_tokenizer",
]
