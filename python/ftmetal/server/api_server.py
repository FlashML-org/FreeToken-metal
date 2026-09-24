"""The FastAPI app and the two threads behind it: the tokenizer worker and the scheduler.

Port of FreeToken server/api_server.py (app, lifespan, run_api_server), with the
message loops of tokenizer/server.py (tokenize_worker) and of scheduler/scheduler.py
(run_forever and its message handling). Divergences: one process; the tokenizer worker
and the scheduler are threads fed by queue.Queue instead of processes over ZMQ, and
the scheduler thread is the only MLX caller; the scheduler is built before the server
accepts requests (no loading gate); a stop string matched by the detokenizer aborts
the request; aborts are not acknowledged, because decode state is created at
tokenization and dropped at the abort, so late tokens have nothing to attach to; no
supervisor, stats, CORS, control or accounting routes.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncIterator
from concurrent.futures import Future
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from jinja2 import TemplateError

from ftmetal import __version__
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
from ftmetal.server.args import ServerArgs
from ftmetal.server.generation import FrontendManager
from ftmetal.server.openai_api import register_openai_routes
from ftmetal.tokenizer import (
    DetokenizeManager,
    TokenizeManager,
    load_eos_token_ids,
    load_sampling_defaults,
)

logger = logging.getLogger(__name__)


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
    frontend = FrontendManager(send_tokenizer=recv_tokenizer)
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

    app = FastAPI(title="ftmetal", version=__version__, lifespan=lifespan)
    register_openai_routes(app, frontend, config, load_sampling_defaults(config.model_path))
    return app


def run_api_server(config: ServerArgs) -> None:
    uvicorn.run(create_app(config), host=config.server_host, port=config.server_port)


__all__ = ["create_app", "run_api_server", "run_scheduler", "run_tokenizer"]
