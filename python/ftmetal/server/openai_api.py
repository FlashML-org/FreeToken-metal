"""OpenAI routes: POST /v1/chat/completions and /v1/completions, streamed or not, and
GET /v1/models.

Port of FreeToken server/openai_api.py. Divergences: chat completions, completions and
the model list only; tools, images, reasoning_effort and the thinking field are refused
or ignored (thinking is toggled through chat_template_kwargs.enable_thinking); no
maintenance gate or request log. As in FreeToken, a completion prompt is raw text with no
chat template or reasoning split, n is ignored (one choice per prompt), and several
prompts run one after another.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ftmetal.core import SamplingParams
from ftmetal.message.tokenizer import ChatMessage, FinishReason, TokenizeMsg
from ftmetal.server.api_models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatResponseMessage,
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    ErrorInfo,
    ErrorResponse,
    Message,
    ModelCard,
    ModelList,
    UsageInfo,
)
from ftmetal.server.args import ServerArgs
from ftmetal.server.generation import (
    ContentDelta,
    EngineUnavailableError,
    FrontendManager,
    GenerationError,
    GenSpec,
    ReasoningDelta,
    generate_events,
    generate_full,
    resolve_sampling,
    submit_generation,
)


def create_error_response(
    message: str,
    status_code: int = 400,
    err_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    error = ErrorResponse(error=ErrorInfo(message=message, type=err_type, param=param, code=code))
    return JSONResponse(status_code=status_code, content=error.model_dump())


def sse(payload: BaseModel) -> bytes:
    return f"data: {payload.model_dump_json()}\n\n".encode()


def usage_info(prompt_tokens: int, completion_tokens: int) -> UsageInfo:
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def chat_message(message: Message) -> ChatMessage:
    if isinstance(message.content, list):
        content = "".join(part.text for part in message.content)
    else:
        content = message.content or ""
    rendered: ChatMessage = {"role": message.role, "content": content}
    if message.reasoning_content is not None:
        rendered["reasoning_content"] = message.reasoning_content
    return rendered


async def stream_chat_completion_chunks(
    uid: int, req: ChatCompletionRequest, spec: GenSpec, frontend: FrontendManager
) -> AsyncGenerator[bytes]:
    """generate_events as chat.completion.chunk server-sent events."""
    completion_id = f"chatcmpl-{uid}"
    created = int(time.time())

    def chunk(delta: DeltaMessage, finish_reason: FinishReason | None = None) -> bytes:
        choice = ChatCompletionChunkChoice(delta=delta, finish_reason=finish_reason)
        return sse(
            ChatCompletionChunk(
                id=completion_id, created=created, model=req.model, choices=[choice]
            )
        )

    yield chunk(DeltaMessage(role="assistant", content=""))
    usage: UsageInfo | None = None
    try:
        async with aclosing(generate_events(uid, spec, frontend)) as events:
            async for event in events:
                if isinstance(event, ReasoningDelta):
                    yield chunk(DeltaMessage(reasoning_content=event.text))
                elif isinstance(event, ContentDelta):
                    yield chunk(DeltaMessage(content=event.text))
                else:
                    usage = usage_info(event.prompt_tokens, event.completion_tokens)
                    yield chunk(DeltaMessage(), event.finish_reason)
    except GenerationError as exc:
        # The headers are out, so a failure can only ride in the stream.
        yield sse(ErrorResponse(error=ErrorInfo(message=str(exc), code=exc.code)))
    if usage is not None and req.stream_options is not None and req.stream_options.include_usage:
        yield sse(
            ChatCompletionChunk(
                id=completion_id, created=created, model=req.model, choices=[], usage=usage
            )
        )
    yield b"data: [DONE]\n\n"


async def stream_completion_chunks(
    uid: int, req: CompletionRequest, frontend: FrontendManager
) -> AsyncGenerator[bytes]:
    """uid's raw text as text_completion.chunk server-sent events."""
    completion_id = f"cmpl-{uid}"
    created = int(time.time())

    def chunk(choices: list[CompletionChoice], usage: UsageInfo | None = None) -> bytes:
        return sse(
            CompletionChunk(
                id=completion_id, created=created, model=req.model, choices=choices, usage=usage
            )
        )

    prompt_tokens = 0
    completion_tokens = 0
    finish_reason: FinishReason | None = None
    error: ErrorResponse | None = None
    async with aclosing(frontend.wait_for_ack(uid)) as replies:
        async for reply in replies:
            if reply.error is not None:
                error = ErrorResponse(error=ErrorInfo(message=reply.error, code=reply.error_code))
            else:
                prompt_tokens += reply.prompt_tokens_delta
                completion_tokens += reply.completion_tokens_delta
                finish_reason = reply.finish_reason
                if reply.incremental_output:
                    yield chunk([CompletionChoice(text=reply.incremental_output)])
    if error is not None:
        yield sse(error)
    else:
        assert finish_reason is not None
        yield chunk([CompletionChoice(text="", finish_reason=finish_reason)])
        if req.stream_options is not None and req.stream_options.include_usage:
            yield chunk([], usage_info(prompt_tokens, completion_tokens))
    yield b"data: [DONE]\n\n"


def register_openai_routes(
    app: FastAPI, frontend: FrontendManager, config: ServerArgs, model_sampling: SamplingParams
) -> None:
    @app.get("/v1/models")
    async def v1_models() -> ModelList:
        model_id = config.served_model_name or config.model_name
        return ModelList(data=[ModelCard(id=model_id, root=config.model_path)])

    @app.post("/v1/chat/completions", response_model=None)
    async def v1_chat_completions(
        req: ChatCompletionRequest,
    ) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
        if req.n != 1:
            return create_error_response("only n=1 is supported", param="n")
        if req.tools:
            return create_error_response("tools are not supported yet", param="tools")
        max_tokens = (
            req.max_tokens if req.max_completion_tokens is None else req.max_completion_tokens
        )
        spec = GenSpec(
            messages=[chat_message(message) for message in req.messages],
            sampling_params=resolve_sampling(
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                max_tokens=max_tokens,
                ignore_eos=req.ignore_eos,
                stop=req.stop,
                model_sampling=model_sampling,
                default_max_tokens=config.max_output_tokens,
            ),
            chat_template_kwargs=req.chat_template_kwargs,
        )
        try:
            uid = submit_generation(spec, frontend)
        except EngineUnavailableError as exc:
            return create_error_response(str(exc), status_code=503, err_type="server_error")
        if req.stream:
            return StreamingResponse(
                stream_chat_completion_chunks(uid, req, spec, frontend),
                media_type="text/event-stream",
            )
        try:
            result = await generate_full(uid, spec, frontend)
        except GenerationError as exc:
            return create_error_response(str(exc), code=exc.code)
        message = ChatResponseMessage(
            content=result.content, reasoning_content=result.reasoning or None
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uid}",
            created=int(time.time()),
            model=req.model,
            choices=[ChatCompletionChoice(message=message, finish_reason=result.finish_reason)],
            usage=usage_info(result.prompt_tokens, result.completion_tokens),
        )

    @app.post("/v1/completions", response_model=None)
    async def v1_completions(
        req: CompletionRequest,
    ) -> CompletionResponse | StreamingResponse | JSONResponse:
        prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
        text_prompts = [prompt for prompt in prompts if isinstance(prompt, str)]
        if len(text_prompts) != len(prompts):
            unsupported = (
                "OpenAI token-id prompt inputs are not supported; pass text prompt strings instead"
            )
        elif req.logprobs is not None:
            unsupported = "logprobs is not supported"
        elif req.echo:
            unsupported = "echo is not supported"
        elif req.suffix is not None:
            unsupported = "suffix is not supported"
        elif req.logit_bias is not None:
            unsupported = "logit_bias is not supported"
        elif (req.response_format or {}).get("type") not in (None, "text"):
            unsupported = (
                "response_format json_object/json_schema is not supported (no constrained decoding)"
            )
        elif req.stream and len(text_prompts) != 1:
            unsupported = "Streaming completions only support a single text prompt"
        else:
            unsupported = None
        if unsupported is not None:
            return create_error_response(unsupported)
        max_tokens = (
            req.max_tokens if req.max_completion_tokens is None else req.max_completion_tokens
        )

        def submit(prompt: str) -> int:
            uid = frontend.new_user()
            # One SamplingParams per request: the scheduler clamps max_tokens in place.
            sampling_params = resolve_sampling(
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                max_tokens=max_tokens,
                ignore_eos=req.ignore_eos,
                stop=req.stop,
                model_sampling=model_sampling,
                default_max_tokens=config.max_output_tokens,
            )
            frontend.send_one(TokenizeMsg(uid=uid, text=prompt, sampling_params=sampling_params))
            return uid

        if req.stream:
            try:
                uid = submit(text_prompts[0])
            except EngineUnavailableError as exc:
                return create_error_response(str(exc), status_code=503, err_type="server_error")
            return StreamingResponse(
                stream_completion_chunks(uid, req, frontend), media_type="text/event-stream"
            )
        else:
            choices: list[CompletionChoice] = []
            prompt_tokens = 0
            completion_tokens = 0
            for index, prompt in enumerate(text_prompts):
                try:
                    uid = submit(prompt)
                except EngineUnavailableError as exc:
                    return create_error_response(str(exc), status_code=503, err_type="server_error")
                text: list[str] = []
                finish_reason: FinishReason | None = None
                async with aclosing(frontend.wait_for_ack(uid)) as replies:
                    async for reply in replies:
                        if reply.error is not None:
                            return create_error_response(reply.error, code=reply.error_code)
                        prompt_tokens += reply.prompt_tokens_delta
                        completion_tokens += reply.completion_tokens_delta
                        finish_reason = reply.finish_reason
                        text.append(reply.incremental_output)
                assert finish_reason is not None
                choices.append(
                    CompletionChoice(index=index, text="".join(text), finish_reason=finish_reason)
                )
            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=req.model,
                choices=choices,
                usage=usage_info(prompt_tokens, completion_tokens),
            )


__all__ = [
    "chat_message",
    "create_error_response",
    "register_openai_routes",
    "sse",
    "stream_chat_completion_chunks",
    "stream_completion_chunks",
    "usage_info",
]
