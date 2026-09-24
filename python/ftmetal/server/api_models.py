"""OpenAI chat-completion request and response schemas.

Port of FreeToken server/api_models.py. Divergences: chat completions and the model
list only; text content only; tools are declared so they can be refused (no tool
parser yet); responses are typed models instead of dicts.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from ftmetal.message.tokenizer import FinishReason


class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[TextContentPart] | None = None
    reasoning_content: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    # -1 (or 0) disables top-k.
    top_k: int | None = Field(default=None, ge=-1)
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    chat_template_kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    tools: list[dict[str, JsonValue]] | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    reasoning_content: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatResponseMessage
    finish_reason: FinishReason


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None

    @model_serializer(mode="wrap")
    def omit_absent_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, str]:
        # A delta carries only the fields that changed.
        return {key: value for key, value in handler(self).items() if value is not None}


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: FinishReason | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: UsageInfo | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "ftmetal"
    root: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorInfo(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorInfo


__all__ = [
    "ChatCompletionChoice",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatResponseMessage",
    "DeltaMessage",
    "ErrorInfo",
    "ErrorResponse",
    "Message",
    "ModelCard",
    "ModelList",
    "StreamOptions",
    "TextContentPart",
    "UsageInfo",
]
