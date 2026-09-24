"""ftm: serve a model over the OpenAI API, or chat with a running server.

    ftm serve --model <checkpoint dir> [--host H] [--port P] [...]
    ftm shell [--server URL]

The shell is an ordinary client of POST /v1/chat/completions, so it gets the prompt
rendering, sampling defaults and reasoning split every other client gets.
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import readline  # noqa: F401 - importing it gives input() line editing and history
import sys
import time
import urllib.parse
from collections.abc import Iterator

from ftmetal.env import read_env
from ftmetal.server.api_models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ErrorResponse,
    Message,
    ModelList,
    StreamOptions,
)
from ftmetal.server.api_server import run_api_server
from ftmetal.server.args import parse_args

DEFAULT_SERVER_URL = "http://127.0.0.1:1919"
# A turn's first token waits for the prefill of the whole conversation.
CHAT_READ_TIMEOUT = 900.0
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DIM = "\033[2m"
RESET = "\033[0m"
SHELL_HELP = """/think   toggle thinking (on by default)
/reset   clear the conversation
/exit    quit (Ctrl-D also works)"""


class ShellError(Exception):
    """The server refused or failed a request; the message is printed as is."""


def open_connection(server_url: str) -> http.client.HTTPConnection:
    parts = urllib.parse.urlsplit(server_url)
    if parts.scheme != "http" or parts.hostname is None:
        raise ShellError(f"expected an http://host:port server URL, got {server_url}")
    return http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=CHAT_READ_TIMEOUT)


def served_model_id(server_url: str) -> str:
    connection = open_connection(server_url)
    try:
        connection.request("GET", "/v1/models")
        models = ModelList.model_validate_json(connection.getresponse().read())
    finally:
        connection.close()
    return models.data[0].id


def stream_chat(server_url: str, request: ChatCompletionRequest) -> Iterator[ChatCompletionChunk]:
    """The chunks of one streamed completion; closing early disconnects, which aborts it."""
    connection = open_connection(server_url)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=request.model_dump_json(exclude_none=True),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise ShellError(f"HTTP {response.status}: {response.read().decode()}")
        for line in response:
            if not line.startswith(b"data: "):
                continue
            payload = line.removeprefix(b"data: ").strip()
            if payload == b"[DONE]":
                return
            event = json.loads(payload)
            if isinstance(event, dict) and "error" in event:
                raise ShellError(ErrorResponse.model_validate(event).error.message)
            yield ChatCompletionChunk.model_validate(event)
    finally:
        connection.close()


def run_turn(server_url: str, request: ChatCompletionRequest, style: bool) -> str | None:
    """Stream one reply to the terminal; returns the content shown, None if the turn failed."""
    dim, reset = (DIM, RESET) if style else ("", "")
    content: list[str] = []
    in_reasoning = False
    started = time.perf_counter()
    first_token_at: float | None = None
    try:
        for chunk in stream_chat(server_url, request):
            if chunk.usage is not None:
                decode_seconds = time.perf_counter() - (first_token_at or started)
                rate = chunk.usage.completion_tokens / max(decode_seconds, 1e-9)
                ttft = (first_token_at or started) - started
                print(
                    f"{dim}[{chunk.usage.prompt_tokens} prompt + "
                    f"{chunk.usage.completion_tokens} completion tokens, "
                    f"TTFT {ttft:.2f} s, {rate:.1f} tok/s]{reset}"
                )
            for choice in chunk.choices:
                delta = choice.delta
                if (delta.reasoning_content or delta.content) and first_token_at is None:
                    first_token_at = time.perf_counter()
                if delta.reasoning_content:
                    in_reasoning = True
                    print(f"{dim}{delta.reasoning_content}{reset}", end="", flush=True)
                # The whitespace between </think> and the answer is not shown.
                text = (delta.content or "") if content else (delta.content or "").lstrip()
                if text:
                    if in_reasoning:
                        in_reasoning = False
                        print("\n")
                    content.append(text)
                    print(text, end="", flush=True)
                if choice.finish_reason is not None:
                    print(f"\n{dim}(finish: {choice.finish_reason}){reset}")
    except KeyboardInterrupt:
        print(f"\n{dim}(cancelled){reset}")
    except (ShellError, OSError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return None
    return "".join(content)


def run_shell(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="ftm shell", description="Chat with a running ftmetal server in the terminal."
    )
    parser.add_argument(
        "--server",
        default=read_env("HOST") or DEFAULT_SERVER_URL,
        help=f"server URL (default: $FTMETAL_HOST, else {DEFAULT_SERVER_URL})",
    )
    server_url = parser.parse_args(argv).server
    try:
        model_id = served_model_id(server_url)
    except (ShellError, OSError) as exc:
        print(f"cannot reach {server_url}: {exc}", file=sys.stderr)
        return 1
    print(f"ftm shell -> {model_id} @ {server_url}  (/help for commands)")
    history: list[Message] = []
    thinking = True
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue
        if not line:
            continue
        elif line == "/exit":
            return 0
        elif line == "/help":
            print(SHELL_HELP)
        elif line == "/reset":
            history = []
            print("(conversation cleared)")
        elif line == "/think":
            thinking = not thinking
            print(f"(thinking {'on' if thinking else 'off'})")
        elif line.startswith("/"):
            print(f"unknown command {line}; try /help")
        else:
            messages = [*history, Message(role="user", content=line)]
            request = ChatCompletionRequest(
                model=model_id,
                messages=messages,
                stream=True,
                stream_options=StreamOptions(include_usage=True),
                chat_template_kwargs={"enable_thinking": thinking},
            )
            answer = run_turn(server_url, request, style=sys.stdout.isatty())
            # The reasoning is never sent back: the template drops it from past turns.
            if answer is not None:
                history = [*messages, Message(role="assistant", content=answer)]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    command, rest = args[0], args[1:]
    if command == "serve":
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
        run_api_server(parse_args(rest, prog="ftm serve"))
        return 0
    elif command == "shell":
        return run_shell(rest)
    else:
        print(f"unknown command {command!r}; expected serve or shell", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
