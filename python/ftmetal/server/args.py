"""Server arguments: the engine config plus the HTTP front's knobs, parsed from argv.

Port of FreeToken server/args.py (ServerArgs, parse_args). Divergences: ServerArgs
extends EngineConfig directly (one process, no scheduler config or TP); only the flags
this engine has; the shell is a client of a running server, so there is no shell mode.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from ftmetal.attention import SUPPORTED_ATTENTION_BACKENDS
from ftmetal.engine.config import EngineConfig


@dataclass(frozen=True)
class ServerArgs(EngineConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    # Model id /v1/models reports; the checkpoint directory name when None.
    served_model_name: str | None = None
    # max_tokens of a request that omits it; the scheduler clamps it to the context left.
    max_output_tokens: int = 32768


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return number


def parse_args(argv: list[str], prog: str) -> ServerArgs:
    parser = argparse.ArgumentParser(prog=prog, description="Serve a model over the OpenAI API.")
    parser.add_argument(
        "--model-path",
        "--model",
        required=True,
        metavar="DIR",
        help="checkpoint directory (MLX format)",
    )
    parser.add_argument(
        "--served-model-name",
        default=ServerArgs.served_model_name,
        metavar="NAME",
        help="model id reported by /v1/models (default: the checkpoint directory name)",
    )
    parser.add_argument(
        "--host", dest="server_host", default=ServerArgs.server_host, metavar="HOST"
    )
    parser.add_argument(
        "--port",
        dest="server_port",
        type=positive_int,
        default=ServerArgs.server_port,
        metavar="PORT",
    )
    parser.add_argument(
        "--max-running-requests",
        "--max-running-req",
        dest="max_running_req",
        type=positive_int,
        metavar="N",
        default=ServerArgs.max_running_req,
        help="requests decoded together",
    )
    parser.add_argument(
        "--max-seq-len",
        "--max-seq-len-override",
        dest="max_seq_len_override",
        type=positive_int,
        metavar="N",
        default=ServerArgs.max_seq_len_override,
        help="context length, prompt plus output (default: the model's)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=positive_int,
        metavar="N",
        default=ServerArgs.max_output_tokens,
        help="max_tokens of a request that omits it",
    )
    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        dest="max_extend_tokens",
        type=positive_int,
        metavar="N",
        default=ServerArgs.max_extend_tokens,
        help="prefill tokens per batch; longer prompts are chunked",
    )
    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=positive_int,
        metavar="N",
        default=ServerArgs.num_page_override,
        help="KV pages to allocate (default: sized from the memory budget)",
    )
    parser.add_argument(
        "--attention-backend",
        "--attn",
        choices=SUPPORTED_ATTENTION_BACKENDS.supported_names(),
        default=ServerArgs.attention_backend,
    )
    namespace = parser.parse_args(argv)
    model_path = os.path.expanduser(namespace.model_path)
    if not os.path.isdir(model_path):
        parser.error(f"--model {model_path} is not a directory")
    return ServerArgs(
        model_path=model_path,
        served_model_name=namespace.served_model_name,
        server_host=namespace.server_host,
        server_port=namespace.server_port,
        max_running_req=namespace.max_running_req,
        max_seq_len_override=namespace.max_seq_len_override,
        max_output_tokens=namespace.max_output_tokens,
        max_extend_tokens=namespace.max_extend_tokens,
        num_page_override=namespace.num_page_override,
        attention_backend=namespace.attention_backend,
    )


__all__ = ["ServerArgs", "parse_args", "positive_int"]
