"""Server arguments: the engine config plus the HTTP front's knobs, parsed from argv.

Port of FreeToken server/args.py (ServerArgs, parse_args, and its tool-call and reasoning
parser inference). Divergences: ServerArgs extends EngineConfig directly (one process, no
scheduler config or TP); only the flags this engine has, and --disable-command-graph for
FreeToken's --cuda-graph-max-bs (the graph ladder always ends at max_running_req); of the
multimodal flags, no --mm-processor-kwargs (no transformers processor),
--mm-embed-cache-device or --mm-encoder-weights (the tower stays resident in unified
memory); --image-max-tokens defaults per image processor; the shell is a client of a running
server, so there is no shell mode; the parser inference reads config.json itself (no
transformers) and is module-level. MoE: --moe-strategy takes auto, fused and offload, and
refuses FreeToken's cpu and hybrid (no host banks on unified memory); --expert-load is
accepted and ignored; --memory-ratio defaults to unset (the working set less a headroom);
no --moe-cpu-threads, --moe-cpu-layers, --moe-hybrid-max-fetch or --moe-prefill-hit-d2d.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from ftmetal.attention import SUPPORTED_ATTENTION_BACKENDS
from ftmetal.engine.config import (
    CacheTypeChoice,
    EngineConfig,
    MoeStrategyChoice,
    command_graph_from_env,
)
from ftmetal.kvcache.linear_state_pool import CONVERSATION_SNAPSHOTS, DEFAULT_GDN_SNAPSHOTS
from ftmetal.mm.config import ENCODER_KINDS, EncoderKind, MultimodalConfig
from ftmetal.tokenizer import read_json_object

TOOL_CALL_PARSER_CHOICES = (
    "auto",
    "llama3",
    "qwen",
    "qwen25",
    "qwen3_coder",
    "mistral",
    "deepseekv32",
    "gemma4",
    "glm47",
    "minimax",
    "minimax_m3",
    "muse_glimmer",
    "gpt_oss",
    "gpt-oss",
)
# FreeToken's strategies that compute experts on the CPU from host banks.
HOST_MOE_STRATEGIES = ("cpu", "hybrid")
MOE_STRATEGY_CHOICES = (*get_args(MoeStrategyChoice), *HOST_MOE_STRATEGIES)

logger = logging.getLogger(__name__)

REASONING_PARSER_CHOICES = (
    "auto",
    "off",
    "deepseekv32",
    "gpt_oss",
    "qwen3",
    "glm",
    "minimax",
    "minimax_m3",
    "muse_glimmer",
    "gemma4",
)


@dataclass(frozen=True)
class ServerArgs(EngineConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    # Model id /v1/models reports; parse_args defaults it to the checkpoint directory name.
    served_model_name: str | None = None
    # Tool-call format the chat routes parse out of the completion.
    tool_call_parser: str = "llama3"
    # Splits the chain-of-thought into reasoning_content; None disables it.
    reasoning_parser: str | None = None
    # max_tokens of a request that omits it; the scheduler clamps it to the context left.
    max_output_tokens: int = 32768
    # Report the prefix-cache hit in each response's usage block (OpenAI
    # prompt_tokens_details.cached_tokens, Anthropic cache_read_input_tokens, Responses
    # input_tokens_details.cached_tokens).
    enable_cache_report: bool = False
    # Comma-separated hostname allowlist for client-supplied image URLs; empty admits any
    # domain.
    allowed_media_domains: str = ""
    # Directory file:// image refs may be read from; empty rejects local files.
    allowed_local_media_path: str = ""
    # Comma-separated CORS allow-list for browser and webview clients; empty disables the
    # CORS headers, "*" allows any origin. Parity: FreeToken's desktop-app origins.
    cors_origins: str = "tauri://localhost,http://tauri.localhost,http://localhost:1420"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return number


def unit_fraction(value: str) -> float:
    """A number in [0, 1] (FreeToken's --moe-cache-rate)."""
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError(f"expected a number in [0, 1], got {value}")
    return number


def memory_fraction(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError(f"expected a number in (0, 1], got {value}")
    return number


class DeprecatedAlias(argparse.Action):
    """An old flag (FreeToken's _DeprecatedAlias): warns when given, stores under the new
    flag's dest."""

    def __init__(
        self,
        option_strings: list[str],
        dest: str,
        *,
        new_flag: str,
        choices: tuple[str, ...],
        default: str,
        help: str,
    ) -> None:
        super().__init__(option_strings, dest, choices=choices, default=default, help=help)
        self.new_flag = new_flag

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        assert isinstance(values, str), f"{option_string} takes one value"
        logger.warning(f"{option_string} is deprecated; use {self.new_flag}")
        setattr(namespace, self.dest, values)


def model_family_marker(model_path: str) -> str:
    """The path, model types and architectures of a checkpoint, lowercased into one string."""
    config = read_json_object(Path(model_path) / "config.json")
    text_config = config.get("text_config")
    text_config = text_config if isinstance(text_config, dict) else {}
    candidates = [
        model_path,
        str(config.get("model_type", "")),
        str(text_config.get("model_type", "")),
    ]
    for architectures in (config.get("architectures"), text_config.get("architectures")):
        if isinstance(architectures, list):
            candidates.append(" ".join(str(name) for name in architectures))
    return " ".join(candidates).lower()


def infer_tool_call_parser(model_path: str) -> str:
    marker = model_family_marker(model_path)
    if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
        return "gpt_oss"
    # M3 first: its marker also contains the bare "minimax" substring, but the
    # namespaced tool grammar is a different protocol from M2's.
    if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
        return "minimax_m3"
    if "minimax" in marker:
        return "minimax"
    if "muse_glimmer" in marker or "muse-glimmer" in marker or "museglimmer" in marker:
        return "muse_glimmer"
    if "gemma4" in marker:
        return "gemma4"
    if "qwen4_exp" in marker or "qwen4exp" in marker or "qwen3.8-flash" in marker:
        return "qwen3_coder"
    if "qwen3_5" in marker or "qwen3.5" in marker or ("qwen3" in marker and "coder" in marker):
        return "qwen3_coder"
    if "qwen" in marker:
        return "qwen25"
    if "deepseek" in marker and ("v4" in marker or "deepseek_v4" in marker):
        return "deepseekv32"
    if "deepseek" in marker and ("v3.2" in marker or "v32" in marker):
        return "deepseekv32"
    if "glm" in marker:
        return "glm47"
    if "mistral" in marker:
        return "mistral"
    return "llama3"


def infer_reasoning_parser(model_path: str) -> str | None:
    marker = model_family_marker(model_path)
    if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
        return "gpt_oss"
    if "deepseek" in marker and any(tag in marker for tag in ("v4", "deepseek_v4", "v3.2", "v32")):
        return "deepseekv32"
    if "qwen4_exp" in marker or "qwen4exp" in marker or "qwen3.8-flash" in marker:
        return "qwen3"
    if "qwen3" in marker or "qwen3.5" in marker or "qwen3_5" in marker:
        return "qwen3"
    if "glm" in marker:
        return "glm"
    # M3 first ("minimax" is a substring): <mm:think> tags + 3 thinking gears,
    # not M2's always-on implicit <think>.
    if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
        return "minimax_m3"
    if "minimax" in marker:
        return "minimax"
    if "muse_glimmer" in marker or "muse-glimmer" in marker or "museglimmer" in marker:
        return "muse_glimmer"
    if "gemma4" in marker:
        return "gemma4"
    return None


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
        "--page-size",
        type=positive_int,
        default=ServerArgs.page_size,
        help="KV page size in tokens (FreeToken's --page-size)",
    )
    parser.add_argument(
        "--cache-type",
        choices=get_args(CacheTypeChoice),
        default=ServerArgs.cache_type,
        help="radix reuses cached prompt prefixes (with GDN state snapshots on a model with "
        "linear attention); naive reuses nothing",
    )
    parser.add_argument(
        "--swa-windows",
        type=positive_int,
        metavar="N",
        default=ServerArgs.swa_windows,
        help="window pool of the radix cache on a sliding-window model, in sliding windows "
        "(default: the fewest that hold the running requests and a prefill, and 8 more)",
    )
    parser.add_argument(
        "--gdn-snapshots",
        type=positive_int,
        metavar="N",
        default=ServerArgs.gdn_snapshots,
        help=f"GDN state snapshots the radix cache keeps on a model with linear attention, "
        f"{CONVERSATION_SNAPSHOTS} per cached conversation, at least {CONVERSATION_SNAPSHOTS} "
        f"(default: {DEFAULT_GDN_SNAPSHOTS})",
    )
    parser.add_argument(
        "--attention-backend",
        "--attn",
        choices=SUPPORTED_ATTENTION_BACKENDS.supported_names(),
        default=ServerArgs.attention_backend,
        help="metal: paged flash-prefill and flash-decode kernels; mlx: SDPA over gathered pages",
    )
    parser.add_argument(
        "--memory-ratio",
        type=memory_fraction,
        metavar="R",
        default=ServerArgs.memory_ratio,
        help="the memory budget for weights, the MoE cache, the KV pools and the activations, "
        "as a fraction of Metal's recommended working set (default: the working set less a "
        "headroom); a small fraction emulates a Mac with less memory",
    )
    parser.add_argument(
        "--moe-strategy",
        choices=MOE_STRATEGY_CHOICES,
        default=ServerArgs.moe_strategy,
        help="how a MoE model's routed experts are served: fused keeps them resident, offload "
        "reads them from the checkpoint into a slot cache, auto (default) keeps them resident "
        "when they fit the memory budget beside --kv-reserve-tokens of KV and offloads "
        "otherwise; cpu and hybrid are not supported on macOS",
    )
    parser.add_argument(
        "--moe-backend",
        dest="moe_strategy",
        action=DeprecatedAlias,
        new_flag="--moe-strategy",
        choices=MOE_STRATEGY_CHOICES,
        default=argparse.SUPPRESS,
        help="[Deprecated] Use --moe-strategy.",
    )
    moe_cache_group = parser.add_mutually_exclusive_group()
    moe_cache_group.add_argument(
        "--moe-cache-size",
        type=non_negative_int,
        metavar="N",
        default=ServerArgs.moe_cache_size,
        help="expert slots of the MoE offload cache (with prefill overlap at least twice the "
        "experts of a layer, else once)",
    )
    moe_cache_group.add_argument(
        "--moe-cache-rate",
        type=unit_fraction,
        metavar="R",
        default=ServerArgs.moe_cache_rate,
        help="the MoE offload cache's slots as a fraction of all routed experts",
    )
    moe_cache_group.add_argument(
        "--moe-cache-auto",
        action="store_true",
        help="size the MoE offload cache from the memory budget, experts first: the KV pools "
        "get --kv-reserve-tokens, the experts the rest (the default of an offload without a "
        "size flag)",
    )
    parser.add_argument(
        "--kv-reserve-tokens",
        type=non_negative_int,
        metavar="N",
        default=ServerArgs.kv_reserve_tokens,
        help="KV tokens kept before the experts by --moe-cache-auto and by auto's "
        f"resident check (default {ServerArgs.kv_reserve_tokens})",
    )
    parser.add_argument(
        "--moe-cache-policy",
        choices=["lru"],
        default=ServerArgs.moe_cache_policy,
        help="the MoE offload cache's eviction policy",
    )
    parser.add_argument(
        "--moe-layer-slots",
        choices=["hot", "equal"],
        default=ServerArgs.moe_layer_slots,
        help="how the MoE offload cache splits its slots among the layers: by the model's "
        "hot-expert table (expert_hot_rank.json in its directory, else the one shipped for the "
        "model) or evenly",
    )
    parser.add_argument(
        "--disable-moe-prefill-overlap",
        action="store_false",
        dest="moe_prefill_overlap",
        help="look up every prefill's experts like a decode's, instead of reading each layer's "
        "ahead into two windows of the cache once a prefill is long; lowers the cache's "
        "floor from twice a layer's experts to once",
    )
    parser.add_argument(
        "--expert-load",
        choices=["auto", "serial", "parallel"],
        default=None,
        help="accepted and ignored: the experts are read from the checkpoint straight into "
        "the offload cache, with no host bank to load",
    )
    parser.add_argument(
        "--disable-command-graph",
        action="store_true",
        help="decode eagerly instead of replaying captured CommandGraphs; without the flag "
        "FTMETAL_COMMAND_GRAPH=0/1 decides (default 1)",
    )
    parser.add_argument(
        "--enable-cache-report",
        action="store_true",
        help="report the prompt tokens served from the prefix cache in each response's usage "
        "(on /v1/messages input_tokens then excludes them, as Anthropic bills)",
    )
    parser.add_argument(
        "--tool-call-parser",
        choices=TOOL_CALL_PARSER_CHOICES,
        default="auto",
        help="tool-call format parsed out of chat completions; auto picks the model family's",
    )
    parser.add_argument(
        "--reasoning-parser",
        choices=REASONING_PARSER_CHOICES,
        default="auto",
        help="splits the chain-of-thought into reasoning_content; auto picks the model "
        "family's (<think> for qwen3, the thought channel for gemma4), off disables it",
    )
    parser.add_argument(
        "--text-model-only",
        action="store_true",
        help="Serve a multimodal checkpoint text-only: no encoder tower is built (its memory goes "
        "to the KV pools) and every multimodal input is rejected. Same as --mm-disable with "
        "every encoder kind.",
    )
    parser.add_argument(
        "--mm-disable",
        nargs="+",
        choices=list(ENCODER_KINDS),
        default=[],
        metavar="{vision,audio}",
        help="Encoder towers to leave unbuilt; every input they would serve is rejected.",
    )
    parser.add_argument(
        "--image-min-tokens",
        type=positive_int,
        metavar="N",
        default=MultimodalConfig.image_min_tokens,
        help="Fewest tokens an image may take: the image processor scales smaller images up to "
        "it, in the family's own units. Default: the processor's own limit.",
    )
    parser.add_argument(
        "--image-max-tokens",
        type=positive_int,
        metavar="N",
        default=MultimodalConfig.image_max_tokens,
        help="Most tokens an image may take: the image processor scales larger images down to "
        "it, in the family's own units (Qwen VL: one token per 32x32 pixels). Default: the "
        "image processor's default (Qwen VL: 4096, a 2048x2048 image), else the checkpoint's "
        "limit.",
    )
    parser.add_argument(
        "--allowed-media-domains",
        default=ServerArgs.allowed_media_domains,
        metavar="DOMAINS",
        help="comma-separated hostname allowlist for client-supplied image URLs; empty "
        "(default) allows any domain",
    )
    parser.add_argument(
        "--allowed-local-media-path",
        default=ServerArgs.allowed_local_media_path,
        metavar="DIR",
        help="directory that file:// image refs may be read from; unset (default) rejects "
        "local files",
    )
    parser.add_argument(
        "--cors-origins",
        default=ServerArgs.cors_origins,
        metavar="ORIGINS",
        help="comma-separated CORS allow-list for browser and webview clients (default: the "
        "local desktop-app origins); '' disables it, '*' allows any origin",
    )
    namespace = parser.parse_args(argv)
    model_path = os.path.expanduser(namespace.model_path)
    if not os.path.isdir(model_path):
        parser.error(f"--model {model_path} is not a directory")
    if namespace.tool_call_parser == "auto":
        tool_call_parser = infer_tool_call_parser(model_path)
    else:
        tool_call_parser = namespace.tool_call_parser
    if namespace.reasoning_parser == "auto":
        reasoning_parser = infer_reasoning_parser(model_path)
    elif namespace.reasoning_parser == "off":
        reasoning_parser = None
    else:
        reasoning_parser = namespace.reasoning_parser
    # A bad media root is a deployment mistake: fail at startup, not per request.
    if namespace.allowed_local_media_path:
        media_root = os.path.realpath(os.path.expanduser(namespace.allowed_local_media_path))
        if not os.path.isdir(media_root):
            parser.error(f"--allowed-local-media-path {media_root} is not a directory")
    else:
        media_root = ""
    image_min_tokens, image_max_tokens = namespace.image_min_tokens, namespace.image_max_tokens
    if (
        image_min_tokens is not None
        and image_max_tokens is not None
        and image_min_tokens > image_max_tokens
    ):
        parser.error(
            f"--image-min-tokens {image_min_tokens} exceeds --image-max-tokens {image_max_tokens}"
        )
    disabled_encoders: set[EncoderKind] = set(ENCODER_KINDS) if namespace.text_model_only else set()
    disabled_encoders.update(namespace.mm_disable)
    if namespace.moe_strategy in HOST_MOE_STRATEGIES:
        parser.error(
            f"--moe-strategy {namespace.moe_strategy} computes experts on the CPU from host "
            "banks, which macOS's unified memory has no use for; use offload or fused"
        )
    if namespace.expert_load is not None:
        logger.warning(
            "--expert-load is ignored: the experts are read from the checkpoint into the "
            "offload cache"
        )
    # FreeToken: an offload without a size flag sizes its cache automatically.
    moe_cache_auto = namespace.moe_cache_auto or (
        namespace.moe_strategy == "offload"
        and namespace.moe_cache_size == 0
        and not namespace.moe_cache_rate
    )
    return ServerArgs(
        model_path=model_path,
        served_model_name=(
            namespace.served_model_name or os.path.basename(os.path.normpath(model_path))
        ),
        tool_call_parser=tool_call_parser,
        reasoning_parser=reasoning_parser,
        enable_cache_report=namespace.enable_cache_report,
        cors_origins=namespace.cors_origins,
        server_host=namespace.server_host,
        server_port=namespace.server_port,
        max_running_req=namespace.max_running_req,
        max_seq_len_override=namespace.max_seq_len_override,
        max_output_tokens=namespace.max_output_tokens,
        max_extend_tokens=namespace.max_extend_tokens,
        num_page_override=namespace.num_page_override,
        page_size=namespace.page_size,
        cache_type=namespace.cache_type,
        swa_windows=namespace.swa_windows,
        gdn_snapshots=namespace.gdn_snapshots,
        attention_backend=namespace.attention_backend,
        command_graph=not namespace.disable_command_graph and command_graph_from_env(),
        memory_ratio=namespace.memory_ratio,
        moe_strategy=namespace.moe_strategy,
        moe_cache_size=namespace.moe_cache_size,
        moe_cache_rate=namespace.moe_cache_rate if namespace.moe_cache_rate else None,
        moe_cache_auto=moe_cache_auto,
        kv_reserve_tokens=namespace.kv_reserve_tokens,
        moe_cache_policy=namespace.moe_cache_policy,
        moe_layer_slots=namespace.moe_layer_slots,
        moe_prefill_overlap=namespace.moe_prefill_overlap,
        mm=MultimodalConfig(
            disabled_encoders=frozenset(disabled_encoders),
            image_min_tokens=image_min_tokens,
            image_max_tokens=image_max_tokens,
        ),
        allowed_media_domains=namespace.allowed_media_domains,
        allowed_local_media_path=media_root,
    )


__all__ = [
    "HOST_MOE_STRATEGIES",
    "MOE_STRATEGY_CHOICES",
    "REASONING_PARSER_CHOICES",
    "TOOL_CALL_PARSER_CHOICES",
    "DeprecatedAlias",
    "ServerArgs",
    "infer_reasoning_parser",
    "infer_tool_call_parser",
    "memory_fraction",
    "model_family_marker",
    "non_negative_int",
    "parse_args",
    "positive_int",
    "unit_fraction",
]
