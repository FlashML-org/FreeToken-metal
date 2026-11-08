"""What a launch knows about the server, the model and the agent's binary, what a preparer
returns, the limits, thinking levels and provider every agent is told, and the y/n prompt and
command probe the launch steps share."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import JsonValue

# Used when the server reports no context length. Guessing low only costs earlier compaction.
FALLBACK_CONTEXT_WINDOW = 128_000
MAX_OUTPUT_TOKENS_CAP = 32_768
# The server accepts any key, and some agents refuse an empty one.
PLACEHOLDER_API_KEY = "ftmetal"
# The variable that carries it for agents that read their key from one.
API_KEY_ENV = "FTMETAL_API_KEY"
PROVIDER_ID = "ftmetal"
CLOUD_PROVIDER_API_KEY_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_OAUTH_TOKEN",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
)
PROBE_TIMEOUT_SECONDS = 5


@dataclass(frozen=True, kw_only=True)
class ServerURL:
    origin: str
    openai_base_url: str


@dataclass(frozen=True, kw_only=True)
class ServedModel:
    model_id: str
    # The longest prompt plus output a request may hold; None when the server does not say.
    context_length: int | None
    accepts_images: bool
    # The effort names the template grades, weakest first; empty when it grades none.
    reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    # Whether a request can switch thinking off and on, and whether it is on by default.
    thinking_toggleable: bool = False
    thinks_by_default: bool = False


@dataclass(frozen=True, kw_only=True)
class LaunchContext:
    server: ServerURL
    model: ServedModel
    extra_args: list[str]
    dry_run: bool
    assume_yes: bool
    # None only for a dry run of an agent that is not installed.
    agent_binary: str | None


class PreLaunch(Protocol):
    """What runs just before the agent starts, given its environment; returns a process the agent
    needs, which then outlives it only until the agent exits."""

    def __call__(self, environment: dict[str, str]) -> subprocess.Popen[bytes] | None: ...


@dataclass(frozen=True, kw_only=True)
class CommandSpec:
    argv: list[str]
    env: dict[str, str]
    # Set only where the user's environment has no value of its own.
    default_env: dict[str, str] = field(default_factory=dict)
    unset_env: tuple[str, ...] = ()
    prelaunch: PreLaunch | None = None


class AgentPreparer(Protocol):
    def __call__(self, context: LaunchContext) -> CommandSpec: ...


def context_window(context: LaunchContext) -> int:
    return context.model.context_length or FALLBACK_CONTEXT_WINDOW


def max_output_tokens(context: LaunchContext) -> int:
    """Scales with the window: agents derive their compaction headroom from this."""
    return max(1, min(context_window(context) // 4, MAX_OUTPUT_TOKENS_CAP))


def input_modalities(context: LaunchContext) -> list[JsonValue]:
    return ["text", "image"] if context.model.accepts_images else ["text"]


def reasoning_levels(context: LaunchContext) -> tuple[list[str], str | None]:
    """The thinking levels an agent may offer, weakest first, and the default among them: the
    graded efforts, led by "none" when thinking turns off; an on/off switch alone is none/high."""
    model = context.model
    if model.reasoning_efforts:
        off = ["none"] if model.thinking_toggleable else []
        levels, default = [*off, *model.reasoning_efforts], model.default_reasoning_effort
    elif model.thinking_toggleable:
        levels, default = ["none", "high"], "high" if model.thinks_by_default else "none"
    else:
        levels, default = [], None
    return levels, default if default in levels else None


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def confirm(question: str) -> bool:
    """A yes/no answer from the terminal, yes on Enter; no terminal, Ctrl-C or Ctrl-D is no."""
    if not is_interactive():
        print(f"{question} requires confirmation; re-run with -y to continue.", file=sys.stderr)
        return False
    else:
        while True:
            print(f"{question} [Y/n] ", end="", file=sys.stderr, flush=True)
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                line = ""
            answer = line.strip().lower()
            if not line:
                print(file=sys.stderr)
                return False
            elif answer in ("", "y", "yes"):
                return True
            elif answer in ("n", "no"):
                return False
            else:
                print("Please answer y or n.", file=sys.stderr)


def command_version(binary: str) -> tuple[int, int] | None:
    """The major and minor version that binary --version prints last; None if it prints none."""
    words = command_output([binary, "--version"]).split()
    parts = words[-1].lstrip("v").split(".") if words else []
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    else:
        return None


def command_output(argv: list[str]) -> str:
    """What a short read-only command prints; empty when it cannot run, fails or hangs."""
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


__all__ = [
    "API_KEY_ENV",
    "CLOUD_PROVIDER_API_KEY_ENV",
    "MAX_OUTPUT_TOKENS_CAP",
    "PLACEHOLDER_API_KEY",
    "PROVIDER_ID",
    "AgentPreparer",
    "CommandSpec",
    "LaunchContext",
    "PreLaunch",
    "ServedModel",
    "ServerURL",
    "command_output",
    "command_version",
    "confirm",
    "context_window",
    "input_modalities",
    "is_interactive",
    "max_output_tokens",
    "reasoning_levels",
]
