"""ftm launch: read the running server's model, install the agent if missing, write its config
and become it.

Port of FreeToken launch.py. Divergences: /v1/models alone describes the model; proxies are
ignored when asking the server, and the agent is told to skip them for it; the process execs the
agent, since on macOS nothing needs a parent left around it, unless the agent needs a helper
process that must stop with it.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import shlex
import signal
import subprocess
import sys
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

from pydantic import ValidationError

from ftmetal.env import read_env
from ftmetal.launch.agents import PREPARERS
from ftmetal.launch.codex import restore_codex
from ftmetal.launch.context import (
    API_KEY_ENV,
    CommandSpec,
    LaunchContext,
    ServedModel,
    ServerURL,
    confirm,
    context_window,
)
from ftmetal.launch.install import (
    AGENT_INSTALLERS,
    AgentCommand,
    ensure_agent_installed,
    resolve_agent_binary,
)
from ftmetal.server.api_models import ModelList

SERVER_TIMEOUT_SECONDS = 5
PROXY_ENV = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
NO_PROXY_ENV = ("NO_PROXY", "no_proxy")
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
AGENT_ALIASES = {"deepseek-harness": "dsh", "clawdbot": "openclaw", "moltbot": "openclaw"}
RESTORERS = {"codex": restore_codex}
# These start only after a yes to "launch now?" once --config has configured them.
LAUNCH_AFTER_CONFIG = ("claude", "codex", "opencode", "openclaw", "pi")
INTERRUPTED_EXIT_STATUS = 130


def resolve_server_url(server: str, default_port: int) -> ServerURL:
    """server, a host, a host:port or a URL whose path may end in /v1, as URLs an agent can
    connect to; a host without a scheme is on default_port, a URL's on its scheme's port."""
    raw = server.strip()
    scheme, has_scheme, rest = raw.partition("://")
    host_port, _, path = (rest if has_scheme else raw).partition("/")
    # A bare IPv6 address has colons a port would be read from.
    try:
        address = ipaddress.IPv6Address(host_port.strip("[]"))
        host_port = f"[{address}]"
    except ValueError:
        pass
    parsed = urlsplit(f"{scheme if has_scheme else 'http'}://{host_port}/{path}")
    prefix = parsed.path.strip("/").removesuffix("v1").strip("/")
    if not parsed.hostname:
        raise ValueError(f"invalid server URL: {server!r}")
    elif parsed.scheme not in ("http", "https"):
        raise ValueError(f"--server takes an http or https URL, got {server!r}")
    else:
        # A listen-all address is where the server binds, not one to connect to.
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(parsed.hostname, parsed.hostname)
        host = f"[{host}]" if ":" in host else host
        if parsed.port is not None:
            netloc = f"{host}:{parsed.port}"
        elif has_scheme:
            netloc = host
        else:
            netloc = f"{host}:{default_port}"
        origin = urlunsplit((parsed.scheme, netloc, f"/{prefix}" if prefix else "", "", ""))
        return ServerURL(origin=origin, openai_base_url=f"{origin}/v1")


def discover_served_model(server: ServerURL, api_key: str | None) -> ServedModel:
    url = f"{server.openai_base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key is not None else {}
    # A proxy in the environment cannot reach the loopback address the server is usually on.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(Request(url, headers=headers), timeout=SERVER_TIMEOUT_SECONDS) as response:
            models = ModelList.model_validate_json(response.read())
    except HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError(
                f"the ftmetal server at {server.origin} needs a valid API key: pass --api-key "
                f"or set {API_KEY_ENV}"
            ) from exc
        raise RuntimeError(f"the ftmetal server at {server.origin} answered {exc}") from exc
    except (HTTPException, OSError) as exc:
        raise RuntimeError(
            f"cannot reach the ftmetal server at {server.origin}: {exc}\n"
            "Start one with: ftm serve --model <dir>"
        ) from exc
    except ValidationError as exc:
        raise RuntimeError(f"{url} did not answer with an ftmetal model list") from exc
    if not models.data:
        raise RuntimeError(f"the ftmetal server at {server.origin} reported no models")
    else:
        card = models.data[0]
        return ServedModel(
            model_id=card.id,
            context_length=card.context_length,
            accepts_images="image" in card.input_modalities,
            # The server lists them strongest first.
            reasoning_efforts=tuple(reversed(card.supported_reasoning_efforts or [])),
            default_reasoning_effort=card.default_reasoning_effort,
            thinking_toggleable=bool(card.thinking_toggleable),
            thinks_by_default=bool(card.default_thinking),
        )


def agent_environment(spec: CommandSpec, server: ServerURL, binary: str | None) -> dict[str, str]:
    """The environment the agent starts with: this process's, changed as spec says, with the
    server kept off any proxy and the binary's directory on PATH."""
    inherited = {key: value for key, value in os.environ.items() if key not in spec.unset_env}
    environment = {**spec.default_env, **inherited, **spec.env}
    bypass_lists = [environment.get(name, "").strip() for name in NO_PROXY_ENV]
    if any(os.environ.get(name) for name in PROXY_ENV) and "*" not in bypass_lists:
        server_host = urlsplit(server.origin).hostname or ""
        # Readers differ in which spelling they take first, so both get the user's entries.
        entries = [entry.strip() for value in bypass_lists for entry in value.split(",")]
        bypass = ",".join(dict.fromkeys(filter(None, [*entries, server_host, *LOOPBACK_HOSTS])))
        bypass_env = dict.fromkeys(NO_PROXY_ENV, bypass)
    else:
        bypass_env = {}
    search_path = environment.get("PATH", "")
    binary_dir = str(Path(binary).parent) if binary is not None else ""
    if binary_dir and binary_dir not in search_path.split(os.pathsep):
        path_env = {"PATH": os.pathsep.join(filter(None, [binary_dir, search_path]))}
    else:
        path_env = {}
    return {**environment, **bypass_env, **path_env}


def print_dry_run(context: LaunchContext, spec: CommandSpec, environment: dict[str, str]) -> None:
    print(f"ftmetal server: {context.server.origin}")
    print(f"OpenAI base URL: {context.server.openai_base_url}")
    print(f"Model: {context.model.model_id}")
    print(f"Context window: {context_window(context)}")
    print(f"Accepts images: {'yes' if context.model.accepts_images else 'no'}")
    print(f"Command: {shlex.join(spec.argv)}")
    changed = {key: value for key, value in environment.items() if os.environ.get(key) != value}
    if changed:
        print("Environment:")
        for key, value in sorted(changed.items()):
            print(f"  {key}={value}")
    if spec.unset_env:
        print("Unset environment:")
        for key in spec.unset_env:
            print(f"  {key}")


def start_agent(command: AgentCommand, spec: CommandSpec, environment: dict[str, str]) -> int:
    """Runs what the agent needs first, then becomes the agent; when a helper process must stop
    with it, runs the agent as a child instead and stops the helper after."""
    argv = [Path(command.binary).name, *command.leading_args, *spec.argv[1:]]
    helper = spec.prelaunch(environment) if spec.prelaunch is not None else None
    # Ctrl-C is the agent's to handle; a handler, unlike ignoring it, is not inherited.
    previous_handler = signal.signal(signal.SIGINT, lambda signum, frame: None) if helper else None
    try:
        if helper is not None:
            return subprocess.run(argv, executable=command.binary, env=environment).returncode
        else:
            # Replaced, so the terminal, signals and exit status are the agent's alone.
            os.execve(command.binary, argv, environment)
    except OSError as exc:
        raise RuntimeError(f"cannot start {command.binary}: {exc}") from exc
    finally:
        if helper is not None:
            helper.terminate()
            helper.wait()
            signal.signal(signal.SIGINT, previous_handler)


def agent_name(name: str) -> str:
    lowered = name.lower()
    return AGENT_ALIASES.get(lowered, lowered)


def run_launch(argv: list[str], default_server_url: str) -> int:
    separator = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(
        prog="ftm launch",
        description="Configure and start a coding agent against a running ftmetal server. "
        "Arguments after -- go to the agent.",
    )
    parser.add_argument("agent", type=agent_name, choices=sorted(PREPARERS))
    parser.add_argument(
        "--server",
        default=read_env("HOST") or default_server_url,
        help=f"server origin or /v1 URL (default: $FTMETAL_HOST, else {default_server_url})",
    )
    parser.add_argument(
        "--api-key",
        default=read_env("API_KEY"),
        help=f"API key of a server started with --api-key (default: ${API_KEY_ENV})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the planned command and change nothing"
    )
    parser.add_argument("-y", "--yes", action="store_true", help="approve every prompt")
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="run the agent's installer even when its CLI is already installed",
    )
    parser.add_argument(
        "--config",
        dest="config_only",
        action="store_true",
        help="install and configure the agent, then ask before starting it",
    )
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="install the agent's CLI if missing, then exit; needs no server",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="remove the files a launch wrote for the agent (codex only)",
    )
    args = parser.parse_args(argv[:separator])
    extra_args = argv[separator + 1 :]
    default_port = urlsplit(default_server_url).port
    assert default_port is not None, default_server_url
    installer = AGENT_INSTALLERS[args.agent]
    try:
        if args.restore and (args.config_only or args.dry_run or extra_args):
            raise ValueError("--restore cannot be combined with --config, --dry-run or extra args")
        elif args.restore and args.agent not in RESTORERS:
            raise ValueError(f"{args.agent} does not support --restore")
        elif args.restore:
            print(RESTORERS[args.agent]())
            return 0
        elif args.install_only and args.dry_run:
            existing = resolve_agent_binary(args.agent)
            if existing is not None and not args.force_reinstall:
                print(f"{installer.display_name} is installed: {existing}")
            else:
                print(f"Install command: {shlex.join(installer.argv)}")
            return 0
        elif args.install_only:
            ensure_agent_installed(args.agent, args.yes, args.force_reinstall)
            return 0
        else:
            server = resolve_server_url(args.server, default_port)
            api_key = args.api_key or None
            model = discover_served_model(server, api_key)
            if args.dry_run:
                binary = resolve_agent_binary(args.agent)
                command = AgentCommand(binary=binary) if binary is not None else None
            else:
                command = ensure_agent_installed(args.agent, args.yes, args.force_reinstall)
            context = LaunchContext(
                server=server,
                model=model,
                extra_args=extra_args,
                dry_run=args.dry_run,
                assume_yes=args.yes,
                agent_binary=command.binary if command is not None else None,
                # The plan a dry run prints never holds the key itself.
                api_key="<api-key>" if args.dry_run and api_key is not None else api_key,
            )
            spec = PREPARERS[args.agent](context)
            environment = agent_environment(spec, server, context.agent_binary)
            if args.dry_run:
                print_dry_run(context, spec, environment)
                return 0
            elif args.config_only and not (
                args.agent in LAUNCH_AFTER_CONFIG
                and (args.yes or confirm(f"Launch {installer.display_name} now?"))
            ):
                return 0
            else:
                assert command is not None, "only a dry run goes on without the agent installed"
                return start_agent(command, spec, environment)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return INTERRUPTED_EXIT_STATUS


__all__ = ["resolve_server_url", "run_launch"]
