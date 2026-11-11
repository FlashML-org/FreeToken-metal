"""OpenClaw: onboarded on first launch, its config patched in place, then run as tui against
its gateway, started first when it is down.

Port of FreeToken launch.py's OpenClaw preparer. Divergences: the config is found the way
OpenClaw finds it (profiles, home variables); a running gateway is not restarted, since the model
edit reloads live; session overrides are not touched, since OpenClaw keeps them in a database.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

from pydantic import JsonValue

from ftmetal.launch.config_files import (
    dict_child,
    env_path,
    load_json_object,
    nested_value,
    write_json_with_backup,
)
from ftmetal.launch.context import (
    CLOUD_PROVIDER_API_KEY_ENV,
    PROVIDER_ID,
    CommandSpec,
    LaunchContext,
    PreLaunch,
    agent_api_key,
    confirm,
    context_window,
    input_modalities,
    is_interactive,
    max_output_tokens,
)

OPENCLAW_GATEWAY_PORT = 18789
PORT_PROBE_SECONDS = 0.5
PORT_POLL_SECONDS = 0.25
OPENCLAW_DAEMON_WAIT_SECONDS = 10
OPENCLAW_GATEWAY_WAIT_SECONDS = 30
OPENCLAW_FIRST_MESSAGE = "Wake up, my friend!"
# OpenClaw's root options that take a value; it never takes one starting with "-" as the value.
OPENCLAW_VALUE_OPTIONS = ("--profile", "--log-level", "--container")
OPENCLAW_PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
# Keys under channels that configure no channel themselves.
OPENCLAW_CHANNEL_SETTINGS = ("defaults", "modelByChannel")


def openclaw_paths(extra_args: list[str]) -> tuple[Path, Path, list[str]]:
    """OpenClaw's state directory, its config file and the root options that pick its profile,
    from its variables and its root --profile or --dev; under gateway and qa matrix those
    options are the command's own."""
    end = extra_args.index("--") if "--" in extra_args else len(extra_args)
    command_words: list[str] = []
    profile = None
    i = 0
    while i < end:
        option, equals, inline_value = extra_args[i].partition("=")
        following = extra_args[i + 1] if i + 1 < end else "-"
        takes_following = (
            option in OPENCLAW_VALUE_OPTIONS and not equals and not following.startswith("-")
        )
        if extra_args[i] == "--dev" and command_words[:1] != ["gateway"]:
            profile = "dev"
        elif option == "--profile" and command_words[:2] != ["qa", "matrix"]:
            profile = (following if takes_following else inline_value).strip()
            if not OPENCLAW_PROFILE_NAME.fullmatch(profile):
                raise RuntimeError(
                    f"OpenClaw refuses the profile name {profile!r}: use letters, numbers, _ and "
                    "- only."
                )
        elif not option.startswith("-"):
            command_words.append(extra_args[i])
        i += 2 if takes_following else 1
    home = env_path("OPENCLAW_HOME", Path.home())
    if profile is None or profile.casefold() == "default":
        profile_state_dir, root_options = home / ".openclaw", []
    elif profile == "dev" and "--dev" in extra_args[:end]:
        profile_state_dir, root_options = home / ".openclaw-dev", ["--dev"]
    else:
        profile_state_dir, root_options = home / f".openclaw-{profile}", ["--profile", profile]
    state_dir = env_path("OPENCLAW_STATE_DIR", profile_state_dir)
    config_path = env_path("OPENCLAW_CONFIG_PATH", state_dir / "openclaw.json")
    return state_dir, config_path, root_options


def patch_openclaw_config(context: LaunchContext, config: dict[str, JsonValue]) -> None:
    model_id = context.model.model_id
    providers = dict_child(dict_child(config, "models"), "providers")
    provider = dict_child(providers, PROVIDER_ID)
    model_entry: dict[str, JsonValue] = {
        "id": model_id,
        "name": model_id,
        "input": input_modalities(context),
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": context_window(context),
        "maxTokens": max_output_tokens(context),
        # Else OpenClaw drops the streamed reasoning and strips it from replayed turns; a custom
        # endpoint keeps the system role either way.
        "reasoning": True,
    }
    existing_models = provider.get("models")
    for existing in existing_models if isinstance(existing_models, list) else []:
        if isinstance(existing, dict) and existing.get("id") == model_id:
            for key, value in existing.items():
                model_entry.setdefault(key, value)
    provider["baseUrl"] = context.server.openai_base_url
    provider["apiKey"] = agent_api_key(context)
    provider["api"] = "openai-completions"
    provider["models"] = [model_entry]
    model_config = dict_child(dict_child(dict_child(config, "agents"), "defaults"), "model")
    model_config["primary"] = f"{PROVIDER_ID}/{model_id}"


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=PORT_PROBE_SECONDS):
            return True
    except OSError:
        return False


def wait_for_port(port: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while not port_is_open(port):
        if time.monotonic() >= deadline:
            return False
        time.sleep(PORT_POLL_SECONDS)
    return True


def openclaw_channels_configured(config_path: Path) -> bool:
    """Whether a channel has settings beyond being enabled; a file that does not parse counts as
    configured, so nothing prompts over it."""
    try:
        channels = nested_value(load_json_object(config_path), "channels")
    except RuntimeError:
        return True
    return isinstance(channels, dict) and any(
        isinstance(channel, dict) and any(key != "enabled" for key in channel)
        for name, channel in channels.items()
        if name not in OPENCLAW_CHANNEL_SETTINGS
    )


def onboard_openclaw(context: LaunchContext, binary: str, root_options: list[str]) -> None:
    """OpenClaw's own non-interactive onboarding, with this server as its custom provider and its
    gateway installed as a service, after the user accepts its security notice."""
    print("\nSecurity\n", file=sys.stderr)
    print("  OpenClaw can read files and run actions when tools are enabled.", file=sys.stderr)
    print("  A bad prompt can trick it into doing unsafe things.\n", file=sys.stderr)
    print("  Learn more: https://docs.openclaw.ai/gateway/security\n", file=sys.stderr)
    if not (context.assume_yes or confirm("I understand the risks. Continue?")):
        raise RuntimeError("OpenClaw setup cancelled")
    print("\nSetting up OpenClaw with ftmetal...", file=sys.stderr)
    print(f"  Model: {context.model.model_id}\n", file=sys.stderr)
    argv = [
        binary,
        *root_options,
        "onboard",
        "--non-interactive",
        "--accept-risk",
        "--auth-choice",
        "custom-api-key",
        "--custom-base-url",
        context.server.openai_base_url,
        "--custom-model-id",
        context.model.model_id,
        "--custom-api-key",
        agent_api_key(context),
        "--custom-provider-id",
        PROVIDER_ID,
        "--custom-compatibility",
        "openai",
        "--custom-image-input" if context.model.accepts_images else "--custom-text-input",
        # The launch starts the gateway itself right after.
        "--skip-health",
        "--skip-channels",
        "--skip-skills",
        "--install-daemon",
    ]
    environment = {k: v for k, v in os.environ.items() if k not in CLOUD_PROVIDER_API_KEY_ENV}
    try:
        subprocess.run(argv, env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"openclaw onboarding failed: {exc}\n\nTry running: openclaw onboard"
        ) from exc


def start_openclaw_gateway(
    binary: str, root_options: list[str], port: int, environment: dict[str, str]
) -> subprocess.Popen[bytes] | None:
    """Starts OpenClaw's gateway service, else a gateway of its own for this launch, which is
    returned; a gateway already up is left running, since it reloads the edited config."""
    if port_is_open(port):
        return None
    else:
        daemon = subprocess.run(
            [binary, *root_options, "daemon", "start"], env=environment, capture_output=True
        )
        if daemon.returncode == 0 and wait_for_port(port, OPENCLAW_DAEMON_WAIT_SECONDS):
            print("Starting gateway...", file=sys.stderr)
            return None
        elif daemon.returncode != 0:
            print(f"  Warning: daemon start failed (exit {daemon.returncode})", file=sys.stderr)
        # Its log would otherwise land on the terminal the tui draws on.
        gateway = subprocess.Popen(
            [binary, *root_options, "gateway", "run", "--force"],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("Starting gateway...", file=sys.stderr)
        if not wait_for_port(port, OPENCLAW_GATEWAY_WAIT_SECONDS):
            gateway.kill()
            gateway.wait()
            raise RuntimeError(f"the OpenClaw gateway did not start on 127.0.0.1:{port}")
        else:
            return gateway


def prepare_openclaw(context: LaunchContext) -> CommandSpec:
    state_dir, config_path, root_options = openclaw_paths(context.extra_args)
    config = load_json_object(config_path)
    is_first_launch = not nested_value(config, "wizard", "lastRunAt")
    model_ref = f"{PROVIDER_ID}/{context.model.model_id}"
    if not context.dry_run:
        binary = context.agent_binary
        assert binary is not None, "only a dry run goes on without the agent installed"
        if is_first_launch:
            onboard_openclaw(context, binary, root_options)
            config = load_json_object(config_path)
        patch_openclaw_config(context, config)
        write_json_with_backup(config_path, config, private=context.api_key is not None)

    # The main agent is keyed under agents.entries, or listed by id in agents.list.
    entries = nested_value(config, "agents", "entries")
    listed = nested_value(config, "agents", "list")
    if isinstance(entries, dict):
        main_agent = entries.get("main")
    elif isinstance(listed, list):
        main_agent = next(
            (
                agent_entry
                for agent_entry in listed
                if isinstance(agent_entry, dict) and agent_entry.get("id") == "main"
            ),
            None,
        )
    else:
        main_agent = None
    main_model = nested_value(main_agent, "model")
    main_primary = (
        main_model if isinstance(main_model, str) else nested_value(main_model, "primary")
    )
    if isinstance(main_primary, str) and main_primary and main_primary != model_ref:
        print(
            f"warning: the main agent's own model ({main_primary}) in {config_path} overrides "
            f"the default model ftm sets, so OpenClaw will not use {model_ref}.",
            file=sys.stderr,
        )
    workspace = nested_value(main_agent, "workspace") or nested_value(
        config, "agents", "defaults", "workspace"
    )
    if isinstance(workspace, str) and workspace.strip():
        workspace_dir = Path(workspace.strip()).expanduser()
    else:
        workspace_dir = env_path("OPENCLAW_WORKSPACE_DIR", state_dir / "workspace")
    print(f"OpenClaw works in {workspace_dir}, not the current directory.", file=sys.stderr)

    port_text = os.environ.get("OPENCLAW_GATEWAY_PORT", "").strip()
    config_port = nested_value(config, "gateway", "port")
    if port_text.isdigit():
        port = int(port_text)
    elif isinstance(config_port, int):
        port = config_port
    else:
        port = OPENCLAW_GATEWAY_PORT
    token = nested_value(config, "gateway", "auth", "token")

    def run_assistant(environment: dict[str, str]) -> subprocess.Popen[bytes] | None:
        """Offers to connect a channel, starts the gateway and says where its web UI is."""
        binary_path = context.agent_binary
        assert binary_path is not None, "the agent starts only once it is installed"
        while not (
            context.assume_yes or not is_interactive() or openclaw_channels_configured(config_path)
        ):
            print(
                "\nYour assistant can message you on WhatsApp, Telegram, Discord, and more.\n",
                file=sys.stderr,
            )
            if not confirm("Connect a channel (messaging app) now?"):
                break
            elif subprocess.run(
                [binary_path, *root_options, "channels", "add"], env=environment
            ).returncode:
                raise RuntimeError(
                    "openclaw channel setup failed\n\nTry running: openclaw channels add"
                )
        print("\nStarting your assistant — this may take a moment...\n", file=sys.stderr)
        gateway = start_openclaw_gateway(binary_path, root_options, port, environment)
        url = f"http://127.0.0.1:{port}"
        if isinstance(token, str) and token:
            url += f"/#token={quote(token, safe='')}"
        print(f"\nOpenClaw is running\n\n  Open the Web UI:\n    {url}\n", file=sys.stderr)
        if is_first_launch:
            print("  Quick start:", file=sys.stderr)
            print("    /help             see all commands", file=sys.stderr)
            print("    openclaw skills   browse and install skills\n", file=sys.stderr)
            print("  The OpenClaw gateway is running in the background.", file=sys.stderr)
            print("  Stop it with: openclaw gateway stop\n", file=sys.stderr)
        return gateway

    def attach_gateway(environment: dict[str, str]) -> subprocess.Popen[bytes] | None:
        assert context.agent_binary is not None, "the agent starts only once it is installed"
        return start_openclaw_gateway(context.agent_binary, root_options, port, environment)

    prelaunch: PreLaunch | None
    if context.extra_args:
        argv = ["openclaw", *context.extra_args]
        prelaunch = attach_gateway if context.extra_args[0] == "tui" else None
    else:
        first_message = ["--message", OPENCLAW_FIRST_MESSAGE] if is_first_launch else []
        argv = ["openclaw", "tui", *first_message]
        prelaunch = run_assistant
    return CommandSpec(argv=argv, env={}, unset_env=CLOUD_PROVIDER_API_KEY_ENV, prelaunch=prelaunch)


__all__ = ["prepare_openclaw"]
