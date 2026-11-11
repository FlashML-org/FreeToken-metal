"""Hermes: the model section of its config.yaml, found the way Hermes finds it, and an offer to
connect a messaging app before an interactive chat.

Port of FreeToken launch.py's Hermes preparer. Divergences: HERMES_HOME and profiles are honoured;
only the model block of the file is rewritten, so the rest keeps its comments.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from ftmetal.launch.config_files import (
    dict_child,
    load_yaml_object,
    read_config_text,
    write_yaml_with_backup,
)
from ftmetal.launch.context import (
    CommandSpec,
    LaunchContext,
    agent_api_key,
    confirm,
    context_window,
    is_interactive,
)

HERMES_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")  # after casefold
HERMES_MIN_CONTEXT_LENGTH = 64_000  # Hermes refuses to start below this
# Any of these, in Hermes's .env or the environment, means a messaging app is connected.
HERMES_MESSAGING_ENV = (
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "SLACK_BOT_TOKEN",
    "SIGNAL_ACCOUNT",
    "EMAIL_ADDRESS",
    "TWILIO_ACCOUNT_SID",
    "MATRIX_ACCESS_TOKEN",
    "MATRIX_PASSWORD",
    "MATTERMOST_TOKEN",
    "WHATSAPP_PHONE_NUMBER_ID",
    "DINGTALK_CLIENT_ID",
    "FEISHU_APP_ID",
    "WECOM_BOT_ID",
    "WEIXIN_ACCOUNT_ID",
    "BLUEBUBBLES_SERVER_URL",
    "WEBHOOK_ENABLED",
)


def hermes_messaging_configured(config_home: Path) -> bool:
    """Whether a messaging token is set, in Hermes's .env or the environment."""
    names: dict[str, str] = {}
    for line in read_config_text(config_home / ".env").splitlines():
        name, has_value, value = line.strip().removeprefix("export ").partition("=")
        if has_value and not name.strip().startswith("#"):
            names[name.strip()] = value.strip().strip("\"'")
    return any(
        (os.environ.get(name) or names.get(name, "")).strip() for name in HERMES_MESSAGING_ENV
    )


def prepare_hermes(context: LaunchContext) -> CommandSpec:
    """Its config.yaml model section takes a custom OpenAI-compatible endpoint: with base_url set
    Hermes calls it directly. Only that section changes, and max_tokens stays unset because
    Hermes budgets compaction from context_length alone."""
    env_home = os.environ.get("HERMES_HOME", "").strip()
    home = Path(os.path.expandvars(env_home)).expanduser() if env_home else Path.home() / ".hermes"
    # A HERMES_HOME of ROOT/profiles/NAME is a profile already: no sticky profile applies.
    is_profile_home = bool(env_home) and home.parent.name == "profiles"
    root = home.parent.parent if is_profile_home else home
    extra_args = context.extra_args
    arguments = extra_args[: extra_args.index("--")] if "--" in extra_args else extra_args
    flag_profile = next(
        (
            argument.removeprefix("--profile=")
            if argument.startswith("--profile=")
            else arguments[i + 1]
            for i, argument in enumerate(arguments)
            if argument.startswith("--profile=")
            or (argument in ("-p", "--profile") and i + 1 < len(arguments))
        ),
        None,
    )
    sticky_path = root / "active_profile"
    if flag_profile is not None:
        profile = flag_profile.strip().casefold()
    elif not is_profile_home and sticky_path.is_file():
        profile = sticky_path.read_text(encoding="utf-8-sig").strip().casefold() or None
    else:
        profile = None
    if profile is None:
        config_home = home
    elif profile == "default":
        config_home = root
    elif not HERMES_PROFILE_NAME.fullmatch(profile):
        raise RuntimeError(
            f"Hermes refuses the profile name {profile!r}: use lowercase letters, numbers, "
            "_ and -, starting with a letter or number."
        )
    elif not (root / "profiles" / profile).is_dir():
        raise RuntimeError(
            f"Hermes has no profile {profile!r} ({root / 'profiles' / profile}); create it "
            f"with: hermes profile create {profile}"
        )
    else:
        config_home = root / "profiles" / profile
    if not context.dry_run:
        config_path = config_home / "config.yaml"
        config = load_yaml_object(config_path)
        model_section = dict_child(config, "model")
        window = context_window(context)
        model_section["default"] = context.model.model_id
        model_section["provider"] = "custom"
        model_section["base_url"] = context.server.openai_base_url
        model_section["api_key"] = agent_api_key(context)
        # Else Hermes takes a window it cached for this URL, stale once the server restarts
        # with another.
        model_section["context_length"] = window
        if window < HERMES_MIN_CONTEXT_LENGTH:
            print(
                f"warning: Hermes refuses to start under {HERMES_MIN_CONTEXT_LENGTH} tokens of "
                f"context and this server holds {window}.",
                file=sys.stderr,
            )
        write_yaml_with_backup(config_path, config, ("model",), private=context.api_key is not None)

    def offer_messaging(environment: dict[str, str]) -> subprocess.Popen[bytes] | None:
        binary_path = context.agent_binary
        assert binary_path is not None, "the agent starts only once it is installed"
        if not (
            context.assume_yes or not is_interactive() or hermes_messaging_configured(config_home)
        ):
            print(
                "\nHermes can message you on Telegram, Discord, Slack, and more.\n", file=sys.stderr
            )
            if (
                confirm("Connect a messaging app now?")
                and subprocess.run([binary_path, "gateway", "setup"], env=environment).returncode
            ):
                raise RuntimeError(
                    "hermes messaging setup failed\n\nTry running: hermes gateway setup"
                )
        return None

    # Hermes takes CUSTOM_BASE_URL over the base_url in config.yaml.
    return CommandSpec(
        argv=["hermes", *(extra_args or ["chat"])],
        env={},
        unset_env=("CUSTOM_BASE_URL",),
        prelaunch=None if extra_args else offer_messaging,
    )


__all__ = ["prepare_hermes"]
