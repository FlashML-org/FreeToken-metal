"""Codex: a launch profile and model catalog in its home, with the provider on its command line.

Port of FreeToken launch.py's Codex preparer. Divergences: no migration of an older Codex
configuration; routing rides -c overrides too; extra arguments that would reroute it are refused.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import JsonValue

from ftmetal.launch.config_files import (
    env_path,
    toml_string,
    write_json_with_backup,
    write_text_with_backup,
)
from ftmetal.launch.context import (
    API_KEY_ENV,
    CLOUD_PROVIDER_API_KEY_ENV,
    CommandSpec,
    LaunchContext,
    agent_api_key,
    command_version,
    context_window,
    input_modalities,
    max_output_tokens,
    reasoning_levels,
)

CODEX_PROFILE = "ftmetal-launch"
CODEX_PROVIDER_NAME = "ftmetal"
CODEX_CATALOG_NAME = "ftmetal-model.json"
CODEX_CLEAR_ENV = (*CLOUD_PROVIDER_API_KEY_ENV, "CODEX_API_KEY")
CODEX_MIN_VERSION = (0, 134)  # the first release whose model catalog takes these entries
# Root config keys ftm sets for Codex; an extra -c on one would route the launch elsewhere.
CODEX_MANAGED_KEYS = ("profile", "model", "model_provider", "model_catalog_json")
CODEX_LEVEL_DESCRIPTIONS = {
    "none": "Turn thinking off",
    "minimal": "Fastest responses with minimal thinking",
    "low": "Fast responses with lighter thinking",
    "medium": "Balances speed and thinking depth for everyday tasks",
    "high": "Greater thinking depth for complex tasks",
    "xhigh": "Extra thinking depth for hard tasks",
    "max": "Maximum thinking depth for the hardest tasks",
}


def codex_files() -> tuple[Path, Path]:
    """The launch profile and model catalog ftm writes in Codex's home."""
    codex_home = env_path("CODEX_HOME", Path.home() / ".codex")
    return codex_home / f"{CODEX_PROFILE}.config.toml", codex_home / CODEX_CATALOG_NAME


def restore_codex() -> str:
    """Removes the files a launch wrote; the user's own config.toml was never touched."""
    for path in codex_files():
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"cannot remove {path}: {exc}") from exc
    return "Codex launch configuration removed."


def prepare_codex(context: LaunchContext) -> CommandSpec:
    profile_path, catalog_path = codex_files()
    model_id = context.model.model_id
    base_url = toml_string(context.server.openai_base_url)
    extra_args = context.extra_args
    for i, argument in enumerate(extra_args):
        option, _, inline_value = argument.partition("=")
        if option in ("-p", "--profile", "-m", "--model") or argument.startswith(("-p", "-m")):
            raise ValueError(
                f"conflicting extra argument {argument!r}: ftm launch codex sets the profile and "
                "the model"
            )
        elif option in ("-c", "--config") or argument.startswith("-c"):
            if option == "--config" and inline_value:
                override = inline_value
            elif argument.startswith("-c") and len(argument) > len("-c"):
                override = argument.removeprefix("-c")
            else:
                override = extra_args[i + 1] if i + 1 < len(extra_args) else ""
            key, has_value, _ = override.strip().partition("=")
            key = key.strip().strip("\"'")
            if has_value and (key in CODEX_MANAGED_KEYS or key.startswith("model_providers.")):
                raise ValueError(
                    f"conflicting extra config {override!r}: ftm launch codex sets the provider "
                    "and the model catalog"
                )
    version = command_version(context.agent_binary) if context.agent_binary is not None else None
    if version is not None and version < CODEX_MIN_VERSION:
        floor = ".".join(str(part) for part in CODEX_MIN_VERSION)
        raise RuntimeError(
            f"codex {version[0]}.{version[1]} is too old, ftm launch needs {floor} or newer; "
            "update it with: ftm launch codex --install-only --force-reinstall"
        )
    if not context.dry_run:
        window = context_window(context)
        levels, default_level = reasoning_levels(context)
        catalog_entry: dict[str, JsonValue] = {
            "slug": model_id,
            "display_name": model_id,
            "description": model_id,
            "context_window": window,
            "max_context_window": window,
            "effective_context_window_percent": 100,
            # Codex sends no output limit, so the reply gets what the prompt leaves.
            "auto_compact_token_limit": window - max_output_tokens(context),
            "shell_type": "default",
            "visibility": "list",
            "supported_in_api": True,
            "priority": 0,
            "additional_speed_tiers": [],
            "service_tiers": [],
            "truncation_policy": {"mode": "bytes", "limit": 10000},
            "input_modalities": input_modalities(context),
            "base_instructions": "",
            "default_reasoning_summary": "none",
            "supported_reasoning_levels": [
                {
                    "effort": level,
                    "description": CODEX_LEVEL_DESCRIPTIONS.get(level, "Thinking effort"),
                }
                for level in levels
            ],
            "default_reasoning_level": default_level,
            "supports_reasoning_summaries": False,
            "experimental_supported_tools": [],
            "supports_search_tool": False,
            "web_search_tool_type": "text",
            "supports_image_detail_original": False,
            "support_verbosity": True,
            "default_verbosity": "low",
            "supports_parallel_tool_calls": False,
            "use_responses_lite": False,
            "model_messages": None,
            "upgrade": None,
        }
        write_json_with_backup(catalog_path, {"models": [catalog_entry]})
        write_text_with_backup(
            profile_path,
            f"model = {toml_string(model_id)}\n"
            f"openai_base_url = {base_url}\n"
            f"model_provider = {toml_string(CODEX_PROFILE)}\n"
            f"model_catalog_json = {toml_string(str(catalog_path))}\n\n"
            f"[model_providers.{CODEX_PROFILE}]\n"
            f"name = {toml_string(CODEX_PROVIDER_NAME)}\n"
            f"base_url = {base_url}\n"
            'wire_api = "responses"\n'
            f"env_key = {toml_string(API_KEY_ENV)}\n",
        )
    # The provider rides the command line too, so routing never depends on Codex finding the
    # profile file.
    provider = f"model_providers.{CODEX_PROFILE}"
    overrides = [
        f"model_provider={toml_string(CODEX_PROFILE)}",
        f"{provider}.name={toml_string(CODEX_PROVIDER_NAME)}",
        f"{provider}.base_url={base_url}",
        f'{provider}.wire_api="responses"',
        f"{provider}.env_key={toml_string(API_KEY_ENV)}",
        f"model_catalog_json={toml_string(str(catalog_path))}",
        'service_tier="default"',
    ]
    return CommandSpec(
        argv=[
            "codex",
            "--profile",
            CODEX_PROFILE,
            *(argument for override in overrides for argument in ("-c", override)),
            "-m",
            model_id,
            *extra_args,
        ],
        env={API_KEY_ENV: agent_api_key(context)},
        unset_env=CODEX_CLEAR_ENV,
    )


__all__ = ["prepare_codex", "restore_codex"]
