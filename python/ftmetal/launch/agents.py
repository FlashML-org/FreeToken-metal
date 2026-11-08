"""What each coding agent needs to reach a running server: command, environment, config files.

Port of FreeToken launch.py's preparers. Divergences: the server accepts any API key, so agents
get a placeholder and there is no key option; one served model, not a list;
Codex, OpenClaw and Hermes live in modules of their own.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import yaml
from pydantic import JsonValue

from ftmetal.launch.codex import prepare_codex
from ftmetal.launch.config_files import (
    dict_child,
    env_path,
    load_json_object,
    write_json_with_backup,
    write_text_with_backup,
)
from ftmetal.launch.context import (
    API_KEY_ENV,
    CLOUD_PROVIDER_API_KEY_ENV,
    MAX_OUTPUT_TOKENS_CAP,
    PLACEHOLDER_API_KEY,
    PROVIDER_ID,
    AgentPreparer,
    CommandSpec,
    LaunchContext,
    command_version,
    context_window,
    input_modalities,
    max_output_tokens,
    reasoning_levels,
)
from ftmetal.launch.hermes import prepare_hermes
from ftmetal.launch.openclaw import prepare_openclaw

PROVIDER_NAME = "ftmetal"
PI_LAUNCH_MARKER = "_launch"  # on the models.json entries a launch wrote
# Pi options that take a value, so a value that looks like --model is not read as one.
PI_VALUE_OPTIONS = (
    "--mode",
    "--api-key",
    "--system-prompt",
    "--append-system-prompt",
    "--session",
    "--fork",
    "--session-dir",
    "--models",
    "--tools",
    "-t",
    "--thinking",
    "--export",
    "--extension",
    "-e",
    "--skill",
    "--prompt-template",
    "--theme",
)
# Each would route Claude Code to a cloud provider ahead of ANTHROPIC_BASE_URL.
CLAUDE_PROVIDER_SELECTORS = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_USE_GATEWAY",
)
OPENCODE_RECENT_MODELS = 10
DSH_MIN_NODE = (24, 9)  # the engines floor in dsh's package.json
DSH_LAUNCH_PATCH_NAME = "ftmetal-launch.cordis.patch.yml"
# dsh's compaction defaults: it keeps this much headroom, and a tail of this fraction verbatim.
DSH_COMPACTION_HEADROOM_TOKENS = 65_536
DSH_RETAIN_RATIO = 0.16
# A key reference nothing sets, so dsh finds no search key and withholds web_search.
DSH_WEB_SEARCH_KEY_ENV = "FTMETAL_DSH_WEB_SEARCH_DISABLED"


def prepare_claude(context: LaunchContext) -> CommandSpec:
    model_id = context.model.model_id
    window = context_window(context)
    user_window = os.environ.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "").strip()
    # The user's own capability entries come after ftm's, so theirs still win.
    user_capabilities = os.environ.get("CLAUDE_CODE_MODEL_CAPABILITIES", "")
    return CommandSpec(
        argv=["claude", "--model", model_id, *context.extra_args],
        env={
            # Honoured only for model ids not starting with "claude-"; else it assumes 200k.
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(
                min(int(user_window), window) if user_window.isdigit() else window
            ),
            "ANTHROPIC_BASE_URL": context.server.origin,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": PLACEHOLDER_API_KEY,
            "ANTHROPIC_MODEL": model_id,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model_id,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model_id,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_id,
            "CLAUDE_CODE_SUBAGENT_MODEL": model_id,
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            # Each of these would reach Anthropic's cloud, which a local launch never asked for.
            "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
            "CLAUDE_CODE_TOTAL_TOKENS_REMINDER": "off",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_FEEDBACK_COMMAND": "1",
            "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1",
            # Provider and auth keys from ~/.claude/settings.json then no longer apply.
            "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST": "1",
            # Reminders then stay inline instead of system turns, which the server merges into
            # the front of the prompt, costing the prefix cache on every turn.
            "CLAUDE_CODE_MODEL_CAPABILITIES": ";".join(
                filter(None, ["-mid_conv_system", user_capabilities])
            ),
        },
        default_env={
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_output_tokens(context)),
            # The server has no auto-mode safety checks, so Claude Code classifies on its own.
            "CLAUDE_CODE_AUTO_MODE_SERVER": "0",
        },
        unset_env=CLAUDE_PROVIDER_SELECTORS,
    )


def prepare_opencode(context: LaunchContext) -> CommandSpec:
    model_id = context.model.model_id
    if not context.dry_run:
        state_home = env_path("XDG_STATE_HOME", Path.home() / ".local" / "state")
        state_path = state_home / "opencode" / "model.json"
        state = load_json_object(state_path)
        recent = state.get("recent")
        kept: list[JsonValue] = [
            entry
            for entry in (recent if isinstance(recent, list) else [])
            if not (
                isinstance(entry, dict)
                and entry.get("providerID") == PROVIDER_ID
                and entry.get("modelID") == model_id
            )
        ]
        state.setdefault("favorite", [])
        state.setdefault("variant", {})
        newest: dict[str, JsonValue] = {"providerID": PROVIDER_ID, "modelID": model_id}
        state["recent"] = [newest, *kept][:OPENCODE_RECENT_MODELS]
        write_json_with_backup(state_path, state)

    # Without limit.context OpenCode reads the window as 0 and stops auto-compacting.
    model_entry: dict[str, JsonValue] = {
        "name": model_id,
        "limit": {"context": context_window(context), "output": max_output_tokens(context)},
    }
    if context.model.accepts_images:
        model_entry["modalities"] = {"input": ["text", "image"], "output": ["text"]}
    levels, default_level = reasoning_levels(context)
    if context.model.reasoning_efforts:
        model_entry["reasoning"] = True
        if default_level is not None:
            model_entry["options"] = {"reasoningEffort": default_level}
        model_entry["variants"] = {level: {"reasoningEffort": level} for level in levels}
    elif levels:
        # OpenCode's own levels mean nothing to an on/off switch, so only "none" stays.
        model_entry["reasoning"] = True
        model_entry["variants"] = {
            "none": {"reasoningEffort": "none"},
            **{level: {"disabled": True} for level in ("low", "medium", "high")},
        }
    config: dict[str, JsonValue] = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": PROVIDER_NAME,
                "options": {"baseURL": context.server.openai_base_url},
                "models": {model_id: model_entry},
            },
        },
        "model": f"{PROVIDER_ID}/{model_id}",
    }
    binary = context.agent_binary
    version = command_version(binary) if binary is not None and not context.extra_args else None
    major = version[0] if version is not None else 0
    if major == 1:
        # 1.x lets an agent's own configured model win over the config's model.
        args = ["--model", f"{PROVIDER_ID}/{model_id}"]
    elif major >= 2:
        # 2.x reuses a background service, which never sees this launch's inline config.
        args = ["--standalone"]
    else:
        args = context.extra_args
    return CommandSpec(
        argv=["opencode", *args],
        env={"OPENCODE_CONFIG_CONTENT": json.dumps(config)},
    )


def prepare_pi(context: LaunchContext) -> CommandSpec:
    """Its models.json takes a custom OpenAI-compatible endpoint. Only this provider's entry
    changes, and flags pick the model unless the user's own do, so the user's default model
    stays theirs."""
    model_id = context.model.model_id
    if not context.dry_run:
        models_path = env_path("PI_CODING_AGENT_DIR", Path.home() / ".pi" / "agent") / "models.json"
        config = load_json_object(models_path)
        model_entry: dict[str, JsonValue] = {
            "id": model_id,
            "name": model_id,
            "input": input_modalities(context),
            "contextWindow": context_window(context),
            "maxTokens": max_output_tokens(context),
            # Pi's thinking levels then reach the server as thinking.type and reasoning_effort,
            # and instructions keep the system role.
            "reasoning": True,
            "compat": {"supportsDeveloperRole": False, "thinkingFormat": "deepseek"},
            PI_LAUNCH_MARKER: True,
        }
        provider = dict_child(dict_child(config, "providers"), PROVIDER_ID)
        provider.setdefault("name", PROVIDER_NAME)
        provider["baseUrl"] = context.server.openai_base_url
        # The model's compat settings are for this API.
        provider["api"] = "openai-completions"
        provider.setdefault("apiKey", PLACEHOLDER_API_KEY)
        existing_models = provider.get("models")
        # Entries a launch wrote are replaced; the user's own are kept, one for this model too.
        user_models = [
            entry
            for entry in (existing_models if isinstance(existing_models, list) else [])
            if not (isinstance(entry, dict) and entry.get(PI_LAUNCH_MARKER) is True)
        ]
        has_user_entry = any(
            isinstance(entry, dict) and entry.get("id") == model_id for entry in user_models
        )
        provider["models"] = user_models if has_user_entry else [*user_models, model_entry]
        write_json_with_backup(models_path, config)
    extra_args = context.extra_args
    has_model_args = False
    i = 0
    while i < len(extra_args) and not has_model_args:
        has_model_args = extra_args[i] in ("--provider", "--model") and i + 1 < len(extra_args)
        i += 2 if extra_args[i] in PI_VALUE_OPTIONS else 1
    model_args = [] if has_model_args else ["--provider", PROVIDER_ID, "--model", model_id]
    return CommandSpec(argv=["pi", *model_args, *extra_args], env={})


def prepare_dsh(context: LaunchContext) -> CommandSpec:
    """dsh 0.2 keeps settings as entry configs in its profile; a --patch overlay, the last layer
    it composes, sets them for this run. llm-deepseek replays tool-call arguments byte-verbatim,
    where llm-pi-ai's JSON round trip can reorder them and cost the prefix cache."""
    patch_path = env_path("DSH_HOME", Path.home() / ".dsh") / DSH_LAUNCH_PATCH_NAME
    model_id = context.model.model_id
    if not context.dry_run:
        node = shutil.which("node")
        node_version = command_version(node) if node is not None else None
        if node_version is not None and node_version < DSH_MIN_NODE:
            floor = ".".join(str(part) for part in DSH_MIN_NODE)
            print(
                f"warning: dsh requires Node.js >={floor}, node reports "
                f"v{node_version[0]}.{node_version[1]}.",
                file=sys.stderr,
            )
        window = context_window(context)
        # dsh reserves the output limit plus its headroom before it compacts; an eighth of the
        # window keeps compaction late in it.
        max_tokens = max(1, min(window // 8, MAX_OUTPUT_TOKENS_CAP))
        message_budget = window - max_tokens
        if (
            int(message_budget * DSH_RETAIN_RATIO)
            >= message_budget - DSH_COMPACTION_HEADROOM_TOKENS
        ):
            print(
                f"warning: a {window}-token window leaves dsh no room beyond its "
                f"{DSH_COMPACTION_HEADROOM_TOKENS}-token compaction headroom, so it will never "
                "compact a long session.",
                file=sys.stderr,
            )
        model_entry: dict[str, JsonValue] = {
            "id": model_id,
            "name": f"{model_id} (ftmetal)",
            "inputModalities": input_modalities(context),
            "contextWindow": window,
            "maxTokens": max_tokens,
        }
        patch: list[JsonValue] = [
            {
                "id": "llm-deepseek",
                "config": {
                    "baseURL": context.server.openai_base_url,
                    # dsh's stored DeepSeek key outranks the environment; nothing stores this one.
                    "apiKeyEnv": API_KEY_ENV,
                    "models": [model_entry],
                },
            },
            {
                "id": "agent-default-model",
                "config": {"provider": "deepseek-official", "model": model_id},
            },
            # Its search would send queries to DeepSeek's cloud, which this server cannot serve.
            {"id": "web-search-deepseek", "config": {"apiKeyEnv": DSH_WEB_SEARCH_KEY_ENV}},
        ]
        write_text_with_backup(
            patch_path, yaml.safe_dump(patch, sort_keys=False, allow_unicode=True)
        )

    # dsh's own grammar: a leading word or --profile picks the profile, the rest is the app's.
    extra_args = context.extra_args
    first = extra_args[0] if extra_args else ""
    if first and not first.startswith("-") and first != "plugin":
        profile, app_args = first, extra_args[1:]
    elif first == "--profile" and len(extra_args) >= 2:
        profile, app_args = extra_args[1], extra_args[2:]
    elif first.startswith("--profile="):
        profile, app_args = first.removeprefix("--profile="), extra_args[1:]
    else:
        profile, app_args = "web", extra_args
    return CommandSpec(
        argv=["dsh", "--profile", profile, "--patch", str(patch_path), *app_args],
        env={API_KEY_ENV: PLACEHOLDER_API_KEY, "DSH_TELEMETRY_DISABLED": "1"},
        unset_env=CLOUD_PROVIDER_API_KEY_ENV,
    )


PREPARERS: dict[str, AgentPreparer] = {
    "codex": prepare_codex,
    "claude": prepare_claude,
    "opencode": prepare_opencode,
    "openclaw": prepare_openclaw,
    "hermes": prepare_hermes,
    "pi": prepare_pi,
    "dsh": prepare_dsh,
}

__all__ = ["PREPARERS"]
