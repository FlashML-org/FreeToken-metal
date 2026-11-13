"""ftm launch against a real server over a scripted scheduler, with fake agent binaries and
os.execve caught, so every written file and the started command can be read back."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import socket
import stat
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI
from pydantic import JsonValue

from ftmetal.cli import main as ftm_main
from ftmetal.env import ENV_PREFIX
from ftmetal.launch import config_files, hermes, openclaw
from ftmetal.launch import context as launch_context
from ftmetal.launch import install as launch_install
from ftmetal.launch import run as launch_run
from ftmetal.launch.agents import PREPARERS
from ftmetal.launch.context import LaunchContext, ServedModel, ServerURL
from ftmetal.launch.install import AGENT_INSTALLERS
from ftmetal.launch.run import resolve_server_url, run_launch
from ftmetal.server.api_server import create_app
from scripted_server import ScriptedScheduler, running_server, tiny_server_args

MODEL_ID = "tiny-model"
UNREACHABLE_SERVER = "http://127.0.0.1:9"
WINDOW = ScriptedScheduler.max_prompt_tokens
CODEX_CATALOG = ".codex/ftmetal-model.json"
OPENCLAW_CONFIG = ".openclaw/openclaw.json"
OPENCLAW_MODEL: list[str | int] = ["models", "providers", "ftmetal", "models", 0]
HERMES_CONFIG = ".hermes/config.yaml"
PI_MODELS = ".pi/agent/models.json"
PI_MODEL: list[str | int] = ["providers", "ftmetal", "models", 0]
DSH_PATCH = ".dsh/ftmetal-launch.cordis.patch.yml"
# Variables that move an agent's files or change what a launch sets; every test starts without.
AGENT_ENV = (
    "CODEX_HOME",
    "HERMES_HOME",
    "OPENCLAW_CONFIG_PATH",
    "OPENCLAW_STATE_DIR",
    "OPENCLAW_HOME",
    "OPENCLAW_WORKSPACE_DIR",
    "XDG_STATE_HOME",
    "DSH_HOME",
    "PI_CODING_AGENT_DIR",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
    "CLAUDE_CODE_MODEL_CAPABILITIES",
    "CUSTOM_BASE_URL",
    "NO_PROXY",
    "no_proxy",
    *launch_run.PROXY_ENV,
)


class AgentStarted(Exception):
    """Stands in for the exec that would replace the process."""

    def __init__(self, binary: str, argv: list[str], env: dict[str, str]) -> None:
        super().__init__(binary)
        self.binary = binary
        self.argv = argv
        self.env = env


def fake_execve(binary: str, argv: list[str], env: dict[str, str]) -> None:
    raise AgentStarted(binary, argv, env)


def make_executable(path: Path, script: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{script}")
    path.chmod(0o755)


@dataclass(frozen=True, kw_only=True)
class Sandbox:
    home: Path
    bin_dir: Path
    system_bin_dir: Path
    server_url: str

    def launch(self, *args: str) -> int:
        return run_launch(list(args), self.server_url)

    def start(self, *args: str) -> AgentStarted:
        with pytest.raises(AgentStarted) as started:
            self.launch(*args)
        return started.value

    def files(self) -> set[str]:
        return {str(p.relative_to(self.home)) for p in self.home.rglob("*") if p.is_file()}


@pytest.fixture
def sandbox(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    tiny_model_dir: Path,
    scripted: type[ScriptedScheduler],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Sandbox]:
    """The server takes the API key the test parametrizes it with, else any."""
    api_key: str | None = getattr(request, "param", None)
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    for agent in PREPARERS:
        make_executable(bin_dir / agent)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(bin_dir))
    for name in (
        *(ENV_PREFIX + variable for variable in ("HOST", "API_KEY")),
        *AGENT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launch_install, "SYSTEM_BIN_DIRS", (tmp_path / "system-bin",))
    monkeypatch.setattr(os, "execve", fake_execve)
    args = dataclasses.replace(tiny_server_args(tiny_model_dir), api_key=api_key)
    with (
        running_server(create_app(args)) as port,
        socket.create_server(("127.0.0.1", 0)) as gateway,
    ):
        # OpenClaw's gateway is up unless a test says otherwise.
        monkeypatch.setenv("OPENCLAW_GATEWAY_PORT", str(gateway.getsockname()[1]))
        yield Sandbox(
            home=home,
            bin_dir=bin_dir,
            system_bin_dir=tmp_path / "system-bin",
            server_url=f"http://127.0.0.1:{port}",
        )


# (agent, the started command's start, the environment it needs).
START_CASES: list[tuple[str, list[str], dict[str, str]]] = [
    ("codex", ["codex", "--profile", "ftmetal-launch"], {"FTMETAL_API_KEY": "ftmetal"}),
    (
        "claude",
        ["claude", "--model", MODEL_ID],
        {
            "ANTHROPIC_MODEL": MODEL_ID,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "ftmetal",
            "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST": "1",
            "CLAUDE_CODE_MODEL_CAPABILITIES": "-mid_conv_system",
            "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
            "DISABLE_ERROR_REPORTING": "1",
            "CLAUDE_CODE_AUTO_MODE_SERVER": "0",
        },
    ),
    ("opencode", ["opencode"], {}),
    ("openclaw", ["openclaw", "tui", "--message", "Wake up, my friend!"], {}),
    ("hermes", ["hermes", "chat"], {}),
    ("pi", ["pi", "--provider", "ftmetal", "--model", MODEL_ID], {}),
    (
        "dsh",
        ["dsh", "--profile", "web", "--patch"],
        {"FTMETAL_API_KEY": "ftmetal", "DSH_TELEMETRY_DISABLED": "1"},
    ),
]


@pytest.mark.parametrize(("agent", "command_start", "env"), START_CASES)
def test_starts_the_agent(
    sandbox: Sandbox, agent: str, command_start: list[str], env: dict[str, str]
) -> None:
    started = sandbox.start(agent, "-y")
    assert started.binary == str(sandbox.bin_dir / agent)
    assert started.argv[: len(command_start)] == command_start
    assert env.items() <= started.env.items()


def test_codex_routing_rides_its_command_line(sandbox: Sandbox) -> None:
    argv = sandbox.start("codex").argv
    overrides = [argv[i + 1] for i, argument in enumerate(argv) if argument == "-c"]
    provider = "model_providers.ftmetal-launch"
    assert overrides == [
        'model_provider="ftmetal-launch"',
        f'{provider}.name="ftmetal"',
        f'{provider}.base_url="{sandbox.server_url}/v1"',
        f'{provider}.wire_api="responses"',
        f'{provider}.env_key="FTMETAL_API_KEY"',
        f'model_catalog_json="{sandbox.home / CODEX_CATALOG}"',
        'service_tier="default"',
    ]


def json_path(document: JsonValue, path: list[str | int]) -> JsonValue:
    for key in path:
        if isinstance(document, list) and isinstance(key, int):
            document = document[key]
        elif isinstance(document, dict) and isinstance(key, str):
            document = document[key]
        else:
            raise AssertionError(f"no {key!r} in {document!r}")
    return document


# (agent, the file in home, where the value is, what it must be).
FILE_VALUE_CASES: list[tuple[str, str, list[str | int], JsonValue]] = [
    ("codex", CODEX_CATALOG, ["models", 0, "context_window"], WINDOW),
    ("codex", CODEX_CATALOG, ["models", 0, "auto_compact_token_limit"], WINDOW - WINDOW // 4),
    # The test template switches thinking and thinks by default.
    ("codex", CODEX_CATALOG, ["models", 0, "supported_reasoning_levels", 0, "effort"], "none"),
    ("codex", CODEX_CATALOG, ["models", 0, "supported_reasoning_levels", 1, "effort"], "high"),
    ("codex", CODEX_CATALOG, ["models", 0, "default_reasoning_level"], "high"),
    ("openclaw", OPENCLAW_CONFIG, [*OPENCLAW_MODEL, "contextWindow"], WINDOW),
    ("openclaw", OPENCLAW_CONFIG, [*OPENCLAW_MODEL, "maxTokens"], WINDOW // 4),
    ("openclaw", OPENCLAW_CONFIG, [*OPENCLAW_MODEL, "reasoning"], True),
    ("hermes", HERMES_CONFIG, ["model", "context_length"], WINDOW),
    ("pi", PI_MODELS, [*PI_MODEL, "contextWindow"], WINDOW),
    ("pi", PI_MODELS, [*PI_MODEL, "maxTokens"], WINDOW // 4),
    ("pi", PI_MODELS, [*PI_MODEL, "reasoning"], True),
    ("pi", PI_MODELS, [*PI_MODEL, "_launch"], True),
    (
        "pi",
        PI_MODELS,
        [*PI_MODEL, "compat"],
        {"supportsDeveloperRole": False, "thinkingFormat": "deepseek"},
    ),
    ("dsh", DSH_PATCH, [0, "config", "models", 0, "contextWindow"], WINDOW),
    ("dsh", DSH_PATCH, [0, "config", "models", 0, "maxTokens"], WINDOW // 8),
]


@pytest.mark.parametrize(("agent", "file_name", "path", "expected"), FILE_VALUE_CASES)
def test_agent_files_hold_what_the_server_serves(
    sandbox: Sandbox, agent: str, file_name: str, path: list[str | int], expected: JsonValue
) -> None:
    sandbox.start(agent, "-y")
    # A JSON file is a YAML document too.
    assert json_path(yaml.safe_load((sandbox.home / file_name).read_text()), path) == expected


def test_agent_environment_holds_the_limit_the_server_enforces(sandbox: Sandbox) -> None:
    claude = sandbox.start("claude").env
    assert claude["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == str(WINDOW)
    assert claude["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == str(WINDOW // 4)
    config = json.loads(sandbox.start("opencode").env["OPENCODE_CONFIG_CONTENT"])
    limit = config["provider"]["ftmetal"]["models"][MODEL_ID]["limit"]
    assert limit == {"context": WINDOW, "output": WINDOW // 4}


def test_opencode_offers_the_thinking_switch(sandbox: Sandbox) -> None:
    config = json.loads(sandbox.start("opencode").env["OPENCODE_CONFIG_CONTENT"])
    model = config["provider"]["ftmetal"]["models"][MODEL_ID]
    assert model["reasoning"] is True
    assert model["variants"]["none"] == {"reasoningEffort": "none"}
    assert model["variants"]["high"] == {"disabled": True}


# (a variable the user set for Claude Code, its value, what Claude Code gets).
CLAUDE_USER_ENV_CASES: list[tuple[str, str, str]] = [
    ("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000", "64000"),
    ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "1000", "1000"),
    ("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "999999", str(WINDOW)),
    ("CLAUDE_CODE_MODEL_CAPABILITIES", "thinking", "-mid_conv_system;thinking"),
    ("CLAUDE_CODE_AUTO_MODE_SERVER", "1", "1"),
]


@pytest.mark.parametrize(("name", "value", "sent"), CLAUDE_USER_ENV_CASES)
def test_claude_keeps_what_the_user_set(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, name: str, value: str, sent: str
) -> None:
    monkeypatch.setenv(name, value)
    assert sandbox.start("claude").env[name] == sent


# (agent, a variable in the user's environment that would route it elsewhere).
CLEARED_ENV_CASES: list[tuple[str, str]] = [
    ("codex", "OPENAI_API_KEY"),
    ("claude", "CLAUDE_CODE_USE_BEDROCK"),
    ("claude", "CLAUDE_CODE_USE_VERTEX"),
    ("openclaw", "ANTHROPIC_API_KEY"),
    ("hermes", "CUSTOM_BASE_URL"),
    ("dsh", "OPENAI_API_KEY"),
]


@pytest.mark.parametrize(("agent", "name"), CLEARED_ENV_CASES)
def test_agent_loses_variables_that_route_elsewhere(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, agent: str, name: str
) -> None:
    monkeypatch.setenv(name, "1")
    monkeypatch.setenv("KEPT_VARIABLE", "kept")
    env = sandbox.start(agent, "-y").env
    assert name not in env and env["KEPT_VARIABLE"] == "kept"


# (agent, the arguments after --, the started command's end).
EXTRA_ARGS_CASES: list[tuple[str, list[str], list[str]]] = [
    ("claude", ["-p", "hi"], ["--model", MODEL_ID, "-p", "hi"]),
    ("opencode", ["run", "hi"], ["run", "hi"]),
    ("openclaw", ["status"], ["status"]),
    ("hermes", ["status"], ["status"]),
    ("pi", ["--print", "hi"], ["--model", MODEL_ID, "--print", "hi"]),
]


@pytest.mark.parametrize(("agent", "extra_args", "command_end"), EXTRA_ARGS_CASES)
def test_arguments_after_the_separator_reach_the_agent(
    sandbox: Sandbox, agent: str, extra_args: list[str], command_end: list[str]
) -> None:
    started = sandbox.start(agent, "-y", "--", *extra_args)
    assert started.argv[1:][-len(command_end) :] == command_end
    assert "chat" not in started.argv


# (what opencode --version prints, the arguments after --, the started command).
OPENCODE_VERSION_CASES: list[tuple[str, list[str], list[str]]] = [
    ("1.19.5", [], ["opencode", "--model", f"ftmetal/{MODEL_ID}"]),
    ("2.2.6", [], ["opencode", "--standalone"]),
    ("1.19.5", ["run", "hi"], ["opencode", "run", "hi"]),
    ("", [], ["opencode"]),
]


@pytest.mark.parametrize(("version", "extra_args", "argv"), OPENCODE_VERSION_CASES)
def test_opencode_command_follows_its_version(
    sandbox: Sandbox, version: str, extra_args: list[str], argv: list[str]
) -> None:
    make_executable(sandbox.bin_dir / "opencode", f"echo {version}\n")
    assert sandbox.start("opencode", "--", *extra_args).argv == argv


def test_dsh_defaults_to_the_served_model(sandbox: Sandbox) -> None:
    started = sandbox.start("dsh")
    patch_path = sandbox.home / DSH_PATCH
    assert started.argv[3:5] == ["--patch", str(patch_path)]
    # Entry ids of dsh-base's bundle; dsh 0.2 ignores settings outside an entry's config.
    entries = {entry["id"]: entry["config"] for entry in yaml.safe_load(patch_path.read_text())}
    assert entries["agent-default-model"] == {"provider": "deepseek-official", "model": MODEL_ID}
    assert entries["llm-deepseek"]["baseURL"] == f"{sandbox.server_url}/v1"
    assert entries["llm-deepseek"]["apiKeyEnv"] == "FTMETAL_API_KEY"
    assert [model["id"] for model in entries["llm-deepseek"]["models"]] == [MODEL_ID]
    assert entries["web-search-deepseek"] == {"apiKeyEnv": "FTMETAL_DSH_WEB_SEARCH_DISABLED"}


# (the arguments after --, the profile dsh boots, the arguments its app gets).
DSH_PROFILE_CASES: list[tuple[list[str], str, list[str]]] = [
    ([], "web", []),
    (["web", "a"], "web", ["a"]),
    (["headless", "run the tests"], "headless", ["run the tests"]),
    (["--profile", "tui", "a"], "tui", ["a"]),
    (["--profile=tui", "a"], "tui", ["a"]),
    (["--resume", "x"], "web", ["--resume", "x"]),
]


@pytest.mark.parametrize(("extra_args", "profile", "app_args"), DSH_PROFILE_CASES)
def test_dsh_takes_its_profile_as_dsh_does(
    sandbox: Sandbox, extra_args: list[str], profile: str, app_args: list[str]
) -> None:
    started = sandbox.start("dsh", "--", *extra_args)
    assert started.argv[1:3] == ["--profile", profile] and started.argv[5:] == app_args


def test_dsh_warns_about_an_old_node(sandbox: Sandbox, capsys: pytest.CaptureFixture[str]) -> None:
    make_executable(sandbox.bin_dir / "node", "echo v22.19.0\n")
    sandbox.start("dsh")
    assert "dsh requires Node.js >=24.9, node reports v22.19" in capsys.readouterr().err


# (the server's window, whether dsh's compaction rule then leaves it unable to compact).
DSH_COMPACTION_CASES: list[tuple[int, bool]] = [
    (WINDOW, True),
    (65_536, True),
    (89_000, True),
    (90_000, False),
    (131_072, False),
    (262_144, False),
]


@pytest.mark.parametrize(("window", "warns"), DSH_COMPACTION_CASES)
def test_dsh_warns_when_its_window_leaves_no_compaction(
    sandbox: Sandbox, capsys: pytest.CaptureFixture[str], window: int, warns: bool
) -> None:
    context = LaunchContext(
        server=ServerURL(origin=sandbox.server_url, openai_base_url=f"{sandbox.server_url}/v1"),
        model=ServedModel(model_id=MODEL_ID, context_length=window, accepts_images=False),
        extra_args=[],
        dry_run=False,
        assume_yes=True,
        agent_binary=str(sandbox.bin_dir / "dsh"),
    )
    PREPARERS["dsh"](context)
    assert (f"a {window}-token window leaves dsh no room" in capsys.readouterr().err) == warns


@pytest.mark.parametrize("agent", sorted(PREPARERS))
def test_dry_run_changes_nothing(
    sandbox: Sandbox, agent: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sandbox.launch(agent, "--dry-run") == 0
    assert sandbox.files() == set()
    output = capsys.readouterr().out
    assert f"Model: {MODEL_ID}" in output and f"Command: {agent}" in output


# (agent, where the started agent finds the server's key: an environment variable, or a file
# in home and the path to the key in it, which must then be owner-only)
API_KEY_CASES: list[tuple[str, str, list[str | int]]] = [
    ("claude", "ANTHROPIC_AUTH_TOKEN", []),
    ("codex", "FTMETAL_API_KEY", []),
    ("dsh", "FTMETAL_API_KEY", []),
    ("opencode", "OPENCODE_CONFIG_CONTENT", ["provider", "ftmetal", "options", "apiKey"]),
    ("openclaw", OPENCLAW_CONFIG, ["models", "providers", "ftmetal", "apiKey"]),
    ("hermes", HERMES_CONFIG, ["model", "api_key"]),
    ("pi", PI_MODELS, ["providers", "ftmetal", "apiKey"]),
]


@pytest.mark.parametrize("sandbox", ["secret"], indirect=True)
@pytest.mark.parametrize(("agent", "location", "path"), API_KEY_CASES)
def test_agent_gets_the_servers_api_key(
    sandbox: Sandbox,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    agent: str,
    location: str,
    path: list[str | int],
) -> None:
    assert sandbox.launch(agent, "-y") == 1
    assert "needs a valid API key" in capsys.readouterr().err
    assert sandbox.launch(agent, "--api-key", "secret", "--dry-run") == 0
    assert "secret" not in capsys.readouterr().out
    monkeypatch.setenv("FTMETAL_API_KEY", "secret")
    started = sandbox.start(agent, "-y")
    if location not in sandbox.files():
        value = started.env[location]
        assert (json_path(json.loads(value), path) if path else value) == "secret"
    else:
        file = sandbox.home / location
        assert json_path(yaml.safe_load(file.read_text()), path) == "secret"
        assert stat.S_IMODE(file.stat().st_mode) == 0o600


# (agent, arguments, whether the agent starts once --config has configured it).
CONFIG_ONLY_CASES: list[tuple[str, list[str], bool]] = [
    ("claude", ["-y"], True),
    ("codex", ["-y"], True),
    ("opencode", ["-y"], True),
    ("openclaw", ["-y"], True),
    ("pi", ["-y"], True),
    ("hermes", ["-y"], False),
    ("dsh", ["-y"], False),
    # No terminal to answer "launch now?" on.
    ("claude", [], False),
]


@pytest.mark.parametrize(("agent", "arguments", "starts"), CONFIG_ONLY_CASES)
def test_config_asks_before_starting(
    sandbox: Sandbox, agent: str, arguments: list[str], starts: bool
) -> None:
    if starts:
        sandbox.start(agent, "--config", *arguments)
    else:
        assert sandbox.launch(agent, "--config", *arguments) == 0
    assert sandbox.files() or agent == "claude"


# (agent, arguments, exit status, what it prints).
RESTORE_CASES: list[tuple[str, list[str], int, str]] = [
    ("codex", [], 0, "Codex launch configuration removed."),
    ("claude", [], 1, "claude does not support --restore"),
    ("codex", ["--config"], 1, "--restore cannot be combined"),
    ("codex", ["--", "exec"], 1, "--restore cannot be combined"),
]


@pytest.mark.parametrize(("agent", "arguments", "status", "message"), RESTORE_CASES)
def test_restore_removes_what_a_launch_wrote(
    sandbox: Sandbox,
    capsys: pytest.CaptureFixture[str],
    agent: str,
    arguments: list[str],
    status: int,
    message: str,
) -> None:
    sandbox.start("codex")
    (sandbox.home / ".codex/config.toml").write_text('model = "mine"\n')
    assert sandbox.launch(agent, "--restore", "--server", UNREACHABLE_SERVER, *arguments) == status
    output = capsys.readouterr()
    assert message in output.out + output.err
    kept = {".codex/config.toml"} if status == 0 else {".codex/config.toml", CODEX_CATALOG}
    assert kept <= sandbox.files() and (status != 0 or sandbox.files() == kept)


# (the agent named on the command line, the agent that starts).
AGENT_NAME_CASES: list[tuple[str, str]] = [
    ("Claude", "claude"),
    ("deepseek-harness", "dsh"),
    ("MoltBot", "openclaw"),
    ("clawdbot", "openclaw"),
]


@pytest.mark.parametrize(("name", "agent"), AGENT_NAME_CASES)
def test_agent_names_and_aliases(sandbox: Sandbox, name: str, agent: str) -> None:
    assert sandbox.start(name, "-y").binary == str(sandbox.bin_dir / agent)


def test_ctrl_c_ends_the_launch_quietly(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(server: ServerURL, api_key: str | None) -> ServedModel:
        raise KeyboardInterrupt

    monkeypatch.setattr(launch_run, "discover_served_model", interrupted)
    assert sandbox.launch("claude") == 130
    assert "Traceback" not in capsys.readouterr().err


# (extra arguments for codex, whether they would reroute it and are refused).
CODEX_EXTRA_ARGS_CASES: list[tuple[list[str], bool]] = [
    (["exec", "hi"], False),
    (["-c", "model_reasoning_effort=high"], False),
    (["-m", "gpt"], True),
    (["--model=gpt"], True),
    (["-p", "work"], True),
    (["--profile", "work"], True),
    (["-c", "model_provider=openai"], True),
    (["-cmodel='x'"], True),
    (["--config=model_providers.openai.base_url=x"], True),
    (["--config", "'profile' = x"], True),
]


@pytest.mark.parametrize(("extra_args", "refused"), CODEX_EXTRA_ARGS_CASES)
def test_codex_refuses_arguments_that_reroute_it(
    sandbox: Sandbox, capsys: pytest.CaptureFixture[str], extra_args: list[str], refused: bool
) -> None:
    if refused:
        assert sandbox.launch("codex", "--", *extra_args) == 1
        assert "conflicting extra" in capsys.readouterr().err
    else:
        assert sandbox.start("codex", "--", *extra_args).argv[-len(extra_args) :] == extra_args


# (what codex --version prints, whether ftm launch refuses it as too old).
CODEX_VERSION_CASES: list[tuple[str, bool]] = [
    ("codex-cli 0.133.9", True),
    ("codex-cli 0.134.0", False),
    ("codex-cli 1.2.0", False),
    ("", False),
]


@pytest.mark.parametrize(("version", "refused"), CODEX_VERSION_CASES)
def test_codex_too_old_is_refused(
    sandbox: Sandbox, capsys: pytest.CaptureFixture[str], version: str, refused: bool
) -> None:
    make_executable(sandbox.bin_dir / "codex", f"echo {version}\n")
    if refused:
        assert sandbox.launch("codex") == 1
        assert "is too old, ftm launch needs 0.134" in capsys.readouterr().err
    else:
        sandbox.start("codex")


# (arguments after --, the started Pi command).
PI_MODEL_ARGS_CASES: list[tuple[list[str], list[str]]] = [
    ([], ["pi", "--provider", "ftmetal", "--model", MODEL_ID]),
    (["--model", "a/b"], ["pi", "--model", "a/b"]),
    (["--provider", "x", "-p", "hi"], ["pi", "--provider", "x", "-p", "hi"]),
    # A value that looks like the option is the other option's value.
    (
        ["--thinking", "--model"],
        ["pi", "--provider", "ftmetal", "--model", MODEL_ID, "--thinking", "--model"],
    ),
    (["--model"], ["pi", "--provider", "ftmetal", "--model", MODEL_ID, "--model"]),
]


@pytest.mark.parametrize(("extra_args", "argv"), PI_MODEL_ARGS_CASES)
def test_pi_takes_the_users_own_model(
    sandbox: Sandbox, extra_args: list[str], argv: list[str]
) -> None:
    assert sandbox.start("pi", "--", *extra_args).argv == argv


def test_pi_keeps_the_users_own_model_entries(sandbox: Sandbox) -> None:
    path = sandbox.home / PI_MODELS
    path.parent.mkdir(parents=True)
    mine = {"id": "mine", "name": "kept"}
    stale = {"id": "old-launch", "_launch": True}
    provider = {"apiKey": "user-key", "headers": {"x": "1"}, "models": [mine, stale]}
    path.write_text(json.dumps({"providers": {"ftmetal": provider}}))
    sandbox.start("pi")
    written = json.loads(path.read_text())["providers"]["ftmetal"]
    assert written["apiKey"] == "user-key" and written["headers"] == {"x": "1"}
    assert [model["id"] for model in written["models"]] == ["mine", MODEL_ID]
    # The user's own entry for the served model wins over the one a launch would write.
    served = {"id": MODEL_ID, "contextWindow": 1}
    path.write_text(json.dumps({"providers": {"ftmetal": {"models": [served]}}}))
    sandbox.start("pi")
    assert json.loads(path.read_text())["providers"]["ftmetal"]["models"] == [served]


def test_backups_keep_the_newest_five(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    for content in "0123456":
        config_files.write_text_with_backup(path, content)
    backups = sorted(tmp_path.glob("config.json.*.bak"))
    assert [backup.read_text() for backup in backups] == list("12345")
    assert path.read_text() == "6"


# (agent, variables naming directories in home, or set as is when they start with $, arguments
# after --, where its config lands).
LOCATION_CASES: list[tuple[str, dict[str, str], list[str], str]] = [
    ("codex", {"CODEX_HOME": "codex-home"}, [], "codex-home/ftmetal-launch.config.toml"),
    ("hermes", {"HERMES_HOME": "hermes-home"}, [], "hermes-home/config.yaml"),
    ("hermes", {"HERMES_HOME": "$HOME/hermes-home"}, [], "hermes-home/config.yaml"),
    ("hermes", {}, ["-p", "work"], ".hermes/profiles/work/config.yaml"),
    ("hermes", {}, ["--profile=work"], ".hermes/profiles/work/config.yaml"),
    (
        "hermes",
        {"HERMES_HOME": "hermes-home"},
        ["--profile", "Work"],
        "hermes-home/profiles/work/config.yaml",
    ),
    (
        "hermes",
        {"HERMES_HOME": "hermes-home/profiles/work"},
        ["-p", "other"],
        "hermes-home/profiles/other/config.yaml",
    ),
    ("openclaw", {"OPENCLAW_CONFIG_PATH": "claw.json"}, [], "claw.json"),
    ("openclaw", {"OPENCLAW_STATE_DIR": "claw-state"}, [], "claw-state/openclaw.json"),
    ("openclaw", {"OPENCLAW_HOME": "claw-home"}, [], "claw-home/.openclaw/openclaw.json"),
    ("openclaw", {}, ["--profile", "work"], ".openclaw-work/openclaw.json"),
    ("openclaw", {}, ["--profile=work"], ".openclaw-work/openclaw.json"),
    ("openclaw", {}, ["--dev"], ".openclaw-dev/openclaw.json"),
    ("openclaw", {}, ["--log-level", "debug", "--dev", "status"], ".openclaw-dev/openclaw.json"),
    # There the options are the command's own, and after -- they are not options at all.
    ("openclaw", {}, ["gateway", "--dev"], OPENCLAW_CONFIG),
    ("openclaw", {}, ["qa", "matrix", "--profile", "work"], OPENCLAW_CONFIG),
    ("openclaw", {}, ["agent", "--", "--dev"], OPENCLAW_CONFIG),
    ("opencode", {"XDG_STATE_HOME": "state"}, [], "state/opencode/model.json"),
    ("pi", {"PI_CODING_AGENT_DIR": "pi-agent"}, [], "pi-agent/models.json"),
    ("dsh", {"DSH_HOME": "dsh-home"}, [], "dsh-home/ftmetal-launch.cordis.patch.yml"),
]


@pytest.mark.parametrize(("agent", "variables", "extra_args", "file_name"), LOCATION_CASES)
def test_config_goes_where_the_agent_reads_it(
    sandbox: Sandbox,
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
    variables: dict[str, str],
    extra_args: list[str],
    file_name: str,
) -> None:
    for name, value in variables.items():
        monkeypatch.setenv(name, value if value.startswith("$") else str(sandbox.home / value))
    path = sandbox.home / file_name
    # Codex wants its home to exist, and Hermes a profile to switch to.
    path.parent.mkdir(parents=True, exist_ok=True)
    sandbox.start(agent, "-y", "--", *extra_args)
    assert path.is_file()
    assert {(sandbox.home / written).parent for written in sandbox.files()} == {path.parent}


# (HERMES_HOME under home, if set, where the config lands while ~/.hermes sticks to "work").
HERMES_STICKY_CASES: list[tuple[dict[str, str], str]] = [
    ({}, ".hermes/profiles/work/config.yaml"),
    ({"HERMES_HOME": ".hermes"}, ".hermes/profiles/work/config.yaml"),
    # A profile's own home is that profile, whatever its root sticks to.
    ({"HERMES_HOME": ".hermes/profiles/other"}, ".hermes/profiles/other/config.yaml"),
]


@pytest.mark.parametrize(("variables", "file_name"), HERMES_STICKY_CASES)
def test_hermes_follows_its_sticky_profile(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], file_name: str
) -> None:
    for name, directory in variables.items():
        monkeypatch.setenv(name, str(sandbox.home / directory))
    (sandbox.home / ".hermes/profiles/work").mkdir(parents=True)
    (sandbox.home / ".hermes/profiles/other").mkdir(parents=True)
    (sandbox.home / ".hermes/active_profile").write_text("work\n")
    sandbox.start("hermes")
    assert sandbox.files() == {".hermes/active_profile", file_name}


# (agent, arguments after --, what the refusal says); a Hermes profile "other" exists.
REFUSED_PROFILE_CASES: list[tuple[str, list[str], str]] = [
    ("hermes", ["-p", "work"], "hermes profile create work"),
    ("hermes", ["--profile="], "Hermes refuses the profile name ''"),
    ("hermes", ["-p", "../other"], "Hermes refuses the profile name '../other'"),
    ("openclaw", ["--profile", "../x"], "OpenClaw refuses the profile name '../x'"),
    ("openclaw", ["--profile="], "OpenClaw refuses the profile name ''"),
    ("openclaw", ["--profile", "--dev"], "OpenClaw refuses the profile name ''"),
]


@pytest.mark.parametrize(("agent", "extra_args", "message"), REFUSED_PROFILE_CASES)
def test_profile_the_agent_would_refuse_is_refused(
    sandbox: Sandbox,
    capsys: pytest.CaptureFixture[str],
    agent: str,
    extra_args: list[str],
    message: str,
) -> None:
    (sandbox.home / ".hermes/profiles/other").mkdir(parents=True)
    assert sandbox.launch(agent, "-y", "--", *extra_args) == 1
    assert message in capsys.readouterr().err
    assert sandbox.files() == set() and not list(sandbox.home.glob(".openclaw*"))


# (the agent's file in home, what the user had in it, the key the launch must keep).
KEPT_CONFIG_CASES: list[tuple[str, str, str, str]] = [
    ("openclaw", OPENCLAW_CONFIG, '{"gateway": {"port": 1}}', "gateway"),
    ("hermes", HERMES_CONFIG, "display:\n  theme: dark\n", "display"),
    ("opencode", ".local/state/opencode/model.json", '{"favorite": ["x"]}', "favorite"),
    ("pi", PI_MODELS, '{"$schema": "x"}', "$schema"),
    # A byte order mark, which Pi and OpenClaw skip.
    ("pi", PI_MODELS, '﻿{"$schema": "x"}', "$schema"),
    ("hermes", HERMES_CONFIG, "﻿display:\n  theme: dark\n", "display"),
]


@pytest.mark.parametrize(("agent", "file_name", "original", "kept_key"), KEPT_CONFIG_CASES)
def test_user_config_is_kept_and_backed_up_once(
    sandbox: Sandbox, agent: str, file_name: str, original: str, kept_key: str
) -> None:
    path = sandbox.home / file_name
    path.parent.mkdir(parents=True)
    path.write_text(original)
    sandbox.start(agent, "-y")
    sandbox.start(agent, "-y")
    assert kept_key in yaml.safe_load(path.read_text())
    backups = list(path.parent.glob(f"{path.name}.*.bak"))
    assert [backup.read_text() for backup in backups] == [original]


# (agent, its file in home, content ftm cannot rewrite without losing some of it, what it says).
REFUSED_CONFIG_CASES: list[tuple[str, str, bytes, str]] = [
    (
        "openclaw",
        OPENCLAW_CONFIG,
        b'{\n  // JSON5\n  "gateway": {"port": 1},\n}\n',
        "syntax ftm cannot rewrite safely",
    ),
    ("pi", PI_MODELS, b'{"providers": {},}\n', "syntax ftm cannot rewrite safely"),
    ("pi", PI_MODELS, b'{"name": "caf\xe9"}\n', "codec can't decode byte 0xe9"),
    ("hermes", HERMES_CONFIG, b"model:\n\tdefault: x\n", "cannot parse"),
    ("hermes", HERMES_CONFIG, b"# caf\xe9\nmodel: {}\n", "codec can't decode byte 0xe9"),
    ("hermes", HERMES_CONFIG, b"- a list\n", "does not hold a YAML mapping"),
    ("opencode", ".local/state/opencode/model.json", b"[]\n", "does not hold a JSON object"),
]


@pytest.mark.parametrize(("agent", "file_name", "original", "message"), REFUSED_CONFIG_CASES)
def test_config_that_does_not_parse_is_left_alone(
    sandbox: Sandbox,
    capsys: pytest.CaptureFixture[str],
    agent: str,
    file_name: str,
    original: bytes,
    message: str,
) -> None:
    path = sandbox.home / file_name
    path.parent.mkdir(parents=True)
    path.write_bytes(original)
    assert sandbox.launch(agent, "-y") == 1
    error = capsys.readouterr().err
    assert str(path) in error and message in error
    assert sandbox.files() == {file_name} and path.read_bytes() == original


HERMES_COMMENTED_CONFIG = """\
# Hermes Agent CLI Configuration
_config_version: 52

# ======================
# Model Configuration
# ======================
model:
  # Default model to use
  default: "anthropic/claude-opus-4.6"
  provider: "auto"  # auto-detect

  # api_key: "your-key-here"
  base_url: "https://openrouter.ai/api/v1"
# providers:
#   meta:
#     base_url: https://api.meta.ai/v1

# ======================
agent:
  system_prompt: "用中文回答"  # the reply language
"""


def test_hermes_keeps_every_line_outside_its_model_section(sandbox: Sandbox) -> None:
    path = sandbox.home / HERMES_CONFIG
    path.parent.mkdir(parents=True)
    path.write_text(HERMES_COMMENTED_CONFIG)
    sandbox.start("hermes")
    head, rest = HERMES_COMMENTED_CONFIG.split("model:\n")
    written = path.read_text()
    assert written.startswith(head) and written.endswith(rest[rest.index("# providers:") :])
    config = yaml.safe_load(written)
    assert config["model"]["default"] == MODEL_ID
    assert config["agent"]["system_prompt"] == "用中文回答"


# (a Hermes config whose model section cannot be replaced as text, a key it must keep).
HERMES_REDUMPED_CASES: list[tuple[str, str]] = [
    ("{model: {default: x}, display: {theme: dark}}\n", "display"),
    ("model: &main\n  default: x\nfallback_model: *main\n", "fallback_model"),
]


@pytest.mark.parametrize(("original", "kept_key"), HERMES_REDUMPED_CASES)
def test_hermes_config_the_text_patch_cannot_keep_is_dumped_whole(
    sandbox: Sandbox, original: str, kept_key: str
) -> None:
    path = sandbox.home / HERMES_CONFIG
    path.parent.mkdir(parents=True)
    path.write_text(original)
    sandbox.start("hermes")
    config = yaml.safe_load(path.read_text())
    assert config["model"]["default"] == MODEL_ID and kept_key in config


def test_non_ascii_text_is_written_as_is(sandbox: Sandbox) -> None:
    path = sandbox.home / PI_MODELS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"providers": {"local": {"name": "小龙虾"}}}, ensure_ascii=False))
    sandbox.start("pi")
    assert "小龙虾" in path.read_text()


def test_failed_write_leaves_the_file_as_it_was(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = sandbox.home / PI_MODELS
    path.parent.mkdir(parents=True)
    path.write_text("{}")

    def failing_replace(source: Path, target: Path) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(config_files.os, "replace", failing_replace)
    assert sandbox.launch("pi") == 1
    error = capsys.readouterr().err
    assert error.startswith(f"cannot write {path}: ") and error.count("\n") == 1
    assert path.read_text() == "{}" and list(path.parent.glob(".*.tmp")) == []


# (the mode of the file a config link points at; read-only, it is still replaced, not opened).
LINKED_CONFIG_MODE_CASES: list[int] = [0o640, 0o444]


@pytest.mark.parametrize("mode", LINKED_CONFIG_MODE_CASES)
def test_rewritten_config_keeps_its_mode_and_its_link(
    sandbox: Sandbox, tmp_path: Path, mode: int
) -> None:
    target = tmp_path / "dotfiles" / "models.json"
    target.parent.mkdir()
    target.write_text("{}")
    target.chmod(mode)
    path = sandbox.home / PI_MODELS
    path.parent.mkdir(parents=True)
    path.symlink_to(target)
    sandbox.start("pi")
    assert path.is_symlink() and "ftmetal" in json.loads(target.read_text())["providers"]
    assert stat.S_IMODE(target.stat().st_mode) == mode
    backups = list(path.parent.glob("models.json.*.bak"))
    assert [backup.read_text() for backup in backups] == ["{}"]
    assert [stat.S_IMODE(backup.stat().st_mode) for backup in backups] == [mode]


def test_pi_keeps_the_other_providers(sandbox: Sandbox) -> None:
    path = sandbox.home / PI_MODELS
    path.parent.mkdir(parents=True)
    ollama = {"baseUrl": "http://localhost:11434/v1", "models": [{"id": "x"}]}
    path.write_text(json.dumps({"providers": {"ollama": ollama}}))
    sandbox.start("pi")
    providers = json.loads(path.read_text())["providers"]
    assert providers["ollama"] == ollama
    assert providers["ftmetal"]["baseUrl"] == f"{sandbox.server_url}/v1"


ONBOARDED: dict[str, JsonValue] = {"wizard": {"lastRunAt": "2026-11-08T00:00:00Z"}}


def openclaw_calls(sandbox: Sandbox, tmp_path: Path, script: str = "") -> Path:
    """Where the fake openclaw logs each command it is given, besides what script does."""
    log = tmp_path / "openclaw-calls"
    make_executable(sandbox.bin_dir / "openclaw", f'echo "$@" >> {log}\n{script}')
    return log


def write_openclaw_config(sandbox: Sandbox, config: dict[str, JsonValue]) -> Path:
    path = sandbox.home / OPENCLAW_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config))
    return path


def test_openclaw_first_launch_onboards_after_its_security_notice(
    sandbox: Sandbox, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = openclaw_calls(sandbox, tmp_path)
    assert sandbox.launch("openclaw") == 1
    error = capsys.readouterr().err
    assert "I understand the risks. Continue? requires confirmation" in error
    assert sandbox.files() == set() and not log.exists()
    assert sandbox.start("openclaw", "-y").argv[1:] == ["tui", "--message", "Wake up, my friend!"]
    [onboard] = log.read_text().splitlines()
    assert onboard.startswith(
        "onboard --non-interactive --accept-risk --auth-choice custom-api-key"
    )
    assert f"--custom-base-url {sandbox.server_url}/v1 --custom-model-id {MODEL_ID}" in onboard
    assert onboard.endswith("--install-daemon")


def test_openclaw_onboarded_starts_without_asking(sandbox: Sandbox, tmp_path: Path) -> None:
    log = openclaw_calls(sandbox, tmp_path)
    write_openclaw_config(sandbox, ONBOARDED)
    assert sandbox.start("openclaw").argv == ["openclaw", "tui"]
    assert not log.exists()


def test_openclaw_keeps_what_the_user_set_on_the_served_model(sandbox: Sandbox) -> None:
    model = {"id": MODEL_ID, "alias": "kept", "contextWindow": 1}
    provider: JsonValue = {"models": [model, {"id": "not-served"}]}
    path = write_openclaw_config(
        sandbox, {**ONBOARDED, "models": {"providers": {"ftmetal": provider}}}
    )
    sandbox.start("openclaw")
    [written] = json.loads(path.read_text())["models"]["providers"]["ftmetal"]["models"]
    assert written["alias"] == "kept" and written["contextWindow"] != 1


def test_openclaw_names_its_web_ui(sandbox: Sandbox, capsys: pytest.CaptureFixture[str]) -> None:
    gateway: JsonValue = {"auth": {"token": "a b"}}
    write_openclaw_config(sandbox, {**ONBOARDED, "gateway": gateway})
    port = os.environ["OPENCLAW_GATEWAY_PORT"]
    sandbox.start("openclaw")
    assert f"http://127.0.0.1:{port}/#token=a%20b" in capsys.readouterr().err


def test_openclaw_gateway_port_comes_from_its_config(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = openclaw_calls(sandbox, tmp_path)
    port = int(os.environ["OPENCLAW_GATEWAY_PORT"])
    monkeypatch.delenv("OPENCLAW_GATEWAY_PORT")
    write_openclaw_config(sandbox, {**ONBOARDED, "gateway": {"port": port}})
    sandbox.start("openclaw")
    assert not log.exists()


# (arguments after --, the commands the fake openclaw saw, the exit status of the launch).
OPENCLAW_GATEWAY_DOWN_CASES: list[tuple[list[str], list[str], int]] = [
    # The service does not start and its own gateway never listens.
    ([], ["daemon start", "gateway run --force"], 1),
    (["tui"], ["daemon start", "gateway run --force"], 1),
    # Any other command starts without one.
    (["status"], [], 0),
]


@pytest.mark.parametrize(("extra_args", "calls", "status"), OPENCLAW_GATEWAY_DOWN_CASES)
def test_openclaw_starts_a_gateway_that_is_down(
    sandbox: Sandbox,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    calls: list[str],
    status: int,
) -> None:
    log = openclaw_calls(sandbox, tmp_path, 'case "$1" in daemon) exit 1;; esac\n')
    write_openclaw_config(sandbox, ONBOARDED)
    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT", "9")
    monkeypatch.setattr(openclaw, "OPENCLAW_DAEMON_WAIT_SECONDS", 0)
    # Long enough for the fake gateway to log its command before it is stopped.
    monkeypatch.setattr(openclaw, "OPENCLAW_GATEWAY_WAIT_SECONDS", 1)
    if status == 0:
        sandbox.start("openclaw", "--", *extra_args)
    else:
        assert sandbox.launch("openclaw", "--", *extra_args) == status
        assert "the OpenClaw gateway did not start on 127.0.0.1:9" in capsys.readouterr().err
    assert (log.read_text().splitlines() if log.exists() else []) == calls


def test_openclaw_gateway_of_its_own_stops_with_it(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = openclaw_calls(
        sandbox,
        tmp_path,
        'case "$1" in daemon) exit 1;; gateway) exec sleep 30;; tui) exit 3;; esac\n',
    )
    write_openclaw_config(sandbox, ONBOARDED)
    monkeypatch.setenv("OPENCLAW_GATEWAY_PORT", "9")
    monkeypatch.setattr(openclaw, "OPENCLAW_DAEMON_WAIT_SECONDS", 0)
    started: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def recorded_popen(
        argv: list[str], env: dict[str, str], stdout: int, stderr: int
    ) -> subprocess.Popen[bytes]:
        started.append(real_popen(argv, env=env, stdout=stdout, stderr=stderr))
        return started[-1]

    monkeypatch.setattr(
        openclaw,
        "subprocess",
        SimpleNamespace(run=subprocess.run, Popen=recorded_popen, DEVNULL=subprocess.DEVNULL),
    )
    monkeypatch.setattr(openclaw, "wait_for_port", lambda port, seconds: bool(started))
    # The agent runs as a child here, and its exit status is the launch's.
    assert sandbox.launch("openclaw") == 3
    assert log.read_text().splitlines() == ["daemon start", "gateway run --force", "tui"]
    assert started[0].poll() is not None


# (answers to the prompts, the channels the fake openclaw was asked to add).
OPENCLAW_CHANNEL_CASES: list[tuple[str, int]] = [
    ("n\n", 0),
    ("\n", 1),
    ("y\ny\nn\n", 2),
]


@pytest.mark.parametrize(("answers", "added"), OPENCLAW_CHANNEL_CASES)
def test_openclaw_offers_to_connect_a_channel(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answers: str, added: int
) -> None:
    log = openclaw_calls(sandbox, tmp_path)
    write_openclaw_config(sandbox, ONBOARDED)
    monkeypatch.setattr(launch_context, "is_interactive", lambda: True)
    monkeypatch.setattr(openclaw, "is_interactive", lambda: True)
    monkeypatch.setattr(sys, "stdin", io.StringIO(answers))
    sandbox.start("openclaw")
    calls = log.read_text().splitlines() if log.exists() else []
    assert calls == ["channels add"] * added


# (the agents section of the default OpenClaw config, arguments after --, what the launch says).
OPENCLAW_NOTICE_CASES: list[tuple[JsonValue, list[str], str]] = [
    (None, [], "OpenClaw works in {home}/.openclaw/workspace,"),
    (None, ["--profile", "work", "status"], "OpenClaw works in {home}/.openclaw-work/workspace,"),
    ({"defaults": {"workspace": "~/claw-work"}}, [], "OpenClaw works in {home}/claw-work,"),
    ({"entries": {"main": {"model": "x/y"}}}, [], "main agent's own model (x/y)"),
    (
        {"list": [{"id": "other"}, {"id": "main", "model": {"primary": "x/y"}}]},
        [],
        "main agent's own model (x/y)",
    ),
]


@pytest.mark.parametrize(("agents", "extra_args", "notice"), OPENCLAW_NOTICE_CASES)
def test_openclaw_names_its_workspace_and_a_main_agent_model(
    sandbox: Sandbox,
    capsys: pytest.CaptureFixture[str],
    agents: JsonValue,
    extra_args: list[str],
    notice: str,
) -> None:
    path = sandbox.home / OPENCLAW_CONFIG
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"agents": agents} if agents is not None else {}))
    sandbox.start("openclaw", "-y", "--", *extra_args)
    assert notice.format(home=sandbox.home) in capsys.readouterr().err


def test_the_cli_starts_an_agent(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FTMETAL_HOST", sandbox.server_url)
    with pytest.raises(AgentStarted):
        ftm_main(["launch", "claude"])


def test_opencode_recent_models_start_with_this_one(sandbox: Sandbox) -> None:
    path = sandbox.home / ".local/state/opencode/model.json"
    path.parent.mkdir(parents=True)
    other = {"providerID": "other", "modelID": "m"}
    this = {"providerID": "ftmetal", "modelID": MODEL_ID}
    path.write_text(json.dumps({"recent": [other, this]}))
    sandbox.start("opencode")
    assert json.loads(path.read_text())["recent"] == [this, other]


SERVER_BYPASS = "127.0.0.1,localhost,::1"
# (NO_PROXY and no_proxy as the user set them, what the agent gets in each; None is unset).
NO_PROXY_CASES: list[tuple[str | None, str | None, str | None, str | None]] = [
    (None, None, SERVER_BYPASS, SERVER_BYPASS),
    ("corp.example", None, f"corp.example,{SERVER_BYPASS}", f"corp.example,{SERVER_BYPASS}"),
    (None, ".corp,*.in", f".corp,*.in,{SERVER_BYPASS}", f".corp,*.in,{SERVER_BYPASS}"),
    (
        "a.example,127.0.0.1",
        "b.example",
        "a.example,127.0.0.1,b.example,localhost,::1",
        "a.example,127.0.0.1,b.example,localhost,::1",
    ),
    ("*", None, "*", None),
    ("corp.example", " * ", "corp.example", " * "),
]


@pytest.mark.parametrize(("user_upper", "user_lower", "upper", "lower"), NO_PROXY_CASES)
def test_proxy_in_the_environment_is_kept_off_the_server(
    sandbox: Sandbox,
    monkeypatch: pytest.MonkeyPatch,
    user_upper: str | None,
    user_lower: str | None,
    upper: str | None,
    lower: str | None,
) -> None:
    monkeypatch.setenv("http_proxy", UNREACHABLE_SERVER)
    for name, value in (("NO_PROXY", user_upper), ("no_proxy", user_lower)):
        if value is not None:
            monkeypatch.setenv(name, value)
    env = sandbox.start("claude").env
    assert (env.get("NO_PROXY"), env.get("no_proxy")) == (upper, lower)


def test_proxy_bypass_needs_a_proxy_and_shows_in_a_dry_run(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env = sandbox.start("claude").env
    assert "NO_PROXY" not in env and "no_proxy" not in env
    monkeypatch.setenv("HTTPS_PROXY", UNREACHABLE_SERVER)
    assert sandbox.launch("claude", "--dry-run") == 0
    assert f"no_proxy={SERVER_BYPASS}" in capsys.readouterr().out


def test_unreachable_server_is_reported(
    sandbox: Sandbox, capsys: pytest.CaptureFixture[str]
) -> None:
    assert sandbox.launch("claude", "--server", UNREACHABLE_SERVER) == 1
    error = capsys.readouterr().err
    assert f"cannot reach the ftmetal server at {UNREACHABLE_SERVER}" in error
    assert "ftm serve --model" in error


def test_server_that_is_not_ftmetal_is_named(
    sandbox: Sandbox, capsys: pytest.CaptureFixture[str]
) -> None:
    app = FastAPI()

    @app.get("/v1/models")
    def list_models() -> dict[str, list[dict[str, str]]]:
        return {"data": [{"id": "llama3", "object": "model"}]}

    with running_server(app) as port:
        assert sandbox.launch("claude", "--server", f"127.0.0.1:{port}") == 1
    error = capsys.readouterr().err
    assert error.count("\n") == 1 and f"http://127.0.0.1:{port}/v1/models" in error


def test_host_environment_picks_the_server(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FTMETAL_HOST", sandbox.server_url)
    with pytest.raises(AgentStarted):
        run_launch(["claude"], UNREACHABLE_SERVER)


# (server, its origin), or None for a server refused.
SERVER_URL_CASES: list[tuple[str, str | None]] = [
    ("http://127.0.0.1:1919", "http://127.0.0.1:1919"),
    ("http://127.0.0.1:1919/", "http://127.0.0.1:1919"),
    ("http://127.0.0.1:1919/v1", "http://127.0.0.1:1919"),
    ("http://127.0.0.1:1919/v1/", "http://127.0.0.1:1919"),
    ("127.0.0.1:8000", "http://127.0.0.1:8000"),
    ("box.local", "http://box.local:1919"),
    # A URL without a port is on its scheme's.
    (" http://box.local ", "http://box.local"),
    ("https://box.local", "https://box.local"),
    ("https://box.local:8443/v1", "https://box.local:8443"),
    ("http://0.0.0.0:1919", "http://127.0.0.1:1919"),
    ("0.0.0.0", "http://127.0.0.1:1919"),
    ("http://[::]:1919", "http://[::1]:1919"),
    ("http://[fe80::1]:1919", "http://[fe80::1]:1919"),
    ("::1", "http://[::1]:1919"),
    ("[::1]", "http://[::1]:1919"),
    # A path is a prefix the server sits under, behind a proxy.
    ("https://proxy.example/ftm/v1", "https://proxy.example/ftm"),
    ("http://127.0.0.1:1919/v2", "http://127.0.0.1:1919/v2"),
    ("http://", None),
    ("ftp://box.local", None),
]


@pytest.mark.parametrize(("server", "origin"), SERVER_URL_CASES)
def test_server_url(server: str, origin: str | None) -> None:
    if origin is None:
        with pytest.raises(ValueError):
            resolve_server_url(server, 1919)
    else:
        resolved = resolve_server_url(server, 1919)
        assert (resolved.origin, resolved.openai_base_url) == (origin, f"{origin}/v1")


@pytest.fixture
def installers_run(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """What the agent installers ran; each one leaves the agent in its fallback directory."""
    runs: list[list[str]] = []

    def fake_run(argv: list[str], check: bool, env: dict[str, str]) -> None:
        runs.append(argv)
        for agent, installer in AGENT_INSTALLERS.items():
            if list(installer.argv) == argv:
                make_executable(sandbox.home / installer.fallback_dirs[-1] / agent)
                assert dict(installer.env).items() <= env.items()
                assert check

    monkeypatch.setattr(
        launch_install,
        "subprocess",
        SimpleNamespace(run=fake_run, CalledProcessError=subprocess.CalledProcessError),
    )
    for dependency in ("curl", "sh", "bash", "git", "npm", "node"):
        make_executable(sandbox.bin_dir / dependency)
    return runs


@pytest.mark.parametrize("agent", sorted(PREPARERS))
def test_missing_agent_is_installed_then_started_from_its_fallback(
    sandbox: Sandbox, installers_run: list[list[str]], agent: str
) -> None:
    (sandbox.bin_dir / agent).unlink()
    installer = AGENT_INSTALLERS[agent]
    started = sandbox.start(agent, "-y")
    assert installers_run == [list(installer.argv)]
    binary_dir = sandbox.home / installer.fallback_dirs[-1]
    assert started.binary == str(binary_dir / agent)
    assert started.argv[0] == agent
    assert started.env["PATH"].split(os.pathsep)[0] == str(binary_dir)


def test_agent_in_a_package_manager_directory_is_started(sandbox: Sandbox) -> None:
    (sandbox.bin_dir / "openclaw").unlink()
    make_executable(sandbox.system_bin_dir / "openclaw")
    started = sandbox.start("openclaw", "-y")
    assert started.binary == str(sandbox.system_bin_dir / "openclaw")
    assert started.env["PATH"].split(os.pathsep)[0] == str(sandbox.system_bin_dir)


def test_npm_agent_is_found_under_the_npm_prefix(sandbox: Sandbox, tmp_path: Path) -> None:
    (sandbox.bin_dir / "pi").unlink()
    make_executable(tmp_path / "npm-prefix" / "bin" / "pi")
    make_executable(sandbox.bin_dir / "npm", f"echo {tmp_path / 'npm-prefix'}\n")
    assert sandbox.start("pi").binary == str(tmp_path / "npm-prefix" / "bin" / "pi")


def test_missing_agent_needs_approval(
    sandbox: Sandbox, installers_run: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    (sandbox.bin_dir / "claude").unlink()
    assert sandbox.launch("claude") == 1
    assert installers_run == []
    error = capsys.readouterr().err
    assert "Claude Code is not installed." in error and "installation cancelled" in error
    assert "re-run with -y" in error


class InterruptedStdin(io.StringIO):
    def readline(self, size: int | None = -1) -> str:
        raise KeyboardInterrupt


# (what the terminal answers the install prompt with, whether the agent is installed).
CONFIRM_CASES: list[tuple[io.StringIO, bool]] = [
    (io.StringIO("\n"), True),
    (io.StringIO("y\n"), True),
    (io.StringIO("maybe\nyes\n"), True),
    (io.StringIO("n\n"), False),
    # Ctrl-D, then Ctrl-C.
    (io.StringIO(""), False),
    (InterruptedStdin(), False),
]


@pytest.mark.parametrize(("stdin", "installs"), CONFIRM_CASES)
def test_prompt_answers(
    sandbox: Sandbox,
    installers_run: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stdin: io.StringIO,
    installs: bool,
) -> None:
    (sandbox.bin_dir / "claude").unlink()
    monkeypatch.setattr(launch_context, "is_interactive", lambda: True)
    monkeypatch.setattr(sys, "stdin", stdin)
    if installs:
        sandbox.start("claude")
        assert installers_run == [list(AGENT_INSTALLERS["claude"].argv)]
    else:
        assert sandbox.launch("claude") == 1
        assert installers_run == [] and "installation cancelled" in capsys.readouterr().err
    assert "Install Claude Code now? [Y/n] " in capsys.readouterr().err or not installs


def test_missing_install_dependency_is_named(
    sandbox: Sandbox, installers_run: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    (sandbox.bin_dir / "claude").unlink()
    (sandbox.bin_dir / "curl").unlink()
    assert sandbox.launch("claude", "-y") == 1
    assert installers_run == []
    assert "curl" in capsys.readouterr().err


def test_install_only_needs_no_server(sandbox: Sandbox, installers_run: list[list[str]]) -> None:
    arguments = ("--install-only", "--server", UNREACHABLE_SERVER, "-y")
    assert sandbox.launch("claude", *arguments) == 0
    assert installers_run == []
    (sandbox.bin_dir / "claude").unlink()
    assert sandbox.launch("claude", *arguments) == 0
    assert installers_run == [list(AGENT_INSTALLERS["claude"].argv)]


def test_install_only_dry_run_installs_nothing(
    sandbox: Sandbox, installers_run: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert sandbox.launch("claude", "--install-only", "--dry-run", "-y") == 0
    assert f"is installed: {sandbox.bin_dir / 'claude'}" in capsys.readouterr().out
    (sandbox.bin_dir / "claude").unlink()
    assert sandbox.launch("claude", "--install-only", "--dry-run", "-y") == 0
    assert installers_run == []
    assert "Install command: bash -c" in capsys.readouterr().out


def test_force_reinstall_runs_the_installer_of_a_present_agent(
    sandbox: Sandbox, installers_run: list[list[str]]
) -> None:
    sandbox.start("claude", "-y", "--force-reinstall")
    assert installers_run == [list(AGENT_INSTALLERS["claude"].argv)]


# (what the installer raises, or None when it exits leaving no binary, what the launch reports).
FAILED_INSTALL_CASES: list[tuple[subprocess.CalledProcessError | None, str]] = [
    (subprocess.CalledProcessError(1, "install"), "failed to install claude"),
    (None, "/.local/bin) to PATH, then re-run: ftm launch claude"),
]


@pytest.mark.parametrize(("install_error", "message"), FAILED_INSTALL_CASES)
def test_failed_install_is_reported(
    sandbox: Sandbox,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    install_error: subprocess.CalledProcessError | None,
    message: str,
) -> None:
    for dependency in ("curl", "bash"):
        make_executable(sandbox.bin_dir / dependency)
    (sandbox.bin_dir / "claude").unlink()

    def unsuccessful_run(argv: list[str], check: bool, env: dict[str, str]) -> None:
        if install_error is not None:
            raise install_error

    monkeypatch.setattr(
        launch_install,
        "subprocess",
        SimpleNamespace(run=unsuccessful_run, CalledProcessError=subprocess.CalledProcessError),
    )
    assert sandbox.launch("claude", "-y") == 1
    assert message in capsys.readouterr().err


def test_agent_that_cannot_start_is_reported(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_execve(binary: str, argv: list[str], env: dict[str, str]) -> None:
        raise OSError(8, "Exec format error")

    monkeypatch.setattr(os, "execve", failing_execve)
    assert sandbox.launch("claude") == 1
    assert f"cannot start {sandbox.bin_dir / 'claude'}" in capsys.readouterr().err


def test_dsh_runs_through_npx_when_npm_cannot_install_it(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused_install(argv: list[str], check: bool, env: dict[str, str]) -> None:
        raise subprocess.CalledProcessError(243, argv)

    monkeypatch.setattr(
        launch_install,
        "subprocess",
        SimpleNamespace(run=refused_install, CalledProcessError=subprocess.CalledProcessError),
    )
    for dependency in ("npm", "node", "npx"):
        make_executable(sandbox.bin_dir / dependency)
    (sandbox.bin_dir / "dsh").unlink()
    started = sandbox.start("dsh", "-y")
    assert started.binary == str(sandbox.bin_dir / "npx")
    assert started.argv[:5] == ["npx", "--yes", "@deepseek-ai/dsh", "--profile", "web"]


def test_pi_from_its_legacy_package_is_replaced(sandbox: Sandbox, tmp_path: Path) -> None:
    prefix = tmp_path / "npm-prefix"
    package = prefix / "lib/node_modules/@mariozechner/pi-coding-agent"
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({"name": "@mariozechner/pi-coding-agent"}))
    make_executable(package / "dist/cli.js")
    (prefix / "bin").mkdir()
    (prefix / "bin/pi").symlink_to(package / "dist/cli.js")
    (sandbox.bin_dir / "pi").unlink()
    log = tmp_path / "npm-calls"
    make_executable(
        sandbox.bin_dir / "npm",
        f'if [ "$1" = prefix ]; then echo {prefix}; else echo "$@" >> {log}; fi\n',
    )
    sandbox.start("pi")
    npm = f"--prefix {prefix}"
    assert log.read_text().splitlines() == [
        f"{npm} install --global --ignore-scripts --force @earendil-works/pi-coding-agent",
        f"{npm} uninstall --global @mariozechner/pi-coding-agent",
        f"{npm} install --global --ignore-scripts @earendil-works/pi-coding-agent",
    ]


# (Hermes's .env, a variable set in the environment, arguments after --, whether it offers to
# connect a messaging app).
HERMES_MESSAGING_CASES: list[tuple[str, dict[str, str], list[str], bool]] = [
    ("", {}, [], True),
    ("# TELEGRAM_BOT_TOKEN=x\n", {}, [], True),
    ("export TELEGRAM_BOT_TOKEN='123:abc'\n", {}, [], False),
    ("", {"SLACK_BOT_TOKEN": "xoxb"}, [], False),
    ("", {}, ["status"], False),
]


@pytest.mark.parametrize(("env_file", "variables", "extra_args", "offers"), HERMES_MESSAGING_CASES)
def test_hermes_offers_to_connect_a_messaging_app(
    sandbox: Sandbox,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_file: str,
    variables: dict[str, str],
    extra_args: list[str],
    offers: bool,
) -> None:
    log = tmp_path / "hermes-calls"
    make_executable(sandbox.bin_dir / "hermes", f'echo "$@" >> {log}\n')
    (sandbox.home / ".hermes").mkdir()
    (sandbox.home / ".hermes/.env").write_text(env_file)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(launch_context, "is_interactive", lambda: True)
    monkeypatch.setattr(hermes, "is_interactive", lambda: True)
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))
    sandbox.start("hermes", "--", *extra_args)
    calls = log.read_text().splitlines() if log.exists() else []
    assert calls == (["gateway setup"] if offers else [])
