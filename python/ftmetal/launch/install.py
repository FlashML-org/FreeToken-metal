"""Finding an agent's binary and installing it when missing: its own installer after a prompt,
an npm package's runner when a global install fails, and a renamed npm package moved over.

Port of FreeToken launch.py's installers. Divergences: Pi installs from npm with its scripts off;
dsh runs through npx when npm cannot install it globally; Pi's legacy package is replaced.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ftmetal.launch.context import command_output, confirm

# Package managers install there, and a fresh install's PATH may not have them yet.
SYSTEM_BIN_DIRS = (Path("/opt/homebrew/bin"), Path("/usr/local/bin"))


@dataclass(frozen=True, kw_only=True)
class AgentInstaller:
    display_name: str
    argv: tuple[str, ...]
    dependencies: tuple[str, ...]
    # Where the installer leaves the binary, relative to home, for a PATH not yet updated.
    fallback_dirs: tuple[str, ...]
    # The npm package the installer puts in npm's global prefix, if it does.
    npm_package: str | None = None
    # Run through npx when the global install fails.
    runs_from_npx: bool = False
    # The package's former name, replaced by npm_package when it is what is installed.
    legacy_npm_package: str | None = None
    env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, kw_only=True)
class AgentCommand:
    """What starts the agent: binary, then leading_args before the agent's own arguments."""

    binary: str
    leading_args: tuple[str, ...] = ()


AGENT_INSTALLERS = {
    "codex": AgentInstaller(
        display_name="Codex CLI",
        argv=("sh", "-c", "curl -fsSL https://chatgpt.com/codex/install.sh | sh"),
        dependencies=("curl", "sh"),
        fallback_dirs=(".local/bin",),
        env=(("CODEX_NON_INTERACTIVE", "1"),),
    ),
    "claude": AgentInstaller(
        display_name="Claude Code",
        argv=("bash", "-c", "curl -fsSL https://claude.ai/install.sh | bash"),
        dependencies=("curl", "bash"),
        fallback_dirs=(".local/bin", ".claude/local"),
    ),
    "opencode": AgentInstaller(
        display_name="OpenCode",
        argv=("bash", "-c", "set -o pipefail; curl -fsSL https://opencode.ai/install | bash"),
        dependencies=("curl", "bash"),
        fallback_dirs=(".opencode/bin",),
    ),
    "openclaw": AgentInstaller(
        display_name="OpenClaw",
        argv=("bash", "-c", "curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard"),
        dependencies=("curl", "bash"),
        fallback_dirs=(".npm-global/bin",),
        npm_package="openclaw",
    ),
    "hermes": AgentInstaller(
        display_name="Hermes",
        argv=(
            "bash",
            "-lc",
            "curl -fsSL https://hermes-agent.nousresearch.com/install.sh"
            " | bash -s -- --skip-setup --non-interactive",
        ),
        dependencies=("bash", "curl", "git"),
        fallback_dirs=(".local/bin",),
    ),
    "pi": AgentInstaller(
        display_name="Pi",
        # Its curl installer offers to start Pi at the end, before this has configured it.
        argv=("npm", "install", "--global", "--ignore-scripts", "@earendil-works/pi-coding-agent"),
        dependencies=("npm", "node"),
        fallback_dirs=(".npm-global/bin",),
        npm_package="@earendil-works/pi-coding-agent",
        legacy_npm_package="@mariozechner/pi-coding-agent",
    ),
    "dsh": AgentInstaller(
        display_name="DeepSeek Harness",
        argv=("npm", "install", "--global", "@deepseek-ai/dsh"),
        dependencies=("npm", "node"),
        fallback_dirs=(".npm-global/bin",),
        npm_package="@deepseek-ai/dsh",
        runs_from_npx=True,
    ),
}


def agent_dirs(agent: str) -> list[Path]:
    """Where the agent's installer may leave it outside PATH, the likeliest first."""
    installer = AGENT_INSTALLERS[agent]
    if installer.npm_package is not None:
        npm_prefix = command_output(["npm", "prefix", "--global"]).strip()
    else:
        npm_prefix = ""
    npm_dirs = [Path(npm_prefix) / "bin"] if npm_prefix else []
    home_dirs = [Path.home() / directory for directory in installer.fallback_dirs]
    return [*npm_dirs, *home_dirs, *SYSTEM_BIN_DIRS]


def resolve_agent_binary(agent: str) -> str | None:
    path = shutil.which(agent)
    if path is not None:
        return path
    else:
        candidates = [directory / agent for directory in agent_dirs(agent)]
        found = (str(file) for file in candidates if file.is_file() and os.access(file, os.X_OK))
        return next(found, None)


def npm_prefix_of_legacy_package(binary: str, package: str) -> Path | None:
    """The npm prefix holding package, when binary is that package's; None when it is not."""
    for directory in Path(os.path.realpath(binary)).parents:
        try:
            name = json.loads((directory / "package.json").read_text()).get("name")
        except (OSError, ValueError, AttributeError):
            continue
        depth = len(Path(package).parts)
        # A global install is PREFIX/lib/node_modules/@scope/name.
        if name == package and directory.parents[depth - 1].name == "node_modules":
            return directory.parents[depth + 1]
    return None


def replace_legacy_package(agent: str, binary: str) -> str:
    """The agent's binary, after moving its npm install from the legacy package name over."""
    installer = AGENT_INSTALLERS[agent]
    legacy, package = installer.legacy_npm_package, installer.npm_package
    prefix = npm_prefix_of_legacy_package(binary, legacy) if legacy is not None else None
    if prefix is None or package is None or legacy is None:
        return binary
    else:
        print(f"Updating {installer.display_name} to {package}...", file=sys.stderr)
        npm = ["npm", "--prefix", str(prefix)]
        steps = [
            [*npm, "install", "--global", "--ignore-scripts", "--force", package],
            [*npm, "uninstall", "--global", legacy],
            [*npm, "install", "--global", "--ignore-scripts", package],
        ]
        for step in steps:
            completed = subprocess.run(step, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"failed to replace {legacy} with {package}: {shlex.join(step)}\n"
                    f"{completed.stderr.strip()}"
                )
        path = resolve_agent_binary(agent)
        if path is None:
            raise RuntimeError(f"{agent} was updated but was not found; restart your shell")
        return path


def ensure_agent_installed(agent: str, assume_yes: bool, force_reinstall: bool) -> AgentCommand:
    """How to start the agent, after running its installer if it has none or a reinstall is
    asked; npx runs it when its npm install fails."""
    existing = resolve_agent_binary(agent)
    if existing is not None and not force_reinstall:
        return AgentCommand(binary=replace_legacy_package(agent, existing))
    else:
        installer = AGENT_INSTALLERS[agent]
        missing = [name for name in installer.dependencies if shutil.which(name) is None]
        if missing:
            joined = "\n  ".join(missing)
            raise RuntimeError(
                f"{installer.display_name} needs these to install:\n  {joined}\n\n"
                f"Install them, then re-run:\n  ftm launch {agent}"
            )
        action = "Reinstall" if existing is not None else "Install"
        if not assume_yes:
            state = "already installed" if existing is not None else "not installed"
            print(f"{installer.display_name} is {state}.", file=sys.stderr)
            print(f"Install command: {shlex.join(installer.argv)}", file=sys.stderr)
            if not confirm(f"{action} {installer.display_name} now?"):
                raise RuntimeError(f"{agent} installation cancelled")
        print(f"\n{action}ing {installer.display_name}...", file=sys.stderr)
        try:
            subprocess.run(
                list(installer.argv), check=True, env={**os.environ, **dict(installer.env)}
            )
        except subprocess.CalledProcessError as exc:
            npx = shutil.which("npx") if installer.runs_from_npx else None
            if npx is None or installer.npm_package is None:
                raise RuntimeError(f"failed to install {agent}: {exc}") from exc
            else:
                print(f"Running {installer.display_name} through npx instead.", file=sys.stderr)
                return AgentCommand(binary=npx, leading_args=("--yes", installer.npm_package))
        path = resolve_agent_binary(agent)
        if path is None:
            raise RuntimeError(
                f"{agent} was installed but was not found. Add the directory it went to "
                f"(usually {agent_dirs(agent)[0]}) to PATH, then re-run: ftm launch {agent}"
            )
        print(f"{installer.display_name} installed successfully\n", file=sys.stderr)
        return AgentCommand(binary=path)


__all__ = [
    "AGENT_INSTALLERS",
    "SYSTEM_BIN_DIRS",
    "AgentCommand",
    "ensure_agent_installed",
    "resolve_agent_binary",
]
