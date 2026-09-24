# Instructions for AI coding agents

## Python environment

Python is managed with [uv](https://docs.astral.sh/uv/), always:

- The environment is `.venv/` at the repository root, created by `uv sync` (or
  `uv venv`) from `pyproject.toml`; it is git-ignored. Run everything through it
  (`uv run pytest tests/`, `uv run ruff check`, `uv run ftm serve ...`). Never
  call a bare `python`, `pip` or a system or global interpreter, and never point
  the repository at a virtual environment that lives outside it.
- Add or change dependencies with `uv add` / `uv remove` so `pyproject.toml` and
  `uv.lock` stay the source of truth; never `pip install` into the environment by
  hand, and never edit the lock file directly.
- `mlx` is pinned in `pyproject.toml`; do not bump it as a side effect of another
  change.
- Do not create a second virtual environment, install into another project's
  environment, or hardcode an interpreter path anywhere in the repository.

## Coding style guidelines

Before writing, modifying, or reviewing code, read and follow
[.claude/skills/code-style/coding-style.md](.claude/skills/code-style/coding-style.md).

## Commit style guidelines

Before writing a commit message, read and follow
[.claude/skills/commit-style/commit-style.md](.claude/skills/commit-style/commit-style.md).
Only commit when the user asks.
