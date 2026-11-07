# Instructions for AI coding agents

## Python environment

Python is managed with [uv](https://docs.astral.sh/uv/), always:

- The environment is `.venv/` at the repository root, created by `uv sync` (or
  `uv venv`) from `pyproject.toml`; it is git-ignored. Run everything through it
  (`uv run pytest tests/`, `uv run ruff check`, `uv run ftm serve ...`). Never
  call a bare `python`, `pip` or a system or global interpreter, and never point
  the repository at a virtual environment that lives outside it.
- Add or change dependencies with `uv add` / `uv remove` so `pyproject.toml` stays
  the source of truth; never `pip install` into the environment by hand. `uv.lock` is
  local to each checkout and git-ignored.
- `mlx`, `mlx-lm`, `openai`, the Hugging Face packages and the build requirements are
  pinned in `pyproject.toml`; do not bump them as a side effect of another change.
- Do not create a second virtual environment, install into another project's
  environment, or hardcode an interpreter path anywhere in the repository. The one
  environment outside it is `~/.ftmetal/venv`, where `install.sh` installs the engine
  to run it, without the dev group; development stays in `.venv/`.

## Coding style guidelines

Before writing, modifying, or reviewing code, read and follow
[.claude/skills/code-style/coding-style.md](.claude/skills/code-style/coding-style.md).

## Kernel layout

Before adding, moving or renaming anything under `python/ftmetal/kernel/`, read and follow
[docs/dev/kernel-layout.md](docs/dev/kernel-layout.md).

## Commit style guidelines

Before writing a commit message, read and follow
[.claude/skills/commit-style/commit-style.md](.claude/skills/commit-style/commit-style.md).
Only commit when the user asks.

## Parallel work with subagents

- Parallelize only when the modules are fully independent: their interface to the
  rest of the code is already agreed and neither side needs to touch the other's
  files. Examples that qualify: optimizing a kernel behind an interface the upper
  layers already call; implementing two models' own parts in `models/<a>/` and
  `models/<b>/` when neither touches shared code. Anything that changes a shared
  interface (`layers/`, `attention/base.py`, `kvcache/`, `core.py`) is done
  serially, by one agent.
- Every subagent works in its own `git worktree` on its own branch; never two
  agents in the same checkout. The parent integrates by merging or rebasing the
  branches, resolves conflicts once, and runs the full test suite before calling
  the work done.

## Model code and shared components

- Packages under `models/` never import from each other. A model's layers, weight
  mapping and attention specifics live in its own package. The one exception is an encoder
  tower that several families load, which gets its own package (`models/qwen3_vl/`, as in
  FreeToken) and serves no model by itself; models may import it.
- When a piece is genuinely shared, extract it into `layers/` (or `attention/`,
  `kvcache/`, `kernel/`) and have every model import the shared version; do not
  reuse across model packages by import.
- Do not extract early. A layer whose generality is uncertain is first
  implemented inside `models/<model>/`; it moves to `layers/` only once a second
  model needs the same thing.
