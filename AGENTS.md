# Instructions for AI coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) first: setup, tests, PR and commit rules. The rules below
add what an agent must follow.

## Rules

- Use the repository's uv environment: `uv sync`, then `uv run ...`. Never a bare `python` or
  `pip`, never `pip install` by hand, never bump a pinned dependency as a side effect
  ([development.md](docs/dev/development.md)).
- Follow [coding-style.md](.claude/skills/code-style/coding-style.md) for every code change or
  review, and [commit-style.md](.claude/skills/commit-style/commit-style.md) for every commit
  message.
- `uv run pre-commit run --all-files` must pass; CI runs the same checks.
- Model packages under `models/` never import from each other (a shared encoder tower such as
  `models/qwen3_vl/` is the exception). Shared code moves to `layers/`, `attention/`, `kvcache/`
  or `kernel/` only once a second model needs it. A model port lands with a parity test against
  mlx-lm (`tests/models/test_parity_*.py`).

## Read before you change

| Before | Read |
|---|---|
| adding, moving or renaming anything under `python/ftmetal/kernel/` | [kernel-layout.md](docs/dev/kernel-layout.md) |
| changing a kernel, a C++ primitive, a model layer or an attention backend that runs in a decode step | [command-graph.md](docs/dev/command-graph.md) |
| upgrading mlx, or changing a C++ primitive or `command_graph/` | [mlx-internals.md](docs/dev/mlx-internals.md) |
| measuring performance or reporting numbers | [benchmark.md](.claude/skills/benchmark/benchmark.md) |
| profiling GPU work | [profiling.md](.claude/skills/metal-profiling/profiling.md) |
