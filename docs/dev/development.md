# Development guide

A first walk through working on FreeToken-metal.

## Set up

You need an Apple Silicon Mac with macOS 15 or later, the Xcode Command Line Tools and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/FlashML-org/FreeToken-metal.git && cd FreeToken-metal
uv sync
uv run pre-commit install
```

`uv sync` creates `.venv/` with the development tools and builds the C++ extensions. Run every
command through `uv run`, so it uses that environment.

## Run it

```bash
uv run ftm serve --model /path/to/Qwen3.6-35B-A3B-MLX
uv run pytest tests/ -q
```

Most tests build tiny models and need no checkpoint. Set `FTMETAL_TEST_MODEL_PATH` to a real
checkpoint to also run the tests that load one.

## Edit and rebuild

Python and `.metal` changes take effect on the next run. After changing a `.cpp`, `.h` or `.mm`
file, rebuild the extensions:

```bash
uv sync --reinstall-package freetoken-metal
```

Add dependencies with `uv add`, never with `pip`.

## Find your way around

The engine lives in `python/ftmetal/` and follows FreeToken's layout; each module's docstring
names the FreeToken file it ports.

| Directory | What is in it |
|---|---|
| `server/`, `launch/` | the HTTP APIs and `ftm launch` |
| `scheduler/`, `engine/` | batching, the decode loop and memory budgeting |
| `kvcache/` | paged KV pools and the prefix caches |
| `models/` | one package per model family |
| `layers/`, `attention/`, `moe/` | shared building blocks, attention backends and MoE offload |
| `kernel/`, `command_graph/` | custom Metal kernels and decode-step capture |

Each model package stands alone: models never import from each other, and code moves to `layers/`
only once a second model needs it. A new model comes with a parity test against mlx-lm
(`tests/models/test_parity_*.py`).

## Before you open a PR

`uv run pre-commit run --all-files` and `uv run pytest tests/ -q` should pass. Some areas have
their own notes: [kernel-layout.md](kernel-layout.md), [command-graph.md](command-graph.md) and
[mlx-internals.md](mlx-internals.md). [CONTRIBUTING.md](../../CONTRIBUTING.md) has the PR rules.
