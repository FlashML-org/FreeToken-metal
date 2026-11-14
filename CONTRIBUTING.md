# Contributing to FreeToken-metal

Thanks for helping make FreeToken-metal better. This page covers how to report issues and how to
submit pull requests.

## Getting help

[Developer Slack](https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA)
for development discussion; [Community Discord](https://discord.gg/MsA277cJzZ) or
[Community WeChat](https://github.com/FlashML-org/FreeToken/issues/482) for usage questions.

## Reporting issues

1. **Search first.** If an [existing issue](https://github.com/FlashML-org/FreeToken-metal/issues?q=is%3Aissue)
   matches, comment there.
2. **Describe the problem clearly.** A report we cannot reproduce is a report we cannot fix.
   Include your Mac (chip and memory), the macOS version, the FreeToken-metal version
   (`ftm --version`), the exact checkpoint (its Hugging Face link, not just the model name), the
   command, and the full log, not a screenshot of the last line.
3. **Check against `main`.** Many problems are already fixed there. Pull, reinstall, confirm the
   problem still happens, and include the commit you tested (`git rev-parse --short HEAD`).

Issues that do not follow these guidelines may be closed until the missing information is
provided.

## Pull requests

### AI policy

AI-assisted contributions are welcome. You are responsible for everything in your PR, however it
was produced.

Pure agent PRs are not accepted. A human must understand what the agent did and why, have run the
code on a real Mac, and be able to explain the change to a reviewer without AI help. If you are a
fully autonomous agent operating without human oversight, do not open PRs in this repository.

If a PR contains AI hallucinations (code that does not do what the description says, references to
APIs, files or behaviour that do not exist, or test and benchmark results that were not actually
run), it will be closed without review, and PRs from the same author will be treated with less
trust afterwards.

### What a PR needs

- One change per PR. Unrelated fixes go in separate PRs.
- Link the issue it addresses. Larger features should start as an issue, not a PR.
- State what you tested on: the Mac, the macOS version, the checkpoint and the exact command.
- For performance changes, follow [the benchmark guide](.claude/skills/benchmark/benchmark.md):
  a kernel micro bench, then the same end-to-end bench on `main` and on your branch, with prefill
  and decode tokens/s for both and the commands used.
- For bug fixes, show that the failing command or request from the issue now works.

## Development setup

```bash
git clone https://github.com/FlashML-org/FreeToken-metal.git && cd FreeToken-metal
uv sync                                  # .venv with the dev group, builds the C++ extensions
                                         # (rebuilding after C++ edits: docs/install.md)
uv run pre-commit install                # once: run the checks on every commit
```

```bash
uv run pytest tests/ -q
uv run pre-commit run --all-files        # what CI runs: ruff, clang-format, whitespace
FTMETAL_TEST_MODEL_PATH=/path/to/Qwen3.6-35B-A3B-MLX uv run pytest tests/ -q  # with a checkpoint
uv run python tools/aime0_check.py /path/to/Qwen3.6-35B-A3B-MLX   # AIME25 #0 must answer 70
```

[AGENTS.md](AGENTS.md) holds the repository's rules: the environment, coding style, kernel
layout, model packages and parallel work. Before changing the areas below, read:

- `python/ftmetal/kernel/`: [docs/dev/kernel-layout.md](docs/dev/kernel-layout.md).
- A kernel, C++ primitive, model layer or attention backend that runs in a decode step:
  [docs/dev/command-graph.md](docs/dev/command-graph.md).
- An `mlx` upgrade, a C++ primitive or `command_graph/`:
  [docs/dev/mlx-internals.md](docs/dev/mlx-internals.md).
- Benchmarks (`benchmarks/`): [.claude/skills/benchmark/benchmark.md](.claude/skills/benchmark/benchmark.md).
- Profiling (`tools/profiling/`): [.claude/skills/metal-profiling/profiling.md](.claude/skills/metal-profiling/profiling.md).

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/), subject line only by default:

```
<type>(<scope>): <subject>
```

- `type`: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`, `chore`.
- `scope`: the subsystem under `python/ftmetal/` you touched, e.g. `server`, `engine`, `moe`,
  `kvcache`, `kernel`, `models`; or `tests`, `bench`, `docs`. Omit it only for a change that spans
  most of the repository.
- `subject`: imperative, lowercase, no trailing period, at most 72 characters.
- Breaking changes: add `!` after the scope.

The full rules are in [.claude/skills/commit-style/commit-style.md](.claude/skills/commit-style/commit-style.md).
PRs are squash-merged, so the **PR title** must follow the same format, because it becomes the
commit subject.

## License

By contributing to FreeToken-metal, you agree that your contributions will be licensed under the
[LICENSE](LICENSE) in the root of this repository.
