# ftmetal

LLM serving engine for one Apple Silicon Mac, built on [MLX](https://github.com/ml-explore/mlx)
and Metal, ported from [FreeToken](https://github.com/FlashML-org/FreeToken).

## Install

```bash
./install.sh                             # builds this checkout into ~/.ftmetal/venv, links ~/.local/bin/ftm
ftm serve --model /path/to/model
```

Needs macOS 15 or later on Apple Silicon and the Xcode Command Line Tools. The script keeps its own
pinned uv in `~/.ftmetal/bin` (linked into `~/.local/bin` if there is no uv yet), uses Python 3.13,
downloading it if missing, and leaves out the dev group. The install is a copy of the sources: rerun
it after any change.

## Development

```bash
uv sync                                  # creates .venv with the dev group, builds the C++ extensions
uv sync --reinstall-package ftmetal      # after editing a .cpp, .h or .mm (not for .metal)
uv run pytest tests/ -q
uv run pre-commit install                # once: run the checks below on every commit
uv run pre-commit run --all-files        # what CI runs: ruff, clang-format (C++, headers), whitespace
uvx pyright python tests tools                                 # 0 errors expected
FTMETAL_TEST_MODEL_PATH=/path/to/Qwen3.6-35B-A3B-MLX uv run pytest tests/models/test_real_checkpoint.py -q
uv run python tools/aime0_check.py /path/to/Qwen3.6-35B-A3B-MLX   # AIME25 #0 must answer 70
uv run python -m benchmarks.bench_serving --dataset sharegpt --tokenizer /path/to/model
                                         # prefill and decode tok/s of a running server
```

Benchmarks live in `benchmarks/` (kernel micro benches, in-process end-to-end benches and the
serving bench); how to run them and report results is in `.claude/skills/benchmark/benchmark.md`.
Profiling tools (per-kernel GPU time, timelines, hardware counters) live in `tools/profiling/`;
how to use them is in `.claude/skills/metal-profiling/profiling.md`.

The extensions `ftmetal.kernel._C` and `ftmetal.command_graph._C` are built by CMake
against the pinned `mlx` wheel (what they rely on in MLX: `docs/dev/mlx-internals.md`); the
Metal kernels under `python/ftmetal/kernel/` (layout in `docs/dev/kernel-layout.md`) are
compiled at first use.

## Serve and chat

```bash
uv run ftm serve --model /path/to/Qwen3.6-35B-A3B-MLX      # OpenAI API on 127.0.0.1:1919
uv run ftm serve --model /path/to/model --port 8000 --max-running-requests 8 --num-token 65536
uv run ftm shell                                            # chat with the running server
uv run ftm shell --server http://127.0.0.1:8000             # or FTMETAL_HOST=http://...
uv run ftm launch claude                                    # a coding agent on the running server
uv run ftm pack --model /path/to/Qwen3.6-35B-A3B-MLX       # expert pack for MoE offload
uv run ftm --version
```

`ftm launch <agent>` (`claude`, `codex`, `opencode`, `openclaw`, `hermes`, `pi` or `dsh`) points a
coding agent at the running server and starts it, the way `ollama launch` does for Ollama. It reads
the model, the longest context a request may hold and whether thinking can be switched off from
`/v1/models`, offers to install the agent if it is missing (`-y` approves every prompt; Enter means
yes), and writes the agent's configuration where the agent reads it (its home variables, such as
`CODEX_HOME` or `HERMES_HOME`, and profiles count), keeping the last five backups of any file it
changes and every setting it does not own (Hermes's `config.yaml` keeps its comments outside the
`model` section). A config file it cannot parse, such as an OpenClaw `openclaw.json` in JSON5, stops
the launch with the file untouched. `--server URL` picks the server (a host without a scheme means
port 1919, a URL its scheme's port, and a path is a prefix the server sits under), `--dry-run`
prints the plan and changes nothing, `--config` configures the agent and asks before starting it,
`--install-only` needs no server, `--restore` removes what a launch wrote for Codex, and arguments
after `--` go to the agent. With a proxy in the environment the agent is told to reach the server
directly (`NO_PROXY`). OpenClaw is onboarded on its first launch (after its security notice), its
gateway is started when it is down, and it works in its own workspace, not the current directory;
OpenClaw and Hermes offer to connect a messaging app before an interactive session.

`ftm serve` exposes FreeToken's routes: the OpenAI `POST /v1/chat/completions` (streamed
or not; `reasoning_content` carries the reasoning, the `<think>` block on Qwen3.6 and Flint and
the thought channel on Gemma4, and `tools` come back as `tool_calls`;
`chat_template_kwargs: {"enable_thinking": false}` or `thinking: {"type": "disabled"}`
turns thinking off, and so does `reasoning_effort: "none"`; any other effort turns it on and
reaches a chat template that grades effort as the nearest level it accepts, which `/v1/models`
lists along with `thinking_toggleable` and `default_thinking`), `POST /v1/completions` and
`GET /v1/models`; the Anthropic
`POST /v1/messages` (thinking, text and tool_use blocks; a system message after the first
turn stays in place, as its own turn when the chat template renders one there and folded into
the turn next to it otherwise, `--anthropic-inline-system`) and `/v1/messages/count_tokens`;
the OpenAI Responses `POST /v1/responses` (stateless: `GET /v1/responses/{id}` and its
`/cancel` answer 404); and `POST /generate` (a raw text prompt, no chat template, streamed
as `{"text": ...}` events and sampled greedily). The control plane is FreeToken's too:
`GET /health`, `GET /v1/stats` (throughput, request counters, token totals, pool usage),
`GET /v1/requests?since=<cursor>` (recent requests), `GET /v1/cache/status`,
`POST /v1/cache/flush?timeout=<s>` (drop the prefix cache once the engine is idle) and, from
loopback only, `POST /v1/admin/prepare-stop` (close admission, drain, abort the rest, seal
the token totals); `FTMETAL_API_LOG_DIR` logs every request body as JSON lines, and
`--cors-origins` sets the CORS allow-list. Any API key is accepted unless `--api-key` (or
`FTMETAL_API_KEY`) sets one, which every route but `/health` then requires as a bearer token
or `x-api-key`; `ftm launch` and `ftm shell` take it the same way. `presence_penalty` and
`frequency_penalty` apply. The reasoning and tool-call parsers follow the model family
(`--reasoning-parser`, `--tool-call-parser` override them); `--enable-cache-report` adds the
prefix-cache hit to each `usage`, and `--enable-timings-report` a per-request `timings` object
(prefill and decode spans and rates; on every streamed chunk with `timings_per_token`).
`ftm shell` is a client of the chat endpoint: `/think` toggles
thinking, `/reset` clears the conversation, `/exit` or Ctrl-D quits, Ctrl-C cancels a
reply. Environment knobs are read as `FTMETAL_<NAME>`.

A MoE checkpoint whose experts do not fit the memory budget, in any format its family loads
(bf16, MLX affine or float modes, the Hugging Face layout, an NVFP4 or FP8 export), serves
them from an offload cache that reads them from the checkpoint on demand
(`--moe-strategy auto|fused|offload`, FreeToken's flags): `--moe-cache-size` or
`--moe-cache-rate` size it (without them the KV pools keep the context first, `--num-token`,
also `--context-length`, or 64K tokens, and the experts get the rest; a `--num-token` the
budget cannot hold stops the start), `--disable-moe-prefill-overlap`
stops long prefills from reading each layer's experts ahead, and `--memory-ratio` sets the
budget as a fraction of Metal's recommended working set, which also emulates a smaller Mac.
`ftm pack --model DIR` writes `DIR/experts.packed`, each routed expert as one record (an
affine checkpoint's scales and biases as codes into a per-layer table of their pairs), which
the offload cache then reads instead of the checkpoint: one read per expert, and smaller
affine slots, which a cache sized from the budget turns into more of them. A pack the checkpoint has
changed under is passed over with a warning; `FTMETAL_MOE_PACK=0` ignores it.

```bash
FTMETAL_TEST_MODEL_PATH=/path/to/Qwen3.6-35B-A3B-MLX uv run pytest tests/server tests/test_tokenizer.py -q
```

## Acknowledgment

ftmetal is the Apple Silicon port of [FreeToken](https://github.com/FlashML-org/FreeToken) and is
built on [MLX](https://github.com/ml-explore/mlx).

## License

[Apache License 2.0](LICENSE).
