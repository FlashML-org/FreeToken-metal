# ftmetal

LLM serving engine for one Apple Silicon Mac: Python + MLX + Metal.

## Install

```bash
./install.sh                             # builds this checkout into ~/.ftmetal/venv, links ~/.local/bin/ftm
ftm serve --model /path/to/model
```

Needs macOS 14 or later on Apple Silicon and the Xcode Command Line Tools. The script keeps its own
pinned uv in `~/.ftmetal/bin` (linked into `~/.local/bin` if there is no uv yet), uses Python 3.13,
downloading it if missing, and leaves out the dev group. The install is a copy of the sources: rerun
it after any change.

## Development

```bash
uv sync                                  # creates .venv with the dev group, builds the C++ extensions
uv sync --reinstall-package ftmetal      # after editing a .cpp or .h (not for .metal)
uv run pytest tests/ -q
uv run ruff check python tests tools && uv run ruff format --check python tests tools
uvx pyright python tests tools                                 # 0 errors expected
FTMETAL_TEST_MODEL_PATH=/path/to/Qwen3.6-35B-A3B-MLX uv run pytest tests/models/test_real_checkpoint.py -q
uv run python tools/aime0_check.py /path/to/Qwen3.6-35B-A3B-MLX   # AIME25 #0 must answer 70
```

The extensions `ftmetal.kernel._C` and `ftmetal.command_graph._C` are built by CMake
against the pinned `mlx` wheel; the Metal kernels under `python/ftmetal/kernel/` (layout in
`docs/dev/kernel-layout.md`) are compiled at first use.

## Serve and chat

```bash
uv run ftm serve --model /path/to/Qwen3.6-35B-A3B-MLX      # OpenAI API on 127.0.0.1:1919
uv run ftm serve --model /path/to/model --port 8000 --max-running-requests 8 --max-seq-len 65536
uv run ftm shell                                            # chat with the running server
uv run ftm shell --server http://127.0.0.1:8000             # or FTMETAL_HOST=http://...
uv run ftm pack --model /path/to/Qwen3.6-35B-A3B-MLX       # expert pack for MoE offload
```

`ftm serve` exposes FreeToken's routes: the OpenAI `POST /v1/chat/completions` (streamed
or not; `reasoning_content` carries the reasoning, the `<think>` block on Qwen3.6 and Flint and
the thought channel on Gemma4, and `tools` come back as `tool_calls`;
`chat_template_kwargs: {"enable_thinking": false}` or `thinking: {"type": "disabled"}`
turns thinking off, and so does `reasoning_effort: "none"`; any other effort turns it on and
reaches a chat template that grades effort as the nearest level it accepts, which `/v1/models`
lists), `POST /v1/completions` and `GET /v1/models`; the Anthropic
`POST /v1/messages` (thinking, text and tool_use blocks) and `/v1/messages/count_tokens`;
the OpenAI Responses `POST /v1/responses` (stateless: `GET /v1/responses/{id}` and its
`/cancel` answer 404); and `POST /generate` (a raw text prompt, no chat template, streamed
as `{"text": ...}` events and sampled greedily). The control plane is FreeToken's too:
`GET /health`, `GET /v1/stats` (throughput, request counters, token totals, pool usage),
`GET /v1/requests?since=<cursor>` (recent requests), `GET /v1/cache/status` and, from
loopback only, `POST /v1/admin/prepare-stop` (close admission, drain, abort the rest, seal
the token totals); `FTMETAL_API_LOG_DIR` logs every request body as JSON lines, and
`--cors-origins` sets the CORS allow-list. Any API key is accepted. The reasoning
and tool-call parsers follow the model family (`--reasoning-parser`, `--tool-call-parser`
override them); `--enable-cache-report` adds the prefix-cache hit to each `usage`. `ftm shell` is a client of the chat endpoint: `/think` toggles
thinking, `/reset` clears the conversation, `/exit` or Ctrl-D quits, Ctrl-C cancels a
reply. Environment knobs are read as `FTMETAL_<NAME>`, falling back to
`FREETOKEN_<NAME>`.

A MoE checkpoint whose experts do not fit the memory budget, in any format its family loads
(bf16, MLX affine or float modes, the Hugging Face layout, an NVFP4 or FP8 export), serves
them from an offload cache that reads them from the checkpoint on demand
(`--moe-strategy auto|fused|offload`, FreeToken's flags): `--moe-cache-size`,
`--moe-cache-rate` or `--moe-cache-auto` size it (auto, the default, gives the KV pools
`--kv-reserve-tokens` first and the experts the rest), `--disable-moe-prefill-overlap`
stops long prefills from reading each layer's experts ahead, and `--memory-ratio` sets the
budget as a fraction of Metal's recommended working set, which also emulates a smaller Mac.
`ftm pack --model DIR` writes `DIR/experts.packed`, each routed expert as one record (an
affine checkpoint's scales and biases as codes into a per-layer table of their pairs), which
the offload cache then reads instead of the checkpoint: one read per expert, and smaller
affine slots, which `--moe-cache-auto` turns into more of them. A pack the checkpoint has
changed under is passed over with a warning; `FTMETAL_MOE_PACK=0` ignores it.

```bash
FTMETAL_TEST_MODEL_PATH=/path/to/Qwen3.6-35B-A3B-MLX uv run pytest tests/server tests/test_tokenizer.py -q
```

## License

Apache-2.0, see [LICENSE](LICENSE). The kernels adapted from MLX and mlx-lm keep their MIT license,
see [LICENSE-MLX](LICENSE-MLX).
