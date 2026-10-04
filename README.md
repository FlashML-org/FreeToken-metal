# ftmetal

LLM serving engine for one Apple Silicon Mac: Python + MLX + Metal.

## Setup

```bash
uv sync                                  # creates .venv, builds the C++ extensions
uv sync --reinstall-package ftmetal      # after editing python/ftmetal/*/csrc
uv run pytest tests/ -q
uv run ruff check python tests tools && uv run ruff format --check python tests tools
uvx pyright python tests tools                                 # 0 errors expected
FTMETAL_TEST_MODEL_PATH=/path/to/Qwen3.6-35B-A3B-MLX uv run pytest tests/models/test_real_qwen3_6.py -q
uv run python tools/aime0_check.py /path/to/Qwen3.6-35B-A3B-MLX   # AIME25 #0 must answer 70
```

The extensions `ftmetal.kernel._C` and `ftmetal.command_graph._C` are built by CMake
against the pinned `mlx` wheel; Metal kernels under `python/ftmetal/kernel/metal/` are
compiled at first use.

## Serve and chat

```bash
uv run ftm serve --model /path/to/Qwen3.6-35B-A3B-MLX      # OpenAI API on 127.0.0.1:1919
uv run ftm serve --model /path/to/model --port 8000 --max-running-requests 8 --max-seq-len 65536
uv run ftm shell                                            # chat with the running server
uv run ftm shell --server http://127.0.0.1:8000             # or FTMETAL_HOST=http://...
```

`ftm serve` exposes `POST /v1/chat/completions` (streamed or not; `reasoning_content`
carries the `<think>` block, `chat_template_kwargs: {"enable_thinking": false}` turns it
off), `POST /v1/completions` and `POST /generate` (a raw text prompt, no chat template;
`/generate` streams `{"text": ...}` events and samples greedily) and `GET /v1/models`;
any API key is accepted. `ftm shell` is a client of the chat endpoint: `/think` toggles
thinking, `/reset` clears the conversation, `/exit` or Ctrl-D quits, Ctrl-C cancels a
reply. Environment knobs are read as `FTMETAL_<NAME>`, falling back to
`FREETOKEN_<NAME>`.

```bash
FTMETAL_TEST_MODEL_PATH=/path/to/Qwen3.6-35B-A3B-MLX uv run pytest tests/server tests/test_tokenizer.py -q
```
