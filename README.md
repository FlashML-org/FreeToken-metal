# ftmetal

LLM serving engine for one Apple Silicon Mac: Python + MLX + Metal.

## Setup

```bash
uv sync                                  # creates .venv, builds the C++ extension
uv sync --reinstall-package ftmetal      # after editing python/ftmetal/kernel/csrc
uv run pytest tests/ -q
uv run ruff check python tests && uv run ruff format --check python tests
FTMETAL_TEST_MODEL_PATH=/path/to/Qwen3.6-35B-A3B-MLX uv run pytest tests/models/test_real_qwen3_6.py -q
uv run python tools/aime0_check.py /path/to/Qwen3.6-35B-A3B-MLX   # AIME25 #0 must answer 70
```

The extension `ftmetal.kernel._C` is built by CMake against the pinned `mlx` wheel;
Metal kernels under `python/ftmetal/kernel/metal/` are compiled at first use.
