"""MLX behaviours the engine relies on; rerun after any mlx upgrade (mlx is pinned)."""

import mlx.core as mx

from ftmetal.kernel._ext import REQUIRED_MLX_VERSION


def test_mlx_version_is_pinned() -> None:
    assert mx.__version__ == REQUIRED_MLX_VERSION


def test_compile_bakes_implicitly_captured_arrays_as_constants() -> None:
    """Why compiled leaves take every array as an argument."""
    captured = mx.array(1.0)

    @mx.compile
    def add_captured(x: mx.array) -> mx.array:
        return x + captured

    x = mx.array(2.0)
    assert add_captured(x).item() == 3.0
    captured = mx.array(10.0)  # noqa: F841  (rebinding does not reach the compiled graph)
    assert add_captured(x).item() == 3.0


def test_python_scatter_copies_the_pool_under_pipelining() -> None:
    """Why pool writes go through the C++ primitives (see kernel/test_inplace.py)."""
    rows, cols = 4096, 8192
    pool_bytes = rows * cols * 2
    pool = mx.zeros((rows, cols), dtype=mx.float16)
    mx.eval(pool)
    copies = 0
    previous = None
    for i in range(8):
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        pool[i : i + 1] = mx.full((1, cols), i, dtype=mx.float16)
        total = pool.sum()
        mx.async_eval(total, pool)
        if previous is not None:
            previous.item()
        previous = total
        if mx.get_peak_memory() - base > pool_bytes // 2:
            copies += 1
    mx.synchronize()
    assert copies > 0


def test_async_eval_keeps_sampled_tokens_lazy_until_read() -> None:
    """The overlap loop: the next step is built from a lazy token and read one step late."""
    logits = mx.random.normal((1, 16))
    token = mx.argmax(logits, axis=-1)
    mx.async_eval(token)
    next_input = token + 1
    mx.async_eval(next_input)
    assert next_input.item() == token.item() + 1


def test_lazy_indexing_with_a_device_array_stays_on_device() -> None:
    """Gathering next-step input ids from lazy sampled tokens issues no host read."""
    token_pool = mx.arange(32, dtype=mx.int32).reshape(4, 8)
    sampled = mx.array([3, 1], dtype=mx.int32)
    gathered = token_pool[mx.array([0, 2], dtype=mx.int32), sampled]
    mx.async_eval(gathered)
    assert gathered.tolist() == [3, 17]
