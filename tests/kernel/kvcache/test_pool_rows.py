"""In-place row primitives: parity with the pure-MLX references, aliasing, no pool copies.

A row whose element count is a multiple of four moves in 4-vectors, any other one by
element; each pool dtype (bf16 K/V and conv, fp32 GDN state, int32 tokens) takes both.
"""

import mlx.core as mx
import pytest

from ftmetal.kernel.inplace import copy_rows, kv_write, scatter_rows, zero_rows
from ftmetal.kernel.ref import copy_rows_ref, scatter_rows_ref, zero_rows_ref

DTYPES = [mx.bfloat16, mx.float32, mx.int32]


def ids(*values: int) -> mx.array:
    return mx.array(list(values), dtype=mx.int32)


def random_rows(shape: tuple[int, ...], dtype: mx.Dtype) -> mx.array:
    if dtype == mx.int32:
        return mx.random.randint(-1000, 1000, shape).astype(dtype)
    return mx.random.normal(shape).astype(dtype)


@pytest.mark.parametrize(
    ("dtype", "row_shape"),
    [
        (mx.bfloat16, (3, 4)),
        (mx.bfloat16, (5,)),
        (mx.float32, (3, 4)),
        (mx.float32, (2, 7)),
        (mx.int32, (3, 4)),
        (mx.int32, (5,)),
    ],
    ids=["bf16-vec", "bf16", "fp32-vec", "fp32", "int32-vec", "int32"],
)
def test_scatter_rows_matches_reference(dtype: mx.Dtype, row_shape: tuple[int, ...]) -> None:
    pool = random_rows((8, *row_shape), dtype)
    rows = random_rows((3, *row_shape), dtype)
    dst = ids(6, 0, 3)
    expected = scatter_rows_ref(pool, rows, dst)
    pool = scatter_rows(pool, rows, dst)
    assert mx.array_equal(pool, expected)


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32", "int32"])
def test_copy_and_zero_rows_match_reference(dtype: mx.Dtype) -> None:
    pool = random_rows((8, 3, 4), dtype)
    expected = copy_rows_ref(pool, ids(1, 2), ids(7, 0))
    pool = copy_rows(pool, ids(1, 2), ids(7, 0))
    assert mx.array_equal(pool, expected)
    expected = zero_rows_ref(pool, ids(7, 4))
    pool = zero_rows(pool, ids(7, 4))
    assert mx.array_equal(pool, expected)


def test_kv_write_matches_reference() -> None:
    k_cache = random_rows((16, 2, 8), mx.bfloat16)
    v_cache = random_rows((16, 2, 8), mx.bfloat16)
    k = random_rows((3, 2, 8), mx.bfloat16)
    v = random_rows((3, 2, 8), mx.bfloat16)
    out_loc = ids(15, 0, 9)
    expected_k = scatter_rows_ref(k_cache, k, out_loc)
    expected_v = scatter_rows_ref(v_cache, v, out_loc)
    k_cache, v_cache = kv_write(k_cache, v_cache, k, v, out_loc)
    assert mx.array_equal(k_cache, expected_k)
    assert mx.array_equal(v_cache, expected_v)


def test_non_contiguous_rows_are_written_correctly() -> None:
    pool = mx.zeros((4, 2, 8), dtype=mx.float32)
    qkv = random_rows((2, 2, 24), mx.float32)
    k = qkv[:, :, 8:16]
    pool = scatter_rows(pool, k, ids(3, 1))
    assert mx.array_equal(pool[3], qkv[0, :, 8:16])
    assert mx.array_equal(pool[1], qkv[1, :, 8:16])


def test_empty_write_is_a_no_op() -> None:
    pool = random_rows((4, 6), mx.float16)
    before = mx.array(pool)
    pool = scatter_rows(pool, mx.zeros((0, 6), dtype=mx.float16), mx.zeros((0,), dtype=mx.int32))
    assert mx.array_equal(pool, before)


def test_rejects_bad_ids() -> None:
    pool = mx.zeros((4, 6), dtype=mx.float16)
    with pytest.raises(ValueError):
        scatter_rows(pool, mx.zeros((1, 6), dtype=mx.float16), mx.array([0], dtype=mx.int64))
    with pytest.raises(ValueError):
        scatter_rows(pool, mx.zeros((1, 5), dtype=mx.float16), ids(0))


def test_pipelined_writes_never_copy_the_pool_and_readers_see_every_write() -> None:
    """The reason the primitives exist: a Python scatter copies the pool under async_eval."""
    rows, cols, steps = 4096, 8192, 30
    pool_bytes = rows * cols * 2
    pool = mx.zeros((rows, cols), dtype=mx.float16)
    mx.eval(pool)
    copies = 0
    previous = None
    for i in range(steps):
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        pool = scatter_rows(pool, mx.full((1, cols), i, dtype=mx.float16), ids(i))
        total = pool.sum()
        mx.async_eval(total, pool)
        if previous is not None:
            previous.item()
        previous = total
        if mx.get_peak_memory() - base > pool_bytes // 2:
            copies += 1
    mx.synchronize()
    assert copies == 0
    expected_row_sums = mx.arange(steps, dtype=mx.float32) * cols
    assert mx.array_equal(pool[:steps].astype(mx.float32).sum(axis=1), expected_row_sums)
