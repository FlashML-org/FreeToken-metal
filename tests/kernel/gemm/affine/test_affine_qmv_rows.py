"""qmv_rows against the one-token mx.quantized_matmul it replaces, bit for bit.

qmv_rows runs MLX's qmv_fast code with one output row per simdgroup, so each output must equal
mx.quantized_matmul's exactly: 4 and 8 bits, group sizes 32 and 64, bf16, fp16 and fp32, a row
count that leaves the last threadgroup half empty. Qwen3.8-Flash-Next's shapes: the fused
hyper-connection down + inject projection (320 + 4 rows, padded) and the final mixer's down
projection over 4 * 2560 inputs, projections of the 2560 hidden size. The hyper-connection
epilogue must equal silu(y / hc_count) in fp32 from the stored y on the down rows, cast once,
and leave the other rows raw.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from ftmetal.kernel.gemm.affine.affine_qmv_rows import qmv_rows, supports_qmv_rows

BF16 = mx.bfloat16
F32 = mx.float32


def check_equal(got: mx.array, expected: mx.array) -> None:
    assert got.shape == expected.shape and got.dtype == expected.dtype
    mismatches = (got != expected).sum().item()
    assert mismatches == 0, f"{mismatches} of {got.size} values differ"


def quantized(
    rows: int, cols: int, bits: int, group_size: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array, mx.array]:
    weight = (0.05 * mx.random.normal((rows, cols))).astype(dtype)
    quantized_weight, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    return quantized_weight, scales, biases


# (rows, cols, bits, group_size, dtype)
CASES = [
    (336, 10240, 4, 32, BF16),  # down + inject, padded to 16 rows
    (320, 10240, 4, 32, F32),  # the final mixer's down
    (12288, 2560, 4, 32, BF16),
    (520, 2560, 8, 64, BF16),
    (72, 1024, 4, 32, F32),  # 36 threadgroups, the last one's second simdgroup idle
    (40, 512, 4, 64, mx.float16),
    (24, 256, 8, 64, F32),  # 8 bits step 256 inputs at a time
    (264, 768, 8, 64, BF16),
]


@pytest.mark.parametrize(("rows", "cols", "bits", "group_size", "dtype"), CASES)
def test_qmv_rows(rows: int, cols: int, bits: int, group_size: int, dtype: mx.Dtype) -> None:
    mx.random.seed(rows + cols)
    weight, scales, biases = quantized(rows, cols, bits, group_size, dtype)
    x = mx.random.normal((1, cols)).astype(dtype)
    assert supports_qmv_rows(x, weight, scales, group_size, bits)
    expected = mx.quantized_matmul(
        x, weight, scales, biases, transpose=True, group_size=group_size, bits=bits
    )
    check_equal(qmv_rows(x, weight, scales, biases, group_size=group_size, bits=bits), expected)


def test_qmv_rows_refuses_other_shapes() -> None:
    """Shapes mx.quantized_matmul runs on another kernel: an output count off its 8-row tile,
    more tokens, an input off qmv_fast's step, other bit widths; scales off x's dtype."""
    x = mx.zeros((1, 1024), BF16)
    weight, scales, _ = quantized(36, 1024, 4, 32, BF16)
    assert not supports_qmv_rows(x, weight, scales, 32, 4)
    weight, scales, _ = quantized(40, 1024, 4, 32, BF16)
    assert supports_qmv_rows(x, weight, scales, 32, 4)
    assert not supports_qmv_rows(mx.zeros((2, 1024), BF16), weight, scales, 32, 4)
    assert not supports_qmv_rows(mx.zeros((1, 1024), F32), weight, scales, 32, 4)
    weight, scales, _ = quantized(40, 768, 4, 32, BF16)
    assert not supports_qmv_rows(mx.zeros((1, 768), BF16), weight, scales, 32, 4)
    weight, scales, _ = quantized(40, 320, 8, 64, BF16)
    assert not supports_qmv_rows(mx.zeros((1, 320), BF16), weight, scales, 64, 8)
    weight, scales, _ = quantized(40, 1024, 6, 64, BF16)
    assert not supports_qmv_rows(x, weight, scales, 64, 6)


# (lowrank, hc_count, cols, rows, dtype, input scale)
EPILOGUE_CASES = [
    (320, 4, 10240, 336, BF16, 1.0),
    (320, 4, 10240, 336, BF16, 20.0),
    (320, 4, 10240, 328, F32, 0.05),
    (64, 4, 1024, 72, BF16, 1.0),
    (64, 2, 1024, 72, F32, 20.0),
]


@pytest.mark.parametrize(("lowrank", "hc_count", "cols", "rows", "dtype", "scale"), EPILOGUE_CASES)
def test_qmv_rows_hc_epilogue(
    lowrank: int, hc_count: int, cols: int, rows: int, dtype: mx.Dtype, scale: float
) -> None:
    mx.random.seed(lowrank + rows)
    weight, scales, biases = quantized(rows, cols, 4, 32, dtype)
    x = (scale * mx.random.normal((1, cols))).astype(dtype)
    raw = mx.quantized_matmul(x, weight, scales, biases, transpose=True, group_size=32, bits=4)
    got = qmv_rows(
        x, weight, scales, biases, group_size=32, bits=4, hc_epilogue=(lowrank, hc_count)
    )
    scaled = raw[:, :lowrank].astype(F32) / hc_count
    check_equal(got[:, :lowrank], (scaled * mx.sigmoid(scaled)).astype(dtype))
    check_equal(got[:, lowrank:], raw[:, lowrank:])
