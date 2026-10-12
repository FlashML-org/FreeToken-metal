"""gather_qmm_sorted against mx.gather_qmm(..., transpose=True, sorted_indices=True).

Each case runs on a resident bank (expert e's rows at row e) or on a larger bank that holds
expert e's rows at a random slot and other random rows everywhere else, read through each
expert's byte offsets, also with the bank behind other bytes in one raw buffer. Where MLX itself
takes its rhs-sorted kernel (B >= 16 and B // E >= 4 on a GPU without matrix units) the
two run the same tile math, so the outputs must be equal bit for bit: these cases fail if
an MLX upgrade changes the kernel the vendored copy came from. Everywhere else MLX runs one
matvec per row, and the two agree within the tolerance table. A large-bank case must also
equal the same rows on the resident bank bit for bit. Edge cases: row tiles cut by a run
boundary or by B, a partial column tile, one expert, mostly empty experts (the first and
last among them), fewer than 16 rows, no rows, affine bits 2-6 and 8 with group sizes 32,
64 and 128, each dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel.fprefill import has_matrix_units
from ftmetal.kernel.gather_qmm_sorted import gather_qmm_sorted

# Worst |error| over a row relative to the row's largest |value|. The rhs kernel stages the
# dequantized weights in x's dtype and MLX's matvec does not, so the two land about one ulp
# apart at the row's top binade (2^-7 for bf16, 2^-10 for fp16); the table allows two. fp32
# differs by summation order alone.
TOLERANCE = {mx.bfloat16: 2**-6, mx.float16: 2**-9, mx.float32: 1e-5}

# uniform: rows on random experts; single: every row on one expert; sparse: rows on three
# experts, the first and last expert empty.
RowPattern = Literal["uniform", "single", "sparse"]


@dataclass(frozen=True)
class Case:
    """num_rows rows of input_size against num_experts experts of [output_size] rows, held
    in a bank of bank_size slots (num_experts: resident)."""

    num_experts: int
    num_rows: int
    bank_size: int | None = None
    output_size: int = 128
    input_size: int = 256
    bits: int = 4
    group_size: int = 64
    dtype: mx.Dtype = mx.bfloat16
    pattern: RowPattern = "uniform"

    @property
    def is_bit_exact(self) -> bool:
        """MLX's own rule for taking the rhs-sorted kernel, on its non-NAX path."""
        is_rhs = self.num_rows >= 16 and self.num_rows // self.num_experts >= 4
        return is_rhs and not has_matrix_units()

    def __str__(self) -> str:
        dtype = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}[self.dtype]
        return (
            f"E{self.num_experts}-S{self.bank_size or self.num_experts}-B{self.num_rows}"
            f"-N{self.output_size}-K{self.input_size}-b{self.bits}-g{self.group_size}"
            f"-{dtype}-{self.pattern}"
        )


CASES = [
    # MLX takes the rhs kernel: bit for bit.
    Case(4, 64),
    Case(4, 37, output_size=80, input_size=192),  # a partial row tile and column tile
    Case(16, 64, bits=8, group_size=32, dtype=mx.float16),
    Case(2, 16, group_size=128, dtype=mx.float32),  # MLX's threshold: B 16, B // E 4
    Case(8, 40, pattern="single"),
    Case(64, 300, output_size=48, pattern="sparse"),
    Case(12, 100, bits=2, group_size=32),
    Case(12, 100, bits=3, dtype=mx.float16),
    Case(12, 100, bits=5, group_size=128),
    Case(12, 100, bits=6, dtype=mx.float32),
    Case(256, 1024, output_size=96, input_size=512),  # a Qwen3.6 layer's 256 experts
    # A larger bank at the experts' byte offsets.
    Case(4, 64, bank_size=64),
    Case(12, 100, bank_size=300, output_size=80, bits=8, group_size=32),
    Case(8, 40, bank_size=50, dtype=mx.float16, pattern="single"),
    Case(64, 300, bank_size=1000, output_size=48, pattern="sparse"),
    Case(256, 1024, bank_size=1280, output_size=96, input_size=512),
    # MLX runs one matvec per row: B // E < 4 or B < 16.
    Case(256, 512, bank_size=2048),
    Case(64, 128, output_size=80, bits=8, group_size=32, dtype=mx.float32),
    Case(16, 7, bank_size=40, output_size=48),
    Case(5, 1, dtype=mx.float16),
    Case(4, 0),
]


def make_bank(
    rng: np.random.Generator, case: Case, num_rows: int
) -> tuple[mx.array, mx.array, mx.array]:
    """num_rows rows of random packed words and small random scales and biases."""
    shape = (num_rows, case.output_size)
    groups = case.input_size // case.group_size
    words = case.input_size * case.bits // 32
    w = mx.array(rng.integers(0, 2**32, (*shape, words), dtype=np.uint32))
    scales = mx.array(0.02 * rng.standard_normal((*shape, groups), dtype=np.float32))
    biases = mx.array(0.02 * rng.standard_normal((*shape, groups), dtype=np.float32))
    return w, scales.astype(case.dtype), biases.astype(case.dtype)


def expert_bytes(
    slots: np.ndarray, bank: tuple[mx.array, mx.array, mx.array], pads: tuple[int, ...] = (0, 0, 0)
) -> mx.array:
    """int64 [E, 3]: the byte offset of each slot's weight, scales and biases rows in the bank
    arrays, each behind pad bytes."""
    rows = [array[0].nbytes for array in bank]
    return mx.array(np.stack([pad + slots * row for pad, row in zip(pads, rows)], axis=1))


def behind(array: mx.array, pad: int) -> mx.array:
    """One raw byte buffer: pad zero bytes, then array's."""
    return mx.concatenate([mx.zeros((pad,), dtype=mx.uint8), array.view(mx.uint8).reshape(-1)])


def expert_rows(rng: np.random.Generator, case: Case) -> np.ndarray:
    """Each row's expert, ascending."""
    if case.pattern == "uniform":
        experts = rng.integers(0, case.num_experts, case.num_rows)
    elif case.pattern == "single":
        experts = np.full(case.num_rows, rng.integers(0, case.num_experts))
    else:
        experts = rng.choice(np.arange(1, case.num_experts - 1), 3, replace=False)
        experts = experts[rng.integers(0, 3, case.num_rows)]
    return np.sort(experts)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_mlx_gather_qmm(case: Case) -> None:
    rng = np.random.default_rng(case.num_experts * 1009 + case.num_rows)
    x = mx.array(rng.standard_normal((case.num_rows, 1, case.input_size), dtype=np.float32))
    x = x.astype(case.dtype)
    experts = expert_rows(rng, case)
    offsets = np.searchsorted(experts, np.arange(case.num_experts + 1))
    expert_offsets = mx.array(offsets.astype(np.int32))
    resident = make_bank(rng, case, case.num_experts)
    expected = mx.gather_qmm(
        x,
        *resident,
        rhs_indices=mx.array(experts.astype(np.uint32)),
        transpose=True,
        group_size=case.group_size,
        bits=case.bits,
        sorted_indices=True,
    )
    options = {"n": case.output_size, "group_size": case.group_size, "bits": case.bits}
    identity = np.arange(case.num_experts, dtype=np.int64)
    out = gather_qmm_sorted(
        x, *resident, expert_offsets, expert_bytes(identity, resident), **options
    )
    if case.bank_size is not None:
        slots = rng.permutation(case.bank_size)[: case.num_experts].astype(np.int64)
        bank = make_bank(rng, case, case.bank_size)
        for array, rows in zip(bank, resident):
            array[mx.array(slots)] = rows
        remapped = gather_qmm_sorted(x, *bank, expert_offsets, expert_bytes(slots, bank), **options)
        assert mx.array_equal(remapped, out)
        pads = (4096, 768, 256)
        raw = [behind(array, pad) for array, pad in zip(bank, pads)]
        offsets = expert_bytes(slots, bank, pads)
        assert mx.array_equal(gather_qmm_sorted(x, *raw, expert_offsets, offsets, **options), out)
    assert out.shape == (case.num_rows, 1, case.output_size) and out.dtype == case.dtype
    if case.is_bit_exact:
        assert mx.array_equal(out, expected)
    else:
        reference = np.array(expected.astype(mx.float32))
        error = np.abs(np.array(out.astype(mx.float32)) - reference).max(axis=2, initial=0.0)
        row_scale = np.abs(reference).max(axis=2, initial=0.0)
        assert (error <= TOLERANCE[case.dtype] * row_scale).all()


def test_rejects_unsupported_arguments() -> None:
    rng = np.random.default_rng(0)
    case = Case(4, 64)
    w, scales, biases = make_bank(rng, case, case.num_experts)
    x = mx.zeros((case.num_rows, 1, case.input_size), dtype=case.dtype)
    offsets = mx.zeros((case.num_experts + 1,), dtype=mx.int32)
    expert_offsets = expert_bytes(np.arange(case.num_experts), (w, scales, biases))

    def call(
        x: mx.array = x,
        offsets: mx.array = offsets,
        byte_offsets: mx.array = expert_offsets,
        n: int = case.output_size,
        group_size: int = 64,
        bits: int = 4,
    ) -> mx.array:
        return gather_qmm_sorted(
            x, w, scales, biases, offsets, byte_offsets, n=n, group_size=group_size, bits=bits
        )

    rejected = [
        lambda: call(group_size=48),
        lambda: call(bits=7),
        lambda: call(x=x[:, 0]),
        lambda: call(n=0),
        lambda: call(offsets=offsets.astype(mx.uint32)),
        lambda: call(offsets=offsets[:-1]),
        lambda: call(byte_offsets=expert_offsets.astype(mx.int32)),
        lambda: call(byte_offsets=expert_offsets[:, :2]),
    ]
    for rejected_call in rejected:
        with pytest.raises(ValueError):
            rejected_call()
