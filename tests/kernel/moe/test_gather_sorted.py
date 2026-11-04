"""gather_qmm_sorted and gather_mm_sorted against mx.gather_qmm(..., transpose=True,
sorted_indices=True) and mx.gather_mm(..., sorted_indices=True).

Each case runs on a resident bank (expert e's rows at row e) or on a larger bank that holds
expert e's rows at a random slot and other random rows everywhere else, read through each
expert's byte offsets, also with the bank behind other bytes in one raw buffer. Where MLX itself
takes its rhs-sorted kernel on a GPU without matrix units (gather_qmm: B >= 16 and B // E >= 4;
gather_mm: any B) the two run the same tile math, so the outputs must be equal bit for bit:
these cases fail if an MLX upgrade changes a kernel the vendored copies came from. Everywhere
else MLX runs one quantized matvec per row, and the two agree within the tolerance table. A
large-bank case must also equal the same rows on the resident bank bit for bit. Edge cases, in
each weight format: row tiles cut by a run boundary or by B, a partial column tile, one expert,
mostly empty experts (the first and last among them), fewer than 16 rows, no rows; affine bits
2-6 and 8 with group sizes 32, 64 and 128, mxfp4, nvfp4, mxfp8 and unquantized weights (also
with a partial K tile), each dtype. gather_qmm_sorted_pair_codes must equal gather_qmm_sorted on
the scales and biases its codes stand for bit for bit, on every affine case with 13-bit codes and
three with 16, 9 and 1, at the experts' byte offsets into a bank and behind other bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel.attention.mha.mha_prefill import has_matrix_units
from ftmetal.kernel.moe.gather_sorted import (
    QuantMode,
    gather_mm_sorted,
    gather_qmm_sorted,
    gather_qmm_sorted_pair_codes,
)
from ftmetal.kernel.ref import pair_code_bank_ref

# Worst |error| over a row relative to the row's largest |value|, where MLX runs its per-row
# matvec instead. The affine rhs kernel stages the dequantized weights in x's dtype and the
# matvec does not, so the two land about one ulp apart at the row's top binade (2^-7 for bf16,
# 2^-10 for fp16); the table allows two. A float block mode's dequantized weights are exact in
# x's dtype, and fp32 differs by summation order alone.
TOLERANCE = {mx.bfloat16: 2**-6, mx.float16: 2**-9, mx.float32: 1e-5}

# MLX's one (bits, group size) for each float block mode.
FP_BLOCK_SHAPES: dict[QuantMode, tuple[int, int]] = {
    "mxfp4": (4, 32),
    "nvfp4": (4, 16),
    "mxfp8": (8, 32),
}

# The pair tables of the rejected-argument cases: 13-bit codes.
MAX_PAIRS = 8192

# uniform: rows on random experts; single: every row on one expert; sparse: rows on three
# experts, the first and last expert empty.
RowPattern = Literal["uniform", "single", "sparse"]


@dataclass(frozen=True)
class Case:
    """num_rows rows of input_size against num_experts experts of [output_size] rows in mode
    (None: unquantized, in dtype), held in a bank of bank_size slots (num_experts: resident).
    bits and group_size are affine's; a float block mode has its own."""

    num_experts: int
    num_rows: int
    bank_size: int | None = None
    output_size: int = 128
    input_size: int = 256
    mode: QuantMode | None = "affine"
    bits: int = 4
    group_size: int = 64
    dtype: mx.Dtype = mx.bfloat16
    pattern: RowPattern = "uniform"

    @property
    def quantization(self) -> tuple[int, int]:
        """(bits, group size)."""
        assert self.mode is not None, "unquantized"
        if self.mode == "affine":
            return self.bits, self.group_size
        else:
            return FP_BLOCK_SHAPES[self.mode]

    @property
    def is_bit_exact(self) -> bool:
        """MLX's own rule for taking the rhs-sorted kernel, on its non-NAX path
        (GatherQMM::eval_gpu and GatherMM::eval_gpu)."""
        if self.mode is None:
            is_rhs = True
        else:
            is_rhs = self.num_rows >= 16 and self.num_rows // self.num_experts >= 4
        return is_rhs and not has_matrix_units()

    def __str__(self) -> str:
        dtype = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}[self.dtype]
        if self.mode == "affine":
            weights = f"affine-b{self.bits}-g{self.group_size}"
        else:
            weights = self.mode or "unquantized"
        return (
            f"E{self.num_experts}-S{self.bank_size or self.num_experts}-B{self.num_rows}"
            f"-N{self.output_size}-K{self.input_size}-{weights}-{dtype}-{self.pattern}"
        )


CASES = [
    # Affine; MLX takes the rhs kernel: bit for bit.
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
    # Float block modes; MLX takes the rhs kernel: bit for bit.
    Case(4, 64, mode="mxfp4"),
    Case(4, 37, output_size=80, input_size=192, mode="mxfp4", dtype=mx.float16),
    Case(2, 16, mode="mxfp4", dtype=mx.float32),
    Case(4, 64, mode="nvfp4"),
    Case(4, 37, output_size=80, input_size=192, mode="nvfp4", dtype=mx.float32),
    Case(8, 40, mode="nvfp4", dtype=mx.float16, pattern="single"),
    Case(256, 1024, output_size=96, input_size=512, mode="nvfp4"),
    Case(4, 64, mode="mxfp8"),
    Case(4, 37, output_size=80, input_size=192, mode="mxfp8", dtype=mx.float32),
    Case(64, 300, output_size=48, mode="mxfp8", dtype=mx.float16, pattern="sparse"),
    # A larger bank at the experts' byte offsets.
    Case(12, 100, bank_size=300, output_size=80, mode="mxfp4"),
    Case(8, 40, bank_size=50, mode="nvfp4", dtype=mx.float16, pattern="single"),
    Case(256, 1024, bank_size=1280, output_size=96, input_size=512, mode="nvfp4"),
    Case(64, 300, bank_size=1000, output_size=48, mode="mxfp8", pattern="sparse"),
    # MLX runs one matvec per row: B // E < 4 or B < 16.
    Case(256, 512, bank_size=1024, mode="nvfp4"),
    Case(64, 128, output_size=80, mode="mxfp8", dtype=mx.float32),
    Case(16, 7, bank_size=40, output_size=48, mode="mxfp4", dtype=mx.float16),
    Case(5, 1, mode="nvfp4", dtype=mx.float16),
    Case(4, 0, mode="mxfp8"),
    # Unquantized; MLX takes the rhs kernel at any B: bit for bit.
    Case(4, 64, mode=None),
    Case(4, 37, output_size=80, input_size=200, mode=None, dtype=mx.float16),  # and a K tile
    Case(2, 16, mode=None, dtype=mx.float32),
    Case(8, 40, mode=None, dtype=mx.float32, pattern="single"),
    Case(64, 300, output_size=48, mode=None, dtype=mx.float16, pattern="sparse"),
    Case(256, 1024, output_size=96, input_size=512, mode=None),
    Case(12, 100, bank_size=300, output_size=80, mode=None),
    Case(8, 40, bank_size=50, mode=None, dtype=mx.float16, pattern="single"),
    Case(64, 300, bank_size=1000, input_size=200, mode=None, dtype=mx.float32, pattern="sparse"),
    Case(256, 512, bank_size=1024, mode=None),
    Case(16, 7, bank_size=40, output_size=48, mode=None, dtype=mx.float16),
    Case(5, 1, mode=None, dtype=mx.float32),
    Case(4, 0, mode=None),
]


def make_bank(rng: np.random.Generator, case: Case, num_rows: int) -> tuple[mx.array, ...]:
    """num_rows experts' weights in the case's format, the weight then the scales and biases
    it has: affine's random packed words with small random scales and biases, a float block
    mode's small random weights quantized by MLX (random codes include NaN), or unquantized
    small random weights."""
    shape = (num_rows, case.output_size)
    if case.mode == "affine":
        groups = case.input_size // case.group_size
        words = case.input_size * case.bits // 32
        w = mx.array(rng.integers(0, 2**32, (*shape, words), dtype=np.uint32))
        scales = mx.array(0.02 * rng.standard_normal((*shape, groups), dtype=np.float32))
        biases = mx.array(0.02 * rng.standard_normal((*shape, groups), dtype=np.float32))
        return w, scales.astype(case.dtype), biases.astype(case.dtype)
    elif case.mode is None:
        w = 0.02 * rng.standard_normal((*shape, case.input_size), dtype=np.float32)
        return (mx.array(w).astype(case.dtype),)
    else:
        w = 0.02 * rng.standard_normal((*shape, case.input_size), dtype=np.float32)
        bits, group_size = case.quantization
        return tuple(mx.quantize(mx.array(w).astype(case.dtype), group_size, bits, mode=case.mode))


def expert_bytes(
    slots: np.ndarray, bank: tuple[mx.array, ...], pads: tuple[int, ...] = (0, 0, 0)
) -> mx.array:
    """int64 [E, len(bank)]: the byte offset of each slot's rows in each bank array, each behind
    pad bytes."""
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


def gather_sorted(
    case: Case,
    x: mx.array,
    bank: tuple[mx.array, ...],
    expert_offsets: mx.array,
    expert_bytes: mx.array,
) -> mx.array:
    """The case's kernel over the bank arrays at expert_bytes."""
    if case.mode is None:
        return gather_mm_sorted(x, bank[0], expert_offsets, expert_bytes, n=case.output_size)
    else:
        bits, group_size = case.quantization
        return gather_qmm_sorted(
            x,
            bank[0],
            bank[1],
            bank[2] if case.mode == "affine" else None,
            expert_offsets,
            expert_bytes,
            n=case.output_size,
            group_size=group_size,
            bits=bits,
            mode=case.mode,
        )


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_mlx(case: Case) -> None:
    rng = np.random.default_rng(case.num_experts * 1009 + case.num_rows)
    x = mx.array(rng.standard_normal((case.num_rows, 1, case.input_size), dtype=np.float32))
    x = x.astype(case.dtype)
    experts = expert_rows(rng, case)
    offsets = np.searchsorted(experts, np.arange(case.num_experts + 1))
    expert_offsets = mx.array(offsets.astype(np.int32))
    resident = make_bank(rng, case, case.num_experts)
    rhs_indices = mx.array(experts.astype(np.uint32))
    if case.mode is None:
        expected = mx.gather_mm(
            x, resident[0].swapaxes(-1, -2), rhs_indices=rhs_indices, sorted_indices=True
        )
    else:
        bits, group_size = case.quantization
        expected = mx.gather_qmm(
            x,
            *resident,
            rhs_indices=rhs_indices,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=case.mode,
            sorted_indices=True,
        )
    identity = np.arange(case.num_experts, dtype=np.int64)
    out = gather_sorted(case, x, resident, expert_offsets, expert_bytes(identity, resident))
    if case.bank_size is not None:
        slots = rng.permutation(case.bank_size)[: case.num_experts].astype(np.int64)
        bank = make_bank(rng, case, case.bank_size)
        for array, rows in zip(bank, resident):
            array[mx.array(slots)] = rows
        remapped = gather_sorted(case, x, bank, expert_offsets, expert_bytes(slots, bank))
        assert mx.array_equal(remapped, out)
        pads = (4096, 768, 256)
        raw = tuple(behind(array, pad) for array, pad in zip(bank, pads))
        offsets = expert_bytes(slots, bank, pads)
        assert mx.array_equal(gather_sorted(case, x, raw, expert_offsets, offsets), out)
    assert out.shape == (case.num_rows, 1, case.output_size) and out.dtype == case.dtype
    if case.is_bit_exact:
        assert mx.array_equal(out, expected)
    else:
        reference = np.array(expected.astype(mx.float32))
        error = np.abs(np.array(out.astype(mx.float32)) - reference).max(axis=2, initial=0.0)
        row_scale = np.abs(reference).max(axis=2, initial=0.0)
        assert (error <= TOLERANCE[case.dtype] * row_scale).all()


AFFINE_CASES = [case for case in CASES if case.mode == "affine"]
# Every affine case with 13-bit codes, and three with 16, 9 and 1 bits.
PAIR_CODE_CASES = [
    *((case, 13) for case in AFFINE_CASES),
    (AFFINE_CASES[0], 16),
    (AFFINE_CASES[len(AFFINE_CASES) // 2], 9),
    (AFFINE_CASES[-1], 1),
]


@pytest.mark.parametrize(
    ("case", "code_bits"),
    PAIR_CODE_CASES,
    ids=[f"{case}-c{bits}" for case, bits in PAIR_CODE_CASES],
)
def test_pair_codes_matches_affine(case: Case, code_bits: int) -> None:
    rng = np.random.default_rng(case.num_experts * 1013 + case.num_rows)
    x = mx.array(rng.standard_normal((case.num_rows, 1, case.input_size), dtype=np.float32))
    x = x.astype(case.dtype)
    experts = expert_rows(rng, case)
    offsets = np.searchsorted(experts, np.arange(case.num_experts + 1))
    expert_offsets = mx.array(offsets.astype(np.int32))
    bank_size = case.bank_size or case.num_experts
    slots = rng.permutation(bank_size)[: case.num_experts].astype(np.int64)
    words = case.input_size * case.bits // 32
    w = mx.array(rng.integers(0, 2**32, (bank_size, case.output_size, words), dtype=np.uint32))
    num_pairs = 1 << code_bits
    pairs = mx.array(0.02 * rng.standard_normal((num_pairs, 2), dtype=np.float32))
    pairs = pairs.astype(case.dtype)
    num_codes = case.output_size * case.input_size // case.group_size
    codes = rng.integers(0, num_pairs, (bank_size, num_codes)).astype(np.uint16)
    rows, scales, biases = pair_code_bank_ref(codes, pairs, case.output_size, code_bits)
    expected = gather_qmm_sorted(
        x,
        w,
        scales,
        biases,
        expert_offsets,
        expert_bytes(slots, (w, scales, biases)),
        n=case.output_size,
        group_size=case.group_size,
        bits=case.bits,
    )

    def pair_codes(
        weight_buffer: mx.array, code_buffer: mx.array, byte_offsets: mx.array
    ) -> mx.array:
        return gather_qmm_sorted_pair_codes(
            x,
            weight_buffer,
            code_buffer,
            pairs,
            expert_offsets,
            byte_offsets,
            n=case.output_size,
            group_size=case.group_size,
            bits=case.bits,
            code_bits=code_bits,
        )

    out = pair_codes(w, rows, expert_bytes(slots, (w, rows)))
    assert out.shape == expected.shape and out.dtype == case.dtype
    assert mx.array_equal(out, expected)
    pads = (4096, 768)
    raw_w, raw_rows = behind(w, pads[0]), behind(rows, pads[1])
    assert mx.array_equal(
        pair_codes(raw_w, raw_rows, expert_bytes(slots, (w, rows), pads)), expected
    )


def test_rejects_unsupported_arguments() -> None:
    rng = np.random.default_rng(0)
    case = Case(4, 64)
    w, scales, biases = make_bank(rng, case, case.num_experts)
    x = mx.zeros((case.num_rows, 1, case.input_size), dtype=case.dtype)
    offsets = mx.zeros((case.num_experts + 1,), dtype=mx.int32)
    affine_bytes = expert_bytes(np.arange(case.num_experts), (w, scales, biases))
    fp_bytes = affine_bytes[:, :2]

    def call(
        x: mx.array = x,
        biases: mx.array | None = biases,
        offsets: mx.array = offsets,
        byte_offsets: mx.array = affine_bytes,
        n: int = case.output_size,
        group_size: int = 64,
        bits: int = 4,
        mode: QuantMode = "affine",
    ) -> mx.array:
        return gather_qmm_sorted(
            x,
            w,
            scales,
            biases,
            offsets,
            byte_offsets,
            n=n,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    pairs = mx.zeros((MAX_PAIRS, 2), dtype=case.dtype)

    def call_pair_codes(
        x: mx.array = x,
        pairs: mx.array = pairs,
        byte_offsets: mx.array = fp_bytes,
        group_size: int = 64,
        bits: int = 4,
        code_bits: int = 13,
    ) -> mx.array:
        return gather_qmm_sorted_pair_codes(
            x,
            w,
            scales.view(mx.uint8),
            pairs,
            offsets,
            byte_offsets,
            n=case.output_size,
            group_size=group_size,
            bits=bits,
            code_bits=code_bits,
        )

    def call_unquantized(
        x: mx.array = x,
        offsets: mx.array = offsets,
        byte_offsets: mx.array = affine_bytes[:, :1],
        n: int = case.output_size,
    ) -> mx.array:
        return gather_mm_sorted(x, w, offsets, byte_offsets, n=n)

    rejected = [
        lambda: call(group_size=48),
        lambda: call(bits=7),
        lambda: call(biases=None),
        lambda: call(x=x[:, 0]),
        lambda: call(n=0),
        lambda: call(offsets=offsets.astype(mx.uint32)),
        lambda: call(offsets=offsets[:-1]),
        lambda: call(byte_offsets=affine_bytes.astype(mx.int32)),
        lambda: call(byte_offsets=fp_bytes),
        lambda: call(biases=None, byte_offsets=fp_bytes, mode="nvfp4", group_size=32),
        lambda: call(biases=None, byte_offsets=fp_bytes, mode="mxfp8", group_size=32),
        lambda: call(byte_offsets=fp_bytes, mode="mxfp4", group_size=32),
        lambda: call(biases=None, mode="mxfp4", group_size=32),
        lambda: call(
            x=x[..., :200], biases=None, byte_offsets=fp_bytes, mode="nvfp4", group_size=16
        ),
        # Whole nvfp4 groups, but a partial float K tile.
        lambda: call(
            x=x[..., :208], biases=None, byte_offsets=fp_bytes, mode="nvfp4", group_size=16
        ),
        lambda: call_pair_codes(group_size=48),
        lambda: call_pair_codes(bits=7),
        lambda: call_pair_codes(x=x[:, 0]),
        lambda: call_pair_codes(x=x[..., :200]),
        lambda: call_pair_codes(pairs=pairs.astype(mx.float32)),
        lambda: call_pair_codes(pairs=pairs.reshape(-1)),
        lambda: call_pair_codes(pairs=mx.zeros((MAX_PAIRS, 3), dtype=case.dtype)),
        lambda: call_pair_codes(byte_offsets=affine_bytes),
        lambda: call_pair_codes(code_bits=0),
        lambda: call_pair_codes(code_bits=17),
        lambda: call_unquantized(x=x.astype(mx.int32)),
        lambda: call_unquantized(x=x[:, 0]),
        lambda: call_unquantized(n=0),
        lambda: call_unquantized(offsets=offsets[:-1]),
        lambda: call_unquantized(byte_offsets=fp_bytes),
    ]
    for rejected_call in rejected:
        with pytest.raises(ValueError):
            rejected_call()
