"""affine_gather_qmv_pair_codes against mx.gather_qmm on the expanded scales and biases, exactly.

The kernel is MLX's gather_qmv code with each group's scale and bias looked up by its code, so
every output must equal mx.gather_qmm's (unsorted, one row per pair) exactly, on the impl MLX
picks for the shape: the fast one (N a multiple of 8, K of its input step) and the other. The
banks hold weights quantized by mx.quantize and random codes into a random pair table. Shapes:
Qwen3.6's and Qwen3.8-Flash-Next's expert projections in decode and copy-miss prefill (whose down
has K = 640, off the fast step), the test models' (tests/tiny_model.py, tests/offload_harness.py),
bits 2-6 and 8 with group sizes 32, 64 and 128 on both impls, each dtype; output rows off the
8-row tile or fewer than one tile; a last code whose high word opens the row's second 64-byte
line; one-pair and nearly full tables, and every table's largest code at a used row's end; codes
of 13 bits (Qwen3.6's and Qwen3.8's), of 16 (an OptiQ checkpoint's 8-bit layers) on both impls,
and of 1, 9, 11 and 12 bits.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel.moe.affine.affine_gather_qmv_pair_codes import affine_gather_qmv_pair_codes
from ftmetal.kernel.ref import pair_code_bank_ref

MAX_PAIRS = 8192


@dataclass(frozen=True)
class Case:
    """tokens x top_k pairs through slots of a [output_size, input_size] projection; x is one
    row per token (gate_up) or, is_per_pair, one per pair (down)."""

    tokens: int
    top_k: int
    output_size: int
    input_size: int
    bits: int = 4
    group_size: int = 64
    dtype: mx.Dtype = mx.bfloat16
    is_per_pair: bool = False
    num_slots: int = 16
    num_pairs: int = MAX_PAIRS
    code_bits: int = 13

    def __str__(self) -> str:
        dtype = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}[self.dtype]
        x_rows = "pair" if self.is_per_pair else "token"
        return (
            f"T{self.tokens}-k{self.top_k}-N{self.output_size}-K{self.input_size}-b{self.bits}"
            f"-g{self.group_size}-{dtype}-x{x_rows}-S{self.num_slots}-P{self.num_pairs}"
            f"-c{self.code_bits}"
        )


CASES = [
    # Qwen3.6 (group 64, top 8): gate_up and down, decode and copy-miss prefill.
    Case(1, 8, 1024, 2048),
    Case(1, 8, 2048, 512, is_per_pair=True),
    Case(22, 8, 1024, 2048, dtype=mx.float16),
    Case(13, 8, 2048, 512, dtype=mx.float32, is_per_pair=True),
    # Qwen3.8-Flash-Next (group 32, top 10); its down's K = 640 takes the other impl.
    Case(1, 10, 1280, 2560, group_size=32),
    Case(1, 10, 2560, 640, group_size=32, is_per_pair=True),
    Case(6, 10, 2560, 640, group_size=32, dtype=mx.float16, is_per_pair=True),
    # tests/tiny_model.py: hidden 64, experts 96 wide, group 32, top 2.
    Case(5, 2, 192, 64, group_size=32, dtype=mx.float32, num_slots=4),
    Case(5, 2, 64, 96, group_size=32, dtype=mx.float32, is_per_pair=True, num_slots=4),
    # tests/offload_harness.py: group 64, hidden 128 / 192 / 2048, 4 and 8 bits.
    Case(3, 2, 128, 128, num_slots=6),
    Case(3, 2, 128, 64, is_per_pair=True, num_slots=6),
    Case(9, 2, 256, 192, bits=8, dtype=mx.float16, num_slots=6),
    Case(9, 2, 192, 128, bits=8, is_per_pair=True, num_slots=6),
    Case(2, 2, 4096, 2048, num_slots=4),
    Case(2, 2, 2048, 2048, dtype=mx.float32, is_per_pair=True, num_slots=4),
    # Each bit width on the fast impl, then on the other.
    Case(4, 4, 256, 512, bits=2, group_size=32),
    Case(4, 4, 256, 1024, bits=3),
    Case(4, 4, 256, 512, bits=5, group_size=128, dtype=mx.float16),
    Case(4, 4, 256, 768, bits=6, dtype=mx.float32, is_per_pair=True),
    Case(4, 4, 256, 256, bits=8, group_size=128),
    Case(4, 4, 96, 384, bits=2, dtype=mx.float32),
    Case(4, 4, 96, 320, bits=3, group_size=32, dtype=mx.float16, is_per_pair=True),
    Case(4, 4, 96, 640, bits=5, group_size=128),
    Case(4, 4, 96, 384, bits=6),
    Case(4, 4, 104, 352, bits=8, group_size=32, dtype=mx.float32),
    # Output rows off the 8-row tile (the last tile moved back), fewer than one tile.
    Case(3, 4, 36, 256),
    Case(3, 4, 6, 128, bits=8, dtype=mx.float32, is_per_pair=True),
    # 8 rows of 5 groups: the last code's high word opens the row's second 64-byte line.
    Case(2, 3, 8, 320, num_slots=5),
    # One pair, and fewer pairs than the codes could index.
    Case(3, 4, 128, 512, num_pairs=1),
    Case(3, 4, 128, 512, dtype=mx.float16, num_pairs=MAX_PAIRS - 3),
    # Other code widths: 16 bits on both impls, then narrower codes that cross words elsewhere.
    Case(1, 8, 1024, 2048, num_pairs=1 << 16, code_bits=16),
    Case(6, 10, 2560, 640, group_size=32, is_per_pair=True, num_pairs=50196, code_bits=16),
    Case(3, 4, 128, 512, dtype=mx.float32, num_pairs=2, code_bits=1),
    Case(3, 4, 128, 512, num_pairs=512, code_bits=9),
    Case(4, 4, 96, 384, bits=6, dtype=mx.float16, num_pairs=2000, code_bits=11),
    Case(2, 3, 8, 320, num_slots=5, num_pairs=4096, code_bits=12),
]


def check_equal(got: mx.array, expected: mx.array) -> None:
    assert got.shape == expected.shape and got.dtype == expected.dtype
    mismatches = (got != expected).sum().item()
    assert mismatches == 0, f"{mismatches} of {got.size} values differ"


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_gather_qmm(case: Case) -> None:
    rng = np.random.default_rng(case.output_size * 31 + case.input_size + case.bits)
    weight = 0.05 * rng.standard_normal(
        (case.num_slots, case.output_size, case.input_size), dtype=np.float32
    )
    w, _, _ = mx.quantize(mx.array(weight).astype(case.dtype), case.group_size, case.bits)
    pair_table = 0.02 * rng.standard_normal((case.num_pairs, 2), dtype=np.float32)
    pairs = mx.array(pair_table).astype(case.dtype)
    slots = rng.integers(0, case.num_slots, (case.tokens, case.top_k), dtype=np.int32)
    num_codes = case.output_size * case.input_size // case.group_size
    codes = rng.integers(0, case.num_pairs, (case.num_slots, num_codes)).astype(np.uint16)
    codes[slots[0, 0], -1] = case.num_pairs - 1
    rows, scales, biases = pair_code_bank_ref(codes, pairs, case.output_size, case.code_bits)
    x_shape = (case.tokens, case.top_k if case.is_per_pair else 1, 1, case.input_size)
    x = mx.array(rng.standard_normal(x_shape, dtype=np.float32)).astype(case.dtype)
    indices = mx.array(slots)
    expected = mx.gather_qmm(
        x,
        w,
        scales,
        biases,
        rhs_indices=indices,
        transpose=True,
        group_size=case.group_size,
        bits=case.bits,
        mode="affine",
        sorted_indices=False,
    )
    got = affine_gather_qmv_pair_codes(
        x,
        w,
        rows,
        pairs,
        indices,
        group_size=case.group_size,
        bits=case.bits,
        code_bits=case.code_bits,
    )
    check_equal(got, expected)
