"""affine_gather_qmv against mx.gather_qmm, exactly, at the pairs whose slot is not negative.

The offload decode's resident gate_up pass over plain scales and biases: every row of a pair with
a slot must equal mx.gather_qmm's (unsorted, one row per pair) on the impl MLX picks for the
shape, the fast one (N a multiple of 8, K of its input step) and the other, while the pairs whose
slot is negative (moe_lookup_issue's pending pairs) are skipped. Shapes: Qwen3.6's and
Qwen3.8-Flash-Next's gate_up and down (whose K = 640 is off the fast step), the test models'
(tests/tiny_model.py, tests/offload_harness.py), bits 2-6 and 8 with group sizes 32, 64 and 128,
each dtype, output rows off the 8-row tile or fewer than one tile, no pending pair and every pair
pending.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel.moe.affine.affine_gather_qmv import affine_gather_qmv


@dataclass(frozen=True)
class Case:
    """tokens x top_k pairs through slots of a [output_size, input_size] projection, num_pending
    of them pending; x is one row per token or, is_per_pair, one per pair."""

    tokens: int
    top_k: int
    output_size: int
    input_size: int
    bits: int = 4
    group_size: int = 64
    dtype: mx.Dtype = mx.bfloat16
    is_per_pair: bool = False
    num_slots: int = 16
    num_pending: int = 3

    def __str__(self) -> str:
        dtype = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}[self.dtype]
        x_rows = "pair" if self.is_per_pair else "token"
        return (
            f"T{self.tokens}-k{self.top_k}-N{self.output_size}-K{self.input_size}-b{self.bits}"
            f"-g{self.group_size}-{dtype}-x{x_rows}-S{self.num_slots}-p{self.num_pending}"
        )


CASES = [
    # Qwen3.6 (group 64, top 8): gate_up and down, one decode row and the decode cap.
    Case(1, 8, 1024, 2048),
    Case(1, 8, 2048, 512, is_per_pair=True),
    Case(22, 8, 1024, 2048, dtype=mx.float16, num_pending=40),
    Case(13, 8, 2048, 512, dtype=mx.float32, is_per_pair=True),
    # Qwen3.8-Flash-Next (group 32, top 10); its down's K = 640 takes the other impl.
    Case(1, 10, 1280, 2560, group_size=32),
    Case(6, 10, 2560, 640, group_size=32, dtype=mx.float16, is_per_pair=True),
    # tests/tiny_model.py: hidden 64, experts 96 wide, group 32, top 2.
    Case(5, 2, 192, 64, group_size=32, dtype=mx.float32, num_slots=4),
    Case(5, 2, 64, 96, group_size=32, dtype=mx.float32, is_per_pair=True, num_slots=4),
    # tests/offload_harness.py: group 64, hidden 128 / 192 / 2048, 4 and 8 bits.
    Case(3, 2, 128, 128, num_slots=6, num_pending=1),
    Case(9, 2, 256, 192, bits=8, dtype=mx.float16, num_slots=6),
    Case(2, 2, 4096, 2048, num_slots=4),
    # Each other bit width on the fast impl, then on the other.
    Case(4, 4, 256, 512, bits=2, group_size=32),
    Case(4, 4, 256, 1024, bits=3),
    Case(4, 4, 256, 512, bits=5, group_size=128, dtype=mx.float16),
    Case(4, 4, 256, 768, bits=6, dtype=mx.float32, is_per_pair=True),
    Case(4, 4, 96, 384, bits=2, dtype=mx.float32),
    Case(4, 4, 96, 320, bits=3, group_size=32, dtype=mx.float16, is_per_pair=True),
    Case(4, 4, 96, 640, bits=5, group_size=128),
    Case(4, 4, 104, 352, bits=8, group_size=32, dtype=mx.float32),
    # Output rows off the 8-row tile (the last tile moved back), fewer than one tile.
    Case(3, 4, 36, 256),
    Case(3, 4, 6, 128, bits=8, dtype=mx.float32, is_per_pair=True),
    # No pending pair; every pair pending.
    Case(3, 8, 1024, 2048, num_pending=0),
    Case(2, 4, 256, 512, num_pending=8),
]


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_gather_qmm(case: Case) -> None:
    rng = np.random.default_rng(case.output_size * 31 + case.input_size + case.bits)
    weight = 0.05 * rng.standard_normal(
        (case.num_slots, case.output_size, case.input_size), dtype=np.float32
    )
    w, scales, biases = mx.quantize(mx.array(weight).astype(case.dtype), case.group_size, case.bits)
    slots = rng.integers(0, case.num_slots, (case.tokens, case.top_k), dtype=np.int32)
    pending = rng.choice(slots.size, case.num_pending, replace=False)
    issued_slots = slots.copy()
    issued_slots.reshape(-1)[pending] = -1 - rng.integers(0, 64, case.num_pending)
    x_shape = (case.tokens, case.top_k if case.is_per_pair else 1, 1, case.input_size)
    x = mx.array(rng.standard_normal(x_shape, dtype=np.float32)).astype(case.dtype)
    expected = mx.gather_qmm(
        x,
        w,
        scales,
        biases,
        rhs_indices=mx.array(slots),
        transpose=True,
        group_size=case.group_size,
        bits=case.bits,
        mode="affine",
        sorted_indices=False,
    )
    got = affine_gather_qmv(
        x,
        w,
        scales,
        biases,
        mx.array(issued_slots),
        group_size=case.group_size,
        bits=case.bits,
    )
    assert got.shape == expected.shape and got.dtype == expected.dtype
    resident = (mx.array(issued_slots) >= 0)[..., None, None]
    mismatches = (resident & (got != expected)).sum().item()
    assert mismatches == 0, f"{mismatches} values of the resident pairs differ"
