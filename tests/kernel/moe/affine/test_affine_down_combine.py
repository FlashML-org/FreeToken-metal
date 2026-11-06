"""affine_down_combine against the ops it fuses, exactly.

The reference is AffineMoEMethod.apply_slots' tail over the same inputs: mx.gather_qmm of act at the
slots through the down bank (on scales and biases, or the pair-code rows' expanded ones), then
(y.squeeze(-2) * weights[..., None]).sum(axis=-2), then layers/activation.py's
sigmoid_gated_add(routed, shared, gate) when the shared expert is given, or routed + shared when
it has no gate. Every output must be equal
bit for bit: the products rounded to the dtype, the sum in MLX's column-reduction order
(min(8, top_k) partial sums from +0, then summed in turn; a copy for top_k 1), the gated add in
fp32 with MLX's sigmoid. Shapes: Qwen3.6's down (hidden 2048, width 512, group 64, top 8) at 1 and
22 rows, Qwen3.8-Flash-Next's (hidden 2560, width 640 off the fast step, group 32, top 10), the
test models' (tests/tiny_model.py: hidden 64, width 96 or 256, group 32, top 2-4;
tests/offload_harness.py: hidden 128 / 192 / 256 / 2048, width 64 / 128 / 2048, group 64), bits
2-6 and 8, each dtype, plain scales and pair codes of 13 and 16 bits, with and without the shared
expert; top_k 1 and 12; routing weights of -0.0, whose products sum to +0 in MLX. Pair codes off
the fast step whose last part-block is not split over both lane halves (a tail over half a block,
or FTMETAL_QMV_HALF_REM=0) take the gathered rows' combine, also at top_k 1 and on -0.0 weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel.moe.affine.affine_down_combine import affine_down_combine
from ftmetal.kernel.moe.affine.affine_glu_pending import (
    AffineScales,
    ExpertScales,
    PairCodeScales,
)
from ftmetal.kernel.ref import pair_code_bank_ref
from ftmetal.layers.activation import sigmoid_gated_add

Weights = Literal["random", "negative_zero"]


@dataclass(frozen=True)
class Case:
    tokens: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    group_size: int = 64
    bits: int = 4
    dtype: mx.Dtype = mx.bfloat16
    # None: plain scales and biases.
    code_bits: int | None = None
    has_shared: bool = True
    has_gate: bool = True
    weights: Weights = "random"
    num_slots: int = 16

    def __str__(self) -> str:
        dtype = {mx.bfloat16: "bf16", mx.float16: "fp16", mx.float32: "fp32"}[self.dtype]
        scales = "plain" if self.code_bits is None else f"c{self.code_bits}"
        return (
            f"T{self.tokens}-k{self.top_k}-H{self.hidden_size}-I{self.intermediate_size}"
            f"-g{self.group_size}-b{self.bits}-{dtype}-{scales}"
            f"-{'routed' if not self.has_shared else 'gated' if self.has_gate else 'plain'}"
            f"-w{self.weights}"
        )


CASES = [
    # Qwen3.6: one decode row and the decode cap, plain and in 13-bit pair codes.
    Case(1, 8, 2048, 512),
    Case(22, 8, 2048, 512, code_bits=13),
    Case(7, 8, 2048, 512, dtype=mx.float16, has_shared=False),
    # Qwen3.8-Flash-Next: K = 640 is off the fast step.
    Case(1, 10, 2560, 640, group_size=32, code_bits=13),
    Case(22, 10, 2560, 640, group_size=32),
    Case(3, 10, 2560, 640, group_size=32, dtype=mx.float32, code_bits=16),
    # Flint: its shared expert has no gate; plain, pair codes, the gathered rows' combine.
    Case(1, 6, 2048, 512, has_gate=False),
    Case(22, 6, 2048, 512, code_bits=13, has_gate=False),
    Case(3, 6, 2048, 512, dtype=mx.float32, has_gate=False),
    Case(2, 8, 256, 704, group_size=32, code_bits=13, has_gate=False),
    # tests/tiny_model.py.
    Case(5, 2, 64, 96, group_size=32, dtype=mx.float32, num_slots=4),
    Case(4, 4, 64, 256, group_size=32, code_bits=13, num_slots=6),
    Case(3, 3, 64, 96, group_size=32, dtype=mx.float16, has_shared=False, num_slots=4),
    # tests/offload_harness.py.
    Case(3, 2, 128, 64, num_slots=6),
    Case(9, 2, 192, 128, bits=8, dtype=mx.float16, num_slots=6),
    Case(2, 2, 2048, 2048, dtype=mx.float32, code_bits=16, num_slots=4),
    Case(4, 2, 256, 128, num_slots=6),
    # Other bit widths, on both impls.
    Case(4, 8, 256, 512, bits=2, group_size=32),
    Case(4, 8, 256, 1024, bits=3, code_bits=13),
    Case(4, 8, 256, 512, bits=5, group_size=128, dtype=mx.float16),
    Case(4, 8, 256, 768, bits=6, dtype=mx.float32),
    Case(4, 8, 256, 256, bits=8, group_size=128),
    Case(4, 8, 96, 320, bits=3, group_size=32, dtype=mx.float16),
    # One routed expert (MLX copies instead of reducing), more pairs than partial sums.
    Case(5, 1, 2048, 512),
    Case(3, 12, 256, 512, has_shared=False),
    # -0.0 routing weights: the products are signed zeros MLX's sum turns into +0.
    Case(3, 8, 256, 512, weights="negative_zero", has_shared=False),
    Case(2, 4, 64, 96, group_size=32, weights="negative_zero", num_slots=4),
    # Pair codes off the fast step: one expert, -0.0 weights; a tail of 192 (gathered).
    Case(3, 1, 64, 96, group_size=32, code_bits=13, num_slots=4),
    Case(2, 4, 64, 96, group_size=32, code_bits=13, weights="negative_zero", num_slots=4),
    Case(2, 8, 256, 704, group_size=32, code_bits=13),
    # A packed Gemma4 26B's: a tail of 192 (gathered), no shared expert.
    Case(2, 8, 2816, 704, code_bits=13, has_shared=False),
]


def bank(case: Case, rng: np.random.Generator) -> tuple[mx.array, ExpertScales, mx.array, mx.array]:
    """The down bank's weight, its scale source, and the scales and biases it stands for."""
    shape = (case.num_slots, case.hidden_size, case.intermediate_size)
    weight = mx.array(0.05 * rng.standard_normal(shape, dtype=np.float32)).astype(case.dtype)
    w, scales, biases = mx.quantize(weight, case.group_size, case.bits)
    if case.code_bits is None:
        return w, AffineScales(scales, biases), scales, biases
    else:
        pair_table = 0.02 * rng.standard_normal((1 << case.code_bits, 2), dtype=np.float32)
        pairs = mx.array(pair_table).astype(case.dtype)
        num_codes = case.hidden_size * case.intermediate_size // case.group_size
        codes = rng.integers(0, 1 << case.code_bits, (case.num_slots, num_codes))
        rows, scales, biases = pair_code_bank_ref(
            codes.astype(np.uint32), pairs, case.hidden_size, case.code_bits
        )
        return w, PairCodeScales(rows, pairs), scales, biases


@pytest.mark.parametrize("case", CASES, ids=str)
def test_matches_apply_slots_tail(case: Case) -> None:
    rng = np.random.default_rng(case.hidden_size * 7 + case.intermediate_size + case.top_k)
    w, scale_source, scales, biases = bank(case, rng)
    shape = (case.tokens, case.top_k)
    slots = mx.array(rng.integers(0, case.num_slots, shape, dtype=np.int32))
    act_shape = (*shape, 1, case.intermediate_size)
    act = mx.array(rng.standard_normal(act_shape, dtype=np.float32)).astype(case.dtype)
    if case.weights == "random":
        weights = mx.array(rng.random(shape, dtype=np.float32)).astype(case.dtype)
    else:
        weights = mx.full(shape, -0.0, dtype=case.dtype)
    shared = mx.array(rng.standard_normal((case.tokens, case.hidden_size), dtype=np.float32))
    gate = mx.array(rng.standard_normal((case.tokens, 1), dtype=np.float32))
    shared, gate = shared.astype(case.dtype), gate.astype(case.dtype)
    y = mx.gather_qmm(
        act,
        w,
        scales,
        biases,
        rhs_indices=slots,
        transpose=True,
        group_size=case.group_size,
        bits=case.bits,
        mode="affine",
    )
    expected = (y.squeeze(-2) * weights[..., None]).sum(axis=-2)
    if case.has_shared and case.has_gate:
        expected = sigmoid_gated_add(expected, shared, gate)
    elif case.has_shared:
        expected = expected + shared
    got = affine_down_combine(
        act,
        w,
        scale_source,
        slots,
        weights,
        group_size=case.group_size,
        bits=case.bits,
        shared=shared if case.has_shared else None,
        shared_gate=gate if case.has_shared and case.has_gate else None,
    )
    assert got.shape == expected.shape and got.dtype == expected.dtype
    # Bit patterns, so that +0 and -0 differ.
    bits_type = mx.uint32 if case.dtype == mx.float32 else mx.uint16
    mismatches = (got.view(bits_type) != expected.view(bits_type)).sum().item()
    assert mismatches == 0, f"{mismatches} of {got.size} values differ"
