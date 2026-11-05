"""layers/projection.py's fused rows on the gates of SharedExpertMoE (fuse_gates): the router gate
and the shared expert gate as one projection of their rows, zero rows up to a qmv_rows tile.

One token whose shape qmv_rows takes: the router's experts and weights (moe_topk reading the
experts' columns in place) equal the unfused gate's bit for bit, and the shared gate's logit is
mx.quantized_matmul's through the fused rows, the unfused one-row GEMV's to rounding (it ran on
MLX's other impl). Several tokens, or a shape qmv_rows does not take: each part's rows
(rows_forward) equal the unfused projection's bit for bit. Shapes: Qwen3.8-Flash-Next's 8-bit
gates (hidden 2560, 512 experts, top 10), Qwen3.6's 4-bit ones (2048, 256, top 8), a tiny model's
(hidden 96, 16 experts, top 4); and gates of two schemes, which stay apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel.moe.moe_topk import moe_topk
from ftmetal.layers.linear import Linear
from ftmetal.layers.projection import fuse_rows, project, qmv_scheme, rows_forward
from ftmetal.layers.quantization.config import AffineScheme

DTYPE = mx.bfloat16
# The shared gate's logit against the unfused GEMV's: bf16 rounding of one fp32 dot product.
SHARED_GATE_TOLERANCE = 2e-2


@dataclass(frozen=True)
class Case:
    hidden: int
    experts: int
    top_k: int
    bits: int
    group_size: int
    tokens: int = 1

    def __str__(self) -> str:
        scheme = f"b{self.bits}-g{self.group_size}"
        return f"H{self.hidden}-E{self.experts}-k{self.top_k}-{scheme}-T{self.tokens}"


CASES = [
    Case(2560, 512, 10, 8, 64),
    Case(2048, 256, 8, 4, 64),
    Case(2048, 256, 8, 4, 64, tokens=5),
    Case(96, 16, 4, 4, 32),
    Case(96, 16, 4, 4, 32, tokens=3),
]


def quantized_linear(rng: np.random.Generator, rows: int, case: Case) -> Linear:
    linear = Linear(case.hidden, rows, scheme=AffineScheme(case.bits, case.group_size), dtype=DTYPE)
    weight = mx.array(0.05 * rng.standard_normal((rows, case.hidden), dtype=np.float32)).astype(
        DTYPE
    )
    w, scales, biases = mx.quantize(weight, case.group_size, case.bits)
    linear["weight"], linear["scales"], linear["biases"] = w, scales, biases
    return linear


@pytest.mark.parametrize("case", CASES, ids=str)
def test_fused_gates_route_as_the_gates_apart(case: Case) -> None:
    rng = np.random.default_rng(case.hidden + case.experts + case.tokens)
    gate = quantized_linear(rng, case.experts, case)
    shared_gate = quantized_linear(rng, 1, case)
    x = mx.array(rng.standard_normal((case.tokens, case.hidden), dtype=np.float32)).astype(DTYPE)
    fused = fuse_rows([gate, shared_gate])
    assert fused is not None and fused["weight"].shape[0] % 8 == 0
    expected_ids, expected_weights = moe_topk(project(gate, x), case.top_k, True)
    expected_shared = project(shared_gate, x)
    if qmv_scheme(fused, x) is not None:
        logits = project(fused, x)
        ids, weights = moe_topk(logits, case.top_k, True, num_experts=case.experts)
        shared = logits[:, case.experts : case.experts + 1]
        scheme = AffineScheme(case.bits, case.group_size)
        through_fused = mx.quantized_matmul(
            x,
            fused["weight"],
            fused["scales"],
            fused["biases"],
            transpose=True,
            group_size=scheme.group_size,
            bits=scheme.bits,
        )[:, case.experts : case.experts + 1]
        assert mx.array_equal(shared, through_fused).item()
        assert mx.allclose(shared, expected_shared, atol=SHARED_GATE_TOLERANCE).item()
    else:
        router = rows_forward(fused, x, 0, case.experts)
        assert mx.array_equal(router, gate.forward(x)).item()
        ids, weights = moe_topk(router, case.top_k, True)
        shared = rows_forward(fused, x, case.experts, case.experts + 1)
        assert mx.array_equal(shared, shared_gate.forward(x)).item()
    assert mx.array_equal(ids, expected_ids).item()
    assert mx.array_equal(weights.view(mx.uint16), expected_weights.view(mx.uint16)).item()


def test_gates_of_two_schemes_stay_apart() -> None:
    rng = np.random.default_rng(0)
    gate = quantized_linear(rng, 16, Case(96, 16, 4, 4, 32))
    shared_gate = quantized_linear(rng, 1, Case(96, 16, 4, 8, 32))
    assert fuse_rows([gate, shared_gate]) is None
