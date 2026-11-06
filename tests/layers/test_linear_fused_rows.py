"""Linear's kernel dispatch and LinearMerged.fuse_rows.

An affine Linear runs one token as qmv_rows where it takes the shape and as MLX's quantized
matmul otherwise, and both equal mx.quantized_matmul bit for bit. The router gate and the shared
expert gate of SharedExpertMoE (fuse_gates) fuse into one LinearMerged of their rows, zero rows
up to a multiple of FUSED_ROW_TILE: one token runs both as one product, the router's logits the
gate's alone bit for bit and the shared gate's logit the one-row product's to rounding (MLX runs a
one-row GEMV on its other impl); several tokens run each part's rows alone, both bit for bit.
Shapes: Qwen3.8-Flash-Next's 8-bit gates (hidden 2560, 512 experts, top 10), Qwen3.6's 4-bit
ones (2048, 256, top 8), a tiny model's (hidden 96, 16 experts, top 4); and gates of two schemes,
which stay apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel.moe.moe_topk import moe_topk
from ftmetal.layers.linear import FUSED_ROW_TILE, Linear, LinearMerged
from ftmetal.layers.quantization import linear as linear_methods
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.layers.quantization.linear import AffineLinearMethod

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


# (case, output rows, qmv_rows_min_rows() as FTMETAL_QMV_ROWS sets it, whether qmv_rows
# runs)
DISPATCH_CASES = [
    (Case(2560, 512, 10, 4, 64), 4096, 1, True),
    (Case(2560, 512, 10, 4, 64), 4096, None, False),
    (Case(2560, 512, 10, 4, 64), 4096, 8192, False),
    (Case(2560, 512, 10, 8, 64), 512, 1, True),
    (Case(2560, 512, 10, 4, 64, tokens=3), 4096, 1, False),
    (Case(96, 16, 4, 4, 32), 16, 1, False),
]


@pytest.mark.parametrize(("case", "rows", "min_rows", "runs_rows"), DISPATCH_CASES)
def test_linear_runs_one_token_as_mlx_does(
    monkeypatch: pytest.MonkeyPatch, case: Case, rows: int, min_rows: int | None, runs_rows: bool
) -> None:
    monkeypatch.setattr(linear_methods, "qmv_rows_min_rows", lambda: min_rows)
    rng = np.random.default_rng(rows + case.tokens)
    linear = quantized_linear(rng, rows, case)
    x = mx.array(rng.standard_normal((case.tokens, case.hidden), dtype=np.float32)).astype(DTYPE)
    method = linear.quant_method
    assert isinstance(method, AffineLinearMethod)
    assert method.uses_qmv_rows(linear, x) == runs_rows
    expected = mx.quantized_matmul(
        x,
        linear["weight"],
        linear["scales"],
        linear["biases"],
        transpose=True,
        group_size=case.group_size,
        bits=case.bits,
    )
    assert mx.array_equal(linear.forward(x), expected).item()


@pytest.mark.parametrize("case", CASES, ids=str)
def test_fused_gates_route_as_the_gates_apart(case: Case) -> None:
    rng = np.random.default_rng(case.hidden + case.experts + case.tokens)
    gate = quantized_linear(rng, case.experts, case)
    shared_gate = quantized_linear(rng, 1, case)
    x = mx.array(rng.standard_normal((case.tokens, case.hidden), dtype=np.float32)).astype(DTYPE)
    fused = LinearMerged.fuse_rows([gate, shared_gate])
    assert fused is not None and fused["weight"].shape[0] % FUSED_ROW_TILE == 0
    router, shared = fused.forward_split(x)
    assert (router.shape, shared.shape) == ((case.tokens, case.experts), (case.tokens, 1))
    assert mx.array_equal(router, gate.forward(x)).item()
    if case.tokens == 1:
        assert mx.allclose(shared, shared_gate.forward(x), atol=SHARED_GATE_TOLERANCE).item()
    else:
        assert mx.array_equal(shared, shared_gate.forward(x)).item()
    expected_ids, expected_weights = moe_topk(gate.forward(x), case.top_k, True)
    ids, weights = moe_topk(router, case.top_k, True)
    assert mx.array_equal(ids, expected_ids).item()
    assert mx.array_equal(weights.view(mx.uint16), expected_weights.view(mx.uint16)).item()


def test_gates_of_two_schemes_stay_apart() -> None:
    rng = np.random.default_rng(0)
    gate = quantized_linear(rng, 16, Case(96, 16, 4, 4, 32))
    shared_gate = quantized_linear(rng, 1, Case(96, 16, 4, 8, 32))
    assert LinearMerged.fuse_rows([gate, shared_gate]) is None
