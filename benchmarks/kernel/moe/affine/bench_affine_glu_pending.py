"""GPU time of one offload decode MoE layer's routed experts: the waiting lookup's kernels against
the split lookup's four.

uv run python -m benchmarks.kernel.moe.affine.bench_affine_glu_pending [--models qwen3.6,qwen3.8]
    [--tokens 1,8,22] [--pending 0,2] [--scales plain,pair_codes] [--slots 128]

waiting: moe_lookup, the gate_up gather (mx.gather_qmm, or affine_gather_qmv_pair_codes on pair-code
rows), the activation, the down gather, (y * weights).sum(-2) and sigmoid_gated_add with the shared
expert's output, where the model has one. split: moe_lookup_issue, the resident gate_up pass
(affine_gather_qmv or affine_gather_qmv_pair_codes with skip_pending), affine_glu_pending and
affine_down_combine. Every expert is resident but --pending of them per layer, which the split path
leaves to affine_glu_pending; the mailbox's shutdown word is set, so nothing is posted, no CPU
answers and both paths give the absent experts' pairs moe_lookup's fallback slot without a wait: the
time is the kernels' alone, the SSD read the split overlaps is not in it. The routed experts of
Qwen3.6-35B-A3B (hidden 2048, width 512, group 64, top 8, 256 experts), Qwen3.8-Flash-Next (hidden
2560, width 640, group 32, top 10, 512 experts) and Gemma4 26B-A4B (hidden 2816, width 704, group
64, top 8, 128 experts, gelu_tanh, no shared expert), affine 4-bit bf16, over a pool of --slots
slots of random weights (and random 13-bit codes into 8192-pair tables). Each call routes every
token to top_k distinct random experts, a new draw per call, and calls chain as layers do: each
call's ids wait on the call before's output (one fused kernel, whose own time is measured alone and
taken off). A run builds the graph of its --calls calls first and times its evaluation alone, as a
replayed decode step pays only that. GPU: the best of --repeats runs in microseconds per call;
exact: the waiting lookup's and the split path's outputs equal bit for bit.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Literal, NamedTuple, Protocol

import mlx.core as mx
import numpy as np

from ftmetal.kernel._ext import _C
from ftmetal.kernel.moe.affine.affine_down_combine import affine_down_combine
from ftmetal.kernel.moe.affine.affine_gather_qmv import affine_gather_qmv
from ftmetal.kernel.moe.affine.affine_gather_qmv_pair_codes import affine_gather_qmv_pair_codes
from ftmetal.kernel.moe.affine.affine_glu_pending import (
    AffineScales,
    ExpertScales,
    PairCodeScales,
    affine_glu_pending,
)
from ftmetal.kernel.moe.moe_lookup import moe_lookup, moe_lookup_issue
from ftmetal.kernel.ref import pair_code_bank_ref
from ftmetal.layers.activation import gelu_tanh_and_mul, sigmoid_gated_add, silu_and_mul

DTYPE = mx.bfloat16
BITS = 4
CODE_BITS = 13


class ExpertShapes(NamedTuple):
    hidden_size: int
    intermediate_size: int
    group_size: int
    top_k: int
    num_experts: int
    activation: Literal["silu", "gelu_tanh"] = "silu"
    has_shared_expert: bool = True


MODELS = {
    "qwen3.6": ExpertShapes(2048, 512, 64, 8, 256),
    "qwen3.8": ExpertShapes(2560, 640, 32, 10, 512),
    "gemma4-26b": ExpertShapes(2816, 704, 64, 8, 128, "gelu_tanh", False),
}


class Projection(NamedTuple):
    weight: mx.array
    scales: ExpertScales


class Layer(NamedTuple):
    """One layer's pool and the lookup tables: slot_of maps every expert but the pending ones."""

    gate_up: Projection
    down: Projection
    slot_of: mx.array
    mailbox: mx.array
    last_use: mx.array
    hit_stats: mx.array


class LayerCall(Protocol):
    def __call__(
        self,
        layer: Layer,
        x: mx.array,
        topk_ids: mx.array,
        weights: mx.array,
        shared: mx.array,
        gate: mx.array,
        moe_step: mx.array,
    ) -> mx.array: ...


def projection(
    shapes: ExpertShapes,
    num_slots: int,
    output_size: int,
    input_size: int,
    is_pair_codes: bool,
    rng: np.random.Generator,
) -> Projection:
    words = input_size * BITS // 32
    w_shape = (num_slots, output_size, words)
    w = mx.random.randint(0, 2**16, w_shape, dtype=mx.uint32) << 16
    w = w | mx.random.randint(0, 2**16, w_shape, dtype=mx.uint32)
    groups = (num_slots, output_size, input_size // shapes.group_size)
    if is_pair_codes:
        table = 0.02 * rng.standard_normal((1 << CODE_BITS, 2), dtype=np.float32)
        pairs = mx.array(table).astype(DTYPE)
        codes = rng.integers(0, 1 << CODE_BITS, (num_slots, groups[1] * groups[2]))
        rows, _, _ = pair_code_bank_ref(codes.astype(np.uint32), pairs, output_size, CODE_BITS)
        return Projection(w, PairCodeScales(rows, pairs))
    else:
        scales = (0.02 * mx.random.normal(groups)).astype(DTYPE)
        biases = (0.02 * mx.random.normal(groups)).astype(DTYPE)
        return Projection(w, AffineScales(scales, biases))


def gather(
    shapes: ExpertShapes, part: Projection, x: mx.array, slots: mx.array, skip_pending: bool
) -> mx.array:
    """part's rows at slots, as the waiting lookup's path or the resident pass gathers them."""
    if isinstance(part.scales, PairCodeScales):
        return affine_gather_qmv_pair_codes(
            x,
            part.weight,
            part.scales.codes,
            part.scales.pairs,
            slots,
            group_size=shapes.group_size,
            bits=BITS,
            code_bits=part.scales.code_bits,
            skip_pending=skip_pending,
        )
    elif skip_pending:
        scales = part.scales
        return affine_gather_qmv(
            x,
            part.weight,
            scales.scales,
            scales.biases,
            slots,
            group_size=shapes.group_size,
            bits=BITS,
        )
    else:
        return mx.gather_qmm(
            x,
            part.weight,
            part.scales.scales,
            part.scales.biases,
            rhs_indices=slots,
            transpose=True,
            group_size=shapes.group_size,
            bits=BITS,
            mode="affine",
        )


def waiting_call(shapes: ExpertShapes) -> LayerCall:
    def call(
        layer: Layer,
        x: mx.array,
        topk_ids: mx.array,
        weights: mx.array,
        shared: mx.array,
        gate: mx.array,
        moe_step: mx.array,
    ) -> mx.array:
        slots, *_ = moe_lookup(
            topk_ids,
            moe_step,
            layer.slot_of,
            layer.mailbox,
            layer.last_use,
            layer.hit_stats,
            layer_id=0,
            slot_base=0,
            kind=_C.MoeLookupKind.DECODE,
        )
        x = mx.expand_dims(x, (-2, -3))
        act_and_mul = silu_and_mul if shapes.activation == "silu" else gelu_tanh_and_mul
        act = act_and_mul(gather(shapes, layer.gate_up, x, slots, False))
        y = gather(shapes, layer.down, act, slots, False)
        routed = (y.squeeze(-2) * weights[..., None]).sum(axis=-2)
        return sigmoid_gated_add(routed, shared, gate) if shapes.has_shared_expert else routed

    return call


def split_call(shapes: ExpertShapes) -> LayerCall:
    def call(
        layer: Layer,
        x: mx.array,
        topk_ids: mx.array,
        weights: mx.array,
        shared: mx.array,
        gate: mx.array,
        moe_step: mx.array,
    ) -> mx.array:
        lookup = moe_lookup_issue(
            topk_ids,
            moe_step,
            layer.slot_of,
            layer.mailbox,
            layer.last_use,
            layer.hit_stats,
            layer_id=0,
            slot_base=0,
        )
        x = mx.expand_dims(x, (-2, -3))
        gate_up = gather(shapes, layer.gate_up, x, lookup.slots, True)
        act, slots, _ = affine_glu_pending(
            x,
            layer.gate_up.weight,
            layer.gate_up.scales,
            gate_up,
            lookup.slots,
            lookup.issued,
            layer.slot_of,
            lookup.mailbox,
            shared if shapes.has_shared_expert else gate_up,
            layer_id=0,
            slot_base=0,
            num_experts=shapes.num_experts,
            group_size=shapes.group_size,
            bits=BITS,
            activation=shapes.activation,
        )
        return affine_down_combine(
            act,
            layer.down.weight,
            layer.down.scales,
            slots,
            weights,
            group_size=shapes.group_size,
            bits=BITS,
            shared=shared if shapes.has_shared_expert else None,
            shared_gate=gate if shapes.has_shared_expert else None,
        )

    return call


class Route(NamedTuple):
    x: mx.array
    topk_ids: mx.array
    weights: mx.array
    shared: mx.array
    gate: mx.array


@mx.compile
def dependent_ids(topk_ids: mx.array, previous: mx.array) -> mx.array:
    """topk_ids, once previous is evaluated, as a router's ids wait for the layer before."""
    return topk_ids + (previous[:1, :1] != previous[:1, :1]).astype(mx.int32) * 0


def chained_seconds(
    call: LayerCall | None, layer: Layer, routes: list[Route], moe_step: mx.array
) -> float:
    """Seconds to evaluate one call per route, each call's ids waiting on the call before; call
    None runs the chain alone."""
    previous = routes[0].topk_ids
    for route in routes:
        topk_ids = dependent_ids(route.topk_ids, previous)
        if call is None:
            previous = topk_ids
        else:
            previous = call(
                layer, route.x, topk_ids, route.weights, route.shared, route.gate, moe_step
            )
    started = time.perf_counter()
    mx.eval(previous)
    return time.perf_counter() - started


def bench_model(
    model: str,
    token_counts: list[int],
    pending_counts: list[int],
    is_pair_codes: bool,
    num_slots: int,
    calls: int,
    repeats: int,
    rng: np.random.Generator,
) -> None:
    shapes = MODELS[model]
    hidden, width = shapes.hidden_size, shapes.intermediate_size
    gate_up = projection(shapes, num_slots, 2 * width, hidden, is_pair_codes, rng)
    down = projection(shapes, num_slots, hidden, width, is_pair_codes, rng)
    mx.eval(gate_up, down)
    waiting, split = waiting_call(shapes), split_call(shapes)
    for tokens in token_counts:
        routes = []
        for _ in range(calls):
            ids = np.stack(
                [rng.choice(shapes.num_experts, shapes.top_k, replace=False) for _ in range(tokens)]
            )
            routes.append(
                Route(
                    (mx.random.normal((tokens, hidden)) * 0.5).astype(DTYPE),
                    mx.array(ids.astype(np.int32)),
                    mx.random.uniform(shape=(tokens, shapes.top_k)).astype(DTYPE),
                    (mx.random.normal((tokens, hidden)) * 0.5).astype(DTYPE),
                    mx.random.normal((tokens, 1)).astype(DTYPE),
                )
            )
        moe_step = mx.array([tokens, 1], dtype=mx.int32)
        mx.eval(routes, moe_step)
        for num_pending in pending_counts:
            # Every expert in a slot of the pool (several per slot), then num_pending absent.
            slot_of = np.arange(shapes.num_experts, dtype=np.int32) % num_slots
            slot_of[rng.choice(shapes.num_experts, num_pending, replace=False)] = -1
            mailbox = mx.zeros((_C.MAILBOX_WORDS,), dtype=mx.uint32)
            layer = Layer(
                gate_up,
                down,
                mx.array(slot_of),
                mailbox,
                mx.zeros((num_slots,), dtype=mx.uint32),
                mx.zeros((1, 2), dtype=mx.uint32),
            )
            mx.eval(layer)
            # Nothing posts, nobody answers: the pending pairs take the fallback slot at once.
            np.asarray(memoryview(mailbox)).view(np.uint32)[_C.MailboxWord.SHUTDOWN] = 1
            route = routes[0]
            outputs = [
                call(
                    layer,
                    route.x,
                    route.topk_ids,
                    route.weights,
                    route.shared,
                    route.gate,
                    moe_step,
                )
                for call in (waiting, split)
            ]
            is_exact = bool(mx.array_equal(outputs[0], outputs[1]))
            variants = (None, waiting, split)
            for call in variants:
                chained_seconds(call, layer, routes, moe_step)
            runs = [
                [chained_seconds(call, layer, routes, moe_step) for call in variants]
                for _ in range(repeats)
            ]
            chain_us, waiting_us, split_us = (
                min(run[i] for run in runs) * 1e6 / calls for i in range(len(variants))
            )
            waiting_us -= chain_us
            split_us -= chain_us
            scales = "pair_codes" if is_pair_codes else "plain"
            exact = "y" if is_exact else "n"
            print(
                f"| {model} | {scales} | {tokens} | {shapes.top_k} | {num_pending} "
                f"| {waiting_us:.1f} | {split_us:.1f} | {split_us / waiting_us:.3f} | {exact} |",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen3.6,qwen3.8,gemma4-26b")
    parser.add_argument("--tokens", default="1,8,22")
    parser.add_argument("--pending", default="0,2")
    parser.add_argument("--scales", default="plain,pair_codes")
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--calls", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    token_counts = [int(tokens) for tokens in args.tokens.split(",")]
    pending_counts = [int(pending) for pending in args.pending.split(",")]
    rng = np.random.default_rng(0)
    mx.random.seed(0)
    print(
        f"affine {BITS}-bit bf16, {args.slots} slots; GPU microseconds per chained MoE layer "
        "(routed experts and any shared expert's gated add)"
    )
    print(
        "| model | scales | T | top_k | pending | waiting us | split us | split / waiting | exact |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for model in args.models.split(","):
        for scales in args.scales.split(","):
            bench_model(
                model,
                token_counts,
                pending_counts,
                scales == "pair_codes",
                args.slots,
                args.calls,
                args.repeats,
                rng,
            )
            mx.clear_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
