"""Per-call time of the offload decode's expert gather: affine_gather_qmv_pair_codes on pair-code
rows against mx.gather_qmm on the scales and biases they stand for.

    uv run python tools/bench_affine_gather_qmv_pair_codes.py [--models qwen3.6,qwen3.8]
        [--projections gate_up,down] [--tokens 1,8,22] [--slots 128] [--bits 4] [--code-bits 13]

The routed expert projections of Qwen3.6-35B-A3B (group 64, top 8) and Qwen3.8-Flash-Next
(group 32, top 10) in affine bf16: gate_up takes one input row per token, down one per
(token, expert) pair. A bank of --slots slots holds random packed weights and random codes into
8192-pair tables; each call routes every token to top_k distinct random slots, a new draw per
call, and reads the next of --tables tables, as each layer reads its own, so the slots and the
table read come from DRAM rather than a cache that kept the last call's. Calls chain as layers
do: each call's x waits on the previous call's output (x + y[..., :1] * 0, a two-op chain whose
own time is measured alone and taken off). A run builds the graph of its --calls calls first and
times its evaluation alone, as a replayed decode step pays only that; runs are long enough for
the GPU clock to settle and interleave the variants. GPU: the best of --repeats runs, in
microseconds per call, with the bytes of the slots a call reads (weights and scales and biases,
or weights and codes) over that time; host: the microseconds of graph building per call, which
an eager step pays on top. impl: the MLX gather_qmv kernel both run (fast, or qmv off its K
step). exact: equal bit for bit.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import NamedTuple, Protocol

import mlx.core as mx
import numpy as np

from ftmetal.kernel.moe.affine.affine_gather_qmv_pair_codes import (
    GATHER_QMV_FAST_INPUT_STEP,
    QMV_TILE_ROWS,
    affine_gather_qmv_pair_codes,
)
from ftmetal.kernel.ref import pair_code_bank_ref

DTYPE = mx.bfloat16


class ExpertShapes(NamedTuple):
    group_size: int
    top_k: int
    # (input size, output size) of each routed expert projection.
    projections: dict[str, tuple[int, int]]


MODELS = {
    "qwen3.6": ExpertShapes(64, 8, {"gate_up": (2048, 1024), "down": (512, 2048)}),
    "qwen3.8": ExpertShapes(32, 10, {"gate_up": (2560, 1280), "down": (640, 2560)}),
}


class Projection(Protocol):
    def __call__(self, x: mx.array, indices: mx.array, pairs: mx.array) -> mx.array: ...


class RunSeconds(NamedTuple):
    build: float
    evaluate: float


def chained_run(
    project: Projection | None,
    x: mx.array,
    routes: list[mx.array],
    tables: list[mx.array],
    y_like: mx.array,
) -> RunSeconds:
    """Seconds to build, then to evaluate, one call per route chained through x, call i on
    table i mod len(tables); project None runs the chain alone."""
    started = time.perf_counter()
    for call, indices in enumerate(routes):
        pairs = tables[call % len(tables)]
        y = y_like if project is None else project(x, indices, pairs)
        x = x + y[:, : x.shape[1], :, :1] * 0
    built = time.perf_counter()
    mx.eval(x)
    return RunSeconds(build=built - started, evaluate=time.perf_counter() - built)


def best_us_per_call(
    projects: list[Projection | None],
    x: mx.array,
    routes: list[mx.array],
    tables: list[mx.array],
    y_like: mx.array,
    repeats: int,
) -> list[RunSeconds]:
    """Each projection's best build and evaluation microseconds per call, the runs interleaved
    so drift hits them alike."""
    for project in projects:
        chained_run(project, x, routes, tables, y_like)
    runs = [
        [chained_run(project, x, routes, tables, y_like) for project in projects]
        for _ in range(repeats)
    ]
    scale = 1e6 / len(routes)
    return [
        RunSeconds(
            build=min(run[i].build for run in runs) * scale,
            evaluate=min(run[i].evaluate for run in runs) * scale,
        )
        for i in range(len(projects))
    ]


def bench_projection(
    model: str,
    projection: str,
    token_counts: list[int],
    num_slots: int,
    num_tables: int,
    bits: int,
    code_bits: int,
    calls: int,
    repeats: int,
    rng: np.random.Generator,
) -> None:
    """One table row per token count, over a bank built once; freed on return."""
    shapes = MODELS[model]
    input_size, output_size = shapes.projections[projection]
    group_size = shapes.group_size
    words = input_size * bits // 32
    w_shape = (num_slots, output_size, words)
    w = mx.random.randint(0, 2**16, w_shape, dtype=mx.uint32) << 16
    w = w | mx.random.randint(0, 2**16, w_shape, dtype=mx.uint32)
    tables = [
        mx.array(0.02 * rng.standard_normal((1 << code_bits, 2), dtype=np.float32)).astype(DTYPE)
        for _ in range(num_tables)
    ]
    num_codes = output_size * input_size // group_size
    codes = rng.integers(0, 1 << code_bits, (num_slots, num_codes)).astype(np.uint16)
    rows, scales, biases = pair_code_bank_ref(codes, tables[0], output_size, code_bits)
    mx.eval(w, rows, scales, biases, tables)
    is_fast = (
        output_size % QMV_TILE_ROWS == 0 and input_size % GATHER_QMV_FAST_INPUT_STEP[bits] == 0
    )
    impl = "fast" if is_fast else "qmv"
    mlx_slot_bytes = w[0].nbytes + scales[0].nbytes + biases[0].nbytes
    pair_codes_slot_bytes = w[0].nbytes + rows[0].nbytes

    def mlx_gather(x: mx.array, indices: mx.array, pairs: mx.array) -> mx.array:
        return mx.gather_qmm(
            x,
            w,
            scales,
            biases,
            rhs_indices=indices,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )

    def pair_codes_gather(x: mx.array, indices: mx.array, pairs: mx.array) -> mx.array:
        return affine_gather_qmv_pair_codes(
            x, w, rows, pairs, indices, group_size=group_size, bits=bits, code_bits=code_bits
        )

    top_k = shapes.top_k
    for tokens in token_counts:
        draws = [
            np.stack([rng.choice(num_slots, top_k, replace=False) for _ in range(tokens)])
            for _ in range(calls)
        ]
        routes = [mx.array(draw.astype(np.int32)) for draw in draws]
        slots_read = float(np.mean([np.unique(draw).size for draw in draws]))
        x_rows = top_k if projection == "down" else 1
        x = mx.random.normal((tokens, x_rows, 1, input_size)).astype(DTYPE)
        y_like = mx.zeros((tokens, top_k, 1, output_size), dtype=DTYPE)
        mx.eval(routes, x, y_like)
        expected = mlx_gather(x, routes[0], tables[0])
        is_exact = bool(mx.array_equal(expected, pair_codes_gather(x, routes[0], tables[0])))
        chain, mlx_run, pair_codes_run = best_us_per_call(
            [None, mlx_gather, pair_codes_gather], x, routes, tables, y_like, repeats
        )
        mlx_us = mlx_run.evaluate - chain.evaluate
        pair_codes_us = pair_codes_run.evaluate - chain.evaluate
        mlx_gbps = slots_read * mlx_slot_bytes / mlx_us / 1e3
        pair_codes_gbps = slots_read * pair_codes_slot_bytes / pair_codes_us / 1e3
        print(
            f"| {model} | {projection} | {tokens} | {top_k} | {output_size} | {input_size} "
            f"| {impl} | {mlx_us:.1f} | {mlx_gbps:.0f} | {pair_codes_us:.1f} "
            f"| {pair_codes_gbps:.0f} "
            f"| {pair_codes_us / mlx_us:.3f} | {mlx_run.build - chain.build:.1f} "
            f"| {pair_codes_run.build - chain.build:.1f} | {'y' if is_exact else 'n'} |",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen3.6,qwen3.8")
    parser.add_argument("--projections", default="gate_up,down")
    parser.add_argument("--tokens", default="1,8,22")
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--code-bits", type=int, default=13)
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    token_counts = [int(tokens) for tokens in args.tokens.split(",")]
    rng = np.random.default_rng(0)
    mx.random.seed(0)
    print(
        f"affine {args.bits}-bit bf16, {args.slots} slots, {args.tables} tables of "
        f"{1 << args.code_bits} pairs ({args.code_bits}-bit codes); GPU microseconds per chained "
        "call and GB/s of the slots read, host microseconds of graph building per call"
    )
    print(
        "| model | proj | T | top_k | N | K | impl | gather_qmm us | GB/s | pair_codes us | GB/s "
        "| pair_codes / gather_qmm | gather_qmm host us | pair_codes host us | exact |"
    )
    print("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for model in args.models.split(","):
        for projection in args.projections.split(","):
            bench_projection(
                model,
                projection,
                token_counts,
                args.slots,
                args.tables,
                args.bits,
                args.code_bits,
                args.calls,
                args.repeats,
                rng,
            )
            mx.clear_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
