"""Per-call time of token sampling: the Metal kernels against the MLX-op composition.

    uv run python -m benchmarks.kernel.sample.bench_sample_tokens [--vocab-sizes 151936,248320]
        [--batch-sizes 1,4,8]

Each call samples one token per row of bf16 logits through Sampler.sample with the
Qwen3.6 recommendation (top_k 20, top_p 0.95; temperature 0.7 here), random draw included.
Calls chain as decode steps do: each call's temperatures wait on the previous call's tokens
(temperature + (tokens < 0), a two-op chain whose own time is measured alone and taken off),
so a run of --calls calls is GPU-bound. Each cell is the best of --repeats runs, in
microseconds per call. Dispatches per call come from running this under the mtlstat dylib.
"""

from __future__ import annotations

import argparse
import sys
import time

import mlx.core as mx

from ftmetal.engine.sample import BatchSamplingArgs, Sampler

TEMPERATURE = 0.7
TOP_K = 20
TOP_P = 0.95


def chained_seconds(
    sampler: Sampler | None, logits: mx.array, args: BatchSamplingArgs, calls: int
) -> float:
    """Wall seconds of calls chained through the temperatures; sampler None times the chain."""
    assert args.temperatures is not None
    temperatures = args.temperatures
    tokens = mx.zeros((logits.shape[0],), dtype=mx.int32)
    started = time.perf_counter()
    for _ in range(calls):
        if sampler is not None:
            step_args = BatchSamplingArgs(temperatures, args.top_k, args.top_p)
            tokens = sampler.sample(logits, step_args)
        temperatures = temperatures + (tokens < 0).astype(mx.float32)
    mx.eval(temperatures)
    return time.perf_counter() - started


def best_us_per_call(
    sampler: Sampler | None, logits: mx.array, args: BatchSamplingArgs, calls: int, repeats: int
) -> float:
    chained_seconds(sampler, logits, args, calls)
    best = min(chained_seconds(sampler, logits, args, calls) for _ in range(repeats))
    return best / calls * 1e6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-sizes", default="151936,248320")
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    mx.random.seed(0)
    print("microseconds per chained call, T 0.7 / top_k 20 / top_p 0.95: MLX ops | Metal kernels")
    for vocab_size in [int(v) for v in args.vocab_sizes.split(",")]:
        for batch_size in [int(b) for b in args.batch_sizes.split(",")]:
            logits = (3.0 * mx.random.normal((batch_size, vocab_size))).astype(mx.bfloat16)
            sampling = BatchSamplingArgs(
                temperatures=mx.full((batch_size,), TEMPERATURE, dtype=mx.float32),
                top_k=mx.full((batch_size,), TOP_K, dtype=mx.int32),
                top_p=mx.full((batch_size,), TOP_P, dtype=mx.float32),
            )
            mx.eval(logits, sampling.temperatures)
            chain = best_us_per_call(None, logits, sampling, args.calls, args.repeats)
            timings = [
                best_us_per_call(
                    Sampler(vocab_size, use_kernel=use_kernel),
                    logits,
                    sampling,
                    args.calls,
                    args.repeats,
                )
                - chain
                for use_kernel in (False, True)
            ]
            cells = " | ".join(f"{us:8.1f}" for us in timings)
            print(f"V={vocab_size:6d} B={batch_size}: {cells}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
