"""Decode tok/s at batch size 1 after prompts of several context depths.

    uv run python tools/bench_decode_depth.py /path/to/Qwen3.6-35B-A3B-MLX \\
        [--attention-backend metal] [--depths 1024,8192,32768] [--decode-steps 64]

The model loads once; each depth then runs one greedy request on a random prompt of
that many tokens and reports the decode rate over its steps after the warm-up, with
the mean wall time per step. Run one backend per process.
"""

from __future__ import annotations

import argparse
import sys
import time
import typing

import numpy as np

from ftmetal.core import SamplingParams
from ftmetal.engine.config import AttentionBackendName, EngineConfig
from ftmetal.scheduler.scheduler import Scheduler

# The first decode steps compile kernels and fill the allocator cache.
WARMUP_STEPS = 8


def decode_step_seconds(
    scheduler: Scheduler, prompt: np.ndarray, params: SamplingParams
) -> list[float]:
    """Run one prompt to completion; the wall time of each decode iteration of the loop."""
    scheduler.add_request(0, prompt, params)
    data = None
    seconds: list[float] = []
    while scheduler.runnable or data is not None:
        started = time.perf_counter()
        data = scheduler.overlap_loop(data)
        if data is not None and data.forward_input.batch.is_decode:
            seconds.append(time.perf_counter() - started)
    scheduler.take_results()
    return seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument(
        "--attention-backend", choices=typing.get_args(AttentionBackendName), default="metal"
    )
    parser.add_argument("--depths", default="1024,8192,32768")
    parser.add_argument("--decode-steps", type=int, default=64)
    args = parser.parse_args()
    depths = [int(depth) for depth in args.depths.split(",")]

    config = EngineConfig(
        model_path=args.model_path,
        max_running_req=1,
        attention_backend=args.attention_backend,
        max_seq_len_override=max(depths) + args.decode_steps,
    )
    scheduler = Scheduler(config, eos_token_ids=set())
    params = SamplingParams(temperature=0.0, max_tokens=args.decode_steps, ignore_eos=True)
    rng = np.random.default_rng(0)
    for depth in depths:
        prompt = rng.integers(0, config.model_config.vocab_size, size=depth, dtype=np.int32)
        steps = decode_step_seconds(scheduler, prompt, params)[WARMUP_STEPS:]
        tokens_per_second = len(steps) / sum(steps)
        print(
            f"{args.attention_backend} depth {depth}: {tokens_per_second:.1f} tok/s over "
            f"{len(steps)} steps ({1000.0 * sum(steps) / len(steps):.2f} ms per step)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
