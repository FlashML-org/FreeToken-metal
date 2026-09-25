"""Prefill tok/s of one request at several prompt depths.

    uv run python tools/bench_prefill.py /path/to/Qwen3.6-35B-A3B-MLX \\
        [--depths 2048,8192] [--repeats 3]

The model loads once and a warm-up prompt compiles the kernels; each depth then runs
fresh random prompts of that many tokens with max_tokens 1 (no prefix-cache hits) and
reports the best wall time of the whole prefill, every chunk plus the first sampled
token, over the repeats.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.scheduler.scheduler import Scheduler

WARMUP_TOKENS = 2048


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--depths", default="2048,8192")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    depths = [int(depth) for depth in args.depths.split(",")]

    config = EngineConfig(
        model_path=args.model_path, max_running_req=1, max_seq_len_override=max(depths) + 1
    )
    scheduler = Scheduler(config, eos_token_ids=set())
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    rng = np.random.default_rng(0)
    vocab_size = config.model_config.vocab_size
    scheduler.generate(rng.integers(0, vocab_size, size=WARMUP_TOKENS, dtype=np.int32), params)
    for depth in depths:
        seconds = []
        for _ in range(args.repeats):
            prompt = rng.integers(0, vocab_size, size=depth, dtype=np.int32)
            started = time.perf_counter()
            scheduler.generate(prompt, params)
            seconds.append(time.perf_counter() - started)
        print(
            f"depth {depth}: {depth / min(seconds):.1f} tok/s prefill "
            f"(best {min(seconds) * 1000.0:.0f} ms, runs "
            + ", ".join(f"{s * 1000.0:.0f}" for s in seconds)
            + " ms)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
