"""Routine correctness check: AIME25 #0 with the checkpoint's recommended sampling.

Passes when every seed closes </think> and answers 70. Token-level agreement with other
engines is not the bar; the bar is that the model still solves the problem.

    uv run python tools/aime0_check.py /path/to/Qwen3.6-35B-A3B-MLX \
        [--seeds 1,2] [--max-tokens 32768]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_tokenizer

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.scheduler.scheduler import Scheduler

PROBLEM = "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$"
ANSWER = "70"


def final_answer(text: str) -> str | None:
    """Last \\boxed{} after </think>, else the last integer of the final answer."""
    if "</think>" not in text:
        return None
    final = text.split("</think>")[-1]
    boxed = re.findall(r"\\boxed\{([^}]*)\}", final)
    if boxed:
        return re.sub(r"[^0-9]", "", boxed[-1])
    numbers = re.findall(r"\d+", final.replace(",", ""))
    return numbers[-1] if numbers else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--max-tokens", type=int, default=32768)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    path = Path(args.model_path)

    tok = load_tokenizer(path)
    gen = json.load(open(path / "generation_config.json"))
    ids = tok.apply_chat_template(
        [{"role": "user", "content": PROBLEM}], add_generation_prompt=True, enable_thinking=True
    )
    params = SamplingParams(
        temperature=gen["temperature"],
        top_k=gen["top_k"],
        top_p=gen["top_p"],
        max_tokens=args.max_tokens,
    )
    mx.random.seed(seeds[0])
    config = EngineConfig(
        model_path=str(path),
        max_running_req=len(seeds),
        max_seq_len_override=len(ids) + args.max_tokens + 64,
    )
    sched = Scheduler(config, set(gen["eos_token_id"]))
    for uid, _ in enumerate(seeds):
        sched.add_request(uid, np.array(ids, dtype=np.int32), params)

    tokens: dict[int, list[int]] = {uid: [] for uid in range(len(seeds))}
    finish: dict[int, str | None] = {}
    t0 = time.perf_counter()
    data = None
    while sched.runnable or data is not None:
        data = sched.overlap_loop(data)
        for event in sched.take_results():
            tokens[event.uid].append(event.token)
            if event.finished:
                finish[event.uid] = event.finish_reason
    elapsed = time.perf_counter() - t0

    ok = True
    for uid, seed in enumerate(seeds):
        answer = final_answer(tok.decode(tokens[uid]))
        passed = answer == ANSWER
        ok &= passed
        print(
            f"seed {seed}: {len(tokens[uid])} tokens, finish={finish[uid]}, "
            f"answer={answer} (expected {ANSWER}) {'PASS' if passed else 'FAIL'}"
        )
    total = sum(len(t) for t in tokens.values())
    verdict = "PASS" if ok else "FAIL"
    print(f"{total} tokens in {elapsed:.0f}s, {total / elapsed:.1f} tok/s aggregate; {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
