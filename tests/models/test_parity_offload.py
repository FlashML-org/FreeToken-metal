"""Numeric parity of MoE offload against resident experts on a tiny Qwen3.5 MoE model.

One checkpoint (affine 4-bit experts, 16 per layer, top-4) is served with every expert
resident and from an offload cache of 2E + 8 slots, which misses and evicts at most steps:
the fp32 greedy tokens must agree and the logits of the first prompt's last prefill chunk
must match. Cases: decode replayed and eager; copy-miss and overlap prefills (the overlap
threshold knob lowered so tiny prompts take it), the chunks of one prompt on both sides of
the threshold, a prompt chunked through overlap prefills; two requests decoding together,
and three, which decode eagerly in two steps each since the cache cannot hold their worst
case in one; prefill overlap off, with a cache below 2E.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from ftmetal.core import SamplingParams
from ftmetal.engine.config import EngineConfig
from ftmetal.models.config import MoeStrategy
from ftmetal.scheduler.scheduler import Scheduler
from tiny_model import TINY_TEXT_CONFIG, build_reference, save_checkpoint

NUM_EXPERTS = 16
OFFLOAD_TEXT_CONFIG = {**TINY_TEXT_CONFIG, "num_experts": NUM_EXPERTS, "num_experts_per_tok": 4}
# Group 32: the tiny widths (64 and 96) are not multiples of 64.
EXPERT_GROUP_SIZE = 32
EXPERT_BITS = 4
CACHE_SIZE = 2 * NUM_EXPERTS + 8
STEPS = 10
# Absolute and relative: the paths differ in summation order only (a weighted sum against
# moe_combine, the sorted tile kernel against the per-row matvec).
LOGITS_TOLERANCE = 1e-4


@dataclass(frozen=True)
class Case:
    name: str
    prompt_lens: tuple[int, ...]
    max_extend_tokens: int = 64
    overlap_min_tokens: int = 128
    command_graph: bool = True
    prefill_overlap: bool = True
    cache_size: int = CACHE_SIZE
    # Whether some prefill reads layers ahead into the windows.
    reads_windows: bool = False


CASES = [
    Case("replayed-decode-copy-miss-prefill", (9,)),
    Case("eager-decode", (9,), command_graph=False),
    Case("overlap-prefill", (40,), overlap_min_tokens=16, reads_windows=True),
    # 64 tokens through the windows, then 36 looked up.
    Case("chunks-across-the-threshold", (100,), overlap_min_tokens=48, reads_windows=True),
    # 64, 64 and 22 tokens through the windows.
    Case("chunked-overlap-prefill", (150,), overlap_min_tokens=16, reads_windows=True),
    Case("two-requests", (13, 29)),
    Case("three-requests-split-decode", (9, 17, 26)),
    Case("overlap-off-small-cache", (40,), prefill_overlap=False, cache_size=NUM_EXPERTS + 8),
]


def is_expert_projection(module_path: str, module: nn.Module) -> bool:
    return "switch_mlp" in module_path and hasattr(module, "to_quantized")


def write_checkpoint(path: Path) -> None:
    """The reference model with its experts alone affine quantized."""
    model = build_reference(seed=11, text_config=OFFLOAD_TEXT_CONFIG)
    nn.quantize(
        model,
        group_size=EXPERT_GROUP_SIZE,
        bits=EXPERT_BITS,
        class_predicate=is_expert_projection,
    )
    mx.eval(model.parameters())
    quantization = {"group_size": EXPERT_GROUP_SIZE, "bits": EXPERT_BITS, "mode": "affine"}
    save_checkpoint(model, path, quantization=quantization)


def make_scheduler(path: Path, case: Case, strategy: MoeStrategy) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=max(2, len(case.prompt_lens)),
        page_size=4,
        num_page_override=128,
        max_extend_tokens=case.max_extend_tokens,
        command_graph=case.command_graph,
        moe_strategy=strategy,
        moe_cache_size=case.cache_size if strategy == "offload" else 0,
        moe_prefill_overlap=case.prefill_overlap,
    )
    return Scheduler(config, eos_token_ids={127})


def serve(
    path: Path, case: Case, strategy: MoeStrategy, prompts: list[list[int]]
) -> tuple[mx.array, list[list[int]], Scheduler]:
    """The logits of the first prompt's last prefill chunk, every prompt's greedy tokens
    decoded together, and the scheduler that decoded them."""
    scheduler = make_scheduler(path, case, strategy)
    scheduler.add_request(0, np.array(prompts[0], dtype=np.int32), SamplingParams(max_tokens=1))
    while True:
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        if forward_input.batch.reqs[0].can_decode:
            logits = scheduler.engine.model.forward(forward_input.batch)
            mx.eval(logits)
            break
        scheduler.process_last_data(scheduler.forward(forward_input))

    scheduler = make_scheduler(path, case, strategy)
    for uid, prompt in enumerate(prompts):
        params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events = scheduler.run_until_idle()
    scheduler.cache_manager.check_integrity()
    tokens = [[event.token for event in events if event.uid == uid] for uid in range(len(prompts))]
    return logits, tokens, scheduler


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_offload_matches_resident_experts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: Case
) -> None:
    monkeypatch.setenv("FTMETAL_MOE_PREFILL_OVERLAP_MIN_TOKENS", str(case.overlap_min_tokens))
    write_checkpoint(tmp_path)
    rng = np.random.default_rng(len(case.name))
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in case.prompt_lens]

    resident_logits, resident_tokens, _ = serve(tmp_path, case, "fused", prompts)
    logits, tokens, scheduler = serve(tmp_path, case, "offload", prompts)
    assert tokens == resident_tokens
    assert mx.allclose(logits, resident_logits, atol=LOGITS_TOLERANCE, rtol=LOGITS_TOLERANCE)

    cache = scheduler.engine.moe_offload_cache
    assert cache is not None and cache.cache_size == case.cache_size
    assert cache.fits_one_step(len(prompts)) == (len(prompts) < 3)
    stats = cache.stats()
    assert sum(stats.misses) > 0
    # A window read counts as an expert read but not as a miss.
    assert (sum(stats.experts_read) > sum(stats.misses)) == case.reads_windows
    cache.check_healthy()
