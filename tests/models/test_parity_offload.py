"""Numeric parity of MoE offload against resident experts on a tiny Qwen3.5 MoE model.

One checkpoint (affine 4-bit experts, 16 per layer, top-4) is served with every expert
resident and from an offload cache of 52 slots (four runs of 10, two windows of 6), which
misses and evicts at most steps: the fp32 greedy tokens must agree and the logits of the
first prompt's last prefill chunk must match. A run of 10 holds two rows' experts, so at
most two requests run at once and a looked-up prefill takes two tokens per forward. Cases:
decode replayed and eager; copy-miss and overlap prefills (the overlap threshold knob
lowered so tiny prompts take it), the chunks of one prompt on both sides of the threshold,
a prompt chunked through overlap prefills; two requests decoding together, and three, of
which two run at once; prefill overlap off, with no windows; up in 8 bits (gate and up two
banks) on a replayed decode and an overlap prefill.
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
CACHE_SIZE = 52
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
    up_bits: int = EXPERT_BITS


CASES = [
    Case("replayed-decode-copy-miss-prefill", (9,)),
    Case("eager-decode", (9,), command_graph=False),
    Case("overlap-prefill", (40,), overlap_min_tokens=16, reads_windows=True),
    # 64 tokens through the windows, then 36 looked up.
    Case("chunks-across-the-threshold", (100,), overlap_min_tokens=48, reads_windows=True),
    # 64, 64 and 22 tokens through the windows.
    Case("chunked-overlap-prefill", (150,), overlap_min_tokens=16, reads_windows=True),
    Case("two-requests", (13, 29)),
    Case("three-requests", (9, 17, 26)),
    Case("overlap-off", (40,), prefill_overlap=False, cache_size=32),
    Case("up-8-bit-replayed-decode", (9,), up_bits=8),
    Case("up-8-bit-overlap-prefill", (40,), overlap_min_tokens=16, reads_windows=True, up_bits=8),
]


def is_expert_projection(module_path: str, module: nn.Module) -> bool:
    return "switch_mlp" in module_path and hasattr(module, "to_quantized")


def write_checkpoint(path: Path, up_bits: int) -> None:
    """The reference model with its experts alone affine quantized, up in up_bits."""
    model = build_reference(seed=11, text_config=OFFLOAD_TEXT_CONFIG)
    up = {"group_size": EXPERT_GROUP_SIZE, "bits": up_bits}

    def predicate(module_path: str, module: nn.Module) -> bool | dict[str, int]:
        if not is_expert_projection(module_path, module):
            return False
        return up if module_path.endswith("up_proj") else True

    nn.quantize(model, group_size=EXPERT_GROUP_SIZE, bits=EXPERT_BITS, class_predicate=predicate)
    mx.eval(model.parameters())
    quantization: dict[str, object] = {
        "group_size": EXPERT_GROUP_SIZE,
        "bits": EXPERT_BITS,
        "mode": "affine",
    }
    num_layers = OFFLOAD_TEXT_CONFIG["num_hidden_layers"]
    assert isinstance(num_layers, int)
    quantization.update(
        {f"language_model.model.layers.{i}.mlp.switch_mlp.up_proj": up for i in range(num_layers)}
    )
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
    write_checkpoint(tmp_path, case.up_bits)
    rng = np.random.default_rng(len(case.name))
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in case.prompt_lens]

    resident_logits, resident_tokens, _ = serve(tmp_path, case, "fused", prompts)
    logits, tokens, scheduler = serve(tmp_path, case, "offload", prompts)
    assert tokens == resident_tokens
    assert mx.allclose(logits, resident_logits, atol=LOGITS_TOLERANCE, rtol=LOGITS_TOLERANCE)

    cache = scheduler.engine.moe_offload_cache
    assert cache is not None and cache.cache_size == case.cache_size
    assert ("gate_up_proj.up_proj.weight" in cache.layout) == (case.up_bits != EXPERT_BITS)
    assert cache.max_rows == 2
    max_running = max(2, len(case.prompt_lens))
    assert len(scheduler.table_manager.free_slots) == min(max_running, cache.max_rows)
    stats = cache.stats()
    assert sum(stats.misses) > 0
    # A window read counts as an expert read but not as a miss.
    assert (sum(stats.experts_read) > sum(stats.misses)) == case.reads_windows
    cache.check_healthy()
