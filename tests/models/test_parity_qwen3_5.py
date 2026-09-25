"""Numeric parity of the paged engine against mlx-lm on a tiny Qwen3.5 MoE model.

The reference (mlx-lm, contiguous caches) and ftmetal (paged KV, GDN state pool,
scheduler, overlap) run the same random weights: prefill logits must match and
greedy decode must agree token for token.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from ftmetal.core import SamplingParams
from ftmetal.engine.config import AttentionBackendName, EngineConfig
from ftmetal.scheduler.scheduler import Scheduler
from tiny_model import build_reference, reference_greedy, save_checkpoint


def make_scheduler(
    path: Path,
    page_size: int,
    max_extend_tokens: int = 64,
    attention_backend: AttentionBackendName = "mlx",
) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=2,
        page_size=page_size,
        attention_backend=attention_backend,
        num_page_override=64,
        max_extend_tokens=max_extend_tokens,
    )
    return Scheduler(config, eos_token_ids={127})


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
@pytest.mark.parametrize("page_size", [1, 4])
def test_greedy_decode_matches_mlx_lm(
    tmp_path: Path, page_size: int, attention_backend: AttentionBackendName
) -> None:
    model = build_reference(seed=0)
    save_checkpoint(model, tmp_path, quantization=None)
    prompt = [int(t) for t in np.random.default_rng(0).integers(1, 120, size=37)]
    steps = 24
    _, expected = reference_greedy(model, prompt, steps)

    scheduler = make_scheduler(tmp_path, page_size, attention_backend=attention_backend)
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    tokens = scheduler.generate(np.array(prompt, dtype=np.int32), params)
    assert tokens == expected
    scheduler.cache_manager.check_integrity()


def test_prefill_logits_match_mlx_lm(tmp_path: Path) -> None:
    model = build_reference(seed=1)
    save_checkpoint(model, tmp_path, quantization=None)
    prompt = [int(t) for t in np.random.default_rng(1).integers(1, 120, size=21)]
    expected_logits, _ = reference_greedy(model, prompt, 1)

    scheduler = make_scheduler(tmp_path, page_size=4)
    scheduler.add_request(0, np.array(prompt, dtype=np.int32), SamplingParams(max_tokens=1))
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    logits = scheduler.engine.model.forward(forward_input.batch)
    assert mx.allclose(logits[0], expected_logits[-1], atol=1e-4, rtol=1e-4)


def test_chunked_prefill_matches_full_prefill(tmp_path: Path) -> None:
    model = build_reference(seed=2)
    save_checkpoint(model, tmp_path, quantization=None)
    prompt = [int(t) for t in np.random.default_rng(2).integers(1, 120, size=50)]
    steps = 8
    _, expected = reference_greedy(model, prompt, steps)

    scheduler = make_scheduler(tmp_path, page_size=4, max_extend_tokens=16)
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    tokens = scheduler.generate(np.array(prompt, dtype=np.int32), params)
    assert tokens == expected


def test_long_prefill_logits_match_mlx_lm(tmp_path: Path) -> None:
    model = build_reference(seed=6)
    save_checkpoint(model, tmp_path, quantization=None)
    prompt = [int(t) for t in np.random.default_rng(6).integers(1, 120, size=200)]
    expected_logits, _ = reference_greedy(model, prompt, 1)

    scheduler = make_scheduler(tmp_path, page_size=4, max_extend_tokens=256)
    scheduler.add_request(0, np.array(prompt, dtype=np.int32), SamplingParams(max_tokens=1))
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    logits = scheduler.engine.model.forward(forward_input.batch)
    assert mx.allclose(logits[0], expected_logits[-1], atol=1e-4, rtol=1e-4)


# 256 tokens: one 200-token prefill (chunked GDN path, ragged last chunk), or 160 chunked
# then 40 on the per-token path.
@pytest.mark.parametrize("max_extend_tokens", [256, 160])
def test_long_prefill_matches_mlx_lm(tmp_path: Path, max_extend_tokens: int) -> None:
    model = build_reference(seed=5)
    save_checkpoint(model, tmp_path, quantization=None)
    prompt = [int(t) for t in np.random.default_rng(5).integers(1, 120, size=200)]
    steps = 8
    _, expected = reference_greedy(model, prompt, steps)

    scheduler = make_scheduler(tmp_path, page_size=4, max_extend_tokens=max_extend_tokens)
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    tokens = scheduler.generate(np.array(prompt, dtype=np.int32), params)
    assert tokens == expected


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
def test_two_requests_batched_match_single(
    tmp_path: Path, attention_backend: AttentionBackendName
) -> None:
    model = build_reference(seed=3)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(3)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in (13, 29)]
    steps = 12
    expected = [reference_greedy(model, prompt, steps)[1] for prompt in prompts]

    scheduler = make_scheduler(tmp_path, page_size=4, attention_backend=attention_backend)
    for uid, prompt in enumerate(prompts):
        params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events = scheduler.run_until_idle()
    got = [[e.token for e in events if e.uid == uid] for uid in range(len(prompts))]
    assert got == expected
    scheduler.cache_manager.check_integrity()


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
def test_quantized_greedy_decode_matches_mlx_lm(
    tmp_path: Path, attention_backend: AttentionBackendName
) -> None:
    model = build_reference(seed=4)

    def predicate(path: str, module: nn.Module) -> bool:
        return hasattr(module, "to_quantized") and not path.endswith(
            ("mlp.gate", "shared_expert_gate")
        )

    nn.quantize(model, group_size=32, bits=4, class_predicate=predicate)
    mx.eval(model.parameters())
    save_checkpoint(model, tmp_path, quantization={"group_size": 32, "bits": 4, "mode": "affine"})
    prompt = [int(t) for t in np.random.default_rng(4).integers(1, 120, size=31)]
    steps = 16
    _, expected = reference_greedy(model, prompt, steps)

    scheduler = make_scheduler(tmp_path, page_size=4, attention_backend=attention_backend)
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    tokens = scheduler.generate(np.array(prompt, dtype=np.int32), params)
    assert tokens == expected
