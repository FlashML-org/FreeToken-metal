"""Greedy agreement with mlx-lm on the real Qwen3.6 checkpoint.

Needs FTMETAL_TEST_MODEL_PATH pointing at an MLX Qwen3.6-35B-A3B checkpoint; skipped
otherwise. bf16 free-running greedy trajectories legitimately diverge after a few
dozen tokens (mlx-lm's own incremental decode differs from its full forward), so the
check is teacher forced: mlx-lm's top-1 at every position of our sequence must equal
our token, except where its top two logits tie. FTMETAL_TEST_ATTENTION_BACKEND (mlx or
metal, default mlx) picks the full-attention backend; one real engine fits per process,
since the global Context keeps the previous engine's pools alive.
"""

from __future__ import annotations

import gc
import os

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm import load

from ftmetal.core import SamplingParams, host_floats, host_ints
from ftmetal.engine.config import AttentionBackendName, EngineConfig
from ftmetal.scheduler.scheduler import Scheduler
from ftmetal.tokenizer import TokenizeManager, load_eos_token_ids

MODEL_PATH = os.environ.get("FTMETAL_TEST_MODEL_PATH")
ATTENTION_BACKENDS: dict[str, AttentionBackendName] = {"mlx": "mlx", "metal": "metal"}
ATTENTION_BACKEND = ATTENTION_BACKENDS[os.environ.get("FTMETAL_TEST_ATTENTION_BACKEND", "mlx")]
PROMPT = "The capital of France is"
STEPS = 32
TIE_MARGIN = 1e-3

pytestmark = pytest.mark.skipif(MODEL_PATH is None, reason="FTMETAL_TEST_MODEL_PATH is not set")


def test_greedy_matches_mlx_lm_teacher_forced() -> None:
    assert MODEL_PATH is not None
    tokenizer = TokenizeManager(MODEL_PATH).tokenizer
    eos = load_eos_token_ids(MODEL_PATH, tokenizer)
    prompt = tokenizer.encode(PROMPT, add_special_tokens=False).ids

    config = EngineConfig(
        model_path=MODEL_PATH, attention_backend=ATTENTION_BACKEND, max_seq_len_override=4096
    )
    scheduler = Scheduler(config, eos)
    params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
    ours = scheduler.generate(np.array(prompt, dtype=np.int32), params)
    assert len(ours) == STEPS
    decode_steps = [s.iteration_ms for s in scheduler.metrics.steps if s.phase == "decode"]
    assert scheduler.metrics.decode_tokens_per_second(last=16) > 40.0, decode_steps
    del scheduler
    gc.collect()
    mx.clear_cache()

    reference = load(MODEL_PATH)[0]
    sequence = prompt + ours[:-1]
    logits = reference(mx.array(sequence)[None])[0, len(prompt) - 1 :].astype(mx.float32)
    top1 = host_ints(mx.argmax(logits, axis=-1))
    sorted_logits = mx.sort(logits, axis=-1)
    margins = host_floats(sorted_logits[:, -1] - sorted_logits[:, -2])
    mismatches = [(i, ours[i], top1[i], margins[i]) for i in range(STEPS) if ours[i] != top1[i]]
    assert all(margin < TIE_MARGIN for _, _, _, margin in mismatches), mismatches
    assert len(mismatches) <= STEPS // 8, mismatches
