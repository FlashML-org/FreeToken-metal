"""Agreement with mlx-lm on a real checkpoint, measured against float32.

Needs FTMETAL_TEST_MODEL_PATH pointing at an MLX checkpoint mlx-lm also loads (Qwen3.6 or
Gemma4); skipped otherwise. The engine answers one chat turn greedily in bfloat16, as served,
and records the logits of every token it samples. mlx-lm runs the same tokens through one
full forward in bfloat16 and one in float32. Each side's error is its mean KL from float32
over the generated positions, and ours may be at most MAX_ERROR_RATIO times mlx-lm's own: how
far bfloat16 rounding alone moves a model varies tenfold with the model and the text, so no
fixed bound or token-for-token check holds everywhere. A fault that persists across steps,
such as one layer's experts read in the wrong order, lands far past the ratio; a sparse one,
such as two experts swapped in one layer, stays inside the noise, and the tiny-model parity
tests catch those. FTMETAL_TEST_ATTENTION_BACKEND (metal or mlx, default metal) picks the
full-attention backend; one real engine fits per process, since the global Context keeps the
previous engine's pools alive.
"""

from __future__ import annotations

import gc
import os

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm import load

from ftmetal.attention import attention_backend_info, validate_attn_backend
from ftmetal.core import Batch, SamplingParams, host_floats, host_ints
from ftmetal.engine.config import EngineConfig
from ftmetal.engine.sample import BatchSamplingArgs
from ftmetal.scheduler.scheduler import Scheduler
from ftmetal.tokenizer import TokenizeManager, load_eos_token_ids

MODEL_PATH = os.environ.get("FTMETAL_TEST_MODEL_PATH")
ATTENTION_BACKEND = validate_attn_backend(os.environ.get("FTMETAL_TEST_ATTENTION_BACKEND", "metal"))
QUESTION = "Describe the main stages of the Apollo 11 mission, from launch to splashdown."
STEPS = 64
# Our mean KL from float32 over mlx-lm's own, on the same tokens: 0.7-1.6 measured 2026-10-26
# with mlx-lm 0.31.3 on Qwen3.6 (affine, OptiQ, UD 3-bit) and Gemma4-26B QAT, metal and mlx
# backends; 17 with one layer's experts shifted by one.
MAX_ERROR_RATIO = 3.0
# Slack for a turn both sides score almost exactly.
KL_FLOOR = 1e-4

pytestmark = pytest.mark.skipif(MODEL_PATH is None, reason="FTMETAL_TEST_MODEL_PATH is not set")


def engine_decode(
    tokenize_manager: TokenizeManager, prompt: list[int], monkeypatch: pytest.MonkeyPatch
) -> tuple[list[int], mx.array]:
    """Our greedy tokens and the float32 logits [STEPS, vocab] each was sampled from."""
    assert MODEL_PATH is not None
    config = EngineConfig(
        model_path=MODEL_PATH,
        attention_backend=ATTENTION_BACKEND,
        max_seq_len_override=4096,
        # Pools for this one turn: the global Context keeps them while mlx-lm runs.
        num_page_override=4096,
    )
    scheduler = Scheduler(config, load_eos_token_ids(MODEL_PATH, tokenize_manager.tokenizer))
    engine = scheduler.engine
    original_sample = engine.sampler.sample
    original_replay = engine.graph_runner.replay
    sampled_logits: list[mx.array] = []
    num_replays = 0

    def record(logits: mx.array, args: BatchSamplingArgs) -> mx.array:
        sampled_logits.append(logits[0].astype(mx.float32))
        return original_sample(logits, args)

    def count(batch: Batch) -> mx.array:
        nonlocal num_replays
        num_replays += 1
        return original_replay(batch)

    monkeypatch.setattr(engine.sampler, "sample", record)
    monkeypatch.setattr(engine.graph_runner, "replay", count)
    params = SamplingParams(temperature=0.0, max_tokens=STEPS, ignore_eos=True)
    scheduler.add_request(0, np.array(prompt, dtype=np.int32), params)
    data = None
    while scheduler.runnable or data is not None:
        data = scheduler.overlap_loop(data)
    tokens = [event.token for event in scheduler.take_results()]
    is_replayed = config.command_graph and attention_backend_info(ATTENTION_BACKEND).can_capture
    # The patches hold the sampler and the graph runner, and through them the engine.
    monkeypatch.undo()
    # Each prefill chunk is sampled too: the last STEPS rows are the tokens'.
    logits = mx.stack(sampled_logits[-STEPS:])
    assert len(tokens) == STEPS
    assert host_ints(mx.argmax(logits, axis=-1)) == tokens, "recorded logits are not the sampled"
    # Decode replays the captured graphs, as served, unless FTMETAL_COMMAND_GRAPH=0.
    assert num_replays == (STEPS - 1 if is_replayed else 0)
    return tokens, logits


def kl_from(exact_logits: mx.array, logits: mx.array) -> np.ndarray:
    """KL(exact || logits) of each row."""
    exact_log_probs = exact_logits - mx.logsumexp(exact_logits, axis=-1, keepdims=True)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    divergence = mx.sum(mx.exp(exact_log_probs) * (exact_log_probs - log_probs), axis=-1)
    return np.array(host_floats(divergence))


def test_decode_matches_mlx_lm_against_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    assert MODEL_PATH is not None
    tokenize_manager = TokenizeManager(MODEL_PATH)
    text = tokenize_manager.render_template([{"role": "user", "content": QUESTION}], None, {})
    prompt = tokenize_manager.tokenizer.encode(text, add_special_tokens=False).ids
    tokens, our_logits = engine_decode(tokenize_manager, prompt, monkeypatch)
    gc.collect()
    mx.clear_cache()

    reference = load(MODEL_PATH)[0]
    sequence = mx.array(prompt + tokens[:-1])[None]

    def reference_logits() -> mx.array:
        logits = reference(sequence)[0, len(prompt) - 1 :].astype(mx.float32)
        mx.eval(logits)
        return logits

    rounded_logits = reference_logits()
    reference.set_dtype(mx.float32)
    exact_logits = reference_logits()
    error = kl_from(exact_logits, our_logits)
    reference_error = kl_from(exact_logits, rounded_logits)
    exact_tokens = np.array(host_ints(mx.argmax(exact_logits, axis=-1)))
    report = (
        f"{ATTENTION_BACKEND}: mean KL from float32 {error.mean():.2e} (max {error.max():.2e} "
        f"at step {error.argmax()}), mlx-lm's {reference_error.mean():.2e} (max "
        f"{reference_error.max():.2e}); top-1 agreement {np.mean(exact_tokens == tokens):.3f}"
    )
    assert error.mean() <= MAX_ERROR_RATIO * reference_error.mean() + KL_FLOOR, report
