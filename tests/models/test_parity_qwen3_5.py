"""Numeric parity of the paged engine against mlx-lm on a tiny Qwen3.5 MoE model.

The reference (mlx-lm, contiguous caches) and ftmetal (paged KV, GDN state pool,
scheduler, overlap) run the same random weights: prefill logits must match and
greedy decode must agree token for token, also for a chat turn resumed from the
snapshot the previous turn left below its generation suffix.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from ftmetal.core import SamplingParams
from ftmetal.engine.config import CacheTypeChoice, EngineConfig
from ftmetal.message.backend import UserMsg
from ftmetal.message.tokenizer import ChatMessage, TokenizeMsg
from ftmetal.scheduler.scheduler import Scheduler
from ftmetal.tokenizer import TokenizeManager
from tiny_model import TINY_TEXT_CONFIG, build_reference, reference_greedy, save_checkpoint


def make_scheduler(
    path: Path,
    page_size: int,
    max_extend_tokens: int = 64,
    attention_backend: str = "mlx",
    num_pages: int = 64,
    max_seq_len: int | None = None,
    cache_type: CacheTypeChoice = "radix",
) -> Scheduler:
    config = EngineConfig(
        model_path=str(path),
        dtype=mx.float32,
        max_running_req=2,
        page_size=page_size,
        attention_backend=attention_backend,
        max_seq_len_override=max_seq_len,
        num_page_override=num_pages,
        max_extend_tokens=max_extend_tokens,
        cache_type=cache_type,
    )
    return Scheduler(config, eos_token_ids={127})


def last_chunk_logits(scheduler: Scheduler, prompt: list[int]) -> mx.array:
    """Logits of the prompt's last prefill chunk; the scheduler runs the chunks before it."""
    scheduler.add_request(0, np.array(prompt, dtype=np.int32), SamplingParams(max_tokens=1))
    while True:
        forward_input = scheduler.schedule_next_batch()
        assert forward_input is not None
        if forward_input.batch.reqs[0].can_decode:
            return scheduler.engine.model.forward(forward_input.batch)
        scheduler.process_last_data(scheduler.forward(forward_input))


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
@pytest.mark.parametrize("page_size", [1, 4])
def test_greedy_decode_matches_mlx_lm(
    tmp_path: Path, page_size: int, attention_backend: str
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


# A short prefill runs the per-token GDN path, a long one the chunked path.
@pytest.mark.parametrize(("prompt_len", "max_extend_tokens"), [(21, 64), (200, 256)])
def test_prefill_logits_match_mlx_lm(
    tmp_path: Path, prompt_len: int, max_extend_tokens: int
) -> None:
    model = build_reference(seed=prompt_len)
    save_checkpoint(model, tmp_path, quantization=None)
    prompt = [int(t) for t in np.random.default_rng(prompt_len).integers(1, 120, size=prompt_len)]
    expected_logits, _ = reference_greedy(model, prompt, 1)

    scheduler = make_scheduler(tmp_path, page_size=4, max_extend_tokens=max_extend_tokens)
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


# One 200-token prefill (chunked GDN path, ragged last chunk), or 160 chunked then 40
# on the per-token path.
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


# A prompt past the 2048-token chunk budget: the metal backend's flash-prefill reads each
# chunk's prefix from the pool, the MLX backend gathers it for SDPA.
LONG_PROMPT_LEN = 5000
LONG_PROMPT_CHUNK = 2048


def test_chunked_long_prefill_metal_matches_mlx_backend(tmp_path: Path) -> None:
    model = build_reference(seed=7)
    save_checkpoint(model, tmp_path, quantization=None)
    prompt = [int(t) for t in np.random.default_rng(7).integers(1, 120, size=LONG_PROMPT_LEN)]
    steps = 8
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    logits: dict[str, mx.array] = {}
    tokens: dict[str, list[int]] = {}
    for backend in ("mlx", "metal"):
        prefill_scheduler, decode_scheduler = (
            make_scheduler(
                tmp_path,
                page_size=1,
                max_extend_tokens=LONG_PROMPT_CHUNK,
                attention_backend=backend,
                num_pages=LONG_PROMPT_LEN + steps,
                max_seq_len=LONG_PROMPT_LEN + steps,
            )
            for _ in range(2)
        )
        logits[backend] = last_chunk_logits(prefill_scheduler, prompt)
        tokens[backend] = decode_scheduler.generate(np.array(prompt, dtype=np.int32), params)
    assert mx.allclose(logits["metal"], logits["mlx"], atol=1e-4, rtol=1e-4)
    assert tokens["metal"] == tokens["mlx"]


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
def test_two_requests_batched_match_single(tmp_path: Path, attention_backend: str) -> None:
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


def test_naive_cache_matches_mlx_lm(tmp_path: Path) -> None:
    """--cache-type naive: one live GDN slot per request and no prefix reuse. Two prompts
    decode together on the pool's two live slots; a chunked third prompt then prefills on a
    slot a finished request left dirty, and its logits must still be mlx-lm's."""
    model = build_reference(seed=6)
    save_checkpoint(model, tmp_path, quantization=None)
    rng = np.random.default_rng(6)
    prompts = [[int(t) for t in rng.integers(1, 120, size=n)] for n in (13, 29)]
    steps = 12
    expected = [reference_greedy(model, prompt, steps)[1] for prompt in prompts]

    scheduler = make_scheduler(tmp_path, page_size=4, attention_backend="metal", cache_type="naive")
    pool = scheduler.engine.ctx.linear_state_pool
    assert scheduler.cache_manager.cache_type == "naive"
    assert pool is not None and pool.num_slots == 3
    for uid, prompt in enumerate(prompts):
        params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
        scheduler.add_request(uid, np.array(prompt, dtype=np.int32), params)
    events = scheduler.run_until_idle()
    got = [[e.token for e in events if e.uid == uid] for uid in range(len(prompts))]
    assert got == expected
    scheduler.cache_manager.check_integrity()

    third = prompts[1] + [int(t) for t in rng.integers(1, 120, size=70)]
    logits = last_chunk_logits(scheduler, third)
    expected_logits, _ = reference_greedy(model, third, 1)
    assert mx.allclose(logits[0], expected_logits[-1], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("attention_backend", ["mlx", "metal"])
def test_quantized_greedy_decode_matches_mlx_lm(tmp_path: Path, attention_backend: str) -> None:
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


def run_turn(
    scheduler: Scheduler, uid: int, user_msg: UserMsg, steps: int
) -> tuple[int, list[int]]:
    """One tokenized turn to completion; returns the prefix its admission found cached and its
    greedy tokens."""
    params = SamplingParams(temperature=0.0, max_tokens=steps, ignore_eos=True)
    scheduler.add_request(uid, user_msg.input_ids, params, user_msg.reuse_len)
    forward_input = scheduler.schedule_next_batch()
    assert forward_input is not None
    cached_len = forward_input.batch.reqs[0].cached_len
    data = scheduler.forward(forward_input)
    while scheduler.runnable or data is not None:
        data = scheduler.overlap_loop(data)
    return cached_len, [event.token for event in scheduler.take_results()]


def test_next_chat_turn_resumes_below_the_generation_suffix(tiny_model_dir: Path) -> None:
    """The first turn's 129 tokens end with the 13-token generation suffix, holding the
    stride boundary at 128; the next turn re-renders the conversation and shares 127. The
    snapshot stays at 64, below the reusable 116, so the next turn resumes there and still
    decodes to mlx-lm's tokens (a snapshot at 128 would leave it nothing to resume from)."""
    model = build_reference(seed=8, text_config={**TINY_TEXT_CONFIG, "vocab_size": 512})
    save_checkpoint(model, tiny_model_dir, quantization=None)
    tokenize_manager = TokenizeManager(str(tiny_model_dir))
    first: list[ChatMessage] = [{"role": "user", "content": "abcd" * 27}]
    second: list[ChatMessage] = [
        *first,
        {"role": "assistant", "content": "a short reply"},
        {"role": "user", "content": "and a follow-up question"},
    ]
    user_msgs = [
        tokenize_manager.tokenize(TokenizeMsg(uid=uid, text=turn, sampling_params=SamplingParams()))
        for uid, turn in enumerate((first, second))
    ]
    assert (len(user_msgs[0].input_ids), user_msgs[0].reuse_len) == (129, 116)
    steps = 8
    scheduler = make_scheduler(tiny_model_dir, page_size=4, max_extend_tokens=256, num_pages=256)
    for uid, (user_msg, cached_len) in enumerate(zip(user_msgs, (0, 64))):
        expected = reference_greedy(model, user_msg.input_ids.tolist(), steps)[1]
        assert run_turn(scheduler, uid, user_msg, steps) == (cached_len, expected)
    scheduler.cache_manager.check_integrity()
