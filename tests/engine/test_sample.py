"""Sampler contract: greedy argmax, top-k / top-p masks, per-row temperatures, both paths."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.core import Batch, Req, SamplingParams
from ftmetal.engine.sample import Sampler, apply_top_k, apply_top_p


def batch_of(params: list[SamplingParams]) -> Batch:
    reqs = [
        Req(np.array([1, 2, 3], dtype=np.int32), i, 0, p.max_tokens, i, p)
        for i, p in enumerate(params)
    ]
    return Batch(reqs=reqs, phase="decode")


def test_greedy_is_argmax() -> None:
    sampler = Sampler(vocab_size=8)
    logits = mx.random.normal((3, 8))
    args = sampler.prepare(batch_of([SamplingParams()] * 3))
    assert args.temperatures is None
    assert mx.array_equal(sampler.sample(logits, args), mx.argmax(logits, axis=-1))


def test_temperature_zero_is_greedy_whatever_top_p_and_top_k() -> None:
    params = [
        SamplingParams(temperature=0.0, top_k=20, top_p=0.95),
        SamplingParams(temperature=0.7, top_k=1, top_p=0.5),
    ]
    assert all(p.is_greedy for p in params)
    assert Sampler(vocab_size=8).prepare(batch_of(params)).temperatures is None


def test_top_k_keeps_exactly_k_candidates() -> None:
    logits = mx.array([[1.0, 5.0, 3.0, 4.0, 2.0], [0.0, 0.0, 9.0, 1.0, 2.0]])
    masked = apply_top_k(logits, mx.array([2, 1], dtype=mx.int32))
    kept = (masked > -mx.inf).tolist()
    assert kept == [[False, True, False, True, False], [False, False, True, False, False]]


def test_top_p_keeps_smallest_prefix_reaching_p() -> None:
    logits = mx.log(mx.array([[0.5, 0.3, 0.15, 0.05]]))
    masked = apply_top_p(logits, mx.array([0.6], dtype=mx.float32))
    assert (masked > -mx.inf).tolist() == [[True, True, False, False]]
    masked = apply_top_p(logits, mx.array([0.9], dtype=mx.float32))
    assert (masked > -mx.inf).tolist() == [[True, True, True, False]]


@pytest.mark.parametrize("use_kernel", [True, False])
def test_mixed_batch_samples_only_from_allowed_tokens(use_kernel: bool) -> None:
    sampler = Sampler(vocab_size=6, use_kernel=use_kernel)
    params = [
        SamplingParams(temperature=1.0, top_k=1),
        SamplingParams(temperature=0.7, top_p=0.5),
        SamplingParams(),
    ]
    args = sampler.prepare(batch_of(params))
    logits = mx.array([[0.0, 8.0, 0.0, 0.0, 0.0, 0.0]] * 3)
    for _ in range(5):
        assert sampler.sample(logits, args).tolist() == [1, 1, 1]


def test_env_selects_the_mlx_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FTMETAL_SAMPLE_KERNEL", "FREETOKEN_SAMPLE_KERNEL"):
        monkeypatch.delenv(name, raising=False)
    assert Sampler(vocab_size=6).use_kernel
    monkeypatch.setenv("FTMETAL_SAMPLE_KERNEL", "0")
    assert not Sampler(vocab_size=6).use_kernel


# (presence and frequency penalties, greedy or nucleus, tokens over four steps of the same
# logits [4, 3, 0, 0, 0, 0])
PENALTY_CASES: list[tuple[float, float, bool, list[int]]] = [
    (0.0, 0.0, True, [0, 0, 0, 0]),
    # Every repeat costs 1.5 more, so the two leaders alternate.
    (0.0, 1.5, True, [0, 1, 0, 1]),
    # One flat cost per token seen: once both leaders paid it, the first leads again.
    (3.0, 0.0, True, [0, 1, 0, 0]),
    # A negative penalty rewards repeats.
    (0.0, -2.0, True, [0, 0, 0, 0]),
    (0.0, 1.5, False, [0, 1, 0, 1]),
]


@pytest.mark.parametrize("use_kernel", [True, False])
@pytest.mark.parametrize(("presence", "frequency", "greedy", "expected"), PENALTY_CASES)
def test_penalties_count_the_tokens_sampled_so_far(
    use_kernel: bool, presence: float, frequency: float, greedy: bool, expected: list[int]
) -> None:
    params = SamplingParams(
        temperature=0.0 if greedy else 1.0,
        top_p=1.0 if greedy else 0.01,
        presence_penalty=presence,
        frequency_penalty=frequency,
    )
    batch = batch_of([params])
    sampler = Sampler(vocab_size=6, use_kernel=use_kernel)
    logits = mx.array([[4.0, 3.0, 0.0, 0.0, 0.0, 0.0]])
    tokens = [sampler.sample(logits, sampler.prepare(batch)).item() for _ in expected]
    assert tokens == expected
    counts = batch.reqs[0].output_token_counts
    assert (counts is None) == (presence == frequency == 0.0)
    if counts is not None:
        assert counts.tolist() == np.bincount(expected, minlength=6).tolist()
