"""Sampler contract: greedy argmax, top-k / top-p masks, per-row temperatures."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

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


def test_mixed_batch_samples_only_from_allowed_tokens() -> None:
    sampler = Sampler(vocab_size=6)
    params = [
        SamplingParams(temperature=1.0, top_k=1),
        SamplingParams(temperature=0.7, top_p=0.5),
        SamplingParams(),
    ]
    args = sampler.prepare(batch_of(params))
    logits = mx.array([[0.0, 8.0, 0.0, 0.0, 0.0, 0.0]] * 3)
    for _ in range(5):
        assert sampler.sample(logits, args).tolist() == [1, 1, 1]
