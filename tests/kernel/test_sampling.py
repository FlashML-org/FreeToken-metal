"""Metal sampling kernels against a float64 numpy reference with FreeToken's semantics.

Every temperature x top-k x top-p combination is drawn at 16-bit (bf16) and 32-bit (fp32)
keys, over one chunk of a small vocabulary and many chunks of Qwen's, at batch sizes that
change the chunking; then each filter alone, any chunk count, ties at both boundaries,
-inf logits and the drawn distribution.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel._ext import _C
from ftmetal.kernel.sample import sample_tokens

TEMPERATURES = [0.1, 0.7, 1.0, 2.0]
# 0 is off; so is any k >= V.
TOP_KS = [1, 2, 20, 256, 0]
TOP_PS = [0.1, 0.5, 0.95, 1.0]
ROW_PARAMS = list(itertools.product(TEMPERATURES, TOP_KS, TOP_PS))
# The kernel sums in fp32: a boundary this close (relative to the kept mass) to the
# decision may fall on either side of it.
BOUNDARY_TOLERANCE = 1e-5
UNIFORM_STEPS = 1 << 24


@dataclass(frozen=True)
class RowParams:
    temperature: float
    top_k: int
    top_p: float


@dataclass(frozen=True)
class ReferenceDraw:
    token: int
    # The token plus the ones a decision within BOUNDARY_TOLERANCE of a boundary reaches.
    acceptable: set[int]
    kept: np.ndarray
    probabilities: np.ndarray


def inverse_cdf(kept_weights: np.ndarray, uniform: float) -> tuple[int, set[int]]:
    """First token whose cumulative kept weight exceeds uniform * total, and its near misses."""
    cumulative = np.cumsum(kept_weights)
    target = uniform * cumulative[-1]
    token = int(np.searchsorted(cumulative, target, side="right"))
    tolerance = BOUNDARY_TOLERANCE * cumulative[-1]
    heavy = np.nonzero(kept_weights > 0)[0]
    before, after = heavy[heavy < token], heavy[heavy > token]
    acceptable = {token}
    if before.size and abs(cumulative[token - 1] - target) < tolerance:
        acceptable.add(int(before.max()))
    if after.size and abs(cumulative[token] - target) < tolerance:
        acceptable.add(int(after.min()))
    return token, acceptable


def reference_draw(logits: np.ndarray, params: RowParams, uniform: float) -> ReferenceDraw:
    """FreeToken's draw of one row: exact top-k ties, top-p over the top-k mass, inverse CDF."""
    vocab = logits.size
    row_max = logits.max()
    with np.errstate(invalid="ignore", over="ignore"):
        weights = np.where(logits == row_max, 1.0, np.exp((logits - row_max) / params.temperature))
    weights = np.nan_to_num(weights, nan=0.0)
    kept = np.ones(vocab, dtype=bool)
    if 0 < params.top_k < vocab:
        kept = logits >= np.partition(logits, vocab - params.top_k)[vocab - params.top_k]
    thresholds = [-np.inf]
    acceptable: set[int] = set()
    if params.top_p < 1.0:
        values, inverse = np.unique(logits[kept], return_inverse=True)
        mass_per_value = np.bincount(inverse, weights=weights[kept])
        mass_at_or_above = np.cumsum(mass_per_value[::-1])[::-1]
        target = params.top_p * mass_at_or_above[0]
        index = int(np.nonzero(mass_at_or_above >= target)[0].max())
        tolerance = BOUNDARY_TOLERANCE * mass_at_or_above[0]
        thresholds = [values[index]]
        if index > 0 and abs(mass_at_or_above[index] - target) < tolerance:
            thresholds.append(values[index - 1])
        if index + 1 < values.size and abs(mass_at_or_above[index + 1] - target) < tolerance:
            thresholds.append(values[index + 1])
    for threshold in thresholds[1:]:
        acceptable |= inverse_cdf(np.where(kept & (logits >= threshold), weights, 0.0), uniform)[1]
    kept &= logits >= thresholds[0]
    kept_weights = np.where(kept, weights, 0.0)
    token, near = inverse_cdf(kept_weights, uniform)
    return ReferenceDraw(token, acceptable | near, kept, kept_weights / kept_weights.sum())


def uniform_bits(rng: np.random.Generator, rows: int) -> tuple[mx.array, np.ndarray]:
    """Random bits for the kernel and the uniforms (bits >> 8) / 2**24 they stand for."""
    steps = rng.integers(0, UNIFORM_STEPS, size=rows, dtype=np.uint32)
    return mx.array(steps << 8), steps.astype(np.float64) / UNIFORM_STEPS


def host_logits(logits: mx.array) -> np.ndarray:
    return np.array(logits.astype(mx.float32)).astype(np.float64)


def run_kernel(
    logits: mx.array,
    params: list[RowParams],
    bits: mx.array,
    *,
    with_top_k: bool = True,
    with_top_p: bool = True,
    num_groups: int | None = None,
) -> list[int]:
    temperatures = mx.array([p.temperature for p in params], dtype=mx.float32)
    top_k = mx.array([p.top_k for p in params], dtype=mx.int32) if with_top_k else None
    top_p = mx.array([p.top_p for p in params], dtype=mx.float32) if with_top_p else None
    if num_groups is None:
        tokens = sample_tokens(logits, temperatures, top_k, top_p, bits)
    else:
        tokens = _C.sample_tokens(logits, temperatures, top_k, top_p, num_groups, bits)
    assert tokens.dtype == mx.int32
    return [int(t) for t in np.array(tokens)]


def check_rows(
    logits: mx.array, params: list[RowParams], bits: mx.array, uniforms: np.ndarray, **kwargs
) -> None:
    tokens = run_kernel(logits, params, bits, **kwargs)
    rows = host_logits(logits)
    for row, (token, row_params) in enumerate(zip(tokens, params)):
        expected = reference_draw(rows[row], row_params, float(uniforms[row]))
        assert token in expected.acceptable, (row, row_params, token, expected.token)


@pytest.mark.parametrize(
    ("dtype", "vocab", "batch_size"),
    [
        (mx.bfloat16, 151936, 8),
        (mx.float32, 151936, 4),
        (mx.bfloat16, 1000, 1),
        (mx.float32, 1000, 8),
    ],
    ids=["bf16-V151936-bs8", "fp32-V151936-bs4", "bf16-V1000-bs1", "fp32-V1000-bs8"],
)
def test_matches_reference(dtype: mx.Dtype, vocab: int, batch_size: int) -> None:
    rng = np.random.default_rng(vocab + batch_size)
    for first in range(0, len(ROW_PARAMS), batch_size):
        params = [
            RowParams(*ROW_PARAMS[(first + row) % len(ROW_PARAMS)]) for row in range(batch_size)
        ]
        logits = mx.array(3.0 * rng.standard_normal((batch_size, vocab)), dtype=mx.float32)
        bits, uniforms = uniform_bits(rng, batch_size)
        check_rows(logits.astype(dtype), params, bits, uniforms)


@pytest.mark.parametrize(
    ("with_top_k", "with_top_p", "vocab"),
    [(True, False, 150001), (False, True, 1000), (False, False, 150001)],
)
def test_absent_filters_drop_their_rounds(with_top_k: bool, with_top_p: bool, vocab: int) -> None:
    rng = np.random.default_rng(vocab)
    batch_size = 8
    for first in range(0, len(ROW_PARAMS), batch_size):
        params = []
        for row in range(batch_size):
            temperature, top_k, top_p = ROW_PARAMS[(first + row) % len(ROW_PARAMS)]
            params.append(
                RowParams(temperature, top_k if with_top_k else 0, top_p if with_top_p else 1.0)
            )
        logits = mx.array(3.0 * rng.standard_normal((batch_size, vocab)), dtype=mx.bfloat16)
        bits, uniforms = uniform_bits(rng, batch_size)
        check_rows(logits, params, bits, uniforms, with_top_k=with_top_k, with_top_p=with_top_p)


@pytest.mark.parametrize("num_groups", [1, 3, 7, 256])
def test_any_chunking_draws_the_same(num_groups: int) -> None:
    rng = np.random.default_rng(num_groups)
    params = [RowParams(*ROW_PARAMS[i]) for i in range(0, len(ROW_PARAMS), 5)]
    logits = mx.array(2.0 * rng.standard_normal((len(params), 1000)), dtype=mx.bfloat16)
    bits, uniforms = uniform_bits(rng, len(params))
    check_rows(logits, params, bits, uniforms, num_groups=num_groups)


def test_boundary_ties_are_kept() -> None:
    vocab = 1000
    # 100 tokens tie at the top: top_k=20 and top_p=0.05 both keep all of them.
    logits = np.zeros(vocab, dtype=np.float32)
    tied = np.arange(3, vocab, 10)[:100]
    logits[tied] = 5.0
    rows = 64
    uniforms = (np.arange(rows) + 0.5) / rows
    bits = mx.array((uniforms * UNIFORM_STEPS).astype(np.uint32) << 8)
    batch = mx.array(np.tile(logits, (rows, 1)), dtype=mx.bfloat16)
    for top_k, top_p in [(20, 1.0), (0, 0.05), (20, 0.05)]:
        params = [RowParams(1.0, top_k, top_p)] * rows
        tokens = run_kernel(batch, params, bits)
        assert set(tokens) <= set(tied.tolist())
        assert max(tokens) > tied[19]
        check_rows(batch, params, bits, uniforms)


def test_negative_infinity_logits() -> None:
    vocab = 151936
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((4, vocab)).astype(np.float32)
    logits[0] = -np.inf
    logits[0, 12345] = 1.0
    logits[1, ::2] = -np.inf
    logits[2] = -np.inf
    logits[2, [7, vocab - 1]] = 0.0
    logits[3] = -np.inf
    batch = mx.array(logits, dtype=mx.bfloat16)
    for params in [RowParams(0.7, 20, 0.95), RowParams(1.0, 0, 1.0), RowParams(2.0, 1, 0.5)]:
        bits, uniforms = uniform_bits(rng, 4)
        tokens = run_kernel(batch, [params] * 4, bits)
        assert tokens[0] == 12345
        assert tokens[1] % 2 == 1
        assert tokens[2] in (7, vocab - 1)
        assert 0 <= tokens[3] < vocab
        check_rows(batch[:3], [params] * 3, bits[:3], uniforms[:3])


def test_draws_follow_the_kept_distribution() -> None:
    vocab = 64
    rng = np.random.default_rng(1)
    params = RowParams(0.7, 20, 0.95)
    logits = rng.standard_normal(vocab).astype(np.float32)
    rows = 1000
    batch = mx.array(np.tile(logits, (rows, 1)), dtype=mx.bfloat16)
    expected = reference_draw(host_logits(batch)[0], params, 0.0)
    counts = np.zeros(vocab)
    mx.random.seed(7)
    for _ in range(20):
        temperatures = mx.full((rows,), params.temperature, dtype=mx.float32)
        top_k = mx.full((rows,), params.top_k, dtype=mx.int32)
        top_p = mx.full((rows,), params.top_p, dtype=mx.float32)
        tokens = np.array(sample_tokens(batch, temperatures, top_k, top_p))
        counts += np.bincount(tokens, minlength=vocab)
    frequencies = counts / counts.sum()
    assert np.all(frequencies[~expected.kept] == 0)
    # 20k draws over at most 20 tokens: the total variation distance is ~0.01.
    assert 0.5 * np.abs(frequencies - expected.probabilities).sum() < 0.03
