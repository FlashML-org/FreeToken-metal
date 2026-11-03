"""QSA kernels against plain MLX references.

Decode selection runs on shuffled pages (page sizes 1, 4 and 16, partial last pages, a
spare page per page-table row). Block scores must match sum_h relu(q_h . k_b) / sqrt(D) in
fp32 over the complete blocks, -inf past them; every index-key slot but a complete block's
last token holds NaN, so a read off those slots shows up. Top-k must pick the topk highest
of the complete blocks, ties toward the lower index, ascending, on hand-made scores with
runs of equal values (relu's zeros) and with fewer, as many or more complete blocks than
topk; the scores past the complete blocks are +inf, so reading them shows up. The key table
must list the selected blocks' and the tail block's slots. The window mean must match the
mx.mean ops bit for bit. QSA prefill must match fp32 masked SDPA where each query sees
random selected blocks plus its own incomplete block, causally, at Qwen3.8-Flash-Next's
attention shape (24 q heads, 2 kv heads, D 256, block 4) and a small one, on both GEMM
paths; the K/V of blocks no query sees and of slots past the keys hold NaN.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.kernel.attention.mha.mha_prefill import MhaPrefillMma, flash_prefill_qsa
from ftmetal.kernel.attention.qsa.qsa_indexer import (
    qsa_block_scores,
    qsa_key_table,
    qsa_topk,
    qsa_window_mean,
)

BLOCK_SIZE = 4
SPARE_PAGES = 2
# Worst |error| over a row relative to the row's largest |value|, by the kernel's output
# dtype. fp32 differs by summation order alone; the bf16 prefill output rounds half an ulp
# at the row's top binade (2^-8), one ulp leaving room for the fp32 summation order.
TOLERANCE = {mx.bfloat16: 2**-7, mx.float32: 1e-5}
# matmul2d needs Metal 4 (macOS 26).
MMAS = [
    "simdgroup",
    pytest.param(
        "mpp",
        marks=pytest.mark.skipif(
            int(platform.mac_ver()[0].split(".")[0]) < 26, reason="matmul2d needs macOS 26"
        ),
    ),
]


def assert_close(actual: mx.array, expected: mx.array) -> None:
    """actual matches the fp32 expected row by row (the last axis) to TOLERANCE[actual.dtype]
    of the row's largest |value|; expected's infinite entries must match exactly."""
    tolerance = TOLERANCE[actual.dtype]
    actual = actual.astype(mx.float32)
    infinite = mx.isinf(expected)
    assert not mx.isnan(actual).any().item()
    assert mx.array_equal(mx.where(infinite, actual, 0), mx.where(infinite, expected, 0))
    error = mx.abs(mx.where(infinite, 0, actual - expected)).max(axis=-1)
    scale = mx.abs(mx.where(infinite, 0, expected)).max(axis=-1)
    assert mx.all(error <= tolerance * scale).item()


def shuffled_page_table(
    seq_lens: tuple[int, ...], page_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """(page_table int32 [bs, width], num_pages): each request's pages shuffled over the pool,
    every row SPARE_PAGES wider than the longest request's pages, padded with a spare page."""
    pages_per_request = [-(-seq_len // page_size) for seq_len in seq_lens]
    num_pages = sum(pages_per_request) + SPARE_PAGES
    pages = rng.permutation(num_pages)
    width = max(pages_per_request) + SPARE_PAGES
    page_table = np.full((len(seq_lens), width), pages[-1], dtype=np.int32)
    start = 0
    for row, num_request_pages in enumerate(pages_per_request):
        page_table[row, :num_request_pages] = pages[start : start + num_request_pages]
        start += num_request_pages
    return page_table, num_pages


def token_slots(
    page_table: np.ndarray, row: int, positions: np.ndarray, page_size: int
) -> np.ndarray:
    """The pool slots of request row's key positions."""
    return page_table[row, positions // page_size] * page_size + positions % page_size


@dataclass(frozen=True)
class ScoresCase:
    """qsa_block_scores over requests of seq_lens keys; heads x head_dim index queries."""

    seq_lens: tuple[int, ...]
    page_size: int
    heads: int = 4
    head_dim: int = 128
    dtype: mx.Dtype = mx.bfloat16

    def __str__(self) -> str:
        lens = "+".join(map(str, self.seq_lens))
        dtype = "bf16" if self.dtype == mx.bfloat16 else "fp32"
        return f"{lens}-p{self.page_size}-H{self.heads}-D{self.head_dim}-{dtype}"


SCORES_CASES = [
    # Qwen3.8-Flash-Next's indexer (4 heads, D 128): an empty padding row, a partial block, a
    # block-aligned length; no complete block; partial last pages; a long context.
    ScoresCase((37, 0, 300), page_size=1),
    ScoresCase((5, 64, 129), page_size=4),
    ScoresCase((3, 17, 1000), page_size=16),
    ScoresCase((4096 + 3,), page_size=16),
    ScoresCase((61, 2048), page_size=4, heads=1, head_dim=64, dtype=mx.float32),
    ScoresCase((130,), page_size=1, heads=8, head_dim=256),
]


@pytest.mark.parametrize("case", SCORES_CASES, ids=str)
def test_block_scores(case: ScoresCase) -> None:
    rng = np.random.default_rng(sum(case.seq_lens))
    page_table, num_pages = shuffled_page_table(case.seq_lens, case.page_size, rng)
    num_blocks = page_table.shape[1] * case.page_size // BLOCK_SIZE
    is_block_key = np.zeros((num_pages * case.page_size, 1), dtype=bool)
    block_slots = np.zeros((len(case.seq_lens), num_blocks), dtype=np.int64)
    for row, seq_len in enumerate(case.seq_lens):
        last_positions = np.arange(num_blocks) * BLOCK_SIZE + BLOCK_SIZE - 1
        block_slots[row] = token_slots(page_table, row, last_positions, case.page_size)
        is_block_key[block_slots[row, : seq_len // BLOCK_SIZE]] = True
    mx.random.seed(len(case.seq_lens))
    shape = (num_pages * case.page_size, case.head_dim)
    index_keys = mx.where(mx.array(is_block_key), mx.random.normal(shape), mx.nan)
    index_keys = index_keys.astype(case.dtype)
    queries = mx.random.normal((len(case.seq_lens), case.heads, case.head_dim)).astype(case.dtype)
    seq_lens = mx.array(case.seq_lens, dtype=mx.int32)
    scores = qsa_block_scores(
        queries, index_keys, mx.array(page_table), seq_lens, case.page_size, BLOCK_SIZE, num_blocks
    )
    keys = index_keys[mx.array(block_slots)].astype(mx.float32)
    products = mx.einsum("bhd,bnd->bhn", queries.astype(mx.float32), keys)
    expected = mx.maximum(products, 0).sum(axis=1) * case.head_dim**-0.5
    is_complete = mx.arange(num_blocks)[None] < (seq_lens // BLOCK_SIZE)[:, None]
    expected = mx.where(is_complete, expected, -mx.inf)
    assert scores.shape == (len(case.seq_lens), num_blocks) and scores.dtype == mx.float32
    assert_close(scores, expected)


ScorePattern = Literal["distinct", "relu", "zeros", "steps"]


@dataclass(frozen=True)
class TopkCase:
    """qsa_topk over scores [len(seq_lens), num_blocks] of the given pattern."""

    seq_lens: tuple[int, ...]
    num_blocks: int
    topk: int
    pattern: ScorePattern

    def __str__(self) -> str:
        lens = "+".join(map(str, self.seq_lens))
        return f"{lens}-n{self.num_blocks}-k{self.topk}-{self.pattern}"

    def scores(self, rng: np.random.Generator) -> np.ndarray:
        """Scores of the pattern on the complete blocks, +inf past them."""
        shape = (len(self.seq_lens), self.num_blocks)
        if self.pattern == "distinct":
            scores = rng.permutation(np.prod(shape)).reshape(shape) / 7.0
        elif self.pattern == "relu":
            scores = np.maximum(rng.standard_normal(shape), 0.0)
        elif self.pattern == "zeros":
            scores = np.zeros(shape)
        else:
            scores = rng.integers(0, 4, shape) * 0.5
        complete = np.minimum(np.asarray(self.seq_lens) // BLOCK_SIZE, self.num_blocks)
        is_complete = np.arange(self.num_blocks)[None] < complete[:, None]
        return np.where(is_complete, scores, np.inf).astype(np.float32)


TOPK_CASES = [
    # More complete blocks than topk: distinct scores; relu's zeros tied across the cut
    # (about half the blocks are 0 and topk reaches into them); all tied; four levels.
    TopkCase((4 * 600 + 1,), num_blocks=700, topk=64, pattern="distinct"),
    TopkCase((4 * 600,), num_blocks=700, topk=400, pattern="relu"),
    TopkCase((4 * 600 + 3,), num_blocks=700, topk=64, pattern="zeros"),
    TopkCase((4 * 600,), num_blocks=640, topk=200, pattern="steps"),
    # As many complete blocks as topk, one more, fewer, none.
    TopkCase((4 * 64 + 3,), num_blocks=100, topk=64, pattern="distinct"),
    TopkCase((4 * 65,), num_blocks=100, topk=64, pattern="zeros"),
    TopkCase((4 * 10 + 2,), num_blocks=100, topk=64, pattern="distinct"),
    TopkCase((3,), num_blocks=100, topk=64, pattern="relu"),
    # Mixed rows at the real budget (512 blocks); many blocks per thread with ties spanning
    # threads; one block selected; a length past the score columns.
    TopkCase((0, 41, 2403, 4 * 5000), num_blocks=6000, topk=512, pattern="relu"),
    TopkCase((4 * 40000, 4 * 30000 + 1), num_blocks=40000, topk=512, pattern="steps"),
    TopkCase((4 * 300,), num_blocks=300, topk=1, pattern="steps"),
    TopkCase((4 * 900,), num_blocks=800, topk=100, pattern="distinct"),
]


def topk_ref(scores: np.ndarray, seq_lens: tuple[int, ...], topk: int) -> np.ndarray:
    """The topk highest complete blocks per row, ties to the lower index, ascending; every
    complete block then zeros while they number at most topk."""
    selected = np.zeros((len(seq_lens), topk), dtype=np.int32)
    for row, seq_len in enumerate(seq_lens):
        complete = min(seq_len // BLOCK_SIZE, scores.shape[1])
        if complete <= topk:
            selected[row, :complete] = np.arange(complete)
        else:
            order = np.lexsort((np.arange(complete), -scores[row, :complete]))
            selected[row] = np.sort(order[:topk])
    return selected


@pytest.mark.parametrize("case", TOPK_CASES, ids=str)
def test_topk(case: TopkCase) -> None:
    scores = case.scores(np.random.default_rng(case.num_blocks + case.topk))
    seq_lens = mx.array(case.seq_lens, dtype=mx.int32)
    selected = qsa_topk(mx.array(scores), seq_lens, case.topk, BLOCK_SIZE)
    assert selected.dtype == mx.int32
    np.testing.assert_array_equal(np.array(selected), topk_ref(scores, case.seq_lens, case.topk))


@dataclass(frozen=True)
class KeyTableCase:
    seq_lens: tuple[int, ...]
    page_size: int
    topk: int

    def __str__(self) -> str:
        return "+".join(map(str, self.seq_lens)) + f"-p{self.page_size}-k{self.topk}"


KEY_TABLE_CASES = [
    # Per row: more complete blocks than topk with a tail, without a tail; fewer; none (a
    # tail only); an empty padding row.
    KeyTableCase((2403, 2400, 41, 3, 0), page_size=1, topk=16),
    KeyTableCase((2403, 2400, 41, 3, 0), page_size=4, topk=16),
    KeyTableCase((2403, 2400, 41, 3, 0), page_size=16, topk=16),
    KeyTableCase((4 * 600 + 1, 4 * 512 + 2), page_size=16, topk=512),
    KeyTableCase((9, 6), page_size=4, topk=1),
]


@pytest.mark.parametrize("case", KEY_TABLE_CASES, ids=str)
def test_key_table(case: KeyTableCase) -> None:
    rng = np.random.default_rng(case.topk)
    page_table, num_pages = shuffled_page_table(case.seq_lens, case.page_size, rng)
    num_blocks = page_table.shape[1] * case.page_size // BLOCK_SIZE
    scores = TopkCase(case.seq_lens, num_blocks, case.topk, "relu").scores(rng)
    selected = topk_ref(scores, case.seq_lens, case.topk)
    key_slots, counts = qsa_key_table(
        mx.array(selected),
        mx.array(page_table),
        mx.array(case.seq_lens, dtype=mx.int32),
        case.page_size,
        BLOCK_SIZE,
    )
    assert key_slots.shape == (len(case.seq_lens), (case.topk + 1) * BLOCK_SIZE)
    key_slots, counts = np.array(key_slots), np.array(counts)
    for row, seq_len in enumerate(case.seq_lens):
        complete = seq_len // BLOCK_SIZE
        blocks = selected[row, : min(complete, case.topk)]
        positions = np.concatenate(
            [
                (blocks[:, None] * BLOCK_SIZE + np.arange(BLOCK_SIZE)).reshape(-1),
                np.arange(complete * BLOCK_SIZE, seq_len),
            ]
        )
        expected = token_slots(page_table, row, positions, case.page_size)
        assert counts[row] == len(expected)
        np.testing.assert_array_equal(key_slots[row, : counts[row]], expected)
    assert ((0 <= key_slots) & (key_slots < num_pages * case.page_size)).all()


@pytest.mark.parametrize(
    ("num_tokens", "window", "head_dim", "dtype", "scale"),
    [
        (1, 4, 128, mx.bfloat16, 1.0),
        (3, 4, 128, mx.bfloat16, 0.001),
        (37, 4, 128, mx.bfloat16, 30.0),
        (2, 4, 64, mx.bfloat16, 3000.0),
        (5, 4, 256, mx.bfloat16, 0.3),
        (9, 4, 128, mx.float16, 1.0),
        (6, 4, 128, mx.float32, 1.0),
    ],
)
def test_window_mean(
    num_tokens: int, window: int, head_dim: int, dtype: mx.Dtype, scale: float
) -> None:
    mx.random.seed(num_tokens * head_dim)
    index_keys = (scale * mx.random.normal((500, head_dim))).astype(dtype)
    window_slots = mx.random.randint(0, 500, (num_tokens, window)).astype(mx.int32)
    expected = mx.mean(index_keys[window_slots].astype(mx.float32), axis=1).astype(dtype)
    assert mx.array_equal(qsa_window_mean(index_keys, window_slots), expected).item()


@dataclass(frozen=True)
class PrefillCase:
    """flash_prefill_qsa of num_queries queries after cached_len keys; a query with more
    than topk complete blocks sees topk random ones, any other every block up to its own."""

    cached_len: int
    num_queries: int
    topk: int
    page_size: int = 1
    num_heads: int = 24
    num_kv_heads: int = 2
    head_dim: int = 256
    dtype: mx.Dtype = mx.bfloat16

    def __str__(self) -> str:
        label = f"{self.cached_len}:{self.num_queries}-k{self.topk}-p{self.page_size}"
        label += f"-H{self.num_heads}/{self.num_kv_heads}-D{self.head_dim}"
        return label + ("-bf16" if self.dtype == mx.bfloat16 else "-fp32")

    def visible(self, rng: np.random.Generator) -> np.ndarray:
        """bool [num_queries, num_blocks]: the selected blocks plus the incomplete own block,
        or every block up to the own one."""
        key_len = self.cached_len + self.num_queries
        visible = np.zeros((self.num_queries, -(-key_len // BLOCK_SIZE)), dtype=bool)
        for i in range(self.num_queries):
            position = self.cached_len + i
            complete = (position + 1) // BLOCK_SIZE
            if complete > self.topk:
                visible[i, rng.choice(complete, self.topk, replace=False)] = True
                # Empty when the own block is complete: it is seen only if selected.
                visible[i, complete : position // BLOCK_SIZE + 1] = True
            else:
                visible[i, : position // BLOCK_SIZE + 1] = True
        return visible


PREFILL_CASES = [
    # Qwen3.8-Flash-Next's attention (G 12: two queries a tile): all sparse with a one-query
    # last tile; dense queries turning sparse; from an empty prefix; a long prefix at the
    # real budget; page sizes 1, 4 and 16.
    PrefillCase(300, 37, topk=16),
    PrefillCase(50, 64, topk=16, page_size=16),
    PrefillCase(0, 21, topk=2, page_size=4),
    PrefillCase(2100, 64, topk=64, page_size=16),
    PrefillCase(3000, 16, topk=512, page_size=16),
    # A small shape (G 2: twelve queries a tile), bf16 and fp32.
    PrefillCase(100, 37, topk=8, num_heads=4, head_dim=64),
    PrefillCase(7, 50, topk=3, page_size=16, num_heads=4, head_dim=64, dtype=mx.float32),
]


@pytest.mark.parametrize("mma", MMAS)
@pytest.mark.parametrize("case", PREFILL_CASES, ids=str)
def test_prefill_matches_masked_sdpa(case: PrefillCase, mma: MhaPrefillMma) -> None:
    rng = np.random.default_rng(case.cached_len + case.num_queries)
    key_len = case.cached_len + case.num_queries
    page_table, num_pages = shuffled_page_table((key_len,), case.page_size, rng)
    visible = case.visible(rng)
    slots = token_slots(page_table, 0, np.arange(key_len), case.page_size)
    is_seen = np.repeat(visible.any(axis=0), BLOCK_SIZE)[:key_len]
    is_key = np.zeros((num_pages * case.page_size, 1, 1), dtype=bool)
    is_key[slots[is_seen]] = True
    shape = (num_pages * case.page_size, case.num_kv_heads, case.head_dim)
    mx.random.seed(key_len)
    k_cache = mx.where(mx.array(is_key), mx.random.normal(shape), mx.nan).astype(case.dtype)
    v_cache = mx.where(mx.array(is_key), mx.random.normal(shape), mx.nan).astype(case.dtype)
    q = mx.random.normal((case.num_queries, case.num_heads, case.head_dim)).astype(case.dtype)
    scale = case.head_dim**-0.5
    out = flash_prefill_qsa(
        q,
        k_cache,
        v_cache,
        mx.array(page_table[:1]),
        case.page_size,
        mx.array(visible),
        case.cached_len,
        BLOCK_SIZE,
        scale,
        mma,
    )
    positions = np.arange(case.cached_len, key_len)
    mask = np.repeat(visible, BLOCK_SIZE, axis=1)[:, :key_len]
    mask &= np.arange(key_len)[None] <= positions[:, None]
    # The NaN rows are masked for every query.
    keys = mx.nan_to_num(k_cache[mx.array(slots)].astype(mx.float32))
    values = mx.nan_to_num(v_cache[mx.array(slots)].astype(mx.float32))
    expected = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32).transpose(1, 0, 2)[None],
        keys.transpose(1, 0, 2)[None],
        values.transpose(1, 0, 2)[None],
        scale=scale,
        mask=mx.array(mask)[None, None],
    )[0].transpose(1, 0, 2)
    assert out.shape == q.shape and out.dtype == case.dtype
    assert_close(out, expected)
