"""Metal attention: parity with the fp32 reference and with the MLX backend.

Random paged pools with shuffled pages. Every pool row outside a request's keys (page
tails, spare pages and the dummy page that pads short page-table rows) holds huge keys
and values, so reading one shows up in the output. Decode runs flash-decode, whose only
numeric test this is: the decode cases cover each decode plan (head groups and
simdgroups per head dim and dtype) and partition regime once. Prefill runs flash-prefill
(its numeric cases are in kernel/attention/mha/test_mha_prefill.py) on the backend's metadata,
one batch of unequal requests or one long prompt chunk by chunk. Replay metadata (a page table
as wide as the longest context, so flash-decode's grid holds the most partitions) must give the
per-length metadata's results. Sliding-window attention runs on a window pool whose slots
back only the keys a step's windows reach; every other full slot maps to the sentinel,
which holds huge keys and values like every unused window slot, and the reference attends
over the whole row under the window mask. Learned sinks join each softmax's denominator, in decode
and prefill, full and windowed, against the reference and the MLX backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.attention.base import AttentionSpec
from ftmetal.attention.full.metal import FullAttnMetalBackend, MetalAttnMetadata
from ftmetal.attention.full.mlx import FullAttnMLXBackend
from ftmetal.attention.metadata import AttnType
from ftmetal.core import Batch, Req, SamplingParams
from ftmetal.kernel.ref import flash_decode_ref, flash_prefill_ref
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.kvcache.swa_pool import SENTINEL_SLOT, SWAKVCache
from ftmetal.models.config import KVCacheGroupSpec, ModelConfig
from ftmetal.models.qwen3_5_moe.config import parse_config

PAGE_SIZE = 32
SPARE_PAGES = 5
UNREAD_VALUE = 1e3
# Sinks about the size of the scores of normal q and k at scale D ** -0.5.
SINK_SCALE = 2.0
# Worst |error| over a row relative to the row's largest |value|, against the fp32
# reference (one bf16 ulp); in decode the MLX backend's own rounding doubles it.
TOLERANCE = {mx.bfloat16: 2**-7, mx.float32: 1e-5}
# Prefill against the MLX backend: at D 256 its SDPA runs unfused, rounding the scores and
# probabilities to bf16, which alone costs up to ~2 ulps (1.7e-2 measured against the fp32
# reference, where flash-prefill stays within one ulp).
PREFILL_MLX_TOLERANCE = {mx.bfloat16: 2**-5, mx.float32: 2e-5}
# One prompt prefilled in chunks of at most PREFILL_CHUNK tokens.
CHUNKED_PROMPT_LEN = 5000
PREFILL_CHUNK = 2048
# A replayed graph's page-table width (the engine's default context), and its batches:
# keys per request and the dummy rows that pad the batch to its graph's size.
REPLAY_MAX_SEQ_LEN = 262144
REPLAY_BATCHES = [
    ([64], 0),
    ([300], 0),
    ([2048], 0),
    ([8192], 0),
    ([32768], 0),
    ([5, 300, 32768, 2049], 0),
    ([2049, 7, 300], 1),
]
WINDOW_FULL_SLOTS = 16384


@dataclass(frozen=True)
class Case:
    """One step: request b has cached_lens[b] keys before extend_lens[b] new tokens."""

    cached_lens: tuple[int, ...]
    extend_lens: tuple[int, ...]
    head_dim: int
    num_heads: int = 16
    num_kv_heads: int = 2
    dtype: mx.Dtype = mx.bfloat16
    window: int | None = None
    has_sinks: bool = False

    def __str__(self) -> str:
        lens = "+".join(f"{c}:{e}" for c, e in zip(self.cached_lens, self.extend_lens))
        label = f"{lens}-D{self.head_dim}-heads{self.num_heads}:{self.num_kv_heads}"
        label += "-bf16" if self.dtype == mx.bfloat16 else "-fp32"
        label += "-sinks" if self.has_sinks else ""
        return label if self.window is None else f"{label}-W{self.window}"

    def attn_spec(self) -> AttentionSpec | None:
        if not self.has_sinks:
            return None
        return AttentionSpec(
            sinks=(SINK_SCALE * mx.random.normal((self.num_heads,))).astype(self.dtype)
        )


def decode_lens(seq_lens: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(cached_lens, extend_lens) of a decode step over seq_lens keys, current token included."""
    return tuple(n - 1 for n in seq_lens), (1,) * len(seq_lens)


# Decode batches: one long row; one key, a page's end and a long row; single keys,
# page-boundary neighbours and long rows across the partition counts.
ONE_LONG = decode_lens((3001,))
SHORT_MIX = decode_lens((1, 33, 1000))
EIGHT = decode_lens((1, 31, 32, 33, 700, 2049, 3001, 4096))
# Window decode batches around a window of 7 and of 1024.
AROUND_SEVEN = decode_lens((7, 8, 9))
AROUND_1024 = decode_lens((1, 1024, 1025, 3001))
# Window prefill batches: requests starting inside, at and before a window; a chunk from an
# empty prefix; a short turn far past the window.
ACROSS = ((1023, 0, 6), (2, 1100, 3))
FROM_EMPTY = ((0,), (2048,))
FAR_TURN = ((5000,), (64,))

FULL_CASES = [
    # Decode. D 128: one head group, 8 simdgroups at bf16 and 4 at fp32.
    Case(*ONE_LONG, 128),
    Case(*EIGHT, 128, dtype=mx.float32),
    # D 256 (Qwen3.6): two head groups; one kv head doubles the partitions the most.
    Case(*EIGHT, 256),
    Case(*SHORT_MIX, 256, dtype=mx.float32),
    Case(*ONE_LONG, 256, num_heads=8, num_kv_heads=1),
    # D 512 (Gemma4 global): four head groups.
    Case(*SHORT_MIX, 512),
    Case(*ONE_LONG, 512, dtype=mx.float32),
    Case(*EIGHT, 512, num_heads=8, num_kv_heads=1),
    # Prefill: one prompt, a short turn, unequal requests, a long chunk beside a long prefix.
    Case((0,), (37,), 128),
    Case((300,), (2,), 256, dtype=mx.float32),
    Case((0, 2049, 7), (100, 33, 1), 256),
    Case((5000, 0), (64, 2048), 512),
    # Learned sinks at Flint's shape (Hq 32, Hkv 4, D 128): one key, long rows, a long prompt.
    Case(*decode_lens((1,)), 128, num_heads=32, num_kv_heads=4, has_sinks=True),
    Case(*EIGHT, 128, num_heads=32, num_kv_heads=4, has_sinks=True),
    Case(*ONE_LONG, 128, num_heads=32, num_kv_heads=4, dtype=mx.float32, has_sinks=True),
    Case((0, 2049, 7), (100, 33, 1), 128, num_heads=32, num_kv_heads=4, has_sinks=True),
]
# Gemma4's window layers: D 256, 16 query heads over 8 kv heads.
WINDOW_CASES = [
    Case(*AROUND_1024, 256, num_kv_heads=8, window=1),
    Case(*AROUND_SEVEN, 256, num_kv_heads=8, dtype=mx.float32, window=7),
    Case(*decode_lens((1,)), 128, num_heads=8, num_kv_heads=1, window=7),
    Case(*AROUND_1024, 256, num_kv_heads=8, window=1024),
    Case(*AROUND_SEVEN, 128, num_heads=8, num_kv_heads=1, dtype=mx.float32, window=1024),
    Case(*ACROSS, 256, num_kv_heads=8, window=1),
    Case(*FROM_EMPTY, 256, num_kv_heads=8, dtype=mx.float32, window=7),
    Case(*FAR_TURN, 128, num_heads=8, num_kv_heads=1, window=7),
    Case(*FROM_EMPTY, 256, num_kv_heads=8, window=1024),
    Case(*FAR_TURN, 256, num_kv_heads=8, dtype=mx.float32, window=1024),
    Case(*ACROSS, 128, num_heads=8, num_kv_heads=1, window=1024),
    # Learned sinks at Flint's window layers.
    Case(*AROUND_1024, 128, num_heads=32, num_kv_heads=4, window=513, has_sinks=True),
    Case(*ACROSS, 128, num_heads=32, num_kv_heads=4, window=513, has_sinks=True),
]


def attention_config(num_heads: int, num_kv_heads: int, head_dim: int) -> ModelConfig:
    return parse_config(
        {
            "num_hidden_layers": 1,
            "full_attention_interval": 1,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_size": num_heads * head_dim,
            "vocab_size": 16,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 8192,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
        }
    )


def new_tokens(
    num_tokens: int, num_heads: int, num_kv_heads: int, head_dim: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array, mx.array]:
    """Random q, k and v of a step's num_tokens new tokens."""
    q = mx.random.normal((num_tokens, num_heads, head_dim)).astype(dtype)
    k = mx.random.normal((num_tokens, num_kv_heads, head_dim)).astype(dtype)
    v = mx.random.normal((num_tokens, num_kv_heads, head_dim)).astype(dtype)
    return q, k, v


def dummy_req(table_idx: int) -> Req:
    """A one-token request on the page table's dummy row, as the engine pads a batch."""
    return Req(
        input_ids=np.zeros(1, dtype=np.int32),
        table_idx=table_idx,
        cached_len=0,
        output_len=1,
        uid=-1,
        sampling_params=SamplingParams(),
    )


def request_batch(
    seq_lens: list[int], extend_lens: list[int], page_table: np.ndarray, num_pages: int
) -> Batch:
    """One request per length, on shuffled pages written into page_table; a decode batch
    when every request extends by one token."""
    rng = np.random.default_rng(len(seq_lens))
    pages = rng.permutation(num_pages)
    offsets = np.arange(PAGE_SIZE, dtype=np.int32)
    reqs = []
    start = 0
    for i, seq_len in enumerate(seq_lens):
        num_req_pages = -(-seq_len // PAGE_SIZE)
        slots = pages[start : start + num_req_pages, None] * PAGE_SIZE + offsets
        page_table[i, : num_req_pages * PAGE_SIZE] = slots.reshape(-1)
        start += num_req_pages
        reqs.append(
            Req(
                input_ids=np.zeros(seq_len, dtype=np.int32),
                table_idx=i,
                cached_len=seq_len - extend_lens[i],
                output_len=1,
                uid=i,
                sampling_params=SamplingParams(),
            )
        )
    return Batch(reqs=reqs, phase="decode" if max(extend_lens) == 1 else "prefill")


def filled_pool(
    seq_lens: list[int],
    num_kv_heads: int,
    head_dim: int,
    dtype: mx.Dtype,
    table_width: int,
    extend_lens: list[int] | None = None,
) -> tuple[MHAKVCache, np.ndarray, Batch]:
    """A pool holding random keys and values for each request's slots, UNREAD_VALUE elsewhere.

    The page table has one row per request plus a last row on the dummy page. The batch
    extends each request by extend_lens tokens (default one: decode).
    """
    num_pages = sum(-(-n // PAGE_SIZE) for n in seq_lens) + SPARE_PAGES
    kv_cache = MHAKVCache(
        num_kv_heads=num_kv_heads,
        num_layers=1,
        head_dim=head_dim,
        num_pages=num_pages + 1,
        page_size=PAGE_SIZE,
        dtype=dtype,
    )
    page_table = np.full((len(seq_lens) + 1, table_width), num_pages * PAGE_SIZE, dtype=np.int32)
    if extend_lens is None:
        extend_lens = [1] * len(seq_lens)
    batch = request_batch(seq_lens, extend_lens, page_table, num_pages)

    num_tokens = kv_cache.num_tokens
    is_key = np.zeros((num_tokens, 1, 1), dtype=bool)
    for i, seq_len in enumerate(seq_lens):
        is_key[page_table[i, :seq_len]] = True
    keys = mx.random.normal((num_tokens, num_kv_heads, head_dim))
    values = mx.random.normal((num_tokens, num_kv_heads, head_dim))
    keys = mx.where(mx.array(is_key), keys, UNREAD_VALUE).astype(dtype)
    values = mx.where(mx.array(is_key), values, UNREAD_VALUE).astype(dtype)
    kv_cache.store_kv(keys, values, mx.arange(num_tokens, dtype=mx.int32), layer_id=0)
    return kv_cache, page_table, batch


def max_row_error(out: mx.array, reference: mx.array) -> float:
    error = mx.abs(out.astype(mx.float32) - reference.astype(mx.float32)).max(axis=(1, 2))
    scale = mx.abs(reference.astype(mx.float32)).max(axis=(1, 2))
    return float((error / scale).max())


@pytest.mark.parametrize("case", FULL_CASES, ids=str)
def test_step_matches_reference_and_mlx_backend(case: Case) -> None:
    cached_lens, extend_lens = list(case.cached_lens), list(case.extend_lens)
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    config = attention_config(case.num_heads, case.num_kv_heads, case.head_dim)
    mx.random.seed(sum(seq_lens) + case.head_dim)
    kv_cache, page_table, batch = filled_pool(
        seq_lens,
        case.num_kv_heads,
        case.head_dim,
        case.dtype,
        max(seq_lens) + PAGE_SIZE,
        extend_lens,
    )
    (spec,) = config.kv_cache_group_specs()
    metal_backend = FullAttnMetalBackend(config, spec, kv_cache, page_table, PAGE_SIZE)
    mlx_backend = FullAttnMLXBackend(config, spec, kv_cache, page_table, PAGE_SIZE)
    metal_backend.prepare_metadata(batch)
    metadata = batch.attn_metadata[AttnType.FULL]
    assert isinstance(metadata, MetalAttnMetadata)

    q, k, v = new_tokens(
        sum(extend_lens), case.num_heads, case.num_kv_heads, case.head_dim, case.dtype
    )
    attn_spec = case.attn_spec()
    sinks = None if attn_spec is None else attn_spec.sinks
    out = metal_backend.forward(q, k, v, 0, batch, attn_spec)
    # The MLX backend stores the same rows again, then reads the same pool.
    expected = mlx_backend.forward(q, k, v, 0, batch, attn_spec)
    pools = (kv_cache.k_cache(0), kv_cache.v_cache(0), metadata.page_table)
    scale = case.head_dim**-0.5
    if batch.is_decode:
        reference = flash_decode_ref(q, *pools, metadata.seq_lens, PAGE_SIZE, scale, sinks)
        mlx_tolerance = 2 * TOLERANCE[case.dtype]
    else:
        reference = flash_prefill_ref(
            q, *pools, extend_lens, cached_lens, PAGE_SIZE, scale, sinks=sinks
        )
        mlx_tolerance = PREFILL_MLX_TOLERANCE[case.dtype]
    assert out.shape == q.shape and out.dtype == case.dtype
    assert max_row_error(out, reference) < TOLERANCE[case.dtype]
    assert max_row_error(out, expected) < mlx_tolerance


@pytest.mark.parametrize(("seq_lens", "num_dummies"), REPLAY_BATCHES)
def test_replay_metadata_matches_reference_and_eager(seq_lens: list[int], num_dummies: int) -> None:
    num_heads, num_kv_heads, head_dim = 16, 2, 256
    size = len(seq_lens)
    config = attention_config(num_heads, num_kv_heads, head_dim)
    mx.random.seed(size + num_dummies)
    kv_cache, page_table, batch = filled_pool(
        seq_lens, num_kv_heads, head_dim, mx.bfloat16, REPLAY_MAX_SEQ_LEN
    )
    (spec,) = config.kv_cache_group_specs()
    backend = FullAttnMetalBackend(config, spec, kv_cache, page_table, PAGE_SIZE)
    padded_size = size + num_dummies
    q, k, v = new_tokens(padded_size, num_heads, num_kv_heads, head_dim, mx.bfloat16)

    backend.prepare_metadata(batch)
    metadata = batch.attn_metadata[AttnType.FULL]
    assert isinstance(metadata, MetalAttnMetadata)
    eager = backend.forward(q[:size], k[:size], v[:size], 0, batch)
    backend.init_capture_graph(REPLAY_MAX_SEQ_LEN, [padded_size])
    batch.padded_reqs = batch.reqs + [dummy_req(size)] * num_dummies
    backend.prepare_metadata(batch)
    backend.prepare_for_replay(batch)
    replayed = backend.forward(q, k, v, 0, batch)[:size]
    reference = flash_decode_ref(
        q[:size],
        kv_cache.k_cache(0),
        kv_cache.v_cache(0),
        metadata.page_table,
        metadata.seq_lens,
        PAGE_SIZE,
        config.head_dim**-0.5,
    )
    assert max_row_error(replayed, reference) < TOLERANCE[mx.bfloat16]
    if num_dummies == 0:
        # Same batch size, so each request gets the same partitions under either grid.
        assert mx.array_equal(replayed, eager)


def test_chunked_prefill_matches_reference_and_mlx_backend() -> None:
    num_heads, num_kv_heads, head_dim = 16, 2, 256
    config = attention_config(num_heads, num_kv_heads, head_dim)
    mx.random.seed(CHUNKED_PROMPT_LEN)
    kv_cache, page_table, _ = filled_pool(
        [CHUNKED_PROMPT_LEN], num_kv_heads, head_dim, mx.bfloat16, CHUNKED_PROMPT_LEN + PAGE_SIZE
    )
    (spec,) = config.kv_cache_group_specs()
    metal_backend = FullAttnMetalBackend(config, spec, kv_cache, page_table, PAGE_SIZE)
    mlx_backend = FullAttnMLXBackend(config, spec, kv_cache, page_table, PAGE_SIZE)
    for start in range(0, CHUNKED_PROMPT_LEN, PREFILL_CHUNK):
        stop = min(CHUNKED_PROMPT_LEN, start + PREFILL_CHUNK)
        # Each chunk attends over the keys the chunks before it stored.
        req = Req(
            input_ids=np.zeros(stop, dtype=np.int32),
            table_idx=0,
            cached_len=start,
            output_len=1,
            uid=0,
            sampling_params=SamplingParams(),
        )
        batch = Batch(reqs=[req], phase="prefill")
        metal_backend.prepare_metadata(batch)
        metadata = batch.attn_metadata[AttnType.FULL]
        assert isinstance(metadata, MetalAttnMetadata)
        q, k, v = new_tokens(stop - start, num_heads, num_kv_heads, head_dim, mx.bfloat16)
        out = metal_backend.forward(q, k, v, 0, batch)
        expected = mlx_backend.forward(q, k, v, 0, batch)
        reference = flash_prefill_ref(
            q,
            kv_cache.k_cache(0),
            kv_cache.v_cache(0),
            metadata.page_table,
            [stop - start],
            [start],
            PAGE_SIZE,
            config.head_dim**-0.5,
        )
        assert max_row_error(out, reference) < TOLERANCE[mx.bfloat16]
        assert max_row_error(out, expected) < PREFILL_MLX_TOLERANCE[mx.bfloat16]


def window_pool(
    cached_lens: list[int],
    extend_lens: list[int],
    num_kv_heads: int,
    head_dim: int,
    dtype: mx.Dtype,
    window: int,
) -> tuple[SWAKVCache, np.ndarray, Batch]:
    """A window pool backing each request's keys from its window start on, random rows in
    those slots and UNREAD_VALUE in every other slot (the sentinel included); the page table
    holds shuffled full slots, one row per request plus the dummy row."""
    spec = KVCacheGroupSpec(
        name="swa",
        attn_type=AttnType.SWA,
        layer_ids=(0,),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=window,
    )
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    num_swa_tokens = sum(seq_lens) + SPARE_PAGES
    pool = SWAKVCache(spec, num_swa_tokens, WINDOW_FULL_SLOTS, dtype)
    rng = np.random.default_rng(sum(seq_lens) + window)
    full_slots = rng.permutation(WINDOW_FULL_SLOTS - 1).astype(np.int32)
    page_table = np.full((len(seq_lens) + 1, max(seq_lens)), WINDOW_FULL_SLOTS - 1, np.int32)
    reqs = []
    start = 0
    for i, (cached_len, seq_len) in enumerate(zip(cached_lens, seq_lens)):
        page_table[i, :seq_len] = full_slots[start : start + seq_len]
        start += seq_len
        pool.alloc_swa(page_table[i, max(0, cached_len - window + 1) : seq_len])
        reqs.append(
            Req(
                input_ids=np.zeros(seq_len, dtype=np.int32),
                table_idx=i,
                cached_len=cached_len,
                output_len=1,
                uid=i,
                sampling_params=SamplingParams(),
            )
        )
    is_key = np.ones((num_swa_tokens, 1, 1), dtype=bool)
    is_key[pool.free_swa_slots] = False
    is_key[SENTINEL_SLOT] = False
    shape = (num_swa_tokens, num_kv_heads, head_dim)
    keys = mx.where(mx.array(is_key), mx.random.normal(shape), UNREAD_VALUE).astype(dtype)
    values = mx.where(mx.array(is_key), mx.random.normal(shape), UNREAD_VALUE).astype(dtype)
    pool.store_kv(keys, values, mx.arange(num_swa_tokens, dtype=mx.int32), layer_id=0)
    phase = "decode" if max(extend_lens) == 1 else "prefill"
    return pool, page_table, Batch(reqs=reqs, phase=phase)


def window_reference(
    q: mx.array,
    pool: SWAKVCache,
    page_table: np.ndarray,
    cached_lens: list[int],
    extend_lens: list[int],
    scale: float,
    sinks: mx.array | None = None,
) -> mx.array:
    """fp32 attention over each request's whole row of window slots (the sentinel before
    its window start) under the absolute-position window mask."""
    rows = [
        pool.translate(page_table[i, : c + e])
        for i, (c, e) in enumerate(zip(cached_lens, extend_lens))
    ]
    table = np.full((len(rows), max(len(row) for row in rows)), SENTINEL_SLOT, np.int32)
    for i, row in enumerate(rows):
        table[i, : len(row)] = row
    return flash_prefill_ref(
        q,
        pool.k_cache(0),
        pool.v_cache(0),
        mx.array(table),
        extend_lens,
        cached_lens,
        1,
        scale,
        pool.sliding_window,
        sinks=sinks,
    )


def window_backends(
    pool: SWAKVCache, page_table: np.ndarray, num_heads: int, head_dim: int
) -> tuple[FullAttnMetalBackend, FullAttnMLXBackend]:
    config = attention_config(num_heads, pool.k_cache(0).shape[1], head_dim)
    spec = KVCacheGroupSpec(
        name="swa",
        attn_type=AttnType.SWA,
        layer_ids=(0,),
        num_kv_heads=pool.k_cache(0).shape[1],
        head_dim=head_dim,
        sliding_window=pool.sliding_window,
    )
    return (
        FullAttnMetalBackend(config, spec, pool, page_table, 1),
        FullAttnMLXBackend(config, spec, pool, page_table, 1),
    )


@pytest.mark.parametrize("case", WINDOW_CASES, ids=str)
def test_window_step_matches_reference_and_mlx_backend(case: Case) -> None:
    cached_lens, extend_lens = list(case.cached_lens), list(case.extend_lens)
    assert case.window is not None
    mx.random.seed(sum(cached_lens) + case.window + case.head_dim)
    pool, page_table, batch = window_pool(
        cached_lens, extend_lens, case.num_kv_heads, case.head_dim, case.dtype, case.window
    )
    metal_backend, mlx_backend = window_backends(pool, page_table, case.num_heads, case.head_dim)
    metal_backend.prepare_metadata(batch)
    metadata = batch.attn_metadata[AttnType.SWA]
    assert isinstance(metadata, MetalAttnMetadata)
    assert max(metadata.seq_lens_host) <= case.window + max(extend_lens) - 1
    q, k, v = new_tokens(
        sum(extend_lens), case.num_heads, case.num_kv_heads, case.head_dim, case.dtype
    )
    attn_spec = case.attn_spec()
    sinks = None if attn_spec is None else attn_spec.sinks
    out = metal_backend.forward(q, k, v, 0, batch, attn_spec)
    expected = mlx_backend.forward(q, k, v, 0, batch, attn_spec)
    scale = case.head_dim**-0.5
    reference = window_reference(q, pool, page_table, cached_lens, extend_lens, scale, sinks)
    assert out.shape == q.shape and out.dtype == case.dtype
    assert max_row_error(out, reference) < TOLERANCE[case.dtype]
    assert max_row_error(out, expected) < PREFILL_MLX_TOLERANCE[case.dtype]


@pytest.mark.parametrize("num_dummies", [0, 2])
def test_window_replay_metadata_matches_reference_and_eager(num_dummies: int) -> None:
    """A replayed window decode step runs on the window-wide row the eager step builds."""
    num_heads, num_kv_heads, head_dim, window = 16, 8, 256, 1024
    seq_lens = [5, 1024, 3001]
    size = len(seq_lens)
    mx.random.seed(size + num_dummies)
    cached_lens = [n - 1 for n in seq_lens]
    pool, page_table, batch = window_pool(
        cached_lens, [1] * size, num_kv_heads, head_dim, mx.bfloat16, window
    )
    backend, _ = window_backends(pool, page_table, num_heads, head_dim)
    padded_size = size + num_dummies
    q, k, v = new_tokens(padded_size, num_heads, num_kv_heads, head_dim, mx.bfloat16)

    backend.prepare_metadata(batch)
    eager = backend.forward(q[:size], k[:size], v[:size], 0, batch)
    backend.init_capture_graph(page_table.shape[1], [padded_size])
    batch.padded_reqs = batch.reqs + [dummy_req(size)] * num_dummies
    backend.prepare_metadata(batch)
    backend.prepare_for_replay(batch)
    metadata = batch.attn_metadata[AttnType.SWA]
    assert isinstance(metadata, MetalAttnMetadata)
    assert (
        metadata.page_table.shape
        == (padded_size, window)
        == (
            padded_size,
            metadata.max_seqlen_k,
        )
    )
    replayed = backend.forward(q, k, v, 0, batch)[:size]
    reference = window_reference(
        q[:size], pool, page_table, cached_lens, [1] * size, head_dim**-0.5
    )
    assert max_row_error(replayed, reference) < TOLERANCE[mx.bfloat16]
    if num_dummies == 0:
        assert mx.array_equal(replayed, eager)
