"""Metal attention: parity with the fp32 reference and with the MLX backend.

Random paged pools with shuffled pages. Every pool row outside a request's keys (page
tails, spare pages and the dummy page that pads short page-table rows) holds huge keys
and values, so reading one shows up in the output. Replay metadata (a page table as wide
as the longest context, so flash-decode's grid holds the most partitions) must give the
per-length metadata's results. Prefill runs flash-prefill over the same pools, one batch
of unequal requests or one long prompt chunk by chunk.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ftmetal.attention.full.common import FullAttnMetadata
from ftmetal.attention.full.metal import FullAttnMetalBackend
from ftmetal.attention.full.mlx import FullAttnMLXBackend
from ftmetal.attention.metadata import AttnType
from ftmetal.core import Batch, Req, SamplingParams
from ftmetal.kernel.ref import flash_decode_ref, flash_prefill_ref
from ftmetal.kvcache.mha_pool import MHAKVCache
from ftmetal.models.config import ModelConfig
from ftmetal.models.qwen3_5_moe.config import parse_config

PAGE_SIZE = 32
SPARE_PAGES = 5
UNREAD_VALUE = 1e3
# Keys per request, current token included: single keys, page-boundary neighbours, long rows.
SEQ_LENS = {
    1: [3001],
    3: [1, 33, 1000],
    8: [1, 31, 32, 33, 700, 2049, 3001, 4096],
}
# Worst |error| over a row relative to the row's largest |value|, against the fp32
# reference (one bf16 ulp); the MLX backend's own rounding doubles it.
TOLERANCE = {mx.bfloat16: 2**-7, mx.float32: 1e-5}
# Prefill against the MLX backend: at D 256 its SDPA runs unfused, rounding the scores and
# probabilities to bf16, which alone costs up to ~2 ulps (1.7e-2 measured against the fp32
# reference, where flash-prefill stays within one ulp).
PREFILL_MLX_TOLERANCE = {mx.bfloat16: 2**-5, mx.float32: 2e-5}
# Prefill batches: cached prefix and new tokens per request.
PREFILL_BATCHES = [
    ([0], [37]),
    ([300], [2]),
    ([0, 2049, 7], [100, 33, 1]),
    ([5000, 0], [64, 2048]),
]
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


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(16, 2), (8, 1)])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("batch_size", sorted(SEQ_LENS))
def test_decode_matches_reference_and_mlx_backend(
    batch_size: int, head_dim: int, num_heads: int, num_kv_heads: int, dtype: mx.Dtype
) -> None:
    seq_lens = SEQ_LENS[batch_size]
    config = attention_config(num_heads, num_kv_heads, head_dim)
    mx.random.seed(batch_size + head_dim + num_heads)
    kv_cache, page_table, batch = filled_pool(
        seq_lens, num_kv_heads, head_dim, dtype, max(seq_lens) + PAGE_SIZE
    )

    (spec,) = config.kv_cache_group_specs()
    metal_backend = FullAttnMetalBackend(config, spec, kv_cache, page_table, PAGE_SIZE)
    mlx_backend = FullAttnMLXBackend(config, spec, kv_cache, page_table, PAGE_SIZE)
    metal_backend.prepare_metadata(batch)
    metadata = batch.attn_metadata[AttnType.FULL]
    assert isinstance(metadata, FullAttnMetadata)

    q = mx.random.normal((batch_size, num_heads, head_dim)).astype(dtype)
    k = mx.random.normal((batch_size, num_kv_heads, head_dim)).astype(dtype)
    v = mx.random.normal((batch_size, num_kv_heads, head_dim)).astype(dtype)
    out = metal_backend.forward(q, k, v, 0, batch)
    # The MLX backend stores the same rows again, then reads the same pool.
    expected = mlx_backend.forward(q, k, v, 0, batch)
    reference = flash_decode_ref(
        q,
        kv_cache.k_cache(0),
        kv_cache.v_cache(0),
        metadata.page_table,
        metadata.seq_lens,
        PAGE_SIZE,
        config.head_dim**-0.5,
    )
    assert out.shape == q.shape and out.dtype == dtype
    assert max_row_error(out, reference) < TOLERANCE[dtype]
    assert max_row_error(out, expected) < 2 * TOLERANCE[dtype]


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize(("seq_lens", "num_dummies"), REPLAY_BATCHES)
def test_replay_metadata_matches_reference_and_eager(
    seq_lens: list[int], num_dummies: int, dtype: mx.Dtype
) -> None:
    num_heads, num_kv_heads, head_dim = 16, 2, 256
    size = len(seq_lens)
    config = attention_config(num_heads, num_kv_heads, head_dim)
    mx.random.seed(size + num_dummies)
    kv_cache, page_table, batch = filled_pool(
        seq_lens, num_kv_heads, head_dim, dtype, REPLAY_MAX_SEQ_LEN
    )
    dummy = Req(
        input_ids=np.zeros(1, dtype=np.int32),
        table_idx=size,
        cached_len=0,
        output_len=1,
        uid=-1,
        sampling_params=SamplingParams(),
    )
    (spec,) = config.kv_cache_group_specs()
    backend = FullAttnMetalBackend(config, spec, kv_cache, page_table, PAGE_SIZE)
    padded_size = size + num_dummies
    q = mx.random.normal((padded_size, num_heads, head_dim)).astype(dtype)
    k = mx.random.normal((padded_size, num_kv_heads, head_dim)).astype(dtype)
    v = mx.random.normal((padded_size, num_kv_heads, head_dim)).astype(dtype)

    backend.prepare_metadata(batch)
    metadata = batch.attn_metadata[AttnType.FULL]
    assert isinstance(metadata, FullAttnMetadata)
    eager = backend.forward(q[:size], k[:size], v[:size], 0, batch)
    backend.init_capture_graph(REPLAY_MAX_SEQ_LEN, [padded_size])
    batch.padded_reqs = batch.reqs + [dummy] * num_dummies
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
    assert max_row_error(replayed, reference) < TOLERANCE[dtype]
    if num_dummies == 0:
        # Same batch size, so each request gets the same partitions under either grid.
        assert mx.array_equal(replayed, eager)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(16, 2), (8, 1)])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize(("cached_lens", "extend_lens"), PREFILL_BATCHES)
def test_prefill_matches_reference_and_mlx_backend(
    cached_lens: list[int],
    extend_lens: list[int],
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: mx.Dtype,
) -> None:
    seq_lens = [c + e for c, e in zip(cached_lens, extend_lens)]
    config = attention_config(num_heads, num_kv_heads, head_dim)
    mx.random.seed(sum(seq_lens) + head_dim + num_heads)
    kv_cache, page_table, batch = filled_pool(
        seq_lens, num_kv_heads, head_dim, dtype, max(seq_lens) + PAGE_SIZE, extend_lens
    )
    (spec,) = config.kv_cache_group_specs()
    metal_backend = FullAttnMetalBackend(config, spec, kv_cache, page_table, PAGE_SIZE)
    mlx_backend = FullAttnMLXBackend(config, spec, kv_cache, page_table, PAGE_SIZE)
    metal_backend.prepare_metadata(batch)
    metadata = batch.attn_metadata[AttnType.FULL]
    assert isinstance(metadata, FullAttnMetadata) and metadata.max_seqlen_q > 1

    num_tokens = sum(extend_lens)
    q = mx.random.normal((num_tokens, num_heads, head_dim)).astype(dtype)
    k = mx.random.normal((num_tokens, num_kv_heads, head_dim)).astype(dtype)
    v = mx.random.normal((num_tokens, num_kv_heads, head_dim)).astype(dtype)
    out = metal_backend.forward(q, k, v, 0, batch)
    expected = mlx_backend.forward(q, k, v, 0, batch)
    reference = flash_prefill_ref(
        q,
        kv_cache.k_cache(0),
        kv_cache.v_cache(0),
        metadata.page_table,
        extend_lens,
        cached_lens,
        PAGE_SIZE,
        config.head_dim**-0.5,
    )
    assert out.shape == q.shape and out.dtype == dtype
    assert max_row_error(out, reference) < TOLERANCE[dtype]
    assert max_row_error(out, expected) < PREFILL_MLX_TOLERANCE[dtype]


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_chunked_prefill_matches_reference_and_mlx_backend(dtype: mx.Dtype) -> None:
    num_heads, num_kv_heads, head_dim = 16, 2, 256
    config = attention_config(num_heads, num_kv_heads, head_dim)
    mx.random.seed(CHUNKED_PROMPT_LEN)
    kv_cache, page_table, _ = filled_pool(
        [CHUNKED_PROMPT_LEN], num_kv_heads, head_dim, dtype, CHUNKED_PROMPT_LEN + PAGE_SIZE
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
        assert isinstance(metadata, FullAttnMetadata)
        q = mx.random.normal((stop - start, num_heads, head_dim)).astype(dtype)
        k = mx.random.normal((stop - start, num_kv_heads, head_dim)).astype(dtype)
        v = mx.random.normal((stop - start, num_kv_heads, head_dim)).astype(dtype)
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
        assert max_row_error(out, reference) < TOLERANCE[dtype]
        assert max_row_error(out, expected) < PREFILL_MLX_TOLERANCE[dtype]
