"""Metal decode attention: parity with the fp32 reference and with the MLX backend.

Random paged pools with shuffled pages. Every pool row outside a request's keys (page
tails, spare pages and the dummy page that pads short page-table rows) holds huge keys
and values, so reading one shows up in the output.
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
from ftmetal.kernel.ref import flash_decode_ref
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


def decode_batch(seq_lens: list[int], page_table: np.ndarray, num_pages: int) -> Batch:
    """One decode request per length, on shuffled pages written into page_table."""
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
                cached_len=seq_len - 1,
                output_len=1,
                uid=i,
                sampling_params=SamplingParams(),
            )
        )
    return Batch(reqs=reqs, phase="decode")


def max_row_error(out: mx.array, reference: mx.array) -> float:
    error = mx.abs(out.astype(mx.float32) - reference.astype(mx.float32)).max(axis=(1, 2))
    scale = mx.abs(reference.astype(mx.float32)).max(axis=(1, 2))
    return (error / scale).max().item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(16, 2), (8, 1)])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("batch_size", sorted(SEQ_LENS))
def test_decode_matches_reference_and_mlx_backend(
    batch_size: int, head_dim: int, num_heads: int, num_kv_heads: int, dtype: mx.Dtype
) -> None:
    seq_lens = SEQ_LENS[batch_size]
    num_pages = sum(-(-n // PAGE_SIZE) for n in seq_lens) + SPARE_PAGES
    config = attention_config(num_heads, num_kv_heads, head_dim)
    kv_cache = MHAKVCache(
        num_kv_heads=num_kv_heads,
        num_layers=1,
        head_dim=head_dim,
        num_pages=num_pages + 1,
        page_size=PAGE_SIZE,
        dtype=dtype,
    )
    page_table = np.zeros((batch_size, max(seq_lens) + PAGE_SIZE), dtype=np.int32)
    batch = decode_batch(seq_lens, page_table, num_pages)

    num_tokens = kv_cache.num_tokens
    is_key = np.zeros((num_tokens, 1, 1), dtype=bool)
    for i, seq_len in enumerate(seq_lens):
        is_key[page_table[i, :seq_len]] = True
    mx.random.seed(batch_size + head_dim + num_heads)
    keys = mx.random.normal((num_tokens, num_kv_heads, head_dim))
    values = mx.random.normal((num_tokens, num_kv_heads, head_dim))
    keys = mx.where(mx.array(is_key), keys, UNREAD_VALUE).astype(dtype)
    values = mx.where(mx.array(is_key), values, UNREAD_VALUE).astype(dtype)
    kv_cache.store_kv(keys, values, mx.arange(num_tokens, dtype=mx.int32), layer_id=0)

    metal_backend = FullAttnMetalBackend(config, kv_cache, page_table, PAGE_SIZE)
    mlx_backend = FullAttnMLXBackend(config, kv_cache, page_table, PAGE_SIZE)
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
