"""Qwen3.8-Flash-Next's PLE branch: a hashed n-gram embedding gating the residual streams,
then a causal dilated short conv.

Port of FreeToken models/qwen4_exp/ple.py (NGramEmbedding, PLELayer) and ple_disk.py
(DiskRowTable). Divergences: the table is the MLX checkpoint's affine shards, never loaded:
a row store reads each forward's rows from the checkpoint files (kernel/service/ple/ple_service.h)
and they are dequantized on the GPU; a prefill hashes its n-grams on the host (the service's
hash) over the request's token ids, its context the ids before them (eos at the start), and
gathers its rows before the layer; a decode step posts its tokens at the forward's start
(ple_post) and the service's coordinator hashes them with the slot's context and gathers the
rows while the GPU runs the layers before this one (ple_wait), so no host code reads the
step's ids and it replays as a CommandGraph (a batch past MAX_PLE_ROWS reads them back and
gathers on the host, so it runs eagerly). The n-gram context (the last ngram_size - 1 ids)
and the conv window live in the request's GDN slot (slot states), so snapshots, restores and
clears carry them.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ftmetal.attention.metadata import AttnType, LinearAttnMetadata
from ftmetal.context import get_global_ctx
from ftmetal.core import Batch
from ftmetal.kernel._ext import _C
from ftmetal.kernel.embedding.ple_lookup import PleRowParts, ple_post, ple_wait
from ftmetal.kernel.kvcache.pool_rows import scatter_rows
from ftmetal.layers.base import BaseOP
from ftmetal.layers.gdn import DepthwiseConv1d
from ftmetal.layers.linear import Linear
from ftmetal.layers.quantization.config import QuantScheme
from ftmetal.mm import restore_placeholder
from ftmetal.models.loader import weight_files
from ftmetal.models.qwen4_exp.config import Qwen4ExpConfig, ple_context_state, ple_conv_state
from ftmetal.models.qwen4_exp.norm import CenteredRMSNorm
from ftmetal.models.qwen4_exp.weight import ple_table_prefix

SHARD_KEY = re.compile(r"ngram_embedding\.shards\.(\d+)\.(weight|scales|biases)$")
TABLE_PARTS = ("weight", "scales", "biases")
# Safetensors dtype of a row part -> the MLX dtype its bytes are viewed as.
PART_DTYPES: dict[str, mx.Dtype] = {
    "U32": mx.uint32,
    "BF16": mx.bfloat16,
    "F16": mx.float16,
    "F32": mx.float32,
}
HASH_TENSORS = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
# The row store's cache, in rows (a power of 2): ~200 MiB of the real table's 100-byte rows.
ROW_CACHE_ROWS = 1 << 21
# Threads reading cold rows beside a gather's caller: the internal SSD serves random reads
# about 20x faster at 32 in flight than at one (M4 Pro, 2026-10).
IO_THREADS = 32
# A decode step's post left unanswered this long fails (the offload lookup's timeout).
STALL_TIMEOUT_S = 10.0


@dataclass(frozen=True, kw_only=True)
class StepLookup:
    """A table's in-step lookup: the service answering the posts, its mailbox, the staging it
    copies rows into."""

    service: _C.PleService
    mailbox: mx.array
    staging: PleRowParts


class NGramTable:
    """One PLE layer's n-gram hash constants and its affine row table in the checkpoint files:
    shards of rows, each row a packed weight, scales and biases part."""

    def __init__(self, model_path: str, layer_id: int, row_width: int) -> None:
        prefix = ple_table_prefix(layer_id)
        files = weight_files(model_path)
        hashes: dict[str, np.ndarray] = {}
        # (shard, part) -> (file index, byte offset of the shard's row 0, row stride).
        locations: dict[tuple[int, int], tuple[int, int, int]] = {}
        shard_rows: dict[int, int] = {}
        self.part_dtypes: list[mx.Dtype] = [mx.uint32] * len(TABLE_PARTS)
        row_bytes = [0] * len(TABLE_PARTS)
        for file_index, file in enumerate(files):
            with open(file, "rb") as handle:
                header_len = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_len))
            for name, entry in header.items():
                if not name.startswith(prefix):
                    continue
                key = name[len(prefix) :]
                start, end = entry["data_offsets"]
                offset = 8 + header_len + start
                if key in HASH_TENSORS:
                    if entry["dtype"] != "I64":
                        raise ValueError(
                            f"{name}: the hash constants are int64, not {entry['dtype']}"
                        )
                    count = (end - start) // 8
                    hashes[key] = np.fromfile(file, dtype=np.int64, count=count, offset=offset)
                    continue
                match = SHARD_KEY.search(key)
                if match is None:
                    raise ValueError(f"unexpected PLE tensor {name}")
                shard, part = int(match.group(1)), TABLE_PARTS.index(match.group(2))
                rows = entry["shape"][0]
                stride = (end - start) // rows
                locations[(shard, part)] = (file_index, offset, stride)
                shard_rows[shard] = rows
                row_bytes[part] = stride
                self.part_dtypes[part] = PART_DTYPES[entry["dtype"]]
        num_shards = len(shard_rows)
        if num_shards == 0 or len(locations) != num_shards * len(TABLE_PARTS):
            raise ValueError(f"{model_path}: no affine PLE table under {prefix}")
        self.multipliers = hashes["layer_multipliers"]
        self.head_sizes = hashes["ngram_heads_vocab_sizes"]
        self.head_offsets = hashes["ngram_heads_offsets"]
        parts = np.array(
            [locations[(shard, part)] for shard in range(num_shards) for part in range(3)],
            dtype=np.int64,
        ).reshape(num_shards, 3, 3)
        counts = [shard_rows[shard] for shard in range(num_shards)]
        shard_starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.row_bytes = row_bytes
        # The row store's cache: each row's parts and its row id.
        self.cache_bytes = ROW_CACHE_ROWS * (sum(row_bytes) + 8)
        self.store = _C.PleRowStore(
            files, parts, shard_starts.tolist(), row_bytes, ROW_CACHE_ROWS, IO_THREADS
        )
        # The packed words and the groups per row give the scheme of row_width-wide rows.
        self.bits = row_bytes[0] * 8 // row_width
        self.group_size = row_width // (row_bytes[1] // self.part_dtypes[1].size)

    def dequantize(self, parts: PleRowParts, dtype: mx.Dtype) -> mx.array:
        """Rows [N, row_width] of their parts, uint8 [N, part bytes] each."""
        weight, scales, biases = (
            part.view(view) for part, view in zip(parts, self.part_dtypes, strict=True)
        )
        rows = mx.dequantize(
            weight, scales, biases, group_size=self.group_size, bits=self.bits, mode="affine"
        )
        return rows.astype(dtype)

    def lookup(self, ids: np.ndarray, dtype: mx.Dtype) -> mx.array:
        """The rows of ids int64 [N], dequantized [N, row_width]."""
        flat = np.ascontiguousarray(ids.reshape(-1), dtype=np.int64)
        parts = [np.empty((flat.size, width), dtype=np.uint8) for width in self.row_bytes]
        self.store.gather(flat, *parts)
        return self.dequantize(PleRowParts(*(mx.array(part) for part in parts)), dtype)

    def start_lookup(self, eos_token_id: int) -> StepLookup:
        """Start the service that answers in-step lookups of this table's rows."""
        heads = self.head_sizes.size
        staging = PleRowParts(
            *(mx.zeros((_C.MAX_PLE_ROWS, heads, width), dtype=mx.uint8) for width in self.row_bytes)
        )
        mailbox = mx.zeros((_C.PLE_MAILBOX_WORDS,), dtype=mx.uint32)
        mx.eval(mailbox, *staging)
        service = _C.PleService(
            mailbox,
            list(staging),
            self.store,
            self.multipliers.tolist(),
            self.head_sizes.tolist(),
            self.head_offsets.tolist(),
            eos_token_id,
            STALL_TIMEOUT_S,
            True,
        )
        return StepLookup(service=service, mailbox=mailbox, staging=staging)


class Qwen4ExpPLE(BaseOP):
    def __init__(
        self,
        config: Qwen4ExpConfig,
        layer_id: int,
        *,
        key_scheme: QuantScheme | None,
        value_scheme: QuantScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.eos_token_id = config.ngram_eos_token_id
        # Content pad ids reach the hash only when the tower is built.
        self.image_token_id = None if config.vision_config is None else config.image_token_id
        self.heads_per_ngram = config.heads_per_ngram
        self.context_len = config.ngram_size - 1
        self.dilation = config.ngram_size
        self.state_len = config.ple_conv_state_len
        self.row_width = config.ple_embed_dim // (self.context_len * self.heads_per_ngram)
        width = config.hc_hidden_size
        eps = config.rms_norm_eps
        self.key_proj = Linear(config.ple_embed_dim, width, scheme=key_scheme, dtype=dtype)
        self.value_proj = Linear(
            config.ple_embed_dim, config.hidden_size, scheme=value_scheme, dtype=dtype
        )
        self.norm_key = CenteredRMSNorm(width, eps, group_size=config.hidden_size, dtype=dtype)
        self.norm_query = CenteredRMSNorm(width, eps, group_size=config.hidden_size, dtype=dtype)
        self.norm_conv = CenteredRMSNorm(width, eps, group_size=config.hidden_size, dtype=dtype)
        self.conv1d = DepthwiseConv1d(width, config.ple_conv_kernel_size, dtype)
        self.context_name = ple_context_state(layer_id)
        self.conv_name = ple_conv_state(layer_id)
        self.embedding_dtype = dtype
        # Opened after the weights load (open_table).
        self.table: NGramTable | None = None
        self._lookup: StepLookup | None = None

    def open_table(self, model_path: str) -> int:
        """Open the table and start its lookup service; returns its row cache bytes."""
        self.table = NGramTable(model_path, self.layer_id, self.row_width)
        self._lookup = self.table.start_lookup(self.eos_token_id)
        return self.table.cache_bytes

    @property
    def lookup(self) -> StepLookup:
        assert self._lookup is not None, "finish_loading was not called"
        return self._lookup

    def post_decode(self, batch: Batch, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        """At a decode forward's start: post the batch's tokens, at most MAX_PLE_ROWS, with
        their slots' contexts (and advance the contexts). Returns the ids to embed, which order
        the post first, and the post's ticket."""
        assert input_ids.shape[0] <= _C.MAX_PLE_ROWS
        metadata = batch.attn_metadata[AttnType.LINEAR]
        assert isinstance(metadata, LinearAttnMetadata)
        pool = get_global_ctx().linear_state_pool
        assert pool is not None
        posted = ple_post(
            input_ids,
            metadata.cache_indices,
            pool.slot_states[self.context_name],
            self.lookup.mailbox,
        )
        pool.slot_states[self.context_name] = posted.context
        return posted.input_ids, posted.ticket

    def host_embeddings(self, batch: Batch, metadata: LinearAttnMetadata) -> mx.array:
        """The batch's n-gram rows [T, ple_embed_dim], hashed and gathered on the host; writes
        each slot's next context (and a tracked snapshot's)."""
        assert self.table is not None, "finish_loading was not called"
        pool = get_global_ctx().linear_state_pool
        assert pool is not None
        context_len = self.context_len
        if batch.is_decode:
            stored = np.array(pool.slot_states[self.context_name][metadata.cache_indices])
            tokens = np.array(batch.input_ids)
            histories = [
                np.concatenate([stored[i], tokens[i : i + 1]]).astype(np.int64)
                for i in range(len(batch.padded_reqs))
            ]
        else:
            histories = []
            for req in batch.padded_reqs:
                before = req.input_ids[max(0, req.cached_len - context_len) : req.cached_len]
                pad = np.full(context_len - before.size, self.eos_token_id)
                chunk = req.input_ids[req.cached_len : req.device_len]
                history = np.concatenate([pad, before, chunk]).astype(np.int64)
                # Image rows hash as the placeholder token, never as their content pad ids.
                if self.image_token_id is None:
                    histories.append(history)
                else:
                    histories.append(restore_placeholder(history, self.image_token_id))
        # (token, previous, before previous) per token, the service's own hash.
        triples = np.concatenate(
            [np.stack([history[2:], history[1:-1], history[:-2]], axis=1) for history in histories]
        )
        ids = self.lookup.service.ngram_ids(np.ascontiguousarray(triples))
        contexts = np.stack([history[-context_len:] for history in histories]).astype(np.int32)
        context_pool = scatter_rows(
            pool.slot_states[self.context_name], mx.array(contexts), metadata.cache_indices
        )
        tracked = [i for i, slot in enumerate(metadata.track_slots_host) if slot >= 0]
        if tracked:
            boundaries = [
                histories[i][metadata.track_seqlens_host[i] - batch.padded_reqs[i].cached_len :][
                    :context_len
                ]
                for i in tracked
            ]
            context_pool = scatter_rows(
                context_pool,
                mx.array(np.stack(boundaries).astype(np.int32)),
                mx.array([metadata.track_slots_host[i] for i in tracked], dtype=mx.int32),
            )
        pool.slot_states[self.context_name] = context_pool
        return self.table.lookup(ids, self.embedding_dtype).reshape(len(ids), -1)

    def step_embeddings(self, hyper: mx.array, ticket: mx.array) -> mx.array:
        """A posted decode step's n-gram rows [T, ple_embed_dim], from the service once the
        step reaches hyper."""
        assert self.table is not None
        num_rows = hyper.shape[0]
        lookup = self.lookup
        parts = ple_wait(ticket, lookup.mailbox, lookup.staging, num_rows, after=[hyper])
        rows = self.table.dequantize(
            PleRowParts(*(part.reshape(-1, part.shape[-1]) for part in parts)),
            self.embedding_dtype,
        )
        return rows.reshape(num_rows, -1)

    def short_conv(self, normed: mx.array, metadata: LinearAttnMetadata, batch: Batch) -> mx.array:
        """silu of the causal dilated depthwise conv of normed [T, width] over each slot's
        window; writes the windows back (and a tracked snapshot's)."""
        pool = get_global_ctx().linear_state_pool
        assert pool is not None
        weight = self.conv1d.weight
        windows = pool.slot_states[self.conv_name]
        if batch.is_decode:
            slots = metadata.cache_indices
            conv_input = mx.concatenate([windows[slots], normed[:, None, :]], axis=1)
            out = mx.conv1d(conv_input, weight, dilation=self.dilation, groups=weight.shape[0])
            pool.slot_states[self.conv_name] = scatter_rows(
                windows, conv_input[:, -self.state_len :], slots
            )
            return nn.silu(out[:, 0])
        else:
            return self.prefill_conv(normed, windows, metadata, batch)

    def prefill_conv(
        self, normed: mx.array, windows: mx.array, metadata: LinearAttnMetadata, batch: Batch
    ) -> mx.array:
        """short_conv of a prefill, one request at a time from its slot's window (zeros for a
        fresh one)."""
        pool = get_global_ctx().linear_state_pool
        assert pool is not None
        weight = self.conv1d.weight
        outputs = []
        start = 0
        for i, seq_len in enumerate(metadata.seq_lens_host):
            req = batch.padded_reqs[i]
            slot = metadata.cache_indices[i : i + 1]
            if req.cached_len > 0:
                state = windows[slot][0]
            else:
                state = mx.zeros((self.state_len, normed.shape[1]), dtype=normed.dtype)
            conv_input = mx.concatenate([state, normed[start : start + seq_len]])
            out = mx.conv1d(
                conv_input[None], weight, dilation=self.dilation, groups=weight.shape[0]
            )
            outputs.append(nn.silu(out[0]))
            windows = scatter_rows(windows, conv_input[-self.state_len :][None], slot)
            if metadata.track_slots_host[i] >= 0:
                split = metadata.track_seqlens_host[i] - req.cached_len
                window = conv_input[split : split + self.state_len]
                windows = scatter_rows(windows, window[None], metadata.track_slots[i : i + 1])
            start += seq_len
        pool.slot_states[self.conv_name] = windows
        return mx.concatenate(outputs) if len(outputs) > 1 else outputs[0]

    def forward(self, hyper: mx.array, batch: Batch, ticket: mx.array | None) -> mx.array:
        """The branch added to the residual streams hyper [T, hc_count * H]; ticket: the decode
        step's post (post_decode), None to look the rows up on the host."""
        metadata = batch.attn_metadata[AttnType.LINEAR]
        assert isinstance(metadata, LinearAttnMetadata)
        num_tokens = hyper.shape[0]
        streams = (num_tokens, self.hc_count, self.hidden_size)
        if ticket is None:
            embeddings = self.host_embeddings(batch, metadata)
        else:
            embeddings = self.step_embeddings(hyper, ticket)
        keys = self.norm_key.forward(self.key_proj.forward(embeddings)).reshape(streams)
        values = self.value_proj.forward(embeddings)
        queries = self.norm_query.forward(hyper).reshape(streams)
        scores = mx.sum(
            keys.astype(mx.float32) * queries.astype(mx.float32), axis=-1, keepdims=True
        )
        scores = scores / math.sqrt(self.hidden_size)
        scores = mx.sign(scores) * mx.sqrt(mx.maximum(mx.abs(scores), 1e-6))
        gated = (mx.sigmoid(scores) * values.astype(mx.float32)[:, None, :]).reshape(hyper.shape)
        gated = gated.astype(hyper.dtype)
        conv = self.short_conv(self.norm_conv.forward(gated), metadata, batch)
        return gated + conv


__all__ = ["NGramTable", "Qwen4ExpPLE", "StepLookup"]
