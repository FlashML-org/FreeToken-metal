"""Request and batch records shared by the scheduler, engine and model.

Port of FreeToken core.py. Divergences: input ids are host numpy arrays and batch
tensors are mx.arrays; the batch is passed to the model instead of living in the
Context, its attention metadata is a dict keyed by AttnType (fla_metadata is the
LINEAR entry); padded_reqs starts as reqs and only the graph runner pads it; a Req's
cache_handle is None while it locks no cached prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx
import numpy as np

from ftmetal.attention.metadata import AttnType, BaseAttnMetadata
from ftmetal.kvcache.hybrid_radix_cache import HybridCacheHandle


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
    stop_strs: list[str] = field(default_factory=list)

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: np.ndarray
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    # The locked prefix-cache match; None when nothing is locked (a miss, or no prefix cache).
    cache_handle: HybridCacheHandle | None = None
    # GDN state slots; None for models without linear attention.
    linear_slot_idx: int | None = None
    # Two track slots that take the prefill snapshots in turn (chunk N + 1 launches before
    # chunk N's snapshot is committed).
    mamba_ping_pong: tuple[int, int] | None = None
    # Which ping-pong slot takes the next snapshot.
    mamba_next_track_idx: int = 0
    # The token count of the snapshot written by the last tracked forward, until committed.
    mamba_last_track_seqlen: int | None = None
    # On a prefix hit: the tree's snapshot slot to copy into the live slot before the first chunk.
    mamba_restore_src: int | None = None
    aborted: bool = False

    def __post_init__(self) -> None:
        self.input_ids = np.asarray(self.input_ids, dtype=np.int32)
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        self.alloc_ids_buf()

    def alloc_ids_buf(self) -> None:
        self.ids_buf = np.empty(self.max_device_len, dtype=np.int32)
        self.ids_buf[: self.device_len] = self.input_ids
        self.input_ids = self.ids_buf[: self.device_len]

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: int) -> None:
        n = len(self.input_ids)
        assert n + 1 <= self.max_device_len
        self.ids_buf[n] = next_token
        self.input_ids = self.ids_buf[: n + 1]

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    @property
    def prompt_len(self) -> int:
        return self.max_device_len - self.output_len

    @property
    def generated_len(self) -> int:
        """Sampled tokens appended so far; unlike device_len it ignores an in-flight step."""
        return len(self.input_ids) - self.prompt_len

    def __repr__(self) -> str:
        return (
            f"Req(uid={self.uid}, table_idx={self.table_idx}, cached_len={self.cached_len}, "
            f"device_len={self.device_len}, max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: list[Req]
    phase: Literal["prefill", "decode"]
    # Set by the scheduler before the forward.
    input_ids: mx.array = field(init=False)
    positions: mx.array = field(init=False)
    out_loc: mx.array = field(init=False)
    # One record per attention backend, set by its prepare_metadata.
    attn_metadata: dict[AttnType, BaseAttnMetadata] = field(default_factory=dict, init=False)
    # reqs followed by the dummy requests that fill a replayed decode batch to its graph's
    # batch size; set by GraphRunner.pad_batch.
    padded_reqs: list[Req] = field(init=False)

    def __post_init__(self) -> None:
        self.padded_reqs = self.reqs

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)

    @property
    def num_tokens(self) -> int:
        return sum(req.extend_len for req in self.reqs)


def host_ints(values: mx.array) -> list[int]:
    """A 1-D integer array as Python ints (one device sync)."""
    listed = values.tolist()
    assert isinstance(listed, list)
    ints = [value for value in listed if isinstance(value, int)]
    assert len(ints) == len(listed), "host_ints needs a 1-D integer array"
    return ints


def host_floats(values: mx.array) -> list[float]:
    """A 1-D floating array as Python floats (one device sync)."""
    listed = values.astype(mx.float32).tolist()
    assert isinstance(listed, list)
    floats = [value for value in listed if isinstance(value, float)]
    assert len(floats) == len(listed), "host_floats needs a 1-D array"
    return floats


__all__ = ["Batch", "Req", "SamplingParams", "host_floats", "host_ints"]
