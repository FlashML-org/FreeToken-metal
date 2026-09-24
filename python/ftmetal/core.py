"""Request and batch records shared by the scheduler, engine and model.

Port of FreeToken core.py. Divergences: input ids are host numpy arrays and batch
tensors are mx.arrays; the batch is passed to the model instead of living in the
Context, its attention metadata is a dict keyed by AttnType (fla_metadata is the
LINEAR entry), and there are no padded request rows (nothing is captured into a graph).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx
import numpy as np

from ftmetal.attention.metadata import AttnType, BaseAttnMetadata


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
    # GDN state slots; None for models without linear attention.
    linear_slot_idx: int | None = None
    mamba_ping_pong: tuple[int, int] | None = None
    mamba_next_track_idx: int = 0
    mamba_last_track_seqlen: int | None = None
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
    def num_tokens(self) -> int:
        return sum(req.extend_len for req in self.reqs)


__all__ = ["Batch", "Req", "SamplingParams"]
