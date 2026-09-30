"""Attention taxonomy and the backend base.

Port of FreeToken attention/base.py. Divergences: only the FULL and LINEAR types
exist, the base backend defines prepare_metadata and the capture hooks only (each
attention type fixes its own forward signature), AttnType and the metadata records live
in attention/metadata.py, and the capture hooks serve CommandGraph: nothing is
allocated for a capture, and a replay rebinds its step inputs, so prepare_for_replay
turns the batch's prepare_metadata record into its bucket's fixed shapes as new arrays
instead of copying it into static buffers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import mlx.core as mx

from ftmetal.attention.metadata import AttnType, BaseAttnMetadata
from ftmetal.core import Batch


@dataclass
class AttentionSpec:
    sliding_window: int | None = None
    sm_scale: float | None = None
    sinks: mx.array | None = None


class BaseAttnBackend(ABC):
    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """Build this step's metadata over batch.padded_reqs and store it on the batch."""

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        """Fix the decode metadata shapes of every graph: max_seq_len keys, bs_list buckets."""

    def prepare_for_capture(self, batch: Batch) -> None:
        """prepare_for_replay for a batch of dummy requests."""
        self.prepare_for_replay(batch)

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """Turn a padded decode batch's metadata into the fixed shapes of its bucket's graph."""


__all__ = ["AttentionSpec", "AttnType", "BaseAttnBackend", "BaseAttnMetadata"]
