"""Attention taxonomy and the backend base.

Port of FreeToken attention/base.py. Divergences: only the FULL and LINEAR types
exist, the base backend defines prepare_metadata and the capture hooks only (each
attention type fixes its own forward signature), AttnType and the metadata records live
in attention/metadata.py, and the capture hooks serve CommandGraph: nothing is
allocated for a capture, and prepare_for_replay builds a padded batch's metadata at the
bucket's fixed shapes as new arrays (the graph rebinds them) instead of copying into
static buffers.
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
        """Build this step's metadata from the batch and store it on the batch."""

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: list[int]) -> None:
        """Fix the decode metadata shapes of every graph: max_seq_len keys, bs_list buckets."""

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None:
        """The fixed-shape decode metadata of a batch of dummy requests, for its capture."""

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """The fixed-shape decode metadata of a padded batch, for a replay of its bucket."""


__all__ = ["AttentionSpec", "AttnType", "BaseAttnBackend", "BaseAttnMetadata"]
