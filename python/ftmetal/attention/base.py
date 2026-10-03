"""Attention taxonomy and the backend base.

Port of FreeToken attention/base.py. Divergences: only the FULL, SWA and LINEAR types
exist, the base backend defines prepare_metadata and the capture hooks only (each
attention type fixes its own forward signature), AttnType and the metadata records live
in attention/metadata.py, and the capture hooks serve CommandGraph: a replay rebinds its
step inputs, so prepare_for_replay turns the batch's prepare_metadata record into its
bucket's fixed shapes as new arrays instead of copying it into static buffers; a
capture batch's replays are compared bit for bit, so a backend whose dummy rows would
race on shared state gives them their own (prepare_for_capture, reset_capture_state)
and returns it after the captures (release_capture).
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
        """Fix the decode metadata shapes of every graph (max_seq_len keys, bs_list buckets),
        and borrow what the capture batches need until release_capture."""

    def prepare_for_capture(self, batch: Batch) -> None:
        """prepare_for_replay for a batch of dummy requests whose replays must agree bit for
        bit: each starts from the state reset_capture_state leaves."""
        self.prepare_for_replay(batch)

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """Turn a padded decode batch's metadata into the fixed shapes of its bucket's graph."""

    def reset_capture_state(self) -> None:
        """Reset the state a capture batch's rows read, before each replay of it."""

    def release_capture(self) -> None:
        """Return what init_capture_graph borrowed; replays need none of it."""


__all__ = ["AttentionSpec", "AttnType", "BaseAttnBackend", "BaseAttnMetadata"]
