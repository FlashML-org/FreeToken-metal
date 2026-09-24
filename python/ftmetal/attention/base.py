"""Attention taxonomy and the backend base.

Port of FreeToken attention/base.py. Divergences: only the FULL and LINEAR types
exist, the base backend defines prepare_metadata alone (each attention type fixes
its own forward signature), the metadata records live in attention/metadata.py and
the CUDA-graph capture hooks are gone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import mlx.core as mx

from ftmetal.attention.metadata import BaseAttnMetadata
from ftmetal.core import Batch


class AttnType(str, Enum):
    FULL = "full"
    LINEAR = "linear"


@dataclass
class AttentionSpec:
    sliding_window: int | None = None
    sm_scale: float | None = None
    sinks: mx.array | None = None


class BaseAttnBackend(ABC):
    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """Build this step's metadata from the batch and store it on the batch."""


__all__ = ["AttentionSpec", "AttnType", "BaseAttnBackend", "BaseAttnMetadata"]
