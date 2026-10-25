"""Embedding methods: a vocab table's arrays for its scheme, its row lookup, and the tied LM
head's product.

Port of FreeToken layers/quantization/embedding/ (EmbeddingConfig, EmbeddingMethod and its
unquantized method). Divergences: as layers/quantization/linear.py's, embedding_method picks the
method and a method has no kernel table or finalize step; MLX affine and float block modes
take the place of FreeToken's GGUF kind; no vocab parallelism; the table is its kind's linear
method's weight, so create_weights and logits are that method's and the tied head has no bias.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import mlx.core as mx

from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.config import AffineScheme, FpBlockScheme, QuantScheme
from ftmetal.layers.quantization.linear import LinearConfig, linear_method


@dataclass(frozen=True, kw_only=True)
class EmbeddingConfig:
    """The table's shape, its scheme, and the float dtype a lookup returns."""

    num_embeddings: int
    embedding_dim: int
    scheme: QuantScheme | None
    dtype: mx.Dtype

    def as_linear(self) -> LinearConfig:
        """The table as a [num_embeddings, embedding_dim] linear weight: the tied head's."""
        return LinearConfig(
            in_features=self.embedding_dim,
            out_features=self.num_embeddings,
            scheme=self.scheme,
            dtype=self.dtype,
        )


class EmbeddingMethod(ABC):
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self.linear = linear_method(cfg.as_linear())

    def create_weights(self, layer: BaseOP) -> None:
        """Set the layer's table arrays: placeholders the loader replaces."""
        self.linear.create_weights(layer)

    @abstractmethod
    def embed(self, weights: Mapping[str, mx.array], input_ids: mx.array) -> mx.array:
        """The table's rows at input_ids."""

    def logits(self, weights: Mapping[str, mx.array], x: mx.array) -> mx.array:
        """x @ table.T: the tied LM head."""
        return self.linear.apply(weights, x)


class UnquantizedEmbeddingMethod(EmbeddingMethod):
    def embed(self, weights: Mapping[str, mx.array], input_ids: mx.array) -> mx.array:
        return weights["weight"][input_ids]


class AffineEmbeddingMethod(EmbeddingMethod):
    def __init__(self, cfg: EmbeddingConfig, scheme: AffineScheme) -> None:
        """scheme: cfg's, as its kind."""
        super().__init__(cfg)
        self.scheme = scheme

    def embed(self, weights: Mapping[str, mx.array], input_ids: mx.array) -> mx.array:
        # Only the looked-up rows are dequantized, never the whole table.
        return mx.dequantize(
            weights["weight"][input_ids],
            weights["scales"][input_ids],
            weights["biases"][input_ids],
            group_size=self.scheme.group_size,
            bits=self.scheme.bits,
            mode="affine",
        )


class FpBlockEmbeddingMethod(EmbeddingMethod):
    def __init__(self, cfg: EmbeddingConfig, scheme: FpBlockScheme) -> None:
        """scheme: cfg's, as its kind."""
        super().__init__(cfg)
        self.scheme = scheme

    def embed(self, weights: Mapping[str, mx.array], input_ids: mx.array) -> mx.array:
        # uint8 scales carry no float dtype for the rows to take.
        return mx.dequantize(
            weights["weight"][input_ids],
            weights["scales"][input_ids],
            group_size=self.scheme.group_size,
            bits=self.scheme.bits,
            mode=self.scheme.mode,
            dtype=self.cfg.dtype,
        )


def embedding_method(cfg: EmbeddingConfig) -> EmbeddingMethod:
    if cfg.scheme is None:
        return UnquantizedEmbeddingMethod(cfg)
    elif isinstance(cfg.scheme, AffineScheme):
        return AffineEmbeddingMethod(cfg, cfg.scheme)
    elif cfg.scheme.has_output_scale:
        raise ValueError("an exported embedding table with a per-tensor scale is not served")
    else:
        return FpBlockEmbeddingMethod(cfg, cfg.scheme)


__all__ = [
    "AffineEmbeddingMethod",
    "EmbeddingConfig",
    "EmbeddingMethod",
    "FpBlockEmbeddingMethod",
    "UnquantizedEmbeddingMethod",
    "embedding_method",
]
