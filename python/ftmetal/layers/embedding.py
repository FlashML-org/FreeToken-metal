"""Token embedding, bf16 or affine quantized, with the tied LM head projection.

Port of FreeToken layers/embedding.py. Divergences: no vocab parallelism, so
VocabParallelEmbedding is Embedding; an untied head is a plain Linear and the tied
head is as_linear here (ParallelLMHead is gone).
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.config import AffineScheme


class Embedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        scheme: AffineScheme | None = None,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.scheme = scheme
        if scheme is None:
            self.weight = mx.zeros((num_embeddings, embedding_dim), dtype=dtype)
        else:
            packed = embedding_dim * scheme.bits // 32
            groups = embedding_dim // scheme.group_size
            self.weight = mx.zeros((num_embeddings, packed), dtype=mx.uint32)
            self.scales = mx.zeros((num_embeddings, groups), dtype=dtype)
            self.biases = mx.zeros((num_embeddings, groups), dtype=dtype)

    def forward(self, input_ids: mx.array) -> mx.array:
        if self.scheme is None:
            return self.weight[input_ids]
        return mx.dequantize(
            self.weight[input_ids],
            self.scales[input_ids],
            self.biases[input_ids],
            group_size=self.scheme.group_size,
            bits=self.scheme.bits,
            mode="affine",
        )

    def as_linear(self, x: mx.array) -> mx.array:
        """x @ weight.T: the tied LM head."""
        if self.scheme is None:
            return x @ self.weight.T
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.scheme.group_size,
            bits=self.scheme.bits,
            mode="affine",
        )


__all__ = ["Embedding"]
