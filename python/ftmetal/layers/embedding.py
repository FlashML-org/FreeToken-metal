"""Token embedding whose table, lookup and tied LM head product are its scheme's embedding
method's.

Port of FreeToken layers/embedding.py. Divergences: no vocab parallelism, so
VocabParallelEmbedding is Embedding; an untied head is a plain Linear and the tied
head is as_linear here (ParallelLMHead is gone); the layer takes its scheme rather than the
quant config and prefix it would be looked up by.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.layers.base import BaseOP
from ftmetal.layers.quantization.config import QuantScheme
from ftmetal.layers.quantization.embedding import EmbeddingConfig, embedding_method


class Embedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        scheme: QuantScheme | None = None,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        cfg = EmbeddingConfig(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, scheme=scheme, dtype=dtype
        )
        self.quant_method = embedding_method(cfg)
        self.quant_method.create_weights(self)

    def forward(self, input_ids: mx.array) -> mx.array:
        return self.quant_method.embed(self, input_ids)

    def as_linear(self, x: mx.array) -> mx.array:
        """x @ weight.T: the tied LM head."""
        return self.quant_method.logits(self, x)


__all__ = ["Embedding"]
