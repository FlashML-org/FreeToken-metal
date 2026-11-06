"""Flint attention for one full or sliding-window layer: q/k RMSNorm per head, rope on the sliding
layers only, a learned attention sink per query head.

Port of FreeToken models/flint/attention.py. Divergences: positions come from the batch and the
backend from the global Context by the layer's attention type (the window is the backend's); the
merged projection is cut with forward_split.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.attention.base import AttentionSpec
from ftmetal.attention.full.common import FullAttnBackend
from ftmetal.attention.metadata import AttnType
from ftmetal.context import get_global_ctx
from ftmetal.core import Batch
from ftmetal.layers.base import BaseOP
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.norm import RMSNorm
from ftmetal.layers.quantization.config import QuantScheme
from ftmetal.layers.rotary import RotaryEmbedding
from ftmetal.models.config import FullAttentionGroupConfig, ModelConfig, SWAAttentionGroupConfig
from ftmetal.models.flint.weight import QKV_PARTS


class FlintAttention(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        qkv_schemes: list[QuantScheme | None],
        o_scheme: QuantScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        group = config.attention_group_for_layer(layer_id)
        assert isinstance(group, (FullAttentionGroupConfig, SWAAttentionGroupConfig))
        self.layer_id = layer_id
        self.attn_type = (
            AttnType.SWA if isinstance(group, SWAAttentionGroupConfig) else AttnType.FULL
        )
        self.num_q = config.num_qo_heads
        self.num_kv = group.num_kv_heads
        self.head_dim = group.head_dim
        q_dim = self.num_q * self.head_dim
        kv_dim = self.num_kv * self.head_dim
        self.qkv_proj = LinearMerged(
            config.hidden_size,
            [q_dim, kv_dim, kv_dim],
            schemes=qkv_schemes,
            part_names=QKV_PARTS,
            dtype=dtype,
        )
        self.o_proj = Linear(q_dim, config.hidden_size, scheme=o_scheme, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.softmax_offset = mx.zeros((self.num_q,), dtype=dtype)
        rotary = group.rotary_config
        self.rotary = (
            RotaryEmbedding(self.head_dim, rotary.rotary_dim, rotary.base)
            if rotary.rotary_dim > 0
            else None
        )

    def forward(self, x: mx.array, batch: Batch) -> mx.array:
        num_tokens = x.shape[0]
        q, k, v = self.qkv_proj.forward_split(x)
        q = self.q_norm.forward(q.reshape(num_tokens, self.num_q, self.head_dim))
        k = self.k_norm.forward(k.reshape(num_tokens, self.num_kv, self.head_dim))
        v = v.reshape(num_tokens, self.num_kv, self.head_dim)
        if self.rotary is not None:
            q = self.rotary.forward(batch.positions, q)
            k = self.rotary.forward(batch.positions, k)
        backend = get_global_ctx().attn_backends[self.attn_type]
        assert isinstance(backend, FullAttnBackend)
        o = backend.forward(
            q, k, v, self.layer_id, batch, attn_spec=AttentionSpec(sinks=self.softmax_offset)
        )
        return self.o_proj.forward(o.reshape(num_tokens, self.num_q * self.head_dim))


__all__ = ["FlintAttention"]
