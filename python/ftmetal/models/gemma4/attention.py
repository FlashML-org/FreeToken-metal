"""Gemma4 attention for one global or sliding-window layer.

Port of FreeToken models/gemma4/attention.py. Divergences: positions come from the
batch and the backend from the global Context by the layer's attention type (the window
and the sm_scale of 1.0 are the backend's, not a per-call AttentionSpec); a k_eq_v
global layer projects q and k only, its V being the K projection's output before the k
norm and rope; the merged projection is cut with forward_split.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.attention.full.common import FullAttnBackend
from ftmetal.attention.metadata import AttnType
from ftmetal.context import get_global_ctx
from ftmetal.core import Batch
from ftmetal.layers.base import BaseOP
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.norm import RMSNorm
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.layers.rotary import RotaryEmbedding
from ftmetal.models.config import FullAttentionGroupConfig, ModelConfig, SWAAttentionGroupConfig


class Gemma4Attention(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        qkv_scheme: AffineScheme | None,
        o_scheme: AffineScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        group = config.attention_group_for_layer(layer_id)
        assert isinstance(group, (FullAttentionGroupConfig, SWAAttentionGroupConfig))
        self.layer_id = layer_id
        self.attn_type = (
            AttnType.SWA if isinstance(group, SWAAttentionGroupConfig) else AttnType.FULL
        )
        self.k_eq_v = isinstance(group, FullAttentionGroupConfig) and group.k_eq_v
        self.num_q = config.num_qo_heads
        self.num_kv = group.num_kv_heads
        self.head_dim = group.head_dim
        q_dim = self.num_q * self.head_dim
        kv_dim = self.num_kv * self.head_dim
        if self.k_eq_v:
            self.qk_proj = LinearMerged(
                config.hidden_size, [q_dim, kv_dim], scheme=qkv_scheme, dtype=dtype
            )
        else:
            self.qkv_proj = LinearMerged(
                config.hidden_size, [q_dim, kv_dim, kv_dim], scheme=qkv_scheme, dtype=dtype
            )
        self.o_proj = Linear(q_dim, config.hidden_size, scheme=o_scheme, dtype=dtype)
        eps = config.rms_norm_eps
        self.q_norm = RMSNorm(self.head_dim, eps=eps, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=eps, dtype=dtype)
        self.v_norm = RMSNorm(self.head_dim, eps=eps, dtype=dtype, with_scale=False)
        rotary = group.rotary_config
        self.rotary = RotaryEmbedding(
            self.head_dim,
            rotary.rotary_dim,
            rotary.base,
            proportional=rotary.rope_type == "proportional",
        )

    def forward(self, x: mx.array, batch: Batch) -> mx.array:
        num_tokens = x.shape[0]
        if self.k_eq_v:
            q, k = self.qk_proj.forward_split(x)
            v = k
        else:
            q, k, v = self.qkv_proj.forward_split(x)
        q = q.reshape(num_tokens, self.num_q, self.head_dim)
        k = k.reshape(num_tokens, self.num_kv, self.head_dim)
        v = v.reshape(num_tokens, self.num_kv, self.head_dim)
        q = self.rotary.forward(batch.positions, self.q_norm.forward(q))
        k = self.rotary.forward(batch.positions, self.k_norm.forward(k))
        v = self.v_norm.forward(v)
        backend = get_global_ctx().attn_backends[self.attn_type]
        assert isinstance(backend, FullAttnBackend)
        o = backend.forward(q, k, v, self.layer_id, batch)
        return self.o_proj.forward(o.reshape(num_tokens, self.num_q * self.head_dim))


__all__ = ["Gemma4Attention"]
