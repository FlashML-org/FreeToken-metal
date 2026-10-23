"""Gated full attention of Qwen3.5: q/k RMSNorm, partial rope, per-head output gate.

Port of FreeToken models/qwen3_5_moe/attention.py. Divergences: positions come
from the batch (its mrope rows when the model serves vision), the backend from the global
Context by attention type; the merged qkv projection is cut with forward_split.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.attention.base import AttnType
from ftmetal.attention.full.common import FullAttnBackend
from ftmetal.context import get_global_ctx
from ftmetal.core import Batch
from ftmetal.layers.activation import sigmoid_gate
from ftmetal.layers.base import BaseOP
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.norm import RMSNorm
from ftmetal.layers.quantization.config import QuantScheme
from ftmetal.layers.rotary import MRotaryEmbedding, RotaryEmbedding
from ftmetal.models.config import ModelConfig
from ftmetal.models.qwen3_5_moe.weight import QKV_PARTS


class Qwen3_5Attention(BaseOP):
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
        head_dim = config.head_dim
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = self.num_q * head_dim
        self.kv_attn_dim = self.num_kv * head_dim
        # The q part is 2x wide: each head's [query | gate].
        self.qkv_proj = LinearMerged(
            config.hidden_size,
            [self.num_q * head_dim * 2, self.kv_attn_dim, self.kv_attn_dim],
            schemes=qkv_schemes,
            part_names=QKV_PARTS,
            dtype=dtype,
        )
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, dtype=dtype)
        rotary = config.rotary_config
        if rotary.mrope_section is None:
            self.rotary = RotaryEmbedding(head_dim, rotary.rotary_dim, rotary.base)
        else:
            self.rotary = MRotaryEmbedding(
                head_dim, rotary.rotary_dim, rotary.base, rotary.mrope_section, rotary.mrope_layout
            )
        self.o_proj = Linear(self.qo_attn_dim, config.hidden_size, scheme=o_scheme, dtype=dtype)

    def forward(self, x: mx.array, batch: Batch) -> mx.array:
        num_tokens = x.shape[0]
        qg, k, v = self.qkv_proj.forward_split(x)
        qg = qg.reshape(num_tokens, self.num_q, 2 * self.head_dim)
        q, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(num_tokens, self.qo_attn_dim)
        positions = batch.get_attn_positions()
        q = self.rotary.forward(positions, self.q_norm.forward(q))
        k = self.rotary.forward(
            positions, self.k_norm.forward(k.reshape(num_tokens, self.num_kv, self.head_dim))
        )
        v = v.reshape(num_tokens, self.num_kv, self.head_dim)
        backend = get_global_ctx().attn_backends[AttnType.FULL]
        assert isinstance(backend, FullAttnBackend)
        o = backend.forward(q, k, v, self.layer_id, batch)
        o = sigmoid_gate(o.reshape(num_tokens, self.qo_attn_dim), gate)
        return self.o_proj.forward(o)


__all__ = ["Qwen3_5Attention"]
