"""Qwen3.8-Flash-Next's QSA layer: Qwen3.5's gated attention plus a block indexer.

Port of FreeToken models/qwen4_exp/attention.py (Qwen4ExpAttention, QSAIndexer).
Divergences: positions come from the batch (its mrope rows when the model serves vision);
the backend from the global Context by attention type, which stores the keys and the index
keys and selects; the indexer hands it normed and roped index queries, raw index keys and
its block key norm (the k_layernorm, then the rope at the block's first position).
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.attention.metadata import AttnType
from ftmetal.attention.qsa.common import QSAAttnBackend, QSAIndexInputs
from ftmetal.context import get_global_ctx
from ftmetal.core import Batch
from ftmetal.kernel.norm.offset_rms_norm import offset_rms_norm_rope, supports_offset_rms_norm_rope
from ftmetal.layers.activation import sigmoid_gate
from ftmetal.layers.base import BaseOP
from ftmetal.layers.linear import Linear, LinearMerged
from ftmetal.layers.quantization.config import QuantScheme
from ftmetal.layers.rotary import MRotaryEmbedding, RotaryEmbedding
from ftmetal.models.config import QSAAttentionGroupConfig, RotaryConfig
from ftmetal.models.qwen4_exp.config import Qwen4ExpConfig
from ftmetal.models.qwen4_exp.norm import CenteredRMSNorm
from ftmetal.models.qwen4_exp.projection import project, project_split
from ftmetal.models.qwen4_exp.weight import QKV_PARTS

# Forwards of at most this many tokens norm and rope each head in one launch (the same
# values as the norm then the rope).
NORM_ROPE_MAX_TOKENS = 4


def rotary_embedding(head_dim: int, rotary: RotaryConfig) -> RotaryEmbedding | MRotaryEmbedding:
    """The partial rope, 3-axis when the model serves vision."""
    if rotary.mrope_section is None:
        return RotaryEmbedding(head_dim, rotary.rotary_dim, rotary.base)
    else:
        return MRotaryEmbedding(
            head_dim, rotary.rotary_dim, rotary.base, rotary.mrope_section, rotary.mrope_layout
        )


def norm_rope(
    norm: CenteredRMSNorm,
    rotary: RotaryEmbedding | MRotaryEmbedding,
    positions: mx.array,
    x: mx.array,
) -> mx.array:
    """rotary(positions, norm(x)) for heads x [T, heads, D]: one launch for a few tokens under
    the 1-D rope."""
    if (
        isinstance(rotary, RotaryEmbedding)
        and x.shape[0] <= NORM_ROPE_MAX_TOKENS
        and supports_offset_rms_norm_rope(x, rotary.rotary_dim)
    ):
        return offset_rms_norm_rope(
            x, norm.weight, norm.eps, positions, rotary.rotary_dim, rotary.base
        )
    else:
        return rotary.forward(positions, norm.forward(x))


class QSAIndexer(BaseOP):
    def __init__(
        self,
        config: Qwen4ExpConfig,
        group: QSAAttentionGroupConfig,
        *,
        scheme: QuantScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        self.num_heads = group.index_num_heads
        self.head_dim = group.index_head_dim
        # The query heads, then the one key head.
        self.index_qk_proj = Linear(
            config.hidden_size, (self.num_heads + 1) * self.head_dim, scheme=scheme, dtype=dtype
        )
        self.q_layernorm = CenteredRMSNorm(self.head_dim, config.rms_norm_eps, dtype=dtype)
        self.k_layernorm = CenteredRMSNorm(self.head_dim, config.rms_norm_eps, dtype=dtype)
        # The attention's partial rope (its rotary dims and frequencies) on index vectors.
        self.rotary = rotary_embedding(self.head_dim, group.rotary_config)

    def block_key_norm(self, pooled: mx.array, positions: mx.array) -> mx.array:
        """Block keys [N, D] of the pooled raw keys [N, D] of blocks starting at positions [N]
        ([3, N] under the 3-axis rope: a block may start on image tokens)."""
        return norm_rope(self.k_layernorm, self.rotary, positions, pooled[:, None, :])[:, 0]

    def forward(self, x: mx.array, positions: mx.array) -> QSAIndexInputs:
        num_tokens = x.shape[0]
        projected = project(self.index_qk_proj, x).reshape(
            num_tokens, self.num_heads + 1, self.head_dim
        )
        queries = norm_rope(
            self.q_layernorm, self.rotary, positions, projected[:, : self.num_heads]
        )
        return QSAIndexInputs(
            queries=queries, keys=projected[:, self.num_heads], block_key_norm=self.block_key_norm
        )


class Qwen4ExpAttention(BaseOP):
    def __init__(
        self,
        config: Qwen4ExpConfig,
        layer_id: int,
        *,
        qkv_schemes: list[QuantScheme | None],
        o_scheme: QuantScheme | None,
        index_scheme: QuantScheme | None,
        dtype: mx.Dtype,
    ) -> None:
        super().__init__()
        group = config.qsa_attention_group()
        assert group is not None
        head_dim = group.head_dim
        self.layer_id = layer_id
        self.num_q = config.num_qo_heads
        self.num_kv = group.num_kv_heads
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
        self.q_norm = CenteredRMSNorm(head_dim, config.rms_norm_eps, dtype=dtype)
        self.k_norm = CenteredRMSNorm(head_dim, config.rms_norm_eps, dtype=dtype)
        self.rotary = rotary_embedding(head_dim, group.rotary_config)
        self.o_proj = Linear(self.qo_attn_dim, config.hidden_size, scheme=o_scheme, dtype=dtype)
        self.indexer = QSAIndexer(config, group, scheme=index_scheme, dtype=dtype)

    def forward(self, x: mx.array, batch: Batch) -> mx.array:
        num_tokens = x.shape[0]
        qg, k, v = project_split(self.qkv_proj, x)
        qg = qg.reshape(num_tokens, self.num_q, 2 * self.head_dim)
        q, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(num_tokens, self.qo_attn_dim)
        positions = batch.get_attn_positions()
        q = norm_rope(self.q_norm, self.rotary, positions, q)
        k = norm_rope(
            self.k_norm, self.rotary, positions, k.reshape(num_tokens, self.num_kv, self.head_dim)
        )
        v = v.reshape(num_tokens, self.num_kv, self.head_dim)
        index = self.indexer.forward(x, positions)
        backend = get_global_ctx().attn_backends[AttnType.QSA]
        assert isinstance(backend, QSAAttnBackend)
        o = backend.qsa_forward(q, k, v, index, self.layer_id, batch)
        o = sigmoid_gate(o.reshape(num_tokens, self.qo_attn_dim), gate)
        return project(self.o_proj, o)


__all__ = ["QSAIndexer", "Qwen4ExpAttention"]
