"""Qwen3.5 / 3.6 hybrid GDN + gated-attention MoE decoder.

Port of FreeToken models/qwen3_5_moe/model.py. Divergences: the batch flows
through forward explicitly; norms are plain RMSNorm because MLX checkpoints store
the (1 + w) shift; each Linear takes its affine scheme from the checkpoint's
quantization map at construction.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.core import Batch
from ftmetal.layers.base import BaseOP
from ftmetal.layers.embedding import Embedding
from ftmetal.layers.linear import Linear
from ftmetal.layers.norm import RMSNorm
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.models.blocks import BaseLLMModel, select_last_rows
from ftmetal.models.config import ModelConfig
from ftmetal.models.qwen3_5_moe.attention import Qwen3_5Attention
from ftmetal.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from ftmetal.models.qwen3_5_moe.moe import Qwen3_5MoE
from ftmetal.models.qwen3_5_moe.weight import (
    GATE_UP_PARTS,
    IN_PROJ_BA_PARTS,
    IN_PROJ_QKVZ_PARTS,
    QKV_PARTS,
    checkpoint_module,
)


def scheme_of(config: ModelConfig, model_key: str) -> AffineScheme | None:
    if config.quant is None:
        return None
    return config.quant.scheme_for(checkpoint_module(model_key))


def fused_scheme_of(
    config: ModelConfig, model_prefix: str, parts: tuple[str, ...]
) -> AffineScheme | None:
    if config.quant is None:
        return None
    names = [checkpoint_module(f"{model_prefix}.{part}") for part in parts]
    return config.quant.scheme_for_fused(names)


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str, dtype: mx.Dtype) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_linear = config.is_linear_layer(layer_id)
        if self.is_linear:
            group = config.linear_attention_group()
            assert group is not None
            gdn_prefix = f"{prefix}.linear_attn"
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config.hidden_size,
                group,
                config.rms_norm_eps,
                layer_id,
                qkvz_scheme=fused_scheme_of(config, gdn_prefix, IN_PROJ_QKVZ_PARTS),
                ba_scheme=fused_scheme_of(config, gdn_prefix, IN_PROJ_BA_PARTS),
                out_scheme=scheme_of(config, f"{gdn_prefix}.out_proj"),
                dtype=dtype,
            )
        else:
            attn_prefix = f"{prefix}.self_attn"
            self.self_attn = Qwen3_5Attention(
                config,
                layer_id,
                qkv_scheme=fused_scheme_of(config, attn_prefix, QKV_PARTS),
                o_scheme=scheme_of(config, f"{attn_prefix}.o_proj"),
                dtype=dtype,
            )
        mlp_prefix = f"{prefix}.mlp"
        self.mlp = Qwen3_5MoE(
            config,
            experts_gate_up_scheme=fused_scheme_of(config, f"{mlp_prefix}.experts", GATE_UP_PARTS),
            experts_down_scheme=scheme_of(config, f"{mlp_prefix}.experts.down_proj"),
            gate_scheme=scheme_of(config, f"{mlp_prefix}.gate"),
            shared_gate_up_scheme=fused_scheme_of(
                config, f"{mlp_prefix}.shared_expert", GATE_UP_PARTS
            ),
            shared_down_scheme=scheme_of(config, f"{mlp_prefix}.shared_expert.down_proj"),
            shared_gate_scheme=scheme_of(config, f"{mlp_prefix}.shared_expert_gate"),
            dtype=dtype,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )

    def forward(self, x: mx.array, batch: Batch) -> mx.array:
        normed = self.input_layernorm.forward(x)
        if self.is_linear:
            r = self.linear_attn.forward(normed, batch)
        else:
            r = self.self_attn.forward(normed, batch)
        h = x + r
        return h + self.mlp.forward(self.post_attention_layernorm.forward(h))


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig, *, dtype: mx.Dtype) -> None:
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            scheme=scheme_of(config, "model.embed_tokens"),
            dtype=dtype,
        )
        self.layers = [
            Qwen3_5DecoderLayer(config, layer_id, prefix=f"model.layers.{layer_id}", dtype=dtype)
            for layer_id in range(config.num_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(self, batch: Batch) -> mx.array:
        x = self.embed_tokens.forward(batch.input_ids)
        for layer in self.layers:
            x = layer.forward(x, batch)
        return self.norm.forward(x)


class Qwen3_5MoeForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig, *, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config, dtype=dtype)
        if not config.tie_word_embeddings:
            self.lm_head = Linear(
                config.hidden_size,
                config.vocab_size,
                scheme=scheme_of(config, "lm_head"),
                dtype=dtype,
            )

    def forward(self, batch: Batch) -> mx.array:
        hidden = select_last_rows(self.model.forward(batch), batch)
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head.forward(hidden)


__all__ = ["Qwen3_5DecoderLayer", "Qwen3_5Model", "Qwen3_5MoeForCausalLM"]
