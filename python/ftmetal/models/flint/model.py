"""Flint decoder: full and sliding-window attention with learned sinks, a leading run of dense MLP
layers, then a sigmoid-routed MoE with an ungated shared expert in every layer.

Port of FreeToken models/flint/model.py. Divergences: the batch flows through forward explicitly;
norms are plain RMSNorm, with no fused residual; each Linear takes its scheme from the checkpoint's
quantization map at construction; the dense MLP and the MoE block are the shared layers', whose
router projects in the engine dtype, as FreeToken's does, and scores in fp32.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.core import Batch
from ftmetal.layers.base import BaseOP
from ftmetal.layers.embedding import Embedding
from ftmetal.layers.linear import Linear
from ftmetal.layers.norm import RMSNorm
from ftmetal.layers.quantization.moe import ExpertSchemes
from ftmetal.layers.shared_expert_moe import GATE_UP_PARTS, SharedExpert, SharedExpertMoE
from ftmetal.models.blocks import BaseLLMModel, embed_input_ids, select_last_rows
from ftmetal.models.config import ModelConfig, scheme_of, schemes_of
from ftmetal.models.flint.attention import FlintAttention
from ftmetal.models.flint.weight import QKV_PARTS


class FlintDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str, dtype: mx.Dtype) -> None:
        """prefix: the layer's checkpoint module path."""
        super().__init__()
        attn_prefix = f"{prefix}.self_attn"
        mlp_prefix = f"{prefix}.mlp"
        self.self_attn = FlintAttention(
            config,
            layer_id,
            qkv_schemes=schemes_of(config, attn_prefix, QKV_PARTS),
            o_scheme=scheme_of(config, f"{attn_prefix}.o_proj"),
            dtype=dtype,
        )
        self.mlp: SharedExpert | SharedExpertMoE
        if layer_id < config.first_k_dense_replace:
            self.mlp = SharedExpert(
                config.hidden_size,
                config.intermediate_size,
                gate_up_schemes=schemes_of(config, mlp_prefix, GATE_UP_PARTS),
                down_scheme=scheme_of(config, f"{mlp_prefix}.down_proj"),
                dtype=dtype,
            )
        else:
            self.mlp = SharedExpertMoE(
                config,
                layer_id,
                experts_schemes=ExpertSchemes(
                    *schemes_of(config, f"{mlp_prefix}.switch_mlp", (*GATE_UP_PARTS, "down_proj")),
                    is_gate_up_apart=config.is_expert_gate_up_apart,
                ),
                gate_scheme=scheme_of(config, f"{mlp_prefix}.gate"),
                shared_gate_up_schemes=schemes_of(
                    config, f"{mlp_prefix}.shared_expert", GATE_UP_PARTS
                ),
                shared_down_scheme=scheme_of(config, f"{mlp_prefix}.shared_expert.down_proj"),
                shared_gate_scheme=None,
                dtype=dtype,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )

    def forward(self, x: mx.array, batch: Batch) -> mx.array:
        h = x + self.self_attn.forward(self.input_layernorm.forward(x), batch)
        return h + self.mlp.forward(self.post_attention_layernorm.forward(h))


class FlintModel(BaseOP):
    def __init__(self, config: ModelConfig, *, dtype: mx.Dtype) -> None:
        super().__init__()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            scheme=scheme_of(config, "model.embed_tokens"),
            dtype=dtype,
        )
        self.layers = [
            FlintDecoderLayer(config, layer_id, prefix=f"model.layers.{layer_id}", dtype=dtype)
            for layer_id in range(config.num_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(self, batch: Batch) -> mx.array:
        x = embed_input_ids(self.embed_tokens, batch)
        for layer in self.layers:
            x = layer.forward(x, batch)
        return self.norm.forward(x)


class FlintForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig, *, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        self.config = config
        self.model = FlintModel(config, dtype=dtype)
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
        else:
            return self.lm_head.forward(hidden)


__all__ = ["FlintDecoderLayer", "FlintForCausalLM", "FlintModel"]
