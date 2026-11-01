"""Qwen3.5 / 3.6 hybrid GDN + gated-attention MoE decoder.

Port of FreeToken models/qwen3_5_moe/model.py, with the encode of its
Qwen3_5MoeForConditionalGeneration. Divergences: the batch flows through forward
explicitly; norms are plain RMSNorm because the MLX layout stores the (1 + w) shift; each
Linear takes its scheme from the checkpoint's quantization map at construction; one
class serves both architectures and builds the vision tower when the config has one, in
the engine dtype.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.core import Batch
from ftmetal.layers.base import BaseOP
from ftmetal.layers.embedding import Embedding
from ftmetal.layers.gdn import IN_PROJ_BA_PARTS, IN_PROJ_QKVZ_PARTS, GatedDeltaNet
from ftmetal.layers.linear import Linear
from ftmetal.layers.norm import RMSNorm
from ftmetal.layers.quantization.config import QuantScheme
from ftmetal.layers.quantization.moe import ExpertSchemes
from ftmetal.layers.shared_expert_moe import GATE_UP_PARTS, SharedExpertMoE
from ftmetal.mm.item import MMItem
from ftmetal.models.blocks import BaseLLMModel, embed_input_ids, select_last_rows
from ftmetal.models.config import ModelConfig
from ftmetal.models.qwen3_5_moe.attention import Qwen3_5Attention
from ftmetal.models.qwen3_5_moe.weight import QKV_PARTS, checkpoint_module
from ftmetal.models.qwen3_vl.vision import Qwen3VLVisionModel


def scheme_of(config: ModelConfig, model_key: str) -> QuantScheme | None:
    if config.quant is None:
        return None
    return config.quant.scheme_for(checkpoint_module(model_key))


def schemes_of(
    config: ModelConfig, model_prefix: str, parts: tuple[str, ...]
) -> list[QuantScheme | None]:
    """The scheme of each part of a fused module; they fuse only when all agree."""
    if config.quant is None:
        return [None] * len(parts)
    return config.quant.schemes_for([checkpoint_module(f"{model_prefix}.{part}") for part in parts])


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str, dtype: mx.Dtype) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_linear = config.is_linear_layer(layer_id)
        if self.is_linear:
            group = config.linear_attention_group()
            assert group is not None
            gdn_prefix = f"{prefix}.linear_attn"
            self.linear_attn = GatedDeltaNet(
                config.hidden_size,
                group,
                config.rms_norm_eps,
                layer_id,
                qkvz_schemes=schemes_of(config, gdn_prefix, IN_PROJ_QKVZ_PARTS),
                ba_schemes=schemes_of(config, gdn_prefix, IN_PROJ_BA_PARTS),
                out_scheme=scheme_of(config, f"{gdn_prefix}.out_proj"),
                dtype=dtype,
            )
        else:
            attn_prefix = f"{prefix}.self_attn"
            self.self_attn = Qwen3_5Attention(
                config,
                layer_id,
                qkv_schemes=schemes_of(config, attn_prefix, QKV_PARTS),
                o_scheme=scheme_of(config, f"{attn_prefix}.o_proj"),
                dtype=dtype,
            )
        mlp_prefix = f"{prefix}.mlp"
        self.mlp = SharedExpertMoE(
            config,
            layer_id,
            experts_schemes=ExpertSchemes(
                *schemes_of(config, f"{mlp_prefix}.experts", (*GATE_UP_PARTS, "down_proj")),
                is_gate_up_apart=config.is_expert_gate_up_apart,
            ),
            gate_scheme=scheme_of(config, f"{mlp_prefix}.gate"),
            shared_gate_up_schemes=schemes_of(config, f"{mlp_prefix}.shared_expert", GATE_UP_PARTS),
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
        x = embed_input_ids(self.embed_tokens, batch)
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
        if config.vision_config is not None:
            self.visual = Qwen3VLVisionModel(config.vision_config)
            self.visual.set_dtype(dtype)

    def forward(self, batch: Batch) -> mx.array:
        hidden = select_last_rows(self.model.forward(batch), batch)
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head.forward(hidden)

    def encode(self, item: MMItem) -> mx.array:
        """[item.num_tokens, out_hidden_size] soft tokens of one image."""
        assert item.feature is not None and item.grid_thw is not None
        return self.visual(mx.array(item.feature).view(mx.bfloat16), [item.grid_thw])


__all__ = ["Qwen3_5DecoderLayer", "Qwen3_5Model", "Qwen3_5MoeForCausalLM"]
