"""Gemma4 decoder: sliding-window and global attention, dense or shared + routed MoE.

Port of FreeToken models/gemma4/model.py (and Gemma4MLP / Gemma4DenseMLP of its moe.py),
with the encode of its Gemma4ForConditionalGeneration. Divergences: the batch flows through
forward explicitly; the feed-forward sandwich runs unfused in the decoder layer, which holds
its norms, router and experts as the checkpoint lays them out; each Linear takes its scheme
from the checkpoint's quantization map at construction; the final logit softcap runs in the
logits' dtype; under the offload strategy the routed experts are an OffloadMoELayer (every
layer routes, so its MoE layer index is the layer's) and the dense MLP stays apart from it; one
class serves both architectures and builds the vision tower when the config has one, in the
engine dtype.
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx

from ftmetal.core import Batch
from ftmetal.layers.base import BaseOP
from ftmetal.layers.embedding import Embedding
from ftmetal.layers.linear import Linear
from ftmetal.layers.moe import MoELayer, OffloadMoELayer
from ftmetal.layers.norm import RMSNorm
from ftmetal.layers.quantization.moe import ExpertSchemes
from ftmetal.mm.item import MMItem
from ftmetal.models.blocks import BaseLLMModel, embed_input_ids, select_last_rows
from ftmetal.models.config import ModelConfig, scheme_of, schemes_of
from ftmetal.models.gemma4.attention import Gemma4Attention
from ftmetal.models.gemma4.moe import Gemma4GatedMLP, Gemma4Router
from ftmetal.models.gemma4.vision import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
    parse_vision_config,
)
from ftmetal.models.gemma4.weight import (
    GATE_UP_PARTS,
    QK_PARTS,
    QKV_PARTS,
    checkpoint_prefix,
    k_eq_v_layer_ids,
)


@partial(mx.compile, shapeless=True)
def logit_softcap(logits: mx.array, cap: mx.array) -> mx.array:
    return mx.tanh(logits / cap) * cap


class Gemma4DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str, dtype: mx.Dtype) -> None:
        """prefix: the layer's checkpoint module path."""
        super().__init__()
        self.layer_id = layer_id
        self.is_moe = config.is_moe
        attn_parts = QK_PARTS if layer_id in k_eq_v_layer_ids(config) else QKV_PARTS
        self.self_attn = Gemma4Attention(
            config,
            layer_id,
            qkv_schemes=schemes_of(config, f"{prefix}.self_attn", attn_parts),
            o_scheme=scheme_of(config, f"{prefix}.self_attn.o_proj"),
            dtype=dtype,
        )
        self.mlp = Gemma4GatedMLP(
            config.hidden_size,
            config.intermediate_size,
            gate_up_schemes=schemes_of(config, f"{prefix}.mlp", GATE_UP_PARTS),
            down_scheme=scheme_of(config, f"{prefix}.mlp.down_proj"),
            dtype=dtype,
        )
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps=eps, dtype=dtype)
        if self.is_moe:
            experts = f"{prefix}.experts.switch_glu"
            self.router = Gemma4Router(
                config, proj_scheme=scheme_of(config, f"{prefix}.router.proj"), dtype=dtype
            )
            schemes = ExpertSchemes(
                *schemes_of(config, experts, (*GATE_UP_PARTS, "down_proj")),
                is_gate_up_apart=config.is_expert_gate_up_apart,
            )
            shape = (
                config.num_experts,
                config.num_experts_per_tok,
                config.hidden_size,
                config.moe_intermediate_size,
            )
            self.experts: MoELayer
            if config.moe_strategy == "offload":
                self.experts = OffloadMoELayer(
                    layer_id, *shape, schemes=schemes, activation="gelu_tanh", dtype=dtype
                )
            else:
                self.experts = MoELayer(
                    *shape, schemes=schemes, activation="gelu_tanh", dtype=dtype
                )
            self.pre_feedforward_layernorm_2 = RMSNorm(hidden, eps=eps, dtype=dtype)
            self.post_feedforward_layernorm_1 = RMSNorm(hidden, eps=eps, dtype=dtype)
            self.post_feedforward_layernorm_2 = RMSNorm(hidden, eps=eps, dtype=dtype)
        self.layer_scalar = mx.ones((1,), dtype=dtype)

    def forward(self, x: mx.array, batch: Batch) -> mx.array:
        attended = self.self_attn.forward(self.input_layernorm.forward(x), batch)
        h = x + self.post_attention_layernorm.forward(attended)
        if self.is_moe:
            shared = self.mlp.forward(self.pre_feedforward_layernorm.forward(h))
            topk_ids, topk_weights = self.router.forward(h)
            routed = self.experts.forward(
                self.pre_feedforward_layernorm_2.forward(h), topk_weights, topk_ids
            )
            ff = self.post_feedforward_layernorm_1.forward(shared)
            ff = ff + self.post_feedforward_layernorm_2.forward(routed)
        else:
            ff = self.mlp.forward(self.pre_feedforward_layernorm.forward(h))
        return (h + self.post_feedforward_layernorm.forward(ff)) * self.layer_scalar


class Gemma4Model(BaseOP):
    def __init__(self, config: ModelConfig, *, dtype: mx.Dtype) -> None:
        super().__init__()
        prefix = f"{checkpoint_prefix(config)}model"
        assert config.embedding_scale is not None
        self.embedding_scale = config.embedding_scale
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            scheme=scheme_of(config, f"{prefix}.embed_tokens"),
            dtype=dtype,
        )
        self.layers = [
            Gemma4DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}", dtype=dtype)
            for layer_id in range(config.num_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(self, batch: Batch) -> mx.array:
        x = embed_input_ids(self.embed_tokens, batch, self.embedding_scale)
        for layer in self.layers:
            x = layer.forward(x, batch)
        return self.norm.forward(x)


class Gemma4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig, *, dtype: mx.Dtype = mx.bfloat16) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config, dtype=dtype)
        if not config.tie_word_embeddings:
            self.lm_head = Linear(
                config.hidden_size,
                config.vocab_size,
                scheme=scheme_of(config, f"{checkpoint_prefix(config)}lm_head"),
                dtype=dtype,
            )
        if config.vision_config is not None:
            vision = parse_vision_config(config.vision_config)
            self.vision_tower = Gemma4VisionModel(config, vision, dtype=dtype)
            self.embed_vision = Gemma4MultimodalEmbedder(config, vision, dtype=dtype)

    def forward(self, batch: Batch) -> mx.array:
        hidden = select_last_rows(self.model.forward(batch), batch)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden)
        else:
            logits = self.lm_head.forward(hidden)
        cap = self.config.final_logit_softcapping
        if cap is None:
            return logits
        return logit_softcap(logits, mx.array(cap, dtype=logits.dtype))

    def encode(self, item: MMItem) -> mx.array:
        """[item.num_soft_tokens, hidden] text-space embeddings of one image."""
        assert item.feature is not None and item.position_ids is not None
        assert item.num_soft_tokens is not None
        soft_tokens = self.vision_tower.forward(
            mx.array(item.feature), item.position_ids, item.num_soft_tokens
        )
        return self.embed_vision.forward(soft_tokens)


__all__ = ["Gemma4DecoderLayer", "Gemma4ForCausalLM", "Gemma4Model"]
