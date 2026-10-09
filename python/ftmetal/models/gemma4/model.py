"""Gemma4 decoder: sliding-window and global attention, dense or shared + routed MoE.

Port of FreeToken models/gemma4/model.py (and Gemma4MLP / Gemma4DenseMLP of its moe.py).
Divergences: the batch flows through forward explicitly; the feed-forward sandwich runs
unfused in the decoder layer, which holds its norms, router and experts as the
checkpoint lays them out; each Linear takes its affine scheme from the checkpoint's
quantization map at construction; the final logit softcap runs in the logits' dtype; the
experts are always resident (no offload strategy yet).
"""

from __future__ import annotations

from functools import partial

import mlx.core as mx

from ftmetal.core import Batch
from ftmetal.layers.base import BaseOP
from ftmetal.layers.embedding import Embedding
from ftmetal.layers.linear import Linear
from ftmetal.layers.moe import MoELayer
from ftmetal.layers.norm import RMSNorm
from ftmetal.layers.quantization.config import AffineScheme
from ftmetal.models.blocks import BaseLLMModel, select_last_rows
from ftmetal.models.config import ModelConfig
from ftmetal.models.gemma4.attention import Gemma4Attention
from ftmetal.models.gemma4.moe import Gemma4GatedMLP, Gemma4Router
from ftmetal.models.gemma4.weight import (
    GATE_UP_PARTS,
    QK_PARTS,
    QKV_PARTS,
    checkpoint_prefix,
    k_eq_v_layer_ids,
)


def scheme_of(config: ModelConfig, checkpoint_module: str) -> AffineScheme | None:
    if config.quant is None:
        return None
    return config.quant.scheme_for(checkpoint_module)


def fused_scheme_of(
    config: ModelConfig, parent_module: str, parts: tuple[str, ...]
) -> AffineScheme | None:
    """The scheme of parts (checkpoint module names under parent_module) fused into one."""
    if config.quant is None:
        return None
    return config.quant.scheme_for_fused([f"{parent_module}.{part}" for part in parts])


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
            qkv_scheme=fused_scheme_of(config, f"{prefix}.self_attn", attn_parts),
            o_scheme=scheme_of(config, f"{prefix}.self_attn.o_proj"),
            dtype=dtype,
        )
        self.mlp = Gemma4GatedMLP(
            config.hidden_size,
            config.intermediate_size,
            gate_up_scheme=fused_scheme_of(config, f"{prefix}.mlp", GATE_UP_PARTS),
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
            experts_scheme = fused_scheme_of(config, experts, GATE_UP_PARTS)
            assert experts_scheme == scheme_of(config, f"{experts}.down_proj"), (
                "expert banks share one scheme"
            )
            self.router = Gemma4Router(
                config, proj_scheme=scheme_of(config, f"{prefix}.router.proj"), dtype=dtype
            )
            self.experts = MoELayer(
                config.num_experts,
                config.num_experts_per_tok,
                config.hidden_size,
                config.moe_intermediate_size,
                scheme=experts_scheme,
                activation="gelu_tanh",
                dtype=dtype,
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
        x = self.embed_tokens.forward(batch.input_ids) * self.embedding_scale
        for layer in self.layers:
            x = layer.forward(x, batch)
        return self.norm.forward(x)


class Gemma4ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig, *, dtype: mx.Dtype = mx.bfloat16) -> None:
        if config.moe_strategy == "offload":
            raise ValueError(
                "MoE offload (--moe-strategy offload) is not supported for Gemma4 yet; serve "
                "its experts resident with --moe-strategy fused"
            )
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


__all__ = ["Gemma4DecoderLayer", "Gemma4ForCausalLM", "Gemma4Model"]
