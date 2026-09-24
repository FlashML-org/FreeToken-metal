"""Qwen3.5 / 3.6 config parsing from an MLX checkpoint's config.json.

Port of FreeToken models/qwen3_5_moe/config.py. Divergences: reads the MLX
checkpoint dictionary (text_config or top level) and its affine quantization map.
"""

from __future__ import annotations

from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)


def layer_types(text: dict[str, object]) -> list[str]:
    types = text.get("layer_types")
    if types is not None:
        return list(types)
    interval = int(text.get("full_attention_interval", 4))
    n = int(text["num_hidden_layers"])
    return ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)]


def parse_config(config: dict[str, object], quant: QuantConfig | None = None) -> ModelConfig:
    text = config.get("text_config", config)
    head_dim = int(text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"])
    num_kv_heads = int(text.get("num_key_value_heads", text["num_attention_heads"]))
    rope_params = text.get("rope_parameters") or {}
    rope_theta = float(rope_params.get("rope_theta", text.get("rope_theta", 10000.0)))
    partial = float(
        rope_params.get("partial_rotary_factor") or text.get("partial_rotary_factor") or 1.0
    )
    rope_type = rope_params.get("type", rope_params.get("rope_type", "default"))
    if rope_type not in (None, "default"):
        raise ValueError(f"unsupported rope type {rope_type!r}")
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(head_dim * partial),
        max_position=int(text["max_position_embeddings"]),
        base=rope_theta,
    )
    types = layer_types(text)
    full_ids = tuple(i for i, t in enumerate(types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(types) if t == "linear_attention")
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=int(text["linear_num_key_heads"]),
        num_value_heads=int(text["linear_num_value_heads"]),
        key_head_dim=int(text["linear_key_head_dim"]),
        value_head_dim=int(text["linear_value_head_dim"]),
        conv_kernel_dim=int(text["linear_conv_kernel_dim"]),
    )
    groups = tuple(g for g in (full_group, linear_group) if g.layer_ids)
    num_experts = int(text.get("num_experts", 0) or 0)
    return ModelConfig(
        num_layers=int(text["num_hidden_layers"]),
        num_qo_heads=int(text["num_attention_heads"]),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=int(text["hidden_size"]),
        vocab_size=int(text["vocab_size"]),
        intermediate_size=int(text.get("intermediate_size", 0) or 0),
        rms_norm_eps=float(text["rms_norm_eps"]),
        rotary_config=rotary,
        hidden_act=str(text.get("hidden_act", "silu")),
        tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
        num_experts=num_experts,
        num_experts_per_tok=int(text.get("num_experts_per_tok", 0) or 0),
        moe_intermediate_size=int(text.get("moe_intermediate_size", 0) or 0),
        norm_topk_prob=bool(text.get("norm_topk_prob", True)),
        model_type=str(config.get("model_type", "qwen3_5_moe")),
        architectures=tuple(config.get("architectures", ("Qwen3_5MoeForCausalLM",))),
        attention_groups=groups,
        quant=quant,
        moe_enabled=num_experts > 0,
        shared_expert_intermediate_size=int(text.get("shared_expert_intermediate_size", 0) or 0),
        use_qk_norm=True,
        attn_output_gate=bool(text.get("attn_output_gate", True)),
    )


__all__ = ["parse_config"]
