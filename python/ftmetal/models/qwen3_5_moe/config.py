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
from ftmetal.models.loader import (
    config_bool,
    config_dict,
    config_float,
    config_int,
    config_optional_float,
    config_optional_int,
    config_optional_str,
    config_str,
    config_str_list,
)


def layer_types(text: dict[str, object]) -> list[str]:
    types = config_str_list(text, "layer_types")
    if types is not None:
        return types
    interval = config_int(text, "full_attention_interval", 4)
    n = config_int(text, "num_hidden_layers")
    return ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)]


def parse_config(config: dict[str, object], quant: QuantConfig | None = None) -> ModelConfig:
    text = config_dict(config, "text_config") or config
    num_heads = config_int(text, "num_attention_heads")
    head_dim = config_optional_int(text, "head_dim") or config_int(text, "hidden_size") // num_heads
    num_kv_heads = config_int(text, "num_key_value_heads", num_heads)
    rope_params = config_dict(text, "rope_parameters") or {}
    rope_theta = config_optional_float(rope_params, "rope_theta")
    if rope_theta is None:
        rope_theta = config_float(text, "rope_theta", 10000.0)
    partial = (
        config_optional_float(rope_params, "partial_rotary_factor")
        or config_optional_float(text, "partial_rotary_factor")
        or 1.0
    )
    rope_type = config_optional_str(rope_params, "type") or config_optional_str(
        rope_params, "rope_type"
    )
    if rope_type not in (None, "default"):
        raise ValueError(f"unsupported rope type {rope_type!r}")
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(head_dim * partial),
        max_position=config_int(text, "max_position_embeddings"),
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
        num_key_heads=config_int(text, "linear_num_key_heads"),
        num_value_heads=config_int(text, "linear_num_value_heads"),
        key_head_dim=config_int(text, "linear_key_head_dim"),
        value_head_dim=config_int(text, "linear_value_head_dim"),
        conv_kernel_dim=config_int(text, "linear_conv_kernel_dim"),
    )
    groups = tuple(g for g in (full_group, linear_group) if g.layer_ids)
    num_experts = config_optional_int(text, "num_experts") or 0
    architectures = config_str_list(config, "architectures") or ["Qwen3_5MoeForCausalLM"]
    return ModelConfig(
        num_layers=config_int(text, "num_hidden_layers"),
        num_qo_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=config_int(text, "hidden_size"),
        vocab_size=config_int(text, "vocab_size"),
        intermediate_size=config_optional_int(text, "intermediate_size") or 0,
        rms_norm_eps=config_float(text, "rms_norm_eps"),
        rotary_config=rotary,
        hidden_act=config_str(text, "hidden_act", "silu"),
        tie_word_embeddings=config_bool(text, "tie_word_embeddings", False),
        num_experts=num_experts,
        num_experts_per_tok=config_optional_int(text, "num_experts_per_tok") or 0,
        moe_intermediate_size=config_optional_int(text, "moe_intermediate_size") or 0,
        norm_topk_prob=config_bool(text, "norm_topk_prob", True),
        model_type=config_str(config, "model_type", "qwen3_5_moe"),
        architectures=tuple(architectures),
        attention_groups=groups,
        quant=quant,
        moe_enabled=num_experts > 0,
        shared_expert_intermediate_size=config_optional_int(text, "shared_expert_intermediate_size")
        or 0,
        use_qk_norm=True,
        attn_output_gate=config_bool(text, "attn_output_gate", True),
    )


__all__ = ["parse_config"]
