"""Gemma4 config parsing from an MLX checkpoint's config.json.

Port of FreeToken models/gemma4/config.py. Divergences: reads the MLX checkpoint
dictionary (text_config or top level) and its affine quantization map; vision_config stays
the raw section, which the tower parses; the global layers' kv heads follow the parity
reference (the global count only with attention_k_eq_v); KV sharing and per-layer inputs
(Gemma4-E2B/E4B) and bidirectional attention on every token are refused.
"""

from __future__ import annotations

from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
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

FULL_TYPE = "full_attention"
SLIDING_TYPE = "sliding_attention"


def layer_types(text: dict[str, object]) -> list[str]:
    types = config_str_list(text, "layer_types")
    if types is not None:
        return types
    pattern = config_int(text, "sliding_window_pattern", 5)
    n = config_int(text, "num_hidden_layers")
    return [FULL_TYPE if (i + 1) % pattern == 0 else SLIDING_TYPE for i in range(n)]


def rotary_config(rope_params: dict[str, object], head_dim: int, max_position: int) -> RotaryConfig:
    rope_type = config_str(rope_params, "rope_type", "default")
    partial = config_optional_float(rope_params, "partial_rotary_factor") or 1.0
    if rope_type not in ("default", "proportional"):
        raise ValueError(f"unsupported rope type {rope_type!r}")
    if config_float(rope_params, "factor", 1.0) != 1.0:
        raise ValueError("scaled proportional rope is not served")
    return RotaryConfig(
        head_dim=head_dim,
        rotary_dim=int(head_dim * partial),
        max_position=max_position,
        base=config_float(rope_params, "rope_theta", 10000.0),
        rope_type="proportional" if rope_type == "proportional" else "default",
    )


def parse_config(config: dict[str, object], quant: QuantConfig | None = None) -> ModelConfig:
    text = config_dict(config, "text_config") or config
    if config_int(text, "num_kv_shared_layers", 0) != 0:
        raise ValueError("Gemma4 KV sharing (num_kv_shared_layers) is not served")
    if config_int(text, "hidden_size_per_layer_input", 0) != 0:
        raise ValueError("Gemma4 per-layer inputs (hidden_size_per_layer_input) are not served")
    bidirectional = config_optional_str(text, "use_bidirectional_attention")
    if bidirectional not in (None, "vision"):
        raise ValueError(f"Gemma4 use_bidirectional_attention {bidirectional!r} is not served")
    types = layer_types(text)
    full_ids = tuple(i for i, layer_type in enumerate(types) if layer_type == FULL_TYPE)
    swa_ids = tuple(i for i, layer_type in enumerate(types) if layer_type != FULL_TYPE)
    max_position = config_int(text, "max_position_embeddings")
    rope_params = config_dict(text, "rope_parameters") or {}
    swa_head_dim = config_int(text, "head_dim")
    full_head_dim = config_optional_int(text, "global_head_dim") or swa_head_dim
    swa_kv_heads = config_int(text, "num_key_value_heads")
    k_eq_v = config_bool(text, "attention_k_eq_v", False)
    full_kv_heads = swa_kv_heads
    if k_eq_v:
        full_kv_heads = config_optional_int(text, "num_global_key_value_heads") or swa_kv_heads
    full_rotary = rotary_config(
        config_dict(rope_params, FULL_TYPE) or {}, full_head_dim, max_position
    )
    swa_rotary = rotary_config(
        config_dict(rope_params, SLIDING_TYPE) or {}, swa_head_dim, max_position
    )
    groups = (
        FullAttentionGroupConfig(
            name="full",
            layer_ids=full_ids,
            num_kv_heads=full_kv_heads,
            head_dim=full_head_dim,
            rotary_config=full_rotary,
            k_eq_v=k_eq_v,
        ),
        SWAAttentionGroupConfig(
            name="swa",
            layer_ids=swa_ids,
            num_kv_heads=swa_kv_heads,
            head_dim=swa_head_dim,
            rotary_config=swa_rotary,
            sliding_window=config_int(text, "sliding_window"),
            # Only the window layers: the global layers stay causal over image tokens too.
            bidirectional_mm_blocks=bidirectional == "vision",
        ),
    )
    num_experts = config_optional_int(text, "num_experts") or 0
    hidden_size = config_int(text, "hidden_size")
    architectures = config_str_list(config, "architectures") or ["Gemma4ForCausalLM"]
    return ModelConfig(
        num_layers=config_int(text, "num_hidden_layers"),
        num_qo_heads=config_int(text, "num_attention_heads"),
        num_kv_heads=full_kv_heads,
        head_dim=full_head_dim,
        hidden_size=hidden_size,
        vocab_size=config_int(text, "vocab_size"),
        intermediate_size=config_int(text, "intermediate_size"),
        rms_norm_eps=config_float(text, "rms_norm_eps", 1e-6),
        rotary_config=full_rotary,
        hidden_act="gelu_tanh",
        tie_word_embeddings=config_bool(text, "tie_word_embeddings", True),
        num_experts=num_experts,
        num_experts_per_tok=config_optional_int(text, "top_k_experts") or 0,
        moe_intermediate_size=config_optional_int(text, "moe_intermediate_size") or 0,
        norm_topk_prob=True,
        model_type=config_optional_str(text, "model_type") or "gemma4_text",
        architectures=tuple(architectures),
        attention_groups=tuple(group for group in groups if group.layer_ids),
        quant=quant,
        moe_enabled=config_bool(text, "enable_moe_block", False) and num_experts > 0,
        use_qk_norm=True,
        attn_sm_scale=1.0,
        final_logit_softcapping=config_optional_float(text, "final_logit_softcapping"),
        embedding_scale=float(hidden_size) ** 0.5,
        vision_config=config_dict(config, "vision_config"),
        image_token_id=config_optional_int(config, "image_token_id"),
    )


__all__ = ["FULL_TYPE", "SLIDING_TYPE", "parse_config"]
