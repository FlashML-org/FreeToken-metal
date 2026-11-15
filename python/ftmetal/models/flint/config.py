"""Flint config parsing from an MLX checkpoint's config.json.

Port of FreeToken models/flint/config.py. Divergences: reads the checkpoint dictionary and its MLX
scheme map, in which the release's export section also declares the embedding's bits (its
quantization section names only the default) and a conversion that nests the text model under
language_model. names its modules with that prefix; no GGUF metadata shim.
"""

from __future__ import annotations

from dataclasses import replace

from ftmetal.layers.quantization.config import AffineScheme, QuantConfig
from ftmetal.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    SWAAttentionGroupConfig,
)
from ftmetal.models.flint.weight import bare_name
from ftmetal.models.loader import (
    config_bool,
    config_dict,
    config_float,
    config_int,
    config_optional_float,
    config_optional_int,
    config_str,
    config_str_list,
)

FULL_TYPE = "full_attention"
SLIDING_TYPE = "sliding_attention"
EMBEDDING_MODULE = "model.embed_tokens"
# The release's config names the router's score under both.
ROUTER_SCORE_KEYS = ("router_score_function", "scoring_func")
# A prefill step's peak above the pools, an upper line through flint-int4's measured 170, 341, 488
# and 691 MiB at chunks of 256, 512, 1024 and 2048 tokens (M4 Pro, 2026-11-06).
PREFILL_ACTIVATION_FIXED_BYTES = 256 << 20
PREFILL_ACTIVATION_BYTES_PER_TOKEN = 256 << 10


def config_layer_ids(config: dict[str, object], key: str) -> tuple[int, ...]:
    value = config.get(key)
    if not isinstance(value, list) or not all(isinstance(layer_id, int) for layer_id in value):
        raise ValueError(f"config {key!r} must be a list of layer indices, got {value!r}")
    return tuple(value)


def with_embedding_bits(config: dict[str, object], quant: QuantConfig | None) -> QuantConfig | None:
    """quant with the embedding's scheme from the export section, unless the checkpoint's
    quantization section names one."""
    export = config_dict(config, "mlx_export")
    embedding_bits = None if export is None else config_optional_int(export, "embedding_bits")
    if (
        quant is None
        or embedding_bits is None
        or EMBEDDING_MODULE in quant.overrides
        or not isinstance(quant.default, AffineScheme)
    ):
        return quant
    else:
        scheme = AffineScheme(embedding_bits, quant.default.group_size)
        return replace(quant, overrides={**quant.overrides, EMBEDDING_MODULE: scheme})


def parse_config(config: dict[str, object], quant: QuantConfig | None = None) -> ModelConfig:
    num_layers = config_int(config, "num_hidden_layers")
    types = config_str_list(config, "layer_types") or []
    if len(types) != num_layers or set(types) - {FULL_TYPE, SLIDING_TYPE}:
        raise ValueError("Flint requires one full_attention or sliding_attention type per layer")
    full_ids = tuple(i for i, layer_type in enumerate(types) if layer_type == FULL_TYPE)
    swa_ids = tuple(i for i, layer_type in enumerate(types) if layer_type == SLIDING_TYPE)
    if not full_ids or not swa_ids:
        raise ValueError("Flint requires both full and sliding attention layers")
    if set(config_layer_ids(config, "no_rope_layer_indices")) != set(full_ids):
        raise ValueError("Flint serves rope on its sliding layers and none on its full layers")
    dense_ids = config_layer_ids(config, "mlp_only_layers")
    if dense_ids != tuple(range(len(dense_ids))) or len(dense_ids) >= num_layers:
        raise ValueError("Flint's dense MLP layers must be a leading run of its layers")
    if config_int(config, "decoder_sparse_step", 1) != 1:
        raise ValueError("Flint requires a MoE block in every layer after its dense ones")
    if config_str(config, "hidden_act", "silu") != "silu":
        raise ValueError("Flint requires SwiGLU experts")
    if {config_str(config, key, "sigmoid") for key in ROUTER_SCORE_KEYS} != {"sigmoid"}:
        raise ValueError("Flint requires a sigmoid router")
    if config_bool(config, "router_expert_bias", False):
        raise ValueError("Flint's router expert bias is not served")
    if config_str(config, "softmax_type", "learnable") != "learnable":
        raise ValueError("Flint requires learned attention sinks")
    if config_bool(config, "attention_bias", False):
        raise ValueError("Flint requires bias-free projections")
    head_dim = config_int(config, "head_dim")
    num_kv_heads = config_int(config, "num_key_value_heads")
    rope_params = config_dict(config, "rope_parameters") or {}
    rope_type = config_str(rope_params, "rope_type", "default")
    if rope_type != "default":
        raise ValueError(f"unsupported rope type {rope_type!r}")
    rope_theta = config_optional_float(rope_params, "rope_theta")
    swa_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=head_dim,
        max_position=config_int(config, "max_position_embeddings"),
        base=config_float(config, "rope_theta", 10000.0) if rope_theta is None else rope_theta,
    )
    architectures = config_str_list(config, "architectures") or ["FlintForCausalLM"]
    # The section names modules as the checkpoint does; mlx_layout gave them bare names.
    bare_quant = (
        None
        if quant is None
        else replace(
            quant,
            overrides={bare_name(module): scheme for module, scheme in quant.overrides.items()},
        )
    )
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=config_int(config, "num_attention_heads"),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=config_int(config, "hidden_size"),
        vocab_size=config_int(config, "vocab_size"),
        intermediate_size=config_int(config, "intermediate_size"),
        rms_norm_eps=config_float(config, "rms_norm_eps", 1e-6),
        rotary_config=swa_rotary,
        hidden_act="silu",
        tie_word_embeddings=config_bool(config, "tie_word_embeddings", False),
        num_experts=config_int(config, "num_experts"),
        num_experts_per_tok=config_int(config, "num_experts_per_tok"),
        moe_intermediate_size=config_int(config, "moe_intermediate_size"),
        norm_topk_prob=config_bool(config, "norm_topk_prob", True),
        model_type=config_str(config, "model_type", "flint"),
        architectures=tuple(architectures),
        attention_groups=(
            SWAAttentionGroupConfig(
                name="swa",
                layer_ids=swa_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=swa_rotary,
                sliding_window=config_int(config, "sliding_window"),
            ),
            FullAttentionGroupConfig(
                name="full",
                layer_ids=full_ids,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_config=replace(swa_rotary, rotary_dim=0),
            ),
        ),
        quant=with_embedding_bits(config, bare_quant),
        moe_enabled=True,
        shared_expert_intermediate_size=config_int(config, "shared_expert_intermediate_size"),
        first_k_dense_replace=len(dense_ids),
        is_sigmoid_router=True,
        has_shared_expert_gate=False,
        use_qk_norm=True,
        prefill_activation_bytes=(
            PREFILL_ACTIVATION_FIXED_BYTES,
            PREFILL_ACTIVATION_BYTES_PER_TOKEN,
        ),
    )


__all__ = ["parse_config"]
