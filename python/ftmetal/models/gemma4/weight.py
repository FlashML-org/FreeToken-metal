"""Checkpoint -> model key rules for Gemma4 MLX checkpoints.

Port of FreeToken models/gemma4/weight.py. Divergences: MLX checkpoints store the experts
stacked per layer (switch_glu gate / up / down), so the rules are renames and fusions
only; a k_eq_v global layer has no v_proj, so its q and k projections are renamed apart
and fused into qk_proj (its V is the K projection's output) instead of duplicating k
into a qkv v slot; the vision tower's projections drop their ".linear" level and fuse as
the text model's do; the audio tower is dropped, and so is the vision tower when it is not
built.
"""

from __future__ import annotations

from ftmetal.layers.quantization.config import QuantScheme
from ftmetal.models.config import FullAttentionGroupConfig, ModelConfig
from ftmetal.models.loader import FuseRule, WeightRules

# The multimodal checkpoint nests the text model under this prefix; a text-only one does not.
CONDITIONAL_GENERATION = "Gemma4ForConditionalGeneration"
LANGUAGE_MODEL_PREFIX = "language_model."
# Checkpoint module paths of the quantization map, per fused model leaf.
QKV_PARTS = ("q_proj", "k_proj", "v_proj")
QK_PARTS = ("q_proj", "k_proj")
GATE_UP_PARTS = ("gate_proj", "up_proj")
# A k_eq_v layer's q_proj / k_proj, renamed so the qkv fusion does not claim them.
K_EQ_V_PARTS = ("k_eq_v_q_proj", "k_eq_v_k_proj")
# The vision tower's weights; the model holds them under the same names.
VISION_PREFIXES = ("vision_tower.", "embed_vision.")
# Towers this engine never builds.
UNSERVED_PREFIXES = ("vision_embedder.", "audio_tower.", "embed_audio.", "multi_modal_projector.")
# The vision tower's projections, stored one level down (their clipping wrapper's linear).
VISION_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def checkpoint_prefix(model_config: ModelConfig) -> str:
    """What precedes model.* in the checkpoint's tensor names."""
    if CONDITIONAL_GENERATION in model_config.architectures:
        return LANGUAGE_MODEL_PREFIX
    return ""


def scheme_of(config: ModelConfig, checkpoint_module: str) -> QuantScheme | None:
    if config.quant is None:
        return None
    return config.quant.scheme_for(checkpoint_module)


def schemes_of(
    config: ModelConfig, parent_module: str, parts: tuple[str, ...]
) -> list[QuantScheme | None]:
    """The scheme of each part (a checkpoint module name under parent_module) of a fused
    module; they fuse only when all agree."""
    if config.quant is None:
        return [None] * len(parts)
    return config.quant.schemes_for([f"{parent_module}.{part}" for part in parts])


def k_eq_v_layer_ids(model_config: ModelConfig) -> tuple[int, ...]:
    group = model_config.full_attention_group()
    if isinstance(group, FullAttentionGroupConfig) and group.k_eq_v:
        return group.layer_ids
    return ()


def weight_rules(model_config: ModelConfig) -> WeightRules:
    prefix = checkpoint_prefix(model_config)
    renames: list[tuple[str, str]] = []
    for layer_id in k_eq_v_layer_ids(model_config):
        attn = f"model.layers.{layer_id}.self_attn"
        for part, renamed in zip(QK_PARTS, K_EQ_V_PARTS):
            renames.append((f"{prefix}{attn}.{part}.", f"{attn}.{renamed}."))
    if prefix:
        renames += [(f"{prefix}model.", "model."), (f"{prefix}lm_head.", "lm_head.")]
    drops = (
        UNSERVED_PREFIXES
        if model_config.vision_config is not None
        else UNSERVED_PREFIXES + VISION_PREFIXES
    )
    segments = [("experts.switch_glu", "experts")]
    segments += [(f"{linear}.linear", linear) for linear in VISION_LINEARS]
    return WeightRules(
        renames=tuple(renames),
        segments=tuple(segments),
        drops=drops,
        fuses=(
            FuseRule("qkv_proj", QKV_PARTS),
            FuseRule("qk_proj", K_EQ_V_PARTS),
            FuseRule("gate_up_proj", GATE_UP_PARTS),
        ),
        expert_segment="experts",
    )


__all__ = [
    "GATE_UP_PARTS",
    "K_EQ_V_PARTS",
    "QKV_PARTS",
    "QK_PARTS",
    "checkpoint_prefix",
    "k_eq_v_layer_ids",
    "scheme_of",
    "schemes_of",
    "weight_rules",
]
