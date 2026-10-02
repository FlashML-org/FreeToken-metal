"""Checkpoint -> model key rules for Gemma4 MLX checkpoints.

Port of FreeToken models/gemma4/weight.py. Divergences: MLX checkpoints store the experts
stacked per layer (switch_glu gate / up / down), so the rules are renames and fusions
only; a k_eq_v global layer has no v_proj, so its q and k projections are renamed apart
and fused into qk_proj (its V is the K projection's output) instead of duplicating k
into a qkv v slot; the vision and audio towers are dropped.
"""

from __future__ import annotations

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
TOWER_PREFIXES = (
    "vision_tower.",
    "embed_vision.",
    "vision_embedder.",
    "audio_tower.",
    "embed_audio.",
    "multi_modal_projector.",
)


def checkpoint_prefix(model_config: ModelConfig) -> str:
    """What precedes model.* in the checkpoint's tensor names."""
    if CONDITIONAL_GENERATION in model_config.architectures:
        return LANGUAGE_MODEL_PREFIX
    return ""


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
    return WeightRules(
        renames=tuple(renames),
        segments=(("experts.switch_glu", "experts"),),
        drops=tuple(TOWER_PREFIXES),
        fuses=(
            FuseRule("qkv_proj", QKV_PARTS),
            FuseRule("qk_proj", K_EQ_V_PARTS),
            FuseRule("gate_up_proj", GATE_UP_PARTS),
        ),
    )


__all__ = [
    "GATE_UP_PARTS",
    "QKV_PARTS",
    "QK_PARTS",
    "checkpoint_prefix",
    "k_eq_v_layer_ids",
    "weight_rules",
]
