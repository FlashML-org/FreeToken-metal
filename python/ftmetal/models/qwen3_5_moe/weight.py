"""Checkpoint -> model key rules for Qwen3.5 / 3.6 MLX checkpoints.

Port of FreeToken models/qwen3_5_moe/weight.py, with the vision rename of
models/qwen3_vl/weight.py. Divergences: MLX checkpoints already store the norm weights
shifted, the conv weight as [conv_dim, kernel, 1] and the experts stacked per layer, so the
rules are renames and fusions only; the tower is read under the HF model.visual. prefix
only (an mlx-vlm vision_tower. checkpoint is served text only).
"""

from __future__ import annotations

from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import FuseRule, WeightRules

# Checkpoint module paths of the quantization map, per fused model leaf.
QKV_PARTS = ("q_proj", "k_proj", "v_proj")
GATE_UP_PARTS = ("gate_proj", "up_proj")
IN_PROJ_QKVZ_PARTS = ("in_proj_qkv", "in_proj_z")
IN_PROJ_BA_PARTS = ("in_proj_b", "in_proj_a")


def weight_rules(model_config: ModelConfig) -> WeightRules:
    fuses: list[FuseRule] = [
        FuseRule("qkv_proj", QKV_PARTS),
        FuseRule("gate_up_proj", GATE_UP_PARTS),
        FuseRule("in_proj_qkvz", IN_PROJ_QKVZ_PARTS),
        FuseRule("in_proj_ba", IN_PROJ_BA_PARTS),
    ]
    renames = (
        ("language_model.model.", "model."),
        ("language_model.lm_head.", "lm_head."),
        ("model.visual.", "visual."),
    )
    # A tower this process does not build leaves its weights unread.
    drops = ("mtp.", "language_model.mtp.", "vision_tower.")
    if model_config.vision_config is None:
        drops += ("model.visual.",)
    return WeightRules(
        renames=renames,
        segments=(("switch_mlp", "experts"),),
        drops=drops,
        fuses=tuple(fuses),
        expert_segment="experts",
    )


def checkpoint_module(model_key: str) -> str:
    """Model attribute path of a Linear -> its checkpoint module path (unfused, unrenamed)."""
    key = model_key.replace(".experts.", ".switch_mlp.")
    if key.startswith("model.") or key.startswith("lm_head"):
        return "language_model." + key
    return key


__all__ = [
    "GATE_UP_PARTS",
    "IN_PROJ_BA_PARTS",
    "IN_PROJ_QKVZ_PARTS",
    "QKV_PARTS",
    "checkpoint_module",
    "weight_rules",
]
