"""Checkpoint -> model key rules for Qwen3.8-Flash-Next MLX checkpoints.

Port of FreeToken models/qwen4_exp/weight.py (its dense-tensor renames and fuses).
Divergences: the rules read the MLX layout (zero-centred norms as stored, [C, K, 1] convs,
switch_mlp expert banks), so they are renames and fusions only; the vision tower loads from
the MLX conversion's vision_tower. prefix when it is built, and is dropped otherwise; the MTP
head is dropped, and so is each PLE layer's n-gram table and its hash constants: the PLE
layer reads them from the checkpoint files itself (ple.py).
"""

from __future__ import annotations

from ftmetal.layers.gdn import IN_PROJ_BA_PARTS, IN_PROJ_QKVZ_PARTS
from ftmetal.layers.shared_expert_moe import GATE_UP_PARTS
from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import FuseRule, WeightRules
from ftmetal.models.qwen4_exp.config import Qwen4ExpConfig

QKV_PARTS = ("q_proj", "k_proj", "v_proj")


def ple_table_prefix(layer_id: int) -> str:
    """The checkpoint prefix of a PLE layer's hash constants and table shards."""
    return f"language_model.model.layers.{layer_id}.ple.ple_embedding."


def weight_rules(model_config: ModelConfig) -> WeightRules:
    assert isinstance(model_config, Qwen4ExpConfig)
    tables = tuple(ple_table_prefix(layer_id) for layer_id in model_config.ple_layer_ids)
    # A tower this process does not build leaves its weights unread.
    tower = ("vision_tower.",) if model_config.vision_config is None else ()
    drops = ("mtp.", "language_model.mtp.", *tables, *tower)
    return WeightRules(
        renames=(
            ("language_model.model.", "model."),
            ("language_model.lm_head.", "lm_head."),
            ("vision_tower.", "visual."),
        ),
        segments=(("switch_mlp", "experts"),),
        drops=drops,
        fuses=(
            FuseRule("qkv_proj", QKV_PARTS),
            FuseRule("gate_up_proj", GATE_UP_PARTS),
            FuseRule("in_proj_qkvz", IN_PROJ_QKVZ_PARTS),
            FuseRule("in_proj_ba", IN_PROJ_BA_PARTS),
        ),
        expert_segment="experts",
    )


def checkpoint_module(model_key: str) -> str:
    """Model attribute path of a Linear -> its checkpoint module path (unfused, unrenamed)."""
    key = model_key.replace(".experts.", ".switch_mlp.")
    if key.startswith("model.") or key.startswith("lm_head"):
        return "language_model." + key
    return key


__all__ = ["QKV_PARTS", "checkpoint_module", "ple_table_prefix", "weight_rules"]
