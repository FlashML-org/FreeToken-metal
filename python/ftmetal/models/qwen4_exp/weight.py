"""Checkpoint -> model key rules for Qwen3.8-Flash-Next MLX checkpoints.

Port of FreeToken models/qwen4_exp/weight.py (its dense-tensor renames and fuses).
Divergences: the rules read the MLX layout (zero-centred norms as stored, [C, K, 1] convs,
switch_mlp expert banks), so they are renames and fusions only; the vision tower and the
MTP head are dropped (text only), and so is each PLE layer's n-gram table and its hash
constants: the PLE layer reads them from the checkpoint files itself (ple.py).
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
    return WeightRules(
        renames=(
            ("language_model.model.", "model."),
            ("language_model.lm_head.", "lm_head."),
        ),
        segments=(("switch_mlp", "experts"),),
        drops=("mtp.", "language_model.mtp.", "vision_tower.", *tables),
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
