"""Checkpoint -> model key rules for Qwen3.5 / 3.6 checkpoints.

Port of FreeToken models/qwen3_5_moe/weight.py, with the vision rename of
models/qwen3_vl/weight.py. Divergences: the rules read the MLX layout, which stores the norm
weights shifted, the conv weight as [conv_dim, kernel, 1] and the experts stacked per layer,
so they are renames and fusions only; mlx_layout rewrites a Hugging Face layout checkpoint
into it first; the tower is read under the HF model.visual. prefix only (an mlx-vlm
vision_tower. checkpoint is served text only).
"""

from __future__ import annotations

import re

import mlx.core as mx

from ftmetal.layers.gdn import IN_PROJ_BA_PARTS, IN_PROJ_QKVZ_PARTS
from ftmetal.layers.shared_expert_moe import GATE_UP_PARTS
from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import (
    ExpertBytes,
    ExpertSlices,
    FuseRule,
    WeightRules,
    stacked_slices,
)

# Checkpoint module paths of the quantization map, per fused model leaf.
QKV_PARTS = ("q_proj", "k_proj", "v_proj")


# One tensor per expert in the Hugging Face layout.
HF_EXPERT = re.compile(r"\.mlp\.experts\.(\d+)\.")
# Each layer's experts in two banks (the bf16 release, every export's MTP head): gate and up
# stacked on the output axis, gate first, and down.
HF_GATE_UP_BANK = ".mlp.experts.gate_up_proj"
HF_DOWN_BANK = ".mlp.experts.down_proj"
# The norms that scale by (1 + w), stored shifted in the MLX layout; linear attention's gated
# norm scales by w.
SHIFTED_NORMS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "language_model.model.norm.weight",
)


def mlx_layout(
    state: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, ExpertBytes]]:
    """A Hugging Face layout checkpoint in the MLX layout: language_model.* names, each layer's
    experts as switch_mlp banks (one tensor per expert stacked, or a gate_up bank split),
    [conv_dim, kernel, 1] convs and shifted norms; and where each expert of every bank lies in
    the checkpoint. An MLX checkpoint, the one with language_model.* names, passes unchanged."""
    if any(key.startswith("language_model.") for key in state):
        return state, {}
    out: dict[str, mx.array] = {}
    expert_bytes: dict[str, ExpertBytes] = {}
    experts: dict[str, dict[int, tuple[str, mx.array]]] = {}
    for hf_key, tensor in state.items():
        if hf_key.startswith("model.language_model."):
            key = "language_model.model." + hf_key.removeprefix("model.language_model.")
        elif hf_key.startswith(("model.", "lm_head.")) and not hf_key.startswith("model.visual."):
            key = "language_model." + hf_key
        else:
            key = hf_key
        expert = HF_EXPERT.search(key)
        if expert is not None:
            bank = f"{key[: expert.start()]}.mlp.switch_mlp.{key[expert.end() :]}"
            experts.setdefault(bank, {})[int(expert.group(1))] = (hf_key, tensor)
        elif key.endswith(HF_GATE_UP_BANK):
            switch_mlp = key.removesuffix(HF_GATE_UP_BANK) + ".mlp.switch_mlp"
            gate, up = f"{switch_mlp}.gate_proj.weight", f"{switch_mlp}.up_proj.weight"
            out[gate], out[up] = mx.split(tensor, 2, axis=-2)
            expert_bytes[gate] = stacked_slices(hf_key, tensor, 0, 2)
            expert_bytes[up] = stacked_slices(hf_key, tensor, 1, 2)
        elif key.endswith(HF_DOWN_BANK):
            down = key.removesuffix(HF_DOWN_BANK) + ".mlp.switch_mlp.down_proj.weight"
            out[down] = tensor
            expert_bytes[down] = stacked_slices(hf_key, tensor)
        elif key.endswith(".conv1d.weight"):
            out[key] = tensor.moveaxis(2, 1)
        elif key.endswith(SHIFTED_NORMS):
            out[key] = tensor + 1
        else:
            out[key] = tensor
    for bank, by_expert in experts.items():
        names, tensors = zip(*(by_expert[expert_id] for expert_id in range(len(by_expert))))
        out[bank] = mx.stack(tensors)
        expert_bytes[bank] = ExpertSlices(
            names=names, offsets=(0,) * len(names), nbytes=tensors[0].nbytes
        )
    return out, expert_bytes


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
    "QKV_PARTS",
    "checkpoint_module",
    "mlx_layout",
    "weight_rules",
]
