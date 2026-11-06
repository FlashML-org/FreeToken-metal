"""Checkpoint -> model key rules for Gemma4 checkpoints.

Port of FreeToken models/gemma4/weight.py. Divergences: the rules read the MLX layout, which
stores the experts stacked per layer (switch_glu gate / up / down), so they are renames and
fusions only; mlx_layout rewrites a Hugging Face layout checkpoint into it first; a k_eq_v
global layer has no v_proj, so its q and k projections are renamed apart and fused into
qk_proj (its V is the K projection's output) instead of duplicating k into a qkv v slot; the
vision tower's projections drop their ".linear" level and fuse as the text model's do; the
audio tower is dropped, and so is the vision tower when it is not built.
"""

from __future__ import annotations

import re

import mlx.core as mx

from ftmetal.models.config import FullAttentionGroupConfig, ModelConfig
from ftmetal.models.loader import (
    ExpertBytes,
    FuseRule,
    WeightRules,
    stack_expert_banks,
    stacked_slices,
)

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
# A multimodal Hugging Face checkpoint nests the text model and the towers under model.
HF_TEXT_PREFIX = "model.language_model."
HF_NESTED_PREFIX = "model."
# Each layer's experts: one tensor per expert (an export), or two banks (the bf16 release),
# gate and up stacked on the output axis, gate first, and down.
HF_EXPERT = re.compile(r"\.experts\.(\d+)\.")
HF_GATE_UP_BANK = ".experts.gate_up_proj"
HF_DOWN_BANK = ".experts.down_proj"
SWITCH_GLU = ".experts.switch_glu"


def checkpoint_prefix(model_config: ModelConfig) -> str:
    """What precedes model.* in the checkpoint's tensor names."""
    if CONDITIONAL_GENERATION in model_config.architectures:
        return LANGUAGE_MODEL_PREFIX
    return ""


def mlx_layout(
    state: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, ExpertBytes]]:
    """A Hugging Face layout checkpoint in the MLX layout: a multimodal one's text model under
    language_model.model., its lm_head under language_model. and its towers at the top level,
    each layer's experts as switch_glu banks (one tensor per expert stacked, or the gate_up
    bank split); and where each expert of every bank lies in the checkpoint. A text-only
    checkpoint keeps its other names, and an MLX one, which names nothing rewritten here,
    passes unchanged."""
    is_nested = any(key.startswith(HF_TEXT_PREFIX) for key in state)
    out: dict[str, mx.array] = {}
    expert_bytes: dict[str, ExpertBytes] = {}
    experts: dict[str, dict[int, tuple[str, mx.array]]] = {}
    for hf_key, tensor in state.items():
        if hf_key.startswith(HF_TEXT_PREFIX):
            key = f"{LANGUAGE_MODEL_PREFIX}model.{hf_key.removeprefix(HF_TEXT_PREFIX)}"
        elif is_nested and hf_key.startswith(HF_NESTED_PREFIX):
            key = hf_key.removeprefix(HF_NESTED_PREFIX)
        elif is_nested and hf_key.startswith("lm_head."):
            key = LANGUAGE_MODEL_PREFIX + hf_key
        else:
            key = hf_key
        expert = HF_EXPERT.search(key)
        if expert is not None:
            bank = f"{key[: expert.start()]}{SWITCH_GLU}.{key[expert.end() :]}"
            experts.setdefault(bank, {})[int(expert.group(1))] = (hf_key, tensor)
        elif key.endswith(HF_GATE_UP_BANK):
            switch_glu = key.removesuffix(HF_GATE_UP_BANK) + SWITCH_GLU
            gate, up = f"{switch_glu}.gate_proj.weight", f"{switch_glu}.up_proj.weight"
            out[gate], out[up] = mx.split(tensor, 2, axis=-2)
            expert_bytes[gate] = stacked_slices(hf_key, tensor, 0, 2)
            expert_bytes[up] = stacked_slices(hf_key, tensor, 1, 2)
        elif key.endswith(HF_DOWN_BANK):
            down = f"{key.removesuffix(HF_DOWN_BANK)}{SWITCH_GLU}.down_proj.weight"
            out[down] = tensor
            expert_bytes[down] = stacked_slices(hf_key, tensor)
        else:
            out[key] = tensor
    banks, slices = stack_expert_banks(experts)
    return out | banks, expert_bytes | slices


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
    # Some exports store the tied head as well.
    tied_head = (f"{prefix}lm_head.",) if model_config.tie_word_embeddings else ()
    segments = [("experts.switch_glu", "experts")]
    segments += [(f"{linear}.linear", linear) for linear in VISION_LINEARS]
    return WeightRules(
        renames=tuple(renames),
        segments=tuple(segments),
        drops=drops + tied_head,
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
    "mlx_layout",
    "weight_rules",
]
