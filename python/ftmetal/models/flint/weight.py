"""Checkpoint -> model key rules for Flint MLX checkpoints.

Port of FreeToken models/flint/weight.py. Divergences: the rules read the MLX layout, which stores
the experts stacked per layer, so they are renames and fusions only; mlx_layout turns the packed
bytes of the flint-int4 release into the words the quantized matmuls read, takes the text model
out from under language_model., where some conversions nest it, and refuses the Hugging Face
layout, so the ModelOpt exports in it too; GGUF is not served.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.layers.shared_expert_moe import GATE_UP_PARTS
from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import ExpertBytes, FuseRule, WeightRules, stacked_slices

# Checkpoint module paths of the quantization map, per fused model leaf.
QKV_PARTS = ("q_proj", "k_proj", "v_proj")
WEIGHT_SUFFIX = ".weight"
# What a layer's experts sit under in the Hugging Face layout and in the MLX layout.
HF_EXPERTS = ".mlp.experts."
MLX_EXPERTS = ".mlp.switch_mlp."
# Some conversions nest the text model's roots under this prefix; the release does not.
LANGUAGE_MODEL_PREFIX = "language_model."
TEXT_ROOTS = ("model", "lm_head")
# Those conversions also store a router bias, all zeros: no Flint config enables one, so a zero
# one is left out and any other refused.
EXPERT_BIAS_SUFFIX = ".mlp.expert_bias"


def bare_name(name: str) -> str:
    """A checkpoint tensor or module name under the release's names: a text model root out from
    under language_model.; any other name as it is."""
    stripped = name.removeprefix(LANGUAGE_MODEL_PREFIX)
    return stripped if stripped.partition(".")[0] in TEXT_ROOTS else name


def mlx_layout(
    state: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, ExpertBytes]]:
    """The checkpoint under the release's names (bare_name), without the zero router bias, and
    with its packed weights as uint32 words, which flint-int4 stores as bytes: the bytes stay
    where they are, so no expert moves; and where each expert of a renamed bank lies in the
    checkpoint. Refuses any other layout, a nonzero router bias and a tensor stored under both
    names."""
    if any(HF_EXPERTS in key for key in state):
        raise ValueError(
            "Flint serves MLX layout checkpoints only (mlp.switch_mlp experts), "
            "not the Hugging Face layout or an export in it"
        )
    out: dict[str, mx.array] = {}
    expert_bytes: dict[str, ExpertBytes] = {}
    for key, tensor in state.items():
        if key.endswith(EXPERT_BIAS_SUFFIX):
            if mx.any(tensor != 0).item():
                raise ValueError(f"{key}: a nonzero router expert bias is not served")
            continue
        name = bare_name(key)
        if name in out:
            raise ValueError(f"{name}: the checkpoint stores it both under language_model. and not")
        is_packed_bytes = (
            tensor.dtype == mx.uint8
            and key.endswith(WEIGHT_SUFFIX)
            and f"{key.removesuffix(WEIGHT_SUFFIX)}.scales" in state
        )
        out[name] = tensor.view(mx.uint32) if is_packed_bytes else tensor
        if name != key and MLX_EXPERTS in key:
            expert_bytes[name] = stacked_slices(key, tensor)
    return out, expert_bytes


def weight_rules(model_config: ModelConfig) -> WeightRules:
    return WeightRules(
        segments=(("switch_mlp", "experts"),),
        fuses=(FuseRule("qkv_proj", QKV_PARTS), FuseRule("gate_up_proj", GATE_UP_PARTS)),
        expert_segment="experts",
    )


__all__ = ["QKV_PARTS", "bare_name", "mlx_layout", "weight_rules"]
