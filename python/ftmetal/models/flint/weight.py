"""Checkpoint -> model key rules for Flint MLX checkpoints.

Port of FreeToken models/flint/weight.py. Divergences: the rules read the MLX layout, which stores
the experts stacked per layer, so they are renames and fusions only; mlx_layout turns the packed
bytes of the flint-int4 release into the words the quantized matmuls read, and refuses the Hugging
Face layout, so the ModelOpt exports in it too; GGUF is not served.
"""

from __future__ import annotations

import mlx.core as mx

from ftmetal.layers.shared_expert_moe import GATE_UP_PARTS
from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import ExpertBytes, FuseRule, WeightRules

# Checkpoint module paths of the quantization map, per fused model leaf.
QKV_PARTS = ("q_proj", "k_proj", "v_proj")
WEIGHT_SUFFIX = ".weight"
# What a layer's experts sit under in the Hugging Face layout; the MLX layout has switch_mlp.
HF_EXPERTS = ".mlp.experts."


def mlx_layout(
    state: dict[str, mx.array],
) -> tuple[dict[str, mx.array], dict[str, ExpertBytes]]:
    """The checkpoint with its packed weights as uint32 words, which flint-int4 stores as bytes:
    the bytes stay where they are, so no expert moves. Refuses any other layout."""
    if any(HF_EXPERTS in key for key in state):
        raise ValueError(
            "Flint serves MLX layout checkpoints only (mlp.switch_mlp experts), "
            "not the Hugging Face layout or an export in it"
        )
    out: dict[str, mx.array] = {}
    for key, tensor in state.items():
        is_packed_bytes = (
            tensor.dtype == mx.uint8
            and key.endswith(WEIGHT_SUFFIX)
            and f"{key.removesuffix(WEIGHT_SUFFIX)}.scales" in state
        )
        out[key] = tensor.view(mx.uint32) if is_packed_bytes else tensor
    return out, {}


def weight_rules(model_config: ModelConfig) -> WeightRules:
    return WeightRules(
        segments=(("switch_mlp", "experts"),),
        fuses=(FuseRule("qkv_proj", QKV_PARTS), FuseRule("gate_up_proj", GATE_UP_PARTS)),
        expert_segment="experts",
    )


__all__ = ["QKV_PARTS", "mlx_layout", "weight_rules"]
