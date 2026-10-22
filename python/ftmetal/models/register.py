"""Architecture name -> model package.

Port of FreeToken models/register.py, reduced to the served families, the two hooks a
package exposes (parse_config and weight_rules) and the multimodal processor and encoder
towers. Divergences: an EncoderSpec names the checkpoint prefix of its tower's weights and
the measured activation sizes its memory reserve is computed from.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace

import mlx.core as mx

from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.mm.config import EncoderKind, Modality
from ftmetal.models.blocks import BaseLLMModel
from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import WeightRules, mixed_fuses


@dataclass(frozen=True)
class EncoderSpec:
    # The name --mm-disable takes.
    kind: EncoderKind
    # The checkpoint config section that builds the tower.
    config_key: str
    # The inputs it serves.
    modalities: tuple[Modality, ...]
    # The checkpoint name prefix the tower's weights load from.
    weight_prefix: str
    # Hidden widths per patch at the tower's peak above its weights (engine/cache_budget.py).
    activation_widths: int
    # The rows its MLP runs at once: the tower's MLP_ROWS.
    mlp_rows: int
    # Intermediate widths each of those rows holds.
    mlp_widths: int
    # The peak's bytes beside the per-patch and MLP parts.
    fixed_bytes: int = 0


# Qwen3.6's tower, measured in bf16: 12.2 widths per patch plus 43 MiB above the weights.
QWEN_VISION = EncoderSpec(
    "vision",
    "vision_config",
    ("image",),
    "model.visual.",
    activation_widths=13,
    mlp_rows=4096,
    mlp_widths=2,
)
# Gemma4's tower (its projection loads from embed_vision.), measured on the 26B-A4B in bf16.
GEMMA4_VISION = EncoderSpec(
    "vision",
    "vision_config",
    ("image",),
    "vision_tower.",
    activation_widths=24,
    mlp_rows=2560,
    mlp_widths=3,
    fixed_bytes=160 << 20,
)


@dataclass(frozen=True)
class ModelSpec:
    module: str
    model_cls: str
    parse_config: str = "parse_config"
    weight_rules: str = "weight_rules"
    # "module:Class" turning the checkpoint's images into items; None: the family takes no
    # multimodal input.
    mm_processor: str | None = None
    encoders: tuple[EncoderSpec, ...] = ()


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Qwen3_5MoeForConditionalGeneration": ModelSpec(
        "ftmetal.models.qwen3_5_moe",
        "Qwen3_5MoeForCausalLM",
        mm_processor="ftmetal.mm.processors.qwen_vl:QwenVLMMProcessor",
        encoders=(QWEN_VISION,),
    ),
    "Qwen3_5MoeForCausalLM": ModelSpec("ftmetal.models.qwen3_5_moe", "Qwen3_5MoeForCausalLM"),
    "Gemma4ForConditionalGeneration": ModelSpec(
        "ftmetal.models.gemma4",
        "Gemma4ForCausalLM",
        mm_processor="ftmetal.mm.processors.gemma4:Gemma4MMProcessor",
        encoders=(GEMMA4_VISION,),
    ),
    "Gemma4ForCausalLM": ModelSpec("ftmetal.models.gemma4", "Gemma4ForCausalLM"),
}


def get_model_spec(architecture: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[architecture]
    except KeyError as exc:
        raise ValueError(f"model architecture {architecture} is not supported") from exc


def parse_model_config(
    architecture: str, config: dict[str, object], quant: QuantConfig | None
) -> ModelConfig:
    spec = get_model_spec(architecture)
    module = importlib.import_module(spec.module)
    return getattr(module, spec.parse_config)(config, quant)


def model_weight_rules(architecture: str, model_config: ModelConfig) -> WeightRules:
    """The model's rules, with the fused modules this checkpoint quantizes unevenly unfused."""
    spec = get_model_spec(architecture)
    module = importlib.import_module(spec.module)
    rules: WeightRules = getattr(module, spec.weight_rules)(model_config)
    return replace(rules, unfused=mixed_fuses(rules, model_config.quant))


def create_model(architecture: str, model_config: ModelConfig, dtype: mx.Dtype) -> BaseLLMModel:
    spec = get_model_spec(architecture)
    module = importlib.import_module(spec.module)
    return getattr(module, spec.model_cls)(model_config, dtype=dtype)


__all__ = [
    "GEMMA4_VISION",
    "MODEL_REGISTRY",
    "QWEN_VISION",
    "EncoderSpec",
    "ModelSpec",
    "create_model",
    "get_model_spec",
    "model_weight_rules",
    "parse_model_config",
]
