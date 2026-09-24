"""Architecture name -> model package.

Port of FreeToken models/register.py, reduced to the served families and the two
hooks a package exposes: parse_config and weight_rules.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import mlx.core as mx

from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.models.blocks import BaseLLMModel
from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import WeightRules


@dataclass(frozen=True)
class ModelSpec:
    module: str
    model_cls: str
    parse_config: str = "parse_config"
    weight_rules: str = "weight_rules"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Qwen3_5MoeForConditionalGeneration": ModelSpec(
        "ftmetal.models.qwen3_5_moe", "Qwen3_5MoeForCausalLM"
    ),
    "Qwen3_5MoeForCausalLM": ModelSpec("ftmetal.models.qwen3_5_moe", "Qwen3_5MoeForCausalLM"),
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
    spec = get_model_spec(architecture)
    module = importlib.import_module(spec.module)
    return getattr(module, spec.weight_rules)(model_config)


def create_model(architecture: str, model_config: ModelConfig, dtype: mx.Dtype) -> BaseLLMModel:
    spec = get_model_spec(architecture)
    module = importlib.import_module(spec.module)
    return getattr(module, spec.model_cls)(model_config, dtype=dtype)


__all__ = [
    "MODEL_REGISTRY",
    "ModelSpec",
    "create_model",
    "get_model_spec",
    "model_weight_rules",
    "parse_model_config",
]
