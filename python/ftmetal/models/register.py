"""Architecture name -> model package.

Port of FreeToken models/register.py, reduced to the served families, the hooks a package
exposes (parse_config, weight_rules and, for a family served from Hugging Face layout
checkpoints too, mlx_layout) and the multimodal processor and encoder towers. Divergences: an
EncoderSpec names the checkpoint prefixes of its tower's weights and the measured activation
sizes its memory reserve is computed from; read_checkpoint brings every checkpoint into the
MLX layout and quantization the weight rules and methods read.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace

import mlx.core as mx

from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.mm.config import EncoderKind, Modality
from ftmetal.models.blocks import BaseLLMModel
from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import (
    ExpertBytes,
    WeightRules,
    config_dict,
    config_str,
    load_checkpoint,
    mixed_fuses,
    read_model_config,
)
from ftmetal.models.quant_export import rewrite_export


@dataclass(frozen=True)
class EncoderSpec:
    # The name --mm-disable takes.
    kind: EncoderKind
    # The checkpoint config section that builds the tower.
    config_key: str
    # The inputs it serves.
    modalities: tuple[Modality, ...]
    # The checkpoint name prefixes the tower's weights load from, one per layout that differs.
    weight_prefixes: tuple[str, ...]
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
    ("model.visual.",),
    activation_widths=13,
    mlp_rows=4096,
    mlp_widths=2,
)
# Qwen3.8-Flash-Next's tower: Qwen3.6's geometry (its merger 2560 wide, not 2048), so its
# measured widths, loaded from the MLX conversion's vision_tower. prefix.
QWEN4_EXP_VISION = replace(QWEN_VISION, weight_prefixes=("vision_tower.",))
# Gemma4's tower (its projection loads from embed_vision.), measured on the 26B-A4B in bf16.
GEMMA4_VISION = EncoderSpec(
    "vision",
    "vision_config",
    ("image",),
    ("vision_tower.", "model.vision_tower."),
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
    # A function rewriting a lazily mapped checkpoint into the MLX layout the weight rules
    # read, with where each expert of the tensors it stacks, splits or renames lies in the
    # checkpoint; None: the family loads MLX layout checkpoints only.
    mlx_layout: str | None = None
    # "module:Class" turning the checkpoint's images into items; None: the family takes no
    # multimodal input.
    mm_processor: str | None = None
    # Whether the family serves its routed experts from an offload cache.
    has_offload: bool = False
    encoders: tuple[EncoderSpec, ...] = ()


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Qwen3_5MoeForConditionalGeneration": ModelSpec(
        "ftmetal.models.qwen3_5_moe",
        "Qwen3_5MoeForCausalLM",
        mlx_layout="mlx_layout",
        mm_processor="ftmetal.mm.processors.qwen_vl:QwenVLMMProcessor",
        encoders=(QWEN_VISION,),
        has_offload=True,
    ),
    "Qwen3_5MoeForCausalLM": ModelSpec(
        "ftmetal.models.qwen3_5_moe",
        "Qwen3_5MoeForCausalLM",
        mlx_layout="mlx_layout",
        has_offload=True,
    ),
    "Gemma4ForConditionalGeneration": ModelSpec(
        "ftmetal.models.gemma4",
        "Gemma4ForCausalLM",
        mlx_layout="mlx_layout",
        mm_processor="ftmetal.mm.processors.gemma4:Gemma4MMProcessor",
        encoders=(GEMMA4_VISION,),
        has_offload=True,
    ),
    "Gemma4ForCausalLM": ModelSpec(
        "ftmetal.models.gemma4",
        "Gemma4ForCausalLM",
        mlx_layout="mlx_layout",
        has_offload=True,
    ),
    "Qwen4ExpForConditionalGeneration": ModelSpec(
        "ftmetal.models.qwen4_exp",
        "Qwen4ExpForCausalLM",
        mm_processor="ftmetal.mm.processors.qwen_vl:QwenVLMMProcessor",
        encoders=(QWEN4_EXP_VISION,),
        has_offload=True,
    ),
    "Qwen4ExpForCausalLM": ModelSpec(
        "ftmetal.models.qwen4_exp", "Qwen4ExpForCausalLM", has_offload=True
    ),
    "FlintForCausalLM": ModelSpec(
        "ftmetal.models.flint", "FlintForCausalLM", mlx_layout="mlx_layout", has_offload=True
    ),
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


@dataclass(frozen=True, kw_only=True)
class Checkpoint:
    # Every tensor, lazily mapped, in the MLX layout and quantization.
    state: dict[str, mx.array]
    quant: QuantConfig
    # Where each expert of a stacked expert tensor of state lies in the checkpoint, for the
    # tensors not stored as they are in state.
    expert_bytes: dict[str, ExpertBytes]


def read_checkpoint(architecture: str, model_path: str) -> Checkpoint:
    """The checkpoint as an MLX one: a Hugging Face layout rewritten by the family's
    mlx_layout, then a ModelOpt or compressed-tensors export's tensors by rewrite_export."""
    spec = get_model_spec(architecture)
    config = read_model_config(model_path)
    state = load_checkpoint(model_path)
    expert_bytes: dict[str, ExpertBytes] = {}
    if spec.mlx_layout is not None:
        mlx_layout = getattr(importlib.import_module(spec.module), spec.mlx_layout)
        state, expert_bytes = mlx_layout(state)
    quantization = config_dict(config, "quantization")
    export_config = config_dict(config, "quantization_config")
    if quantization is not None or export_config is None:
        quant = QuantConfig.from_checkpoint(quantization, list(state))
        return Checkpoint(state=state, quant=quant, expert_bytes=expert_bytes)
    elif spec.mlx_layout is None:
        method = config_str(export_config, "quant_method", "")
        raise ValueError(f"{architecture} serves MLX checkpoints only, not a {method!r} export")
    else:
        rewritten, schemes, expert_bytes = rewrite_export(state, export_config, expert_bytes)
        quant = QuantConfig.from_export(schemes)
        return Checkpoint(state=rewritten, quant=quant, expert_bytes=expert_bytes)


def create_model(architecture: str, model_config: ModelConfig, dtype: mx.Dtype) -> BaseLLMModel:
    spec = get_model_spec(architecture)
    module = importlib.import_module(spec.module)
    return getattr(module, spec.model_cls)(model_config, dtype=dtype)


__all__ = [
    "GEMMA4_VISION",
    "MODEL_REGISTRY",
    "QWEN4_EXP_VISION",
    "QWEN_VISION",
    "Checkpoint",
    "EncoderSpec",
    "ModelSpec",
    "create_model",
    "get_model_spec",
    "model_weight_rules",
    "parse_model_config",
    "read_checkpoint",
]
