"""Qwen3.5 / 3.6 MoE family: the registry reads parse_config, weight_rules, mlx_layout and the
model class."""

from ftmetal.models.qwen3_5_moe.config import parse_config
from ftmetal.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from ftmetal.models.qwen3_5_moe.weight import mlx_layout, weight_rules

__all__ = ["Qwen3_5MoeForCausalLM", "mlx_layout", "parse_config", "weight_rules"]
