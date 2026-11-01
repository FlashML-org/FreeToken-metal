"""Qwen3.8-Flash-Next (qwen4_exp) family: the registry reads parse_config, weight_rules and the
model class."""

from ftmetal.models.qwen4_exp.config import parse_config
from ftmetal.models.qwen4_exp.model import Qwen4ExpForCausalLM
from ftmetal.models.qwen4_exp.weight import weight_rules

__all__ = ["Qwen4ExpForCausalLM", "parse_config", "weight_rules"]
