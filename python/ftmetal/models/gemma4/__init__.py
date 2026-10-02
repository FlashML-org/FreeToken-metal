"""Gemma4 family: the registry reads parse_config, weight_rules and the model class."""

from ftmetal.models.gemma4.config import parse_config
from ftmetal.models.gemma4.model import Gemma4ForCausalLM
from ftmetal.models.gemma4.weight import weight_rules

__all__ = ["Gemma4ForCausalLM", "parse_config", "weight_rules"]
