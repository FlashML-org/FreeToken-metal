"""Engine configuration and the model config it derives from the checkpoint.

Port of FreeToken engine/config.py (EngineConfig), reduced to the knobs this engine
has: one process, one GPU, MLX checkpoints.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import mlx.core as mx

from ftmetal.env import read_env
from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import config_dict, config_str_list, read_model_config, weight_files
from ftmetal.models.register import parse_model_config

# Prefix reuse: radix (the hybrid radix on a model with linear attention) or naive (none).
CacheTypeChoice = Literal["naive", "radix"]


def checkpoint_tensor_names(model_path: str) -> list[str]:
    names: list[str] = []
    for file in weight_files(model_path):
        with open(file, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        names.extend(name for name in header if name != "__metadata__")
    return names


def command_graph_from_env() -> bool:
    """FTMETAL_COMMAND_GRAPH: 1 (default) replays decode steps from captured graphs, 0 not."""
    value = read_env("COMMAND_GRAPH")
    if value not in (None, "0", "1"):
        raise ValueError(f"FTMETAL_COMMAND_GRAPH must be 0 or 1, got {value!r}")
    return value != "0"


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    dtype: mx.Dtype = mx.bfloat16
    max_running_req: int = 4
    page_size: int = 1
    # Name in attention.SUPPORTED_ATTENTION_BACKENDS: metal runs the paged flash-prefill and
    # flash-decode kernels, mlx runs MLX SDPA over gathered pages.
    attention_backend: str = "metal"
    max_seq_len_override: int | None = None
    num_page_override: int | None = None
    # Prefill token budget per batch; longer prompts are chunked.
    max_extend_tokens: int = 2048
    # Decode steps between status log lines.
    decode_log_interval: int = 40
    cache_type: CacheTypeChoice = "radix"
    # GDN snapshot cache of the hybrid radix cache: int(ratio * max_running_req) slots (at
    # least 4) on top of the running requests' own slots.
    linear_state_cache_ratio: float = 2.0
    # Capture a decode graph per batch-size bucket at startup and replay decode from it.
    command_graph: bool = field(default_factory=command_graph_from_env)

    @cached_property
    def raw_config(self) -> dict[str, object]:
        return read_model_config(self.model_path)

    @cached_property
    def architecture(self) -> str:
        architectures = config_str_list(self.raw_config, "architectures")
        assert architectures, "config.json names no architecture"
        return architectures[0]

    @cached_property
    def model_config(self) -> ModelConfig:
        quant = QuantConfig.from_checkpoint(
            config_dict(self.raw_config, "quantization"), checkpoint_tensor_names(self.model_path)
        )
        return parse_model_config(self.architecture, self.raw_config, quant)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def model_name(self) -> str:
        return os.path.basename(os.path.normpath(self.model_path))


__all__ = ["CacheTypeChoice", "EngineConfig", "checkpoint_tensor_names", "command_graph_from_env"]
