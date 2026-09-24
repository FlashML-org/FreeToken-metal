"""Engine configuration and the model config it derives from the checkpoint.

Port of FreeToken engine/config.py (EngineConfig), reduced to the knobs this engine
has: one process, one GPU, MLX checkpoints.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import mlx.core as mx

from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import read_model_config, weight_files
from ftmetal.models.register import parse_model_config

# Kernel set of the full-attention backend; prefill runs on MLX SDPA under both.
AttentionBackendName = Literal["mlx", "metal"]


def checkpoint_tensor_names(model_path: str) -> list[str]:
    names: list[str] = []
    for file in weight_files(model_path):
        with open(file, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        names.extend(name for name in header if name != "__metadata__")
    return names


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    dtype: mx.Dtype = mx.bfloat16
    max_running_req: int = 4
    page_size: int = 32
    attention_backend: AttentionBackendName = "metal"
    max_seq_len_override: int | None = None
    num_page_override: int | None = None
    # Prefill token budget per batch; longer prompts are chunked.
    max_extend_tokens: int = 2048

    @cached_property
    def raw_config(self) -> dict[str, object]:
        return read_model_config(self.model_path)

    @cached_property
    def architecture(self) -> str:
        return str(self.raw_config["architectures"][0])

    @cached_property
    def model_config(self) -> ModelConfig:
        quant = QuantConfig.from_checkpoint(
            self.raw_config.get("quantization"), checkpoint_tensor_names(self.model_path)
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


__all__ = ["AttentionBackendName", "EngineConfig", "checkpoint_tensor_names"]
