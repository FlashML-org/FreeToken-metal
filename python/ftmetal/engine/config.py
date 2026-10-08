"""Engine configuration and the model config it derives from the checkpoint.

Port of FreeToken engine/config.py (EngineConfig), reduced to the knobs this engine
has: one process, one GPU, MLX checkpoints. Divergence: served_modalities reads the
encoders from mm.processor.active_encoders (no active_encoders property) and is empty for
a checkpoint without config.json; model_config drops the unbuilt encoder sections from the
raw config dictionary.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

import mlx.core as mx

from ftmetal.env import read_env
from ftmetal.layers.quantization.config import QuantConfig
from ftmetal.mm.config import ENCODER_SECTIONS, Modality, MultimodalConfig
from ftmetal.mm.processor import active_encoders
from ftmetal.models.config import ModelConfig
from ftmetal.models.loader import (
    checkpoint_tensor_names,
    config_dict,
    config_str_list,
    read_model_config,
)
from ftmetal.models.register import parse_model_config

# Prefix reuse: radix (the hybrid radix on a model with linear attention) or naive (none).
CacheTypeChoice = Literal["naive", "radix"]


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
    # GDN snapshot cache of the hybrid radix cache, in slots on top of the running requests'
    # own, at least one conversation's two; None: kvcache.linear_state_pool's default.
    gdn_snapshots: int | None = None
    # Window pool of the SWA radix cache, in sliding windows, at least what the running
    # requests and a prefill hold; None: kvcache.swa_pool's default, that floor and 8 more.
    swa_windows: int | None = None
    # Capture a decode graph per batch-size bucket at startup and replay decode from it.
    command_graph: bool = field(default_factory=command_graph_from_env)
    # Runtime knobs of the multimodal path; the architecture side lives in the checkpoint
    # config.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)

    @cached_property
    def raw_config(self) -> dict[str, object]:
        return read_model_config(self.model_path)

    @cached_property
    def served_modalities(self) -> frozenset[Modality]:
        """Modalities this process accepts."""
        if not (Path(self.model_path) / "config.json").is_file():
            return frozenset()
        else:
            encoders = active_encoders(self.model_path, self.raw_config, self.mm)
            return frozenset(modality for encoder in encoders for modality in encoder.modalities)

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
        # The parser sees no section of a tower this process does not build (for the vision
        # tower that also means 1-D rope).
        encoders = active_encoders(self.model_path, self.raw_config, self.mm)
        built = {encoder.config_key for encoder in encoders}
        raw_config = {
            key: value
            for key, value in self.raw_config.items()
            if key in built or key not in ENCODER_SECTIONS
        }
        return parse_model_config(self.architecture, raw_config, quant)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def model_name(self) -> str:
        return os.path.basename(os.path.normpath(self.model_path))


__all__ = ["CacheTypeChoice", "EngineConfig", "command_graph_from_env"]
